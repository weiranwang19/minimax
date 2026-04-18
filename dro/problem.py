from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dro.models import classifier_logits
from utils import compute_norm


def identity_prox(v, coeff):
    del coeff
    return v


def positive_eta_prox(eta_min):
    def prox(v, coeff):
        del coeff
        return torch.clamp(v, min=eta_min)

    return prox


def simplex_prox(project_onto_simplex):
    def prox(v, coeff):
        del coeff
        return project_onto_simplex(v, dim=-1)

    return prox


@dataclass
class DROBlockView:
    backbone_params: list
    eta: torch.Tensor
    classifier: torch.Tensor
    train_simplex: torch.Tensor
    val_simplex: torch.Tensor
    classifier_copy: torch.Tensor
    train_simplex_copy: torch.Tensor


class CelebADROProblem:
    def __init__(
        self,
        backbone,
        num_groups,
        train_provider,
        val_provider,
        train_monitor_batches,
        val_monitor_batches,
        rho,
        device,
    ):
        self.backbone = backbone
        self.num_groups = num_groups
        self.train_provider = train_provider
        self.val_provider = val_provider
        self.train_monitor_batches = train_monitor_batches
        self.val_monitor_batches = val_monitor_batches
        self.rho = rho
        self.device = device
        self.backbone_param_count = len(list(backbone.parameters()))
        self.uniform_weights = torch.full((num_groups,), 1.0 / num_groups, device=device)
        self.backbone.eval()

    def _view(self, min_block, max_block):
        if len(min_block) != self.backbone_param_count + 3:
            raise ValueError(
                f"Expected {self.backbone_param_count + 3} min-block tensors, got {len(min_block)}"
            )
        if len(max_block) != 3:
            raise ValueError(f"Expected 3 max-block tensors, got {len(max_block)}")
        return DROBlockView(
            backbone_params=min_block[: self.backbone_param_count],
            eta=min_block[self.backbone_param_count],
            classifier=min_block[self.backbone_param_count + 1],
            train_simplex=min_block[self.backbone_param_count + 2],
            val_simplex=max_block[0],
            classifier_copy=max_block[1],
            train_simplex_copy=max_block[2],
        )

    def _feature_forward(self, images):
        self.backbone.eval()
        return self.backbone(images)

    def _group_average(self, values, groups):
        group_means = []
        group_counts = []
        for group_idx in range(self.num_groups):
            mask = groups == group_idx
            count = mask.sum()
            group_counts.append(count)
            if count.item() == 0:
                group_means.append(torch.zeros((), device=values.device, dtype=values.dtype))
            else:
                group_means.append(values[mask].mean())
        return torch.stack(group_means), torch.stack(group_counts)

    def _eta_regularizer(self, simplex_weights, eta):
        centered = simplex_weights - self.uniform_weights
        return 0.5 * eta.reshape(()).to(centered.dtype) * torch.sum(centered.square())

    def _group_weighted_ce(self, logits, labels, groups, simplex_weights, eta=None):
        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
        group_loss, group_counts = self._group_average(per_sample_loss, groups)
        weighted = torch.dot(simplex_weights, group_loss)
        if eta is not None:
            centered = simplex_weights - self.uniform_weights
            weighted = weighted - 0.5 * eta.reshape(()).to(weighted.dtype) * torch.sum(centered.square())
        return weighted, group_loss, group_counts

    def _objective_from_batches(self, train_batch, val_batch, min_block, max_block):
        block_view = self._view(min_block, max_block)
        train_inputs, train_labels, train_groups = train_batch
        val_inputs, val_labels, val_groups = val_batch

        train_features = self._feature_forward(train_inputs)
        val_features = self._feature_forward(val_inputs)

        train_logits = classifier_logits(train_features, block_view.classifier)
        train_copy_logits = classifier_logits(train_features, block_view.classifier_copy)
        val_logits = classifier_logits(val_features, block_view.classifier)

        train_min_term, train_min_group_loss, _ = self._group_weighted_ce(
            train_logits,
            train_labels,
            train_groups,
            block_view.train_simplex_copy,
            eta=block_view.eta,
        )
        train_max_term, train_max_group_loss, _ = self._group_weighted_ce(
            train_copy_logits,
            train_labels,
            train_groups,
            block_view.train_simplex,
            eta=block_view.eta,
        )
        train_primal_term, train_primal_group_loss, _ = self._group_weighted_ce(
            train_logits,
            train_labels,
            train_groups,
            block_view.train_simplex,
            eta=block_view.eta,
        )
        val_term, val_group_loss, _ = self._group_weighted_ce(
            val_logits,
            val_labels,
            val_groups,
            block_view.val_simplex,
            eta=None,
        )
        train_min_reg = self._eta_regularizer(block_view.train_simplex_copy, block_view.eta)
        train_max_reg = self._eta_regularizer(block_view.train_simplex, block_view.eta)
        train_primal_reg = self._eta_regularizer(block_view.train_simplex, block_view.eta)
        h_value = val_term + self.rho * (train_min_term - train_max_term)
        return {
            "h": h_value,
            "train_min_term": train_min_term,
            "train_max_term": train_max_term,
            "train_primal_term": train_primal_term,
            "val_term": val_term,
            "train_min_reg": train_min_reg,
            "train_max_reg": train_max_reg,
            "train_primal_reg": train_primal_reg,
            "train_primal_group_loss": train_primal_group_loss,
            "train_min_group_loss": train_min_group_loss,
            "train_max_group_loss": train_max_group_loss,
            "val_group_loss": val_group_loss,
        }

    def h_func(self, min_block, max_block):
        def objective():
            train_batch = self.train_provider.next_batch()
            val_batch = self.val_provider.next_batch()
            return self._objective_from_batches(train_batch, val_batch, min_block, max_block)["h"]

        return objective

    def monitor_objective(self, min_block, max_block):
        total = torch.zeros((), device=self.device)
        num_pairs = min(len(self.train_monitor_batches), len(self.val_monitor_batches))
        for batch_idx in range(num_pairs):
            train_batch = tuple(t.to(self.device) for t in self.train_monitor_batches[batch_idx])
            val_batch = tuple(t.to(self.device) for t in self.val_monitor_batches[batch_idx])
            total = total + self._objective_from_batches(train_batch, val_batch, min_block, max_block)["h"]
        return total / float(num_pairs)

    def monitor_metrics(self, min_block, max_block):
        block_view = self._view(min_block, max_block)
        num_pairs = min(len(self.train_monitor_batches), len(self.val_monitor_batches))
        totals = {
            "dro/h": 0.0,
            "dro/train_loss": 0.0,
            "dro/val_loss": 0.0,
            "dro/train_min_term": 0.0,
            "dro/train_max_term": 0.0,
            "dro/reg_loss": 0.0,
            "dro/train_min_reg": 0.0,
            "dro/train_max_reg": 0.0,
        }
        train_group_loss = torch.zeros(self.num_groups, device=self.device)
        val_group_loss = torch.zeros(self.num_groups, device=self.device)
        for batch_idx in range(num_pairs):
            train_batch = tuple(t.to(self.device) for t in self.train_monitor_batches[batch_idx])
            val_batch = tuple(t.to(self.device) for t in self.val_monitor_batches[batch_idx])
            stats = self._objective_from_batches(train_batch, val_batch, min_block, max_block)
            totals["dro/h"] += float(stats["h"].item())
            totals["dro/train_loss"] += float(stats["train_primal_term"].item())
            totals["dro/val_loss"] += float(stats["val_term"].item())
            totals["dro/train_min_term"] += float(stats["train_min_term"].item())
            totals["dro/train_max_term"] += float(stats["train_max_term"].item())
            totals["dro/reg_loss"] += float(stats["train_primal_reg"].item())
            totals["dro/train_min_reg"] += float(stats["train_min_reg"].item())
            totals["dro/train_max_reg"] += float(stats["train_max_reg"].item())
            train_group_loss += stats["train_primal_group_loss"]
            val_group_loss += stats["val_group_loss"]

        denom = float(num_pairs)
        metrics = {key: value / denom for key, value in totals.items()}
        metrics["dro/eta"] = float(block_view.eta.item())
        metrics["dro/backbone_norm"] = float(compute_norm(block_view.backbone_params).item())
        metrics["dro/classifier_norm"] = float(torch.linalg.vector_norm(block_view.classifier).item())
        metrics["dro/classifier_copy_gap"] = float(
            torch.linalg.vector_norm(block_view.classifier - block_view.classifier_copy).item()
        )
        for group_idx in range(self.num_groups):
            metrics[f"dro/train_group_loss_{group_idx}"] = float((train_group_loss[group_idx] / denom).item())
            metrics[f"dro/val_group_loss_{group_idx}"] = float((val_group_loss[group_idx] / denom).item())
            metrics[f"dro/train_simplex_{group_idx}"] = float(block_view.train_simplex[group_idx].item())
            metrics[f"dro/train_simplex_copy_{group_idx}"] = float(
                block_view.train_simplex_copy[group_idx].item()
            )
            metrics[f"dro/val_simplex_{group_idx}"] = float(block_view.val_simplex[group_idx].item())
        return metrics

    @torch.no_grad()
    def evaluate_loader(self, classifier, loader, prefix="eval"):
        was_training = self.backbone.training
        self.backbone.eval()

        total_correct = 0
        total_count = 0
        total_loss = 0.0
        group_correct = torch.zeros(self.num_groups, device=self.device)
        group_count = torch.zeros(self.num_groups, device=self.device)
        group_loss_sum = torch.zeros(self.num_groups, device=self.device)

        for images, labels, groups in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            groups = groups.to(self.device, non_blocking=True)

            logits = classifier_logits(self._feature_forward(images), classifier)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            predictions = logits.argmax(dim=1)
            correct = (predictions == labels).to(dtype=torch.float32)

            total_correct += int(correct.sum().item())
            total_count += int(labels.numel())
            total_loss += float(per_sample_loss.sum().item())

            for group_idx in range(self.num_groups):
                mask = groups == group_idx
                count = int(mask.sum().item())
                if count == 0:
                    continue
                group_count[group_idx] += count
                group_correct[group_idx] += correct[mask].sum()
                group_loss_sum[group_idx] += per_sample_loss[mask].sum()

        if was_training:
            self.backbone.train()

        metrics = {
            f"{prefix}/accuracy": total_correct / max(total_count, 1),
            f"{prefix}/loss": total_loss / max(total_count, 1),
        }
        valid_group_acc = []
        for group_idx in range(self.num_groups):
            count = float(group_count[group_idx].item())
            if count > 0:
                group_acc = float((group_correct[group_idx] / group_count[group_idx]).item())
                group_loss = float((group_loss_sum[group_idx] / group_count[group_idx]).item())
                valid_group_acc.append(group_acc)
            else:
                group_acc = float("nan")
                group_loss = float("nan")
            metrics[f"{prefix}/group_accuracy_{group_idx}"] = group_acc
            metrics[f"{prefix}/group_loss_{group_idx}"] = group_loss
            metrics[f"{prefix}/group_count_{group_idx}"] = count
        metrics[f"{prefix}/worst_group_accuracy"] = min(valid_group_acc) if valid_group_acc else float("nan")
        return metrics
