import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dro.data import CelebASplitDataset, WaterbirdsSplitDataset
from dro.models import build_resnet50_backbone, classifier_logits
from dro.problem import identity_prox, simplex_prox
from minimax import SAPD_SCSC
from utils import project_onto_simplex


torch.set_default_dtype(torch.float32)

LOWER_LEVEL_EVAL_ITER_CAP = 1
DATASET_DEFAULTS = {
    "celeba": {
        "dataset_class": CelebASplitDataset,
        "target_name": "Blond_Hair",
        "confounder_name": "Male",
    },
    "waterbirds": {
        "dataset_class": WaterbirdsSplitDataset,
        "target_name": "waterbird_complete95",
        "confounder_name": "forest2water2",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a directory of DRO checkpoints and log to wandb")
    parser.add_argument("--dataset", choices=DATASET_DEFAULTS.keys(), required=True)
    parser.add_argument("checkpoint_dir", type=Path)
    return parser.parse_args()


def require_key(container, key, context):
    if key not in container:
        raise KeyError(f"Missing required key '{key}' in {context}")
    return container[key]


def resolve_runtime_device(checkpoint_args):
    requested = str(checkpoint_args.get("device", "cpu"))
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested), f"using checkpoint-requested device {requested}"
        return (
            torch.device("cpu"),
            f"checkpoint requested {requested} but CUDA is unavailable; falling back to cpu",
        )
    return torch.device(requested), f"using checkpoint-requested device {requested}"


def resolve_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is required for folder evaluation logging. Run this script in an environment with wandb installed."
        ) from exc
    return wandb


def to_device_tensor(value, device):
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor in checkpoint, got {type(value).__name__}")
    return value.detach().clone().to(device=device, dtype=torch.float32)


def build_eval_loader(dataset, batch_size, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def build_datasets(dataset_name, checkpoint_args):
    dataset_defaults = DATASET_DEFAULTS[dataset_name]
    dataset_class = dataset_defaults["dataset_class"]
    dataset_root = Path(require_key(checkpoint_args, "dataset_root", "checkpoint args"))
    target_name = checkpoint_args.get("target_name", dataset_defaults["target_name"])
    confounder_name = checkpoint_args.get("confounder_name", dataset_defaults["confounder_name"])
    seed = int(require_key(checkpoint_args, "seed", "checkpoint args"))
    train_fraction = float(checkpoint_args.get("train_fraction", 1.0))
    val_fraction = float(checkpoint_args.get("val_fraction", 1.0))
    test_fraction = float(checkpoint_args.get("test_fraction", 1.0))

    common = {
        "root_dir": dataset_root,
        "target_name": target_name,
        "confounder_name": confounder_name,
        "augment": False,
        "seed": seed,
        "eval_mode": True,
    }
    train_dataset = dataset_class(split="train", fraction=train_fraction, **common)
    val_dataset = dataset_class(split="val", fraction=val_fraction, **common)
    test_dataset = dataset_class(split="test", fraction=test_fraction, **common)
    return train_dataset, val_dataset, test_dataset


@torch.no_grad()
def cache_split_features(backbone, loader, device, split_name, num_groups):
    backbone.eval()
    cached_batches = []
    for images, labels, groups in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        groups = groups.to(device, non_blocking=True)
        features = backbone(images).detach()
        cached_batches.append((features, labels.detach(), groups.detach()))

    counts = torch.zeros(num_groups, device=device, dtype=torch.float32)
    for _, _, groups in cached_batches:
        for group_idx in range(num_groups):
            counts[group_idx] += (groups == group_idx).sum().to(dtype=counts.dtype)
    missing = [str(group_idx) for group_idx in range(num_groups) if counts[group_idx].item() <= 0]
    if missing:
        raise ValueError(
            f"{split_name} split is missing groups {', '.join(missing)}; cannot evaluate full-group objectives"
        )
    return cached_batches, counts


def summarize_cached_split(cached_batches, classifier, num_groups):
    dtype = classifier.dtype
    device = classifier.device
    group_loss_sums = [torch.zeros((), device=device, dtype=dtype) for _ in range(num_groups)]
    group_counts = torch.zeros(num_groups, device=device, dtype=dtype)
    group_correct = torch.zeros(num_groups, device=device, dtype=dtype)
    total_correct = torch.zeros((), device=device, dtype=dtype)
    total_count = torch.zeros((), device=device, dtype=dtype)

    for features, labels, groups in cached_batches:
        logits = classifier_logits(features, classifier)
        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
        predictions = logits.argmax(dim=1)
        correct = (predictions == labels).to(dtype=dtype)
        total_correct = total_correct + correct.sum()
        total_count = total_count + torch.tensor(float(labels.numel()), device=device, dtype=dtype)
        for group_idx in range(num_groups):
            mask = groups == group_idx
            count = mask.sum()
            if count.item() == 0:
                continue
            group_loss_sums[group_idx] = group_loss_sums[group_idx] + per_sample_loss[mask].sum()
            group_counts[group_idx] = group_counts[group_idx] + count.to(dtype=dtype)
            group_correct[group_idx] = group_correct[group_idx] + correct[mask].sum()

    if torch.any(group_counts <= 0):
        raise ValueError("Encountered an empty group while computing full-dataset group means")

    group_mean_losses = torch.stack(group_loss_sums) / group_counts
    group_accuracies = group_correct / group_counts
    return {
        "group_mean_losses": group_mean_losses,
        "group_counts": group_counts,
        "group_accuracies": group_accuracies,
        "accuracy": total_correct / torch.clamp(total_count, min=1.0),
        "worst_group_accuracy": group_accuracies.min(),
    }


def classifier_l2_penalty(classifier, weight_decay):
    return 0.5 * weight_decay * torch.sum(classifier.square())


def simplex_eta_penalty(simplex_weights, eta_value):
    uniform = torch.full_like(simplex_weights, 1.0 / simplex_weights.numel())
    centered = simplex_weights - uniform
    return 0.5 * eta_value.reshape(()).to(centered.dtype) * torch.sum(centered.square())


def ldro_value_from_summary(split_summary, classifier, simplex_weights, eta_value, weight_decay):
    objective = torch.dot(simplex_weights, split_summary["group_mean_losses"])
    objective = objective - simplex_eta_penalty(simplex_weights, eta_value)
    objective = objective + classifier_l2_penalty(classifier, weight_decay)
    return objective


def train_ldro_value(cached_batches, classifier, simplex_weights, eta_value, weight_decay):
    split_summary = summarize_cached_split(cached_batches, classifier, simplex_weights.numel())
    objective = ldro_value_from_summary(split_summary, classifier, simplex_weights, eta_value, weight_decay)
    return objective, split_summary


def upper_objective(split_summary):
    group_mean_losses = split_summary["group_mean_losses"]
    max_group_loss = group_mean_losses.max()
    worst_group_index = int(torch.nonzero(group_mean_losses == max_group_loss, as_tuple=False)[0].item())
    return max_group_loss, worst_group_index


def solve_lower_level_problem(
    cached_batches,
    classifier_init,
    simplex_init,
    eta_value,
    weight_decay,
    lip,
    lip_tau,
    theta,
    max_iter,
    log_every,
):
    mu_x = float(weight_decay)
    mu_y = float(eta_value.item())
    if mu_x < 0:
        raise ValueError(f"weight_decay must be positive for SAPD+, got {mu_x}")
    if mu_y < 0:
        raise ValueError(f"eta must be positive for SAPD+, got {mu_y}")

    classifier_var = torch.nn.Parameter(classifier_init.detach().clone())
    simplex_var = torch.nn.Parameter(simplex_init.detach().clone())

    def objective():
        value, _ = train_ldro_value(
            cached_batches,
            classifier_var,
            simplex_var,
            eta_value,
            weight_decay,
        )
        return value

    solver = SAPD_SCSC(
        params_x=[classifier_var],
        params_y=[simplex_var],
        h_bar=objective,
        mu_x=mu_x,
        mu_y=mu_y,
        lip=float(lip),
        prox_x=[identity_prox],
        prox_y=[simplex_prox(project_onto_simplex)],
        lip_tau=None if lip_tau is None else float(lip_tau),
        theta=float(theta),
        max_iter=int(max_iter),
        verbose=False,
        log_every=int(log_every),
    )
    solver_stats = solver.run()
    solved_value, _ = train_ldro_value(
        cached_batches,
        classifier_var,
        simplex_var,
        eta_value,
        weight_decay,
    )
    return {
        "solver_stats": solver_stats,
        "optimized_value": float(solved_value.detach().item()),
    }


def try_build_feature_caches(backbone, train_loader, val_loader, test_loader, device, num_groups):
    try:
        train_cache, train_counts = cache_split_features(backbone, train_loader, device, "train", num_groups)
        val_cache, val_counts = cache_split_features(backbone, val_loader, device, "val", num_groups)
        test_cache, test_counts = cache_split_features(backbone, test_loader, device, "test", num_groups)
        return device, backbone, train_cache, train_counts, val_cache, val_counts, test_cache, test_counts, None
    except RuntimeError as exc:
        if device.type != "cuda" or "out of memory" not in str(exc).lower():
            raise
        print("device_note=cuda OOM during feature caching; retrying on cpu", flush=True)
        torch.cuda.empty_cache()
        cpu_device = torch.device("cpu")
        backbone = backbone.to(cpu_device)
        train_cache, train_counts = cache_split_features(backbone, train_loader, cpu_device, "train", num_groups)
        val_cache, val_counts = cache_split_features(backbone, val_loader, cpu_device, "val", num_groups)
        test_cache, test_counts = cache_split_features(backbone, test_loader, cpu_device, "test", num_groups)
        return (
            cpu_device,
            backbone,
            train_cache,
            train_counts,
            val_cache,
            val_counts,
            test_cache,
            test_counts,
            exc,
        )


def collect_checkpoint_paths(checkpoint_dir):
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Expected a checkpoint directory, got: {checkpoint_dir}")

    checkpoint_paths = sorted(
        {path for pattern in ("*.pt", "*.pth") for path in checkpoint_dir.rglob(pattern)}
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No .pt or .pth files found under {checkpoint_dir}")
    return checkpoint_paths


def collect_checkpoint_records(checkpoint_dir):
    records = []
    for checkpoint_path in collect_checkpoint_paths(checkpoint_dir):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        completed_outer_iters = int(require_key(checkpoint, "completed_outer_iters", f"checkpoint {checkpoint_path}"))
        record = {
            "path": checkpoint_path,
            "completed_outer_iters": completed_outer_iters,
        }
        if checkpoint.get("call_cumulative_scsc_inner_iters") is not None:
            record["call_cumulative_scsc_inner_iters"] = int(checkpoint["call_cumulative_scsc_inner_iters"])
        records.append(record)
    records.sort(key=lambda item: (item["completed_outer_iters"], str(item["path"])))
    return records


def define_wandb_metrics(run, metric_keys):
    if not hasattr(run, "define_metric"):
        return
    step_metric = "outer_iteration"
    run.define_metric(step_metric)
    for key in metric_keys:
        if key == step_metric:
            continue
        run.define_metric(key, step_metric=step_metric)


def build_run_metrics(
    checkpoint_path,
    device,
    device_note,
    train_summary,
    val_summary,
    test_summary,
    train_counts,
    val_counts,
    test_counts,
    current_train_ldro,
    lower_level_solution,
    lower_level_skip_reason,
    requested_solver_max_outer_iters,
    applied_solver_max_outer_iters,
    completed_outer_iters,
    call_cumulative_scsc_inner_iters,
    num_groups,
):
    val_upper_objective, val_worst_group_index = upper_objective(val_summary)
    test_upper_objective, test_worst_group_index = upper_objective(test_summary)

    metrics = {
        "outer_iteration": int(completed_outer_iters),
        "train_accuracy": float(train_summary["accuracy"].detach().item()),
        "train_worst_group_accuracy": float(train_summary["worst_group_accuracy"].detach().item()),
        "val_accuracy": float(val_summary["accuracy"].detach().item()),
        "val_worst_group_accuracy": float(val_summary["worst_group_accuracy"].detach().item()),
        "test_accuracy": float(test_summary["accuracy"].detach().item()),
        "test_worst_group_accuracy": float(test_summary["worst_group_accuracy"].detach().item()),
        "val_upper_objective": float(val_upper_objective.detach().item()),
        "val_worst_group_index": val_worst_group_index,
        "test_upper_objective": float(test_upper_objective.detach().item()),
        "test_worst_group_index": test_worst_group_index,
        "lower_level_eval_skipped": float(lower_level_solution is None),
        "sapd_requested_max_iters": requested_solver_max_outer_iters,
    }
    if call_cumulative_scsc_inner_iters is not None:
        metrics["cumulative_inner_iters"] = int(call_cumulative_scsc_inner_iters)

    if lower_level_solution is None:
        metrics["train_ldro_current"] = float("nan")
        metrics["train_ldro_optimized"] = float("nan")
        metrics["train_lower_gap"] = float("nan")
        metrics["sapd_applied_max_iters"] = 0
        metrics["sapd_num_outer_iters"] = 0
        metrics["sapd_num_inner_iters"] = 0
        metrics["sapd_final_delta"] = float("nan")
        metrics["gap_note_code"] = float("nan")
    else:
        lower_level_gap = float(current_train_ldro.detach().item()) - lower_level_solution["optimized_value"]
        metrics["train_ldro_current"] = float(current_train_ldro.detach().item())
        metrics["train_ldro_optimized"] = lower_level_solution["optimized_value"]
        metrics["train_lower_gap"] = lower_level_gap
        metrics["sapd_applied_max_iters"] = applied_solver_max_outer_iters
        metrics["sapd_num_outer_iters"] = lower_level_solution["solver_stats"]["num_outer_iters"]
        metrics["sapd_num_inner_iters"] = lower_level_solution["solver_stats"]["num_inner_iters"]
        metrics["sapd_final_delta"] = lower_level_solution["solver_stats"]["final_delta"]
        if lower_level_gap < 0.0 and lower_level_gap >= -1e-4:
            metrics["gap_note_code"] = 1.0
        elif lower_level_gap < -1e-4:
            metrics["gap_note_code"] = 2.0
        else:
            metrics["gap_note_code"] = 0.0

    for group_idx in range(num_groups):
        metrics[f"train_group_accuracy_{group_idx}"] = float(train_summary["group_accuracies"][group_idx].detach().item())
        metrics[f"train_group_loss_{group_idx}"] = float(train_summary["group_mean_losses"][group_idx].detach().item())
        metrics[f"train_group_count_{group_idx}"] = int(train_counts[group_idx].item())
        metrics[f"val_group_accuracy_{group_idx}"] = float(val_summary["group_accuracies"][group_idx].detach().item())
        metrics[f"val_group_loss_{group_idx}"] = float(val_summary["group_mean_losses"][group_idx].detach().item())
        metrics[f"val_group_count_{group_idx}"] = int(val_counts[group_idx].item())
        metrics[f"test_group_accuracy_{group_idx}"] = float(test_summary["group_accuracies"][group_idx].detach().item())
        metrics[f"test_group_loss_{group_idx}"] = float(test_summary["group_mean_losses"][group_idx].detach().item())
        metrics[f"test_group_count_{group_idx}"] = int(test_counts[group_idx].item())

    metadata = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "resolved_device": str(device),
        "device_note": device_note,
    }
    if lower_level_skip_reason is not None:
        metadata["evaluation_note"] = lower_level_skip_reason
    return metrics, metadata


def print_checkpoint_metrics(metadata, metrics):
    print(f"checkpoint_path={metadata['checkpoint_path']}")
    print(f"resolved_device={metadata['resolved_device']}")
    print(f"device_note={metadata['device_note']}")
    if metadata.get("evaluation_note"):
        print(f"evaluation_note={metadata['evaluation_note']}")
    for key, value in metrics.items():
        print(f"{key}={value}")


def evaluate_checkpoint(checkpoint_path, dataset_name):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = require_key(checkpoint, "args", f"checkpoint {checkpoint_path}")
    if not isinstance(checkpoint_args, dict):
        raise TypeError(f"checkpoint['args'] must be a dict for {checkpoint_path}")

    device, device_note = resolve_runtime_device(checkpoint_args)
    backbone_state_dict = require_key(checkpoint, "backbone_state_dict", f"checkpoint {checkpoint_path}")
    classifier_tensor = to_device_tensor(require_key(checkpoint, "classifier", f"checkpoint {checkpoint_path}"), device)
    eta_tensor = to_device_tensor(require_key(checkpoint, "eta", f"checkpoint {checkpoint_path}"), device)
    train_simplex_tensor = to_device_tensor(
        require_key(checkpoint, "train_simplex", f"checkpoint {checkpoint_path}"),
        device,
    )
    val_simplex_tensor = to_device_tensor(
        require_key(checkpoint, "val_simplex", f"checkpoint {checkpoint_path}"),
        device,
    )
    classifier_copy_tensor = to_device_tensor(checkpoint.get("classifier_copy", checkpoint["classifier"]), device)
    train_simplex_copy_tensor = to_device_tensor(
        checkpoint.get("train_simplex_copy", checkpoint["train_simplex"]),
        device,
    )

    backbone, feature_dim = build_resnet50_backbone(pretrained=False)
    backbone.load_state_dict(backbone_state_dict)
    if classifier_tensor.ndim != 2 or classifier_tensor.shape[0] != feature_dim:
        raise ValueError(
            f"Checkpoint classifier has shape {tuple(classifier_tensor.shape)} but backbone feature_dim is {feature_dim}"
        )

    backbone = backbone.to(device)
    train_dataset, val_dataset, test_dataset = build_datasets(dataset_name, checkpoint_args)
    num_groups = train_dataset.num_groups
    if train_simplex_tensor.numel() != num_groups:
        raise ValueError(f"Expected {num_groups} train groups, got {train_simplex_tensor.numel()}")
    if val_simplex_tensor.numel() != num_groups:
        raise ValueError(f"Expected {num_groups} validation groups, got {val_simplex_tensor.numel()}")

    batch_size = max(int(checkpoint_args.get("batch_size", 1)), int(checkpoint_args.get("eval_batch_size", 1)))
    num_workers = int(checkpoint_args.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    train_loader = build_eval_loader(train_dataset, batch_size, num_workers, pin_memory)
    val_loader = build_eval_loader(val_dataset, batch_size, num_workers, pin_memory)
    test_loader = build_eval_loader(test_dataset, batch_size, num_workers, pin_memory)

    (
        device,
        backbone,
        train_cache,
        train_counts,
        val_cache,
        val_counts,
        test_cache,
        test_counts,
        cache_fallback_exc,
    ) = try_build_feature_caches(
        backbone,
        train_loader,
        val_loader,
        test_loader,
        device,
        num_groups,
    )
    if cache_fallback_exc is not None:
        classifier_tensor = classifier_tensor.to(device)
        eta_tensor = eta_tensor.to(device)
        train_simplex_tensor = train_simplex_tensor.to(device)
        val_simplex_tensor = val_simplex_tensor.to(device)
        classifier_copy_tensor = classifier_copy_tensor.to(device)
        train_simplex_copy_tensor = train_simplex_copy_tensor.to(device)
        device_note = f"{device_note}; feature cache fallback forced cpu execution"

    weight_decay = float(require_key(checkpoint_args, "weight_decay", f"checkpoint args {checkpoint_path}"))
    solver_lip_h = float(require_key(checkpoint_args, "solver_lip_h", f"checkpoint args {checkpoint_path}"))
    solver_theta = float(require_key(checkpoint_args, "solver_theta", f"checkpoint args {checkpoint_path}"))
    requested_solver_max_outer_iters = int(
        require_key(checkpoint_args, "solver_max_outer_iters", f"checkpoint args {checkpoint_path}")
    )
    applied_solver_max_outer_iters = min(requested_solver_max_outer_iters, LOWER_LEVEL_EVAL_ITER_CAP)
    solver_lip_tau = checkpoint_args.get("solver_lip_tau")
    solver_log_every = int(checkpoint_args.get("log_every", 1))

    train_summary = summarize_cached_split(train_cache, classifier_tensor, num_groups)
    val_summary = summarize_cached_split(val_cache, classifier_tensor, num_groups)
    test_summary = summarize_cached_split(test_cache, classifier_tensor, num_groups)
    current_train_ldro = None
    lower_level_solution = None
    lower_level_skip_reason = None
    if weight_decay <= 0.0:
        lower_level_skip_reason = (
            f"skipped lower-level evaluation because weight_decay={weight_decay} "
            "makes the SAPD+ strong-convexity parameter mu_x non-positive"
        )
    else:
        current_train_ldro = ldro_value_from_summary(
            train_summary,
            classifier_tensor,
            train_simplex_tensor,
            eta_tensor,
            weight_decay,
        )
        lower_level_solution = solve_lower_level_problem(
            train_cache,
            classifier_copy_tensor,
            train_simplex_copy_tensor,
            eta_tensor,
            weight_decay,
            lip=solver_lip_h,
            lip_tau=solver_lip_tau,
            theta=solver_theta,
            max_iter=applied_solver_max_outer_iters,
            log_every=solver_log_every,
        )

    metrics, metadata = build_run_metrics(
        checkpoint_path=checkpoint_path,
        device=device,
        device_note=device_note,
        train_summary=train_summary,
        val_summary=val_summary,
        test_summary=test_summary,
        train_counts=train_counts,
        val_counts=val_counts,
        test_counts=test_counts,
        current_train_ldro=current_train_ldro,
        lower_level_solution=lower_level_solution,
        lower_level_skip_reason=lower_level_skip_reason,
        requested_solver_max_outer_iters=requested_solver_max_outer_iters,
        applied_solver_max_outer_iters=applied_solver_max_outer_iters,
        completed_outer_iters=int(require_key(checkpoint, "completed_outer_iters", f"checkpoint {checkpoint_path}")),
        call_cumulative_scsc_inner_iters=checkpoint.get("call_cumulative_scsc_inner_iters"),
        num_groups=num_groups,
    )
    return metrics, metadata


def main():
    args = parse_args()
    checkpoint_records = collect_checkpoint_records(args.checkpoint_dir)
    wandb = resolve_wandb()
    run = wandb.init(
        project="minimax",
        name=f"dro_eval_{args.dataset}_{args.checkpoint_dir.name}",
        config={
            "dataset": args.dataset,
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "num_checkpoints": len(checkpoint_records),
            "lower_level_eval_iter_cap": LOWER_LEVEL_EVAL_ITER_CAP,
        },
        reinit=True,
    )

    metrics_defined = False
    for checkpoint_record in checkpoint_records:
        checkpoint_path = checkpoint_record["path"]
        print(
            f"evaluating checkpoint={checkpoint_path} outer_iteration={checkpoint_record['completed_outer_iters']}",
            flush=True,
        )
        metrics, metadata = evaluate_checkpoint(checkpoint_path, args.dataset)
        if not metrics_defined:
            define_wandb_metrics(run, metrics.keys())
            metrics_defined = True
        run.log(metrics)
        print_checkpoint_metrics(metadata, metrics)

    if hasattr(run, "summary"):
        run.summary["last_outer_iteration"] = checkpoint_records[-1]["completed_outer_iters"]
        run.summary["last_checkpoint_path"] = str(checkpoint_records[-1]["path"].resolve())
        run.summary["num_checkpoints_evaluated"] = len(checkpoint_records)
    run.finish()


if __name__ == "__main__":
    main()
