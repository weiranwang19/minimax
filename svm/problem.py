"""SVM bilevel problem definition and metric helpers."""

import math

import torch

from .data import stable_seed


def _with_bias_matrix(x_mat):
    ones = torch.ones((x_mat.shape[0], 1), dtype=x_mat.dtype, device=x_mat.device)
    return torch.cat([x_mat, ones], dim=1)


def _compute_spectral_norm(mat):
    if mat.numel() == 0:
        return 0.0
    return float(torch.linalg.matrix_norm(mat, ord=2).item())


def positive_part(v):
    return torch.clamp(v, min=0.0)


def binomial_deviance(labels, scores):
    return torch.nn.functional.softplus(-labels * scores)


class SVMProblem:
    """Packed Section 3.3 SVM hyperparameter-tuning instance."""

    def __init__(self, split, seed):
        self.dataset_name = split["dataset_name"]
        self.split_idx = int(split["split_idx"])
        self.seed = int(seed)
        self.x_train = split["x_train"].clone()
        self.y_train = split["y_train"].clone()
        self.x_val = split["x_val"].clone()
        self.y_val = split["y_val"].clone()
        self.train_indices = split["train_indices"].clone()
        self.val_indices = split["val_indices"].clone()
        self.n_train = int(self.x_train.shape[0])
        self.n_val = int(self.x_val.shape[0])
        self.q = int(self.x_train.shape[1])
        self.y_dim = self.q + 1 + self.n_train

        train_aug = _with_bias_matrix(self.x_train)
        val_aug = _with_bias_matrix(self.x_val)
        constraint_mat = torch.cat(
            [
                -(self.y_train[:, None] * self.x_train),
                -self.y_train[:, None],
                -torch.eye(self.n_train, dtype=self.x_train.dtype),
            ],
            dim=1,
        )
        self.lower_constraint_jac_np = (
            torch.cat(
                [
                    self.y_train[:, None] * self.x_train,
                    self.y_train[:, None],
                    torch.eye(self.n_train, dtype=self.x_train.dtype),
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        self.l_g = _compute_spectral_norm(constraint_mat)
        row_bounds = 22.0 + torch.sum(torch.abs(self.x_train), dim=1)
        self.g_hi = float(torch.linalg.vector_norm(row_bounds).item())
        self.l_grad_f = 0.25 * (_compute_spectral_norm(val_aug) ** 2) / max(1, self.n_val)
        self.l_grad_lower = max(1.0, 0.25 * (_compute_spectral_norm(train_aug) ** 2))

    def unpack_lower(self, y_vec):
        w = y_vec[: self.q]
        b = y_vec[self.q]
        xi = y_vec[self.q + 1 :]
        return w, b, xi

    def pack_lower(self, w, b, xi):
        b_vec = b.reshape(1) if torch.is_tensor(b) else torch.tensor([b], dtype=torch.get_default_dtype())
        return torch.cat([w, b_vec.to(dtype=w.dtype, device=w.device), xi])

    def prox_c(self, v, coeff):
        del coeff
        return torch.clamp(v, min=0.0, max=10.0)

    def prox_lower(self, v, coeff):
        del coeff
        projected = v.clone()
        projected[: self.q + 1] = torch.clamp(projected[: self.q + 1], min=-1.0, max=1.0)
        projected[self.q + 1 :] = torch.clamp(projected[self.q + 1 :], min=0.0, max=20.0)
        return projected

    def project_lower(self, v):
        return self.prox_lower(v, coeff=1.0)

    def upper_smooth(self, x_params, y_params):
        del x_params
        w, b, _ = self.unpack_lower(y_params[0])
        scores = self.x_val @ w + b
        return torch.mean(binomial_deviance(self.y_val, scores))

    def lower_smooth(self, x_params, z_params):
        c_vec = x_params[0]
        w, b, xi = self.unpack_lower(z_params[0])
        scores = self.x_train @ w + b
        return torch.sum(binomial_deviance(self.y_train, scores)) + torch.dot(c_vec, xi)

    def lower_constraints(self, x_params, z_params):
        del x_params
        w, b, xi = self.unpack_lower(z_params[0])
        return 1.0 - xi - self.y_train * (self.x_train @ w + b)

    def upper_objective(self, c_vec, y_vec):
        del c_vec
        with torch.no_grad():
            w, b, _ = self.unpack_lower(y_vec)
            scores = self.x_val @ w + b
            return float(torch.mean(binomial_deviance(self.y_val, scores)).item())

    def lower_objective(self, c_vec, y_vec):
        with torch.no_grad():
            w, b, xi = self.unpack_lower(y_vec)
            scores = self.x_train @ w + b
            value = torch.sum(binomial_deviance(self.y_train, scores)) + torch.dot(c_vec, xi)
            return float(value.item())

    def feasibility_norm(self, c_vec, y_vec):
        del c_vec
        with torch.no_grad():
            empty_x = torch.empty(0, dtype=y_vec.dtype, device=y_vec.device)
            residual = positive_part(self.lower_constraints([empty_x], [y_vec]))
            return float(torch.linalg.vector_norm(residual).item())

    def validation_accuracy(self, y_vec):
        return self.classification_counts(y_vec, self.x_val, self.y_val)["accuracy"]

    def train_accuracy(self, y_vec):
        return self.classification_counts(y_vec, self.x_train, self.y_train)["accuracy"]

    def classification_counts(self, y_vec, x_mat, labels):
        with torch.no_grad():
            w, b, _ = self.unpack_lower(y_vec)
            scores = x_mat @ w + b
            preds = torch.where(scores >= 0.0, torch.ones_like(scores), -torch.ones_like(scores))
            num_errors = int(torch.sum(preds != labels).item())
            return {
                "num_samples": int(labels.numel()),
                "num_errors": num_errors,
                "num_correct": int(labels.numel()) - num_errors,
                "accuracy": float(torch.mean((preds == labels).to(torch.get_default_dtype())).item()),
            }

    def compute_d_y(self):
        return math.sqrt(4.0 * (self.q + 1) + 400.0 * self.n_train)

    def compute_minimax_d_y(self, lagrange_bound):
        return math.sqrt(self.compute_d_y() ** 2 + (float(lagrange_bound) ** 2) * self.n_train)

    def make_initial_iterates(self):
        generator = torch.Generator()
        generator.manual_seed(stable_seed(self.seed, self.dataset_name, self.split_idx, "initial"))
        c0 = torch.zeros(self.n_train)
        y0 = torch.rand(self.y_dim, generator=generator)
        return c0, self.project_lower(y0)
