import math

import torch, tqdm
from torch.optim import Optimizer

from utils import clone_vals, zero_vals, scale_vals, add_vals, blend_vals, assign_vals, compute_prox, compute_norm


def _validate_gradient_result(result, min_vals, max_vals):
    """Check the optional callback contract without synchronizing tensor values."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("Gradient callbacks must return two lists of tensors")
    for gradients, values in zip(result, (min_vals, max_vals)):
        if not isinstance(gradients, (tuple, list)) or len(gradients) != len(values):
            raise ValueError("Gradient callback block lengths must match the input blocks")
        for gradient, value in zip(gradients, values):
            if (not torch.is_tensor(gradient) or gradient.shape != value.shape
                    or gradient.dtype != value.dtype or gradient.device != value.device):
                raise ValueError("Gradient callback tensors must match input shape, dtype and device")
    return tuple([g.detach() for g in block] for block in result)


def _validate_box_bound_pair(bounds):
    try:
        valid = (len(bounds) == 2 and all(math.isfinite(v) for v in bounds)
                 and bounds[0] < bounds[1])
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("box_bounds must contain finite lo < hi")
    return tuple(bounds)


def _normalize_box_bounds(bounds, num_min, num_max):
    """Expand a common box or validate one box per parameter tensor."""
    if bounds is None:
        return None, None
    if not isinstance(bounds, dict):
        pair = _validate_box_bound_pair(bounds)
        return [pair] * num_min, [pair] * num_max
    if set(bounds) != {"min", "max"}:
        raise ValueError("Per-block box_bounds must have exactly 'min' and 'max' keys")
    result = []
    for name, count in (("min", num_min), ("max", num_max)):
        block = bounds[name]
        if not isinstance(block, (list, tuple)) or len(block) != count:
            raise ValueError(f"box_bounds['{name}'] must contain {count} bounds pairs")
        result.append([_validate_box_bound_pair(pair) for pair in block])
    return tuple(result)


def _box_stationarity_residual(values, gradients, bounds, maximize=False):
    """Minimum-norm KKT residual at a feasible point in the declared boxes.

    Boundaries are compared exactly: a nearby interior point has no normal
    cone to cancel its gradient. Maximization uses the negative gradient.
    """
    residual = []
    for value, gradient, (lo, hi) in zip(values, gradients, bounds):
        if not torch.all(torch.isfinite(value)) or not torch.all(torch.isfinite(gradient)):
            raise RuntimeError("Minimax_SCSC encountered non-finite inner certificate values")
        if torch.any((value < lo) | (value > hi)):
            raise RuntimeError("Minimax_SCSC inner candidate is outside declared box_bounds")
        g = -gradient if maximize else gradient
        residual.append(torch.where(value == lo, g.clamp_max(0),
                                    torch.where(value == hi, g.clamp_min(0), g)))
    return residual


def _box_projection_quotient(values, gradients, step, bounds, maximize=False):
    """Exact box-projection quotient, avoiding subtraction of nearby iterates.

    For bounds (lo, hi), min: (u - clip(u-step*g))/step
    = clip(g, (u-hi)/step, (u-lo)/step). For max, the quotient
    (clip(v+step*g) - v)/step = clip(g, (lo-v)/step, (hi-v)/step).
    These identities also hold for extrapolated points outside the box.
    """
    lo, hi = bounds
    if maximize:
        return [torch.clamp(g, min=(lo-v)/step, max=(hi-v)/step)
                for v, g in zip(values, gradients)]
    return [torch.clamp(g, min=(v-hi)/step, max=(v-lo)/step)
            for v, g in zip(values, gradients)]


def _regularize_gradient_callbacks(gradient_func, gradient_difference_func,
                                   min_anchor, max_anchor, lip_h, sigma_y):
    """Add Algorithm 6 / equation (102) regularization to explicit gradients."""
    @torch.no_grad()
    def gradient(min_vals, max_vals):
        gx, gy = _validate_gradient_result(
            gradient_func(min_vals, max_vals), min_vals, max_vals,
        )
        return ([g + 2 * lip_h * (v-a) for g, v, a in zip(gx, min_vals, min_anchor)],
                [g - sigma_y * (v-a) for g, v, a in zip(gy, max_vals, max_anchor)])

    @torch.no_grad()
    def difference(old_min, old_max, new_min, new_max):
        dx, dy = _validate_gradient_result(
            gradient_difference_func(old_min, old_max, new_min, new_max), new_min, new_max,
        )
        return ([g + 2 * lip_h * (n-o) for g, n, o in zip(dx, new_min, old_min)],
                [g - sigma_y * (n-o) for g, n, o in zip(dy, new_max, old_max)])

    return (gradient if gradient_func is not None else None,
            difference if gradient_difference_func is not None else None)


class Minimax_SCSC(Optimizer):
    """
    Algorithm 5 of FOP for strongly-convex-strongly-concave minimax problems.
    Algorithm B.1 of SMO, inner solver reused by the higher-level wrappers.
    References below are to Algorithm 5 in arXiv:2301.01716v2, p. 25:
    https://arxiv.org/pdf/2301.01716#page=25

    Optional gradient callbacks operate on supplied tensor lists, without
    assigning parameters. The difference callback returns new-minus-old.
    box_bounds declares box projection prox operators: either a shared
    (lo, hi) pair or {'min': [pairs...], 'max': [pairs...]} with one pair per
    parameter tensor. Callers must ensure their prox operators agree.
    Every feasible inner candidate is checked for stationarity, using box
    normals when declared and the proximal subgradients otherwise.
    """

    def __init__(self, params_x, params_y, h_bar, sigma_x, sigma_y, lip, prox_x, prox_y, tau, lr=1, max_iter=300000, verbose=False, log_every=1, gradient_func=None, gradient_difference_func=None, box_bounds=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if tau <= 0.0:
            raise ValueError(f"Invalid tau: {tau}")
        if max_iter <= 0:
            raise ValueError(f"Invalid max_iter: {max_iter}")
        if log_every <= 0:
            raise ValueError(f"Invalid log_every: {log_every}")
        for callback in (gradient_func, gradient_difference_func):
            if callback is not None and not callable(callback):
                raise TypeError("Gradient callbacks must be callable or None")
        defaults = dict(lr=lr)
        super().__init__([{"params": params_x}, {"params": params_y}], defaults)
        if len(self.param_groups) != 2:
            raise ValueError("Minimax_SCSC expects exactly two parameter groups")

        self.h_bar = h_bar
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        # Algorithm 5, Input: coefficients (including zeta and zeta_hat).
        self.alpha_bar = min(1, math.sqrt(8 * sigma_y / sigma_x))
        self.eta_z = sigma_x / 2
        self.eta_y = min(1 / (2 * sigma_y), 4 / (self.alpha_bar * sigma_x))
        self.lip = lip
        self.zeta = 1 / (2 * math.sqrt(5) * (1 + 8 * lip / sigma_x))
        self.gamma_x = self.gamma_y = (8 / sigma_x)
        self.zeta_hat = min(sigma_x, sigma_y) / (lip ** 2)
        self.lr = lr
        self.tau = tau
        self.max_iter = max_iter
        self.prox_x = prox_x
        self.prox_y = prox_y
        self.verbose = verbose
        self.log_every = log_every
        self.gradient_func = gradient_func
        self.gradient_difference_func = gradient_difference_func
        self.box_bounds = box_bounds
        self._min_box_bounds, self._max_box_bounds = _normalize_box_bounds(
            box_bounds, len(self.get_x()), len(self.get_y()),
        )

    def _log(self, msg, force=False):
        if self.verbose and (force or True):
            print(msg, flush=True)

    def get_x(self):
        return self.param_groups[0]["params"]

    def get_y(self):
        return self.param_groups[1]["params"]

    @torch.no_grad()
    def get_x_grad(self):
        return [p.grad for p in self.get_x()]

    @torch.no_grad()
    def get_y_grad(self):
        return [p.grad for p in self.get_y()]

    @torch.no_grad()
    def assign_x(self, val):
        assign_vals(self.param_groups[0]["params"], val)

    @torch.no_grad()
    def assign_y(self, val):
        assign_vals(self.param_groups[1]["params"], val)

    def compute_h_bar_gradient(self, x_val, y_val):
        if self.gradient_func is not None:
            # Same gradient oracle as Algorithm 5; no graph or parameter copies.
            with torch.no_grad():
                return _validate_gradient_result(self.gradient_func(x_val, y_val), x_val, y_val)
        self.assign_x(x_val)
        self.assign_y(y_val)

        loss = self.h_bar()
        self.zero_grad()
        loss.backward()
        x_grad = self.get_x_grad()
        y_grad = self.get_y_grad()
        return x_grad, y_grad

    def compute_a_k(self, x_val, y_val, z_g_k, y_g_k, return_h_hat_grad=False):
        x_grad, y_grad = self.compute_h_bar_gradient(x_val, y_val)
        return self._compute_a_k_from_gradient(
            x_val, y_val, z_g_k, y_g_k, x_grad, y_grad, return_h_hat_grad,
        )

    def _compute_a_k_from_gradient(self, x_val, y_val, z_g_k, y_g_k,
                                 x_grad, y_grad, return_h_hat_grad=False):
        # Definitions preceding Algorithm 5: h_hat = h_bar - sigma_x||x||^2/2
        # + sigma_y||y||^2/2, followed by the definitions of a_x^k and a_y^k.
        x_grad_hat = add_vals(x_grad, scale_vals(x_val, -self.sigma_x))
        y_grad_hat = add_vals(y_grad, scale_vals(y_val, self.sigma_y))
        if return_h_hat_grad:
            return x_grad_hat, y_grad_hat

        x_tmp = add_vals(x_val, scale_vals(z_g_k, -1 / self.sigma_x))
        a_x_k = add_vals(x_grad_hat, scale_vals(x_tmp, self.sigma_x / 2))

        y_tmp = add_vals(y_val, scale_vals(y_g_k, -1))
        y_tmp = add_vals(
            scale_vals(y_val, self.sigma_y),
            scale_vals(y_tmp, self.sigma_x / 8),
        )
        a_y_k = add_vals(scale_vals(y_grad_hat, -1), y_tmp)
        return a_x_k, a_y_k

    def compute_inner_stationarity_residual(self, x_val, y_val, x_grad, y_grad,
                                           b_x, b_y):
        """Certificate for h_bar + p - q at the current proximal candidate.

        Proximal optimality gives b_x in partial p(x), b_y in partial q(y).
        Thus (grad_x h_bar + b_x, -grad_y h_bar + b_y) is a valid residual.
        Declared boxes allow the minimum-norm normal without subtracting
        nearby iterates to recover b. No gradient evaluation is needed here.
        """
        if self._min_box_bounds is not None:
            return (
                _box_stationarity_residual(x_val, x_grad, self._min_box_bounds),
                _box_stationarity_residual(y_val, y_grad, self._max_box_bounds, maximize=True),
            )
        for value in x_val + y_val + x_grad + y_grad + b_x + b_y:
            if not torch.all(torch.isfinite(value)):
                raise RuntimeError("Minimax_SCSC encountered non-finite inner certificate values")
        return add_vals(x_grad, b_x), add_vals(scale_vals(y_grad, -1), b_y)

    def compute_stationarity_residual(self, x_val, y_val, x_hat, y_hat, x_grad, y_grad):
        """Algorithm 5, line 25, retaining the original norm and tolerance.

        Original: ((x-x_hat)/eta, (y_hat-y)/eta) - (g_old-g_new).
        Box path: use the equivalent clipped-gradient quotients above and
        add g_new-g_old. Unlike subtracting nearby coordinates, these
        quotients preserve interior gradients when eta is extremely small.
        Explicit differences additionally cancel common constant gradient
        terms algebraically. Generic prox callers retain the old quotients.
        """
        if self.gradient_difference_func is None:
            gx_hat, gy_hat = self.compute_h_bar_gradient(x_hat, y_hat)
            dx = [new-old for new, old in zip(gx_hat, x_grad)]
            dy = [new-old for new, old in zip(gy_hat, y_grad)]
        else:
            with torch.no_grad():
                dx, dy = _validate_gradient_result(
                    self.gradient_difference_func(x_val, y_val, x_hat, y_hat), x_val, y_val,
                )
        if self.box_bounds is None:
            qx = scale_vals(add_vals(x_val, scale_vals(x_hat, -1)), 1 / self.zeta_hat)
            qy = scale_vals(add_vals(y_hat, scale_vals(y_val, -1)), 1 / self.zeta_hat)
        else:
            qx = [_box_projection_quotient([v], [g], self.zeta_hat, bounds)[0]
                  for v, g, bounds in zip(x_val, x_grad, self._min_box_bounds)]
            qy = [_box_projection_quotient([v], [g], self.zeta_hat, bounds, maximize=True)[0]
                  for v, g, bounds in zip(y_val, y_grad, self._max_box_bounds)]
        return add_vals(qx, dx), add_vals(qy, dy)

    def run(self):
        # Algorithm 5, Input: z^0 = z_f^0 = -sigma_x*x^0, y^0 = y_f^0.
        z_k = z_f_k = scale_vals(clone_vals(self.get_x()), -self.sigma_x)
        y_k = y_f_k = clone_vals(self.get_y())

        num_outer_iters = 0
        num_inner_iters = 0
        final_delta = None
        terminated = False
        termination_reason = "max_iter"

        # Algorithm 5, line 1: this implementation adds a finite iteration cap.
        for k in range(self.max_iter):
            # Line 2: extrapolated (z_g^k, y_g^k).
            z_g_k = add_vals(
                scale_vals(z_k, self.alpha_bar),
                scale_vals(z_f_k, 1 - self.alpha_bar),
            )
            y_g_k = add_vals(
                scale_vals(y_k, self.alpha_bar),
                scale_vals(y_f_k, 1 - self.alpha_bar),
            )

            # Line 3: (x^{k,-1}, y^{k,-1}).
            x_k_m1 = scale_vals(z_g_k, -1 / self.sigma_x)
            y_k_m1 = clone_vals(y_g_k)

            # Lines 4-5: initial proximal steps.
            a_x_k, a_y_k = self.compute_a_k(x_k_m1, y_k_m1, z_g_k, y_g_k)
            x_tmp = add_vals(x_k_m1, scale_vals(a_x_k, -self.zeta * self.gamma_x))
            y_tmp = add_vals(y_k_m1, scale_vals(a_y_k, -self.zeta * self.gamma_y))
            x_k_0 = x_k_t = compute_prox(x_tmp, self.prox_x, self.zeta * self.gamma_x)
            y_k_0 = y_k_t = compute_prox(y_tmp, self.prox_y, self.zeta * self.gamma_y)
            # Lines 6-7: initial proximal normal vectors.
            b_x_k_t = scale_vals(
                add_vals(x_tmp, scale_vals(x_k_t, -1)),
                1 / (self.zeta * self.gamma_x),
            )
            b_y_k_t = scale_vals(
                add_vals(y_tmp, scale_vals(y_k_t, -1)),
                1 / (self.zeta * self.gamma_y),
            )

            # Line 8: initialize the inner iteration.
            t = 0
            while True:
                # Input: beta_t = 2/(t+3); existing lr multiplier is an extension.
                beta_t = self.lr * 2 / (t + 3)

                # Certify the original SCSC objective at this feasible candidate,
                # before asking the auxiliary relative test for another update.
                # Reuse these gradients for a_k if the certificate does not pass.
                x_grad_t, y_grad_t = self.compute_h_bar_gradient(x_k_t, y_k_t)
                inner_rx, inner_ry = self.compute_inner_stationarity_residual(
                    x_k_t, y_k_t, x_grad_t, y_grad_t, b_x_k_t, b_y_k_t,
                )
                inner_delta = compute_norm(inner_rx + inner_ry)
                if not torch.isfinite(inner_delta):
                    raise RuntimeError(
                        f"Minimax_SCSC encountered non-finite inner stationarity delta at outer={k}, inner={t}"
                    )
                # The following is therefore a heuristic
                if inner_delta <= self.tau:
                    self.assign_x(x_k_t)
                    self.assign_y(y_k_t)
                    final_delta = float(inner_delta.item())
                    self._log(
                        f"Minimax_SCSC outer={k} inner_t={t} delta={final_delta:.3e} "
                        f"tau={self.tau:.3e} termination_reason=inner_certificate"
                    )
                    return {
                        "num_outer_iters": k + 1,
                        "num_inner_iters": num_inner_iters + t,
                        "final_delta": final_delta,
                        "terminated": True,
                        "termination_reason": "inner_certificate",
                    }

                # Line 9: stop the inner loop when the residual inequality holds.
                a_x_t, a_y_t = self._compute_a_k_from_gradient(
                    x_k_t, y_k_t, z_g_k, y_g_k, x_grad_t, y_grad_t,
                )
                x_tmp = add_vals(a_x_t, b_x_k_t)
                y_tmp = add_vals(a_y_t, b_y_k_t)
                lhs = self.gamma_x * compute_norm(x_tmp) ** 2 + self.gamma_y * compute_norm(y_tmp) ** 2
                x_diff = add_vals(x_k_t, scale_vals(x_k_m1, -1))
                y_diff = add_vals(y_k_t, scale_vals(y_k_m1, -1))
                rhs = (1 / self.gamma_x) * compute_norm(x_diff) ** 2 + (1 / self.gamma_y) * compute_norm(y_diff) ** 2
                if not torch.isfinite(lhs) or not torch.isfinite(rhs):
                    raise RuntimeError(
                        f"Minimax_SCSC encountered non-finite line-search values "
                        f"at outer={k}, inner={t}: lhs={float(lhs.item())}, rhs={float(rhs.item())}"
                    )
                if lhs <= rhs:
                    break

                # Lines 10-11: half-step iterates.
                x_diff = add_vals(x_k_0, scale_vals(x_k_t, -1))
                y_diff = add_vals(y_k_0, scale_vals(y_k_t, -1))
                x_k_t_half = add_vals(x_k_t, scale_vals(x_diff, beta_t))
                x_k_t_half = add_vals(x_k_t_half, scale_vals(x_tmp, -self.zeta * self.gamma_x))
                y_k_t_half = add_vals(y_k_t, scale_vals(y_diff, beta_t))
                y_k_t_half = add_vals(y_k_t_half, scale_vals(y_tmp, -self.zeta * self.gamma_y))

                # Lines 12-13: full proximal steps using the half-step gradients.
                a_x_t_half, a_y_t_half = self.compute_a_k(x_k_t_half, y_k_t_half, z_g_k, y_g_k)
                x_k_tp1_before_prox = add_vals(x_k_t, scale_vals(x_diff, beta_t))
                x_k_tp1_before_prox = add_vals(
                    x_k_tp1_before_prox,
                    scale_vals(a_x_t_half, -self.zeta * self.gamma_x),
                )
                x_k_tp1 = compute_prox(x_k_tp1_before_prox, self.prox_x, self.zeta * self.gamma_x)
                y_k_tp1_before_prox = add_vals(y_k_t, scale_vals(y_diff, beta_t))
                y_k_tp1_before_prox = add_vals(
                    y_k_tp1_before_prox,
                    scale_vals(a_y_t_half, -self.zeta * self.gamma_y),
                )
                y_k_tp1 = compute_prox(y_k_tp1_before_prox, self.prox_y, self.zeta * self.gamma_y)

                # Lines 14-15: update proximal normal vectors.
                b_x_k_tp1 = scale_vals(
                    add_vals(x_k_tp1_before_prox, scale_vals(x_k_tp1, -1)),
                    1 / (self.zeta * self.gamma_x),
                )
                b_y_k_tp1 = scale_vals(
                    add_vals(y_k_tp1_before_prox, scale_vals(y_k_tp1, -1)),
                    1 / (self.zeta * self.gamma_y),
                )

                # Lines 16-17: advance the inner iteration.
                t += 1
                x_k_t = x_k_tp1
                y_k_t = y_k_tp1
                b_x_k_t = b_x_k_tp1
                b_y_k_t = b_y_k_tp1

                if self.verbose and t % max(1000, self.log_every * 1000) == 0:
                    self._log(
                        f"Minimax_SCSC inner k={k} t={t} lhs={float(lhs.item()):.3e} rhs={float(rhs.item()):.3e}"
                    )

            # Line 18: accept the inner solution.
            x_f_kp1, y_f_kp1 = x_k_t, y_k_t

            # Line 19: (z_f^{k+1}, w_f^{k+1}).
            x_grad_hat, y_grad_hat = self.compute_a_k(x_f_kp1, y_f_kp1, None, None, return_h_hat_grad=True)
            z_f_kp1 = add_vals(x_grad_hat, b_x_k_t)
            w_f_kp1 = add_vals(scale_vals(y_grad_hat, -1), b_y_k_t)

            # Line 20: update z^{k+1}.
            z_diff = add_vals(z_f_kp1, scale_vals(z_k, -1))
            z_kp1 = add_vals(z_k, scale_vals(z_diff, self.eta_z / self.sigma_x))
            z_tmp = add_vals(x_f_kp1, scale_vals(z_f_kp1, 1 / self.sigma_x))
            z_kp1 = add_vals(z_kp1, scale_vals(z_tmp, -self.eta_z))

            # Line 21: update y^{k+1}.
            y_diff = add_vals(y_f_kp1, scale_vals(y_k, -1))
            y_kp1 = add_vals(y_k, scale_vals(y_diff, self.eta_y * self.sigma_y))
            y_tmp = add_vals(w_f_kp1, scale_vals(y_f_kp1, self.sigma_y))
            y_kp1 = add_vals(y_kp1, scale_vals(y_tmp, -self.eta_y))

            # Line 22: recover x^{k+1}.
            x_kp1 = scale_vals(z_kp1, -1 / self.sigma_x)

            # Lines 23-24: projected certificate points (unchanged step size).
            x_grad, y_grad = self.compute_h_bar_gradient(x_kp1, y_kp1)
            x_hat_kp1 = add_vals(x_kp1, scale_vals(x_grad, -self.zeta_hat))
            x_hat_kp1 = compute_prox(x_hat_kp1, self.prox_x, self.zeta_hat)
            y_hat_kp1 = add_vals(y_kp1, scale_vals(y_grad, self.zeta_hat))
            y_hat_kp1 = compute_prox(y_hat_kp1, self.prox_y, self.zeta_hat)

            # Preserve caller-visible outputs even when the gradient oracle is pure.
            self.assign_x(x_hat_kp1)
            self.assign_y(y_hat_kp1)

            # Line 25: optional algebraic reformulation documented in the helper.
            delta_x, delta_y = self.compute_stationarity_residual(
                x_kp1, y_kp1, x_hat_kp1, y_hat_kp1, x_grad, y_grad,
            )

            delta = compute_norm(delta_x + delta_y)
            if not torch.isfinite(delta):
                raise RuntimeError(f"Minimax_SCSC encountered non-finite stationarity delta at outer={k}")
            final_delta = float(delta.item())
            num_outer_iters = k + 1
            num_inner_iters+= t

            if self.verbose and (
                k % self.log_every == 0 or final_delta < self.tau or num_outer_iters == self.max_iter
            ):
                self._log(
                    f"Minimax_SCSC outer={k} inner_t={t} delta={final_delta:.3e} tau={self.tau:.3e}"
                )

            # Line 25.
            if delta <= self.tau:
                terminated = True
                termination_reason = "outer_certificate"
                break

            # Line 26: carry the state into the next outer iteration.
            z_k = z_kp1
            y_k = y_kp1
            z_f_k = z_f_kp1
            y_f_k = y_f_kp1

        return {
            "num_outer_iters": num_outer_iters,
            "num_inner_iters": num_inner_iters,
            "final_delta": final_delta,
            "terminated": terminated,
            "termination_reason": termination_reason,
        }


def _evaluate_objective(objective_func, params_x, params_y):
    if objective_func is None:
        return None
    with torch.no_grad():
        value = objective_func(params_x, params_y)
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def _evaluate_metrics(metrics_func, params_x, params_y):
    if metrics_func is None:
        return None
    with torch.no_grad():
        metrics = metrics_func(params_x, params_y)
    if metrics is None:
        return None
    if not isinstance(metrics, dict):
        raise TypeError("metrics_func must return a dict or None")
    normalized = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            normalized[key] = float(value.item())
        else:
            normalized[key] = float(value)
    return normalized


class SAPD_SCSC(Optimizer):
    """
    Algorithm 1 of SAPD+:
        https://arxiv.org/pdf/2205.15084
    """

    def __init__(
        self,
        params_x,
        params_y,
        h_bar,
        mu_x,
        mu_y,
        lip,
        prox_x,
        prox_y,
        lip_tau=None,
        theta=1.0,
        max_iter=1000,
        verbose=False,
        log_every=1,
    ):
        if max_iter <= 0:
            raise ValueError(f"Invalid max_iter: {max_iter}")
        if log_every <= 0:
            raise ValueError(f"Invalid log_every: {log_every}")
        if theta < 0:
            raise ValueError(f"Invalid theta: {theta}")

        defaults = {}
        super().__init__([{"params": params_x}, {"params": params_y}], defaults)
        if len(self.param_groups) != 2:
            raise ValueError("SAPD_SCSC expects exactly two parameter groups")

        self.h_bar = h_bar
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.lip = lip
        self.prox_x = prox_x
        self.prox_y = prox_y
        self.verbose = verbose
        self.log_every = log_every

        self.sigma =  1 / lip
        self.tau = 1 / lip_tau if lip_tau is not None else 1 / lip
        self.theta = theta
        theoretical_num_iter = 33 * max(4/(mu_x * self.tau), 8 / (mu_y * self.sigma))
        theoretical_num_iter = int( max(theoretical_num_iter, 1) )
        self.max_iter = int( min(max_iter, theoretical_num_iter) )
        print(f"theoretical_num_iter={theoretical_num_iter}, max_iter={self.max_iter}, lip={self.lip}")

    def get_x(self):
        return self.param_groups[0]["params"]

    def get_y(self):
        return self.param_groups[1]["params"]

    @torch.no_grad()
    def assign_x(self, val):
        assign_vals(self.param_groups[0]["params"], val)

    @torch.no_grad()
    def assign_y(self, val):
        assign_vals(self.param_groups[1]["params"], val)

    def compute_h_and_gradient(self, x_val, y_val, var='x'):
        self.assign_x(x_val)
        self.assign_y(y_val)

        loss = self.h_bar()
        if var=='x':
            output_grad = torch.autograd.grad(loss, self.get_x(), allow_unused=True)
        else:
            output_grad = torch.autograd.grad(loss, self.get_y(), allow_unused=True)
        return loss, [p.detach() for p in output_grad]

    def run(self):
        x_k = clone_vals(self.get_x())
        y_k = clone_vals(self.get_y())
        q_k = zero_vals(self.get_y())

        x_output = clone_vals(x_k)
        y_output = clone_vals(y_k)
        final_delta = None
        terminated = False

        for k in range(self.max_iter):
            # line 5
            _, grad_y = self.compute_h_and_gradient(x_k, y_k, var='y')
            s_k = add_vals(grad_y, scale_vals(q_k, self.theta))

            # line 6
            y_kp1 = add_vals(y_k, scale_vals(s_k, self.sigma))
            y_kp1 = compute_prox(y_kp1, self.prox_y, self.sigma)

            # line 7
            _, grad_x = self.compute_h_and_gradient(x_k, y_kp1, var='x')
            x_kp1 = add_vals(x_k, scale_vals(grad_x, - self.tau))
            x_kp1 = compute_prox(x_kp1, self.prox_x, self.tau)

            # line 8
            _, grad_y_kp1 = self.compute_h_and_gradient(x_kp1, y_kp1, var='y')
            q_k = add_vals(grad_y_kp1, scale_vals(grad_y, -1))
            final_delta = float(
                compute_norm(
                    add_vals(x_kp1, scale_vals(x_k, -1))
                    + add_vals(y_kp1, scale_vals(y_k, -1))
                ).item()
            )
            x_k = x_kp1
            y_k = y_kp1

            # Compute running average.
            N = k+1
            x_output = blend_vals(x_output, x_kp1, N / (N+1), 1 / (N+1))
            y_output = blend_vals(y_output, y_kp1, N / (N+1), 1 / (N+1))

        # add metrics here.
        self.assign_x(x_output)
        self.assign_y(y_output)
        return {
            "num_outer_iters": self.max_iter,
            "num_inner_iters": self.max_iter,
            "final_delta": final_delta,
            "terminated": terminated,
        }


def optimize_NCWC(
    params_x,
    params_y,
    h_func,
    lip_h,
    D_y,
    prox_x,
    prox_y,
    epsilon,
    epsilon_0,
    sub_routine='deterministic',
    lr=1,
    max_iter=1000,
    subproblem_max_iter=None,
    verbose=False,
    log_every=1,
    objective_func=None,
    metrics_func=None,
    progress_callback=None,
    lip_tau=None,
    theta=1.0,
    gradient_func=None,
    gradient_difference_func=None,
    box_bounds=None,
):
    """
    Algorithm 6 of FOP for non-convex-weakly-concave minimax problems.
    Algorithm B.2 of SMO outer wrapper for the sigma_y = 0 case: it repeatedly regularizes the minimax problem and calls the strongly-convex-strongly-concave inner solver above.
    Optional callbacks describe h_func BEFORE regularization. They use the
    same tensor-list contract as Minimax_SCSC; only deterministic mode
    supports explicit gradients and declared box projections.
    """
    if lip_h <= 0:
        raise ValueError(f"Invalid lip_h: {lip_h}")
    if D_y <= 0:
        raise ValueError(f"Invalid D_y: {D_y}")
    if epsilon <= 0:
        raise ValueError(f"Invalid epsilon: {epsilon}")
    if epsilon_0 <= 0:
        raise ValueError(f"Invalid epsilon_0: {epsilon_0}")
    if max_iter <= 0:
        raise ValueError(f"Invalid max_iter: {max_iter}")
    if subproblem_max_iter is not None and subproblem_max_iter <= 0:
        raise ValueError(f"Invalid subproblem_max_iter: {subproblem_max_iter}")
    if log_every <= 0:
        raise ValueError(f"Invalid log_every: {log_every}")
    if lip_tau is not None and lip_tau <= 0:
        raise ValueError(f"Invalid lip_tau: {lip_tau}")
    if objective_func is not None and not callable(objective_func):
        raise TypeError("objective_func must be callable")
    if metrics_func is not None and not callable(metrics_func):
        raise TypeError("metrics_func must be callable")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    for callback in (gradient_func, gradient_difference_func):
        if callback is not None and not callable(callback):
            raise TypeError("Gradient callbacks must be callable or None")
    if sub_routine != 'deterministic' and any(
        option is not None for option in (gradient_func, gradient_difference_func, box_bounds)
    ):
        raise ValueError("Explicit gradients and box residuals require deterministic NCWC")

    params_x = list(params_x)
    params_y = list(params_y)
    x_k = clone_vals(params_x)
    y_0 = clone_vals(params_y)
    inner_max_iter = max_iter if subproblem_max_iter is None else subproblem_max_iter

    num_outer_iters = 0
    num_inner_iters = 0
    final_diff = None
    terminated = False

    if progress_callback is not None:
        progress_callback(
            {
                "completed_outer_iters": 0,
                "epsilon_k": None,
                "final_diff": None,
                "terminated": False,
                "inner_terminated": False,
                "ncwc_terminated": False,
                "scsc_num_outer_iters": 0,
                "scsc_num_inner_iters": 0,
                "scsc_final_delta": None,
                "scsc_termination_reason": None,
                "call_cumulative_scsc_inner_iters": 0,
                "objective": _evaluate_objective(objective_func, params_x, params_y),
                "metrics": _evaluate_metrics(metrics_func, params_x, params_y),
            }
        )

    for k in tqdm.trange(max_iter, desc="optimize_NCWC", unit="outer iter", disable=not verbose):

        def h_alg6():
            f = h_func()
            s_y = 0
            for p, p0 in zip(params_y, y_0):
                s_y += torch.sum(torch.square(p - p0))
            s_x = 0
            for p, p0 in zip(params_x, x_k):
                s_x += torch.sum(torch.square(p - p0))
            return f - (epsilon * s_y) / (4 * D_y) + lip_h * s_x

        epsilon_k = epsilon_0 / (k + 1)
        sigma_y = epsilon / (2 * D_y)
        lip_k = 3 * lip_h + epsilon / (2 * D_y)
        if sub_routine == 'deterministic':
            grad_k, grad_diff_k = _regularize_gradient_callbacks(
                gradient_func, gradient_difference_func, x_k, y_0, lip_h, sigma_y,
            )
            solver = Minimax_SCSC(
                params_x,
                params_y,
                h_alg6,
                sigma_x=lip_h,
                sigma_y=sigma_y,
                lip=lip_k,
                prox_x=prox_x,
                prox_y=prox_y,
                tau=epsilon_k,
                lr=lr,
                max_iter=inner_max_iter,
                verbose=verbose,
                log_every=log_every,
                gradient_func=grad_k,
                gradient_difference_func=grad_diff_k,
                box_bounds=box_bounds,
            )
        elif sub_routine == 'stochastic':
            solver = SAPD_SCSC(
                params_x,
                params_y,
                h_alg6,
                mu_x=lip_h,
                mu_y=sigma_y,
                lip=lip_k,
                prox_x=prox_x,
                prox_y=prox_y,
                lip_tau=lip_tau,
                theta=theta,
                max_iter=inner_max_iter,
                verbose=verbose,
                log_every=log_every,
            )
        else:
            raise NotImplementedError("This sub_routine is not yet available.")
        solver_stats = solver.run()

        x_kp1 = [p.clone().detach() for p in params_x]
        diff = compute_norm(add_vals(x_kp1, scale_vals(x_k, -1)))
        final_diff = float(diff.item())
        num_outer_iters = k + 1
        num_inner_iters += solver_stats["num_inner_iters"]
        exit_criteria_gap = diff - epsilon / (4 * lip_h)
        ncwc_terminated = exit_criteria_gap <= 0

        if progress_callback is not None:
            progress_callback(
                {
                    "completed_outer_iters": num_outer_iters,
                    "epsilon_k": epsilon_k,
                    "final_diff": final_diff,
                    "terminated": solver_stats["terminated"],
                    "inner_terminated": solver_stats["terminated"],
                    "ncwc_terminated": ncwc_terminated,
                    "scsc_num_outer_iters": solver_stats["num_outer_iters"],
                    "scsc_num_inner_iters": solver_stats["num_inner_iters"],
                    "scsc_final_delta": solver_stats.get("final_delta"),
                    "scsc_termination_reason": solver_stats.get("termination_reason"),
                    "call_cumulative_scsc_inner_iters": num_inner_iters,
                    "objective": _evaluate_objective(objective_func, params_x, params_y),
                    "metrics": _evaluate_metrics(metrics_func, params_x, params_y),
                }
            )

        if verbose and (
            k % log_every == 0 or final_diff <= epsilon / (4 * lip_h) or num_outer_iters == max_iter
        ):
            print(
                f"optimize_NCWC outer={k} epsilon_k={epsilon_k:.3e} exit_criteria_gap={exit_criteria_gap:.3e} "
                f"inner_terminated={solver_stats['terminated']}",
                flush=True,
            )

        if ncwc_terminated:
            terminated = True
            break

        x_k = x_kp1

    # TODO: add metrics here and stop early.

    return {
        "num_outer_iters": num_outer_iters,
        "num_inner_iters": num_inner_iters,
        "final_diff": final_diff,
        "terminated": terminated,
    }
