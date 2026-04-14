import math

import torch, tqdm
from torch.optim import Optimizer

from utils import clone_vals, zero_vals, scale_vals, add_vals, blend_vals, assign_vals, compute_prox, compute_norm


class Minimax_SCSC(Optimizer):
    """
    Algorithm 5 of FOP for strongly-convex-strongly-concave minimax problems.
    Algorithm B.1 of SMO, inner solver reused by the higher-level wrappers.
    """

    def __init__(self, params_x, params_y, h_bar, sigma_x, sigma_y, lip, prox_x, prox_y, tau, lr=1, max_iter=1000, verbose=False, log_every=1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if tau <= 0.0:
            raise ValueError(f"Invalid tau: {tau}")
        if max_iter <= 0:
            raise ValueError(f"Invalid max_iter: {max_iter}")
        if log_every <= 0:
            raise ValueError(f"Invalid log_every: {log_every}")

        defaults = dict(lr=lr)
        super().__init__([{"params": params_x}, {"params": params_y}], defaults)
        if len(self.param_groups) != 2:
            raise ValueError("Minimax_SCSC expects exactly two parameter groups")

        self.h_bar = h_bar
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
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

    def run(self):
        z_k = z_f_k = scale_vals(clone_vals(self.get_x()), -self.sigma_x)
        y_k = y_f_k = clone_vals(self.get_y())

        num_outer_iters = 0
        num_inner_iters = 0
        final_delta = None
        terminated = False

        for k in range(self.max_iter):
            z_g_k = add_vals(
                scale_vals(z_k, self.alpha_bar),
                scale_vals(z_f_k, 1 - self.alpha_bar),
            )
            y_g_k = add_vals(
                scale_vals(y_k, self.alpha_bar),
                scale_vals(y_f_k, 1 - self.alpha_bar),
            )

            x_k_m1 = scale_vals(z_g_k, -1 / self.sigma_x)
            y_k_m1 = clone_vals(y_g_k)

            a_x_k, a_y_k = self.compute_a_k(x_k_m1, y_k_m1, z_g_k, y_g_k)
            x_tmp = add_vals(x_k_m1, scale_vals(a_x_k, -self.zeta * self.gamma_x))
            y_tmp = add_vals(y_k_m1, scale_vals(a_y_k, -self.zeta * self.gamma_y))
            x_k_0 = x_k_t = compute_prox(x_tmp, self.prox_x, self.zeta * self.gamma_x)
            y_k_0 = y_k_t = compute_prox(y_tmp, self.prox_y, self.zeta * self.gamma_y)
            b_x_k_t = scale_vals(
                add_vals(x_tmp, scale_vals(x_k_t, -1)),
                1 / (self.zeta * self.gamma_x),
            )
            b_y_k_t = scale_vals(
                add_vals(y_tmp, scale_vals(y_k_t, -1)),
                1 / (self.zeta * self.gamma_y),
            )

            t = 0
            while True:
                beta_t = self.lr * 2 / (t + 3)

                a_x_t, a_y_t = self.compute_a_k(x_k_t, y_k_t, z_g_k, y_g_k)
                x_tmp = add_vals(a_x_t, b_x_k_t)
                y_tmp = add_vals(a_y_t, b_y_k_t)
                lhs = self.gamma_x * compute_norm(x_tmp) ** 2 + self.gamma_y * compute_norm(y_tmp) ** 2
                x_diff = add_vals(x_k_t, scale_vals(x_k_m1, -1))
                y_diff = add_vals(y_k_t, scale_vals(y_k_m1, -1))
                rhs = (1 / self.gamma_x) * compute_norm(x_diff) ** 2 + (1 / self.gamma_y) * compute_norm(y_diff) ** 2
                if lhs <= rhs + 1e-8 or t > 10000:
                    break

                x_diff = add_vals(x_k_0, scale_vals(x_k_t, -1))
                y_diff = add_vals(y_k_0, scale_vals(y_k_t, -1))
                x_k_t_half = add_vals(x_k_t, scale_vals(x_diff, beta_t))
                x_k_t_half = add_vals(x_k_t_half, scale_vals(x_tmp, -self.zeta * self.gamma_x))
                y_k_t_half = add_vals(y_k_t, scale_vals(y_diff, beta_t))
                y_k_t_half = add_vals(y_k_t_half, scale_vals(y_tmp, -self.zeta * self.gamma_y))

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

                b_x_k_tp1 = scale_vals(
                    add_vals(x_k_tp1_before_prox, scale_vals(x_k_tp1, -1)),
                    1 / (self.zeta * self.gamma_x),
                )
                b_y_k_tp1 = scale_vals(
                    add_vals(y_k_tp1_before_prox, scale_vals(y_k_tp1, -1)),
                    1 / (self.zeta * self.gamma_y),
                )

                t += 1
                x_k_t = x_k_tp1
                y_k_t = y_k_tp1
                b_x_k_t = b_x_k_tp1
                b_y_k_t = b_y_k_tp1

                if self.verbose and t % max(1000, self.log_every * 1000) == 0:
                    self._log(
                        f"Minimax_SCSC inner k={k} t={t} lhs={float(lhs.item()):.3e} rhs={float(rhs.item()):.3e}"
                    )

            x_f_kp1, y_f_kp1 = x_k_t, y_k_t

            x_grad_hat, y_grad_hat = self.compute_a_k(x_f_kp1, y_f_kp1, None, None, return_h_hat_grad=True)
            z_f_kp1 = add_vals(x_grad_hat, b_x_k_t)
            w_f_kp1 = add_vals(scale_vals(y_grad_hat, -1), b_y_k_t)

            z_diff = add_vals(z_f_kp1, scale_vals(z_k, -1))
            z_kp1 = add_vals(z_k, scale_vals(z_diff, self.eta_z / self.sigma_x))
            z_tmp = add_vals(x_f_kp1, scale_vals(z_f_kp1, 1 / self.sigma_x))
            z_kp1 = add_vals(z_kp1, scale_vals(z_tmp, -self.eta_z))

            y_diff = add_vals(y_f_kp1, scale_vals(y_k, -1))
            y_kp1 = add_vals(y_k, scale_vals(y_diff, self.eta_y * self.sigma_y))
            y_tmp = add_vals(w_f_kp1, scale_vals(y_f_kp1, self.sigma_y))
            y_kp1 = add_vals(y_kp1, scale_vals(y_tmp, -self.eta_y))

            x_kp1 = scale_vals(z_kp1, -1 / self.sigma_x)

            x_grad, y_grad = self.compute_h_bar_gradient(x_kp1, y_kp1)
            x_hat_kp1 = add_vals(x_kp1, scale_vals(x_grad, -self.zeta_hat))
            x_hat_kp1 = compute_prox(x_hat_kp1, self.prox_x, self.zeta_hat)
            y_hat_kp1 = add_vals(y_kp1, scale_vals(y_grad, self.zeta_hat))
            y_hat_kp1 = compute_prox(y_hat_kp1, self.prox_y, self.zeta_hat)

            self.assign_x(x_hat_kp1)
            self.assign_y(y_hat_kp1)

            x_grad_hat, y_grad_hat = self.compute_h_bar_gradient(x_hat_kp1, y_hat_kp1)
            grad_delta_x = add_vals(x_grad, scale_vals(x_grad_hat, -1))
            delta_x = add_vals(x_kp1, scale_vals(x_hat_kp1, -1))
            delta_x = add_vals(
                scale_vals(delta_x, 1 / self.zeta_hat),
                scale_vals(grad_delta_x, -1),
            )

            grad_delta_y = add_vals(y_grad, scale_vals(y_grad_hat, -1))
            delta_y = add_vals(y_hat_kp1, scale_vals(y_kp1, -1))
            delta_y = add_vals(
                scale_vals(delta_y, 1 / self.zeta_hat),
                scale_vals(grad_delta_y, -1),
            )

            delta = compute_norm(delta_x + delta_y)
            final_delta = float(delta.item())
            num_outer_iters = k + 1
            num_inner_iters+= t

            # if self.verbose and (
            #     k % self.log_every == 0 or final_delta < self.tau or num_outer_iters == self.max_iter
            # ):
            #     self._log(
            #         f"Minimax_SCSC outer={k} inner_t={t} delta={final_delta:.3e} tau={self.tau:.3e}"
            #     )

            if delta < self.tau:
                terminated = True
                break

            z_k = z_kp1
            y_k = y_kp1
            z_f_k = z_f_kp1
            y_f_k = y_f_kp1

        return {
            "num_outer_iters": num_outer_iters,
            "num_inner_iters": num_inner_iters,
            "final_delta": final_delta,
            "terminated": terminated,
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

    def __init__(self, params_x, params_y, h_bar, mu_x, mu_y, lip, prox_x, prox_y, max_iter=1000, verbose=False, log_every=1):
        if max_iter <= 0:
            raise ValueError(f"Invalid max_iter: {max_iter}")
        if log_every <= 0:
            raise ValueError(f"Invalid log_every: {log_every}")

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
        self.tau = 1 / lip
        self.theta = 1
        theoretical_num_iter = 33 * max(4/(mu_x * self.tau), 8 / (mu_y * self.sigma))
        theoretical_num_iter = int( max(theoretical_num_iter, 1) )
        self.max_iter = int( min(max_iter, theoretical_num_iter) )
        print(f"theoretical_num_iter={theoretical_num_iter}, max_iter={self.max_iter}")

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
        print(f"k={0}: x_0={x_k}, y_0={y_k}")

        x_output = clone_vals(x_k)
        y_output = clone_vals(y_k)

        for k in range(self.max_iter):
            # line 5
            grad_y = self.compute_h_and_gradient(x_k, y_k, var='y')
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
            x_k = x_kp1
            y_k = y_kp1
            # print(f"k={k}: x_kp1={x_kp1}, y_kp1={y_kp1}")

            # Compute running average.
            N = k+1
            x_output = blend_vals(x_output, x_kp1, N / (N+1), 1 / (N+1))
            y_output = blend_vals(y_output, y_kp1, N / (N+1), 1 / (N+1))

        # add metrics here.
        self.assign_x(x_output)
        self.assign_y(y_output)
        print(f"k={k}: x_output={x_output}, y_output={y_output}")
        return {"num_inner_iters": k}


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
    verbose=False,
    log_every=1,
    objective_func=None,
    metrics_func=None,
    progress_callback=None,
):
    """
    Algorithm 6 of FOP for non-convex-weakly-concave minimax problems.
    Algorithm B.2 of SMO outer wrapper for the sigma_y = 0 case: it repeatedly regularizes the minimax problem and calls the strongly-convex-strongly-concave inner solver above.
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
    if log_every <= 0:
        raise ValueError(f"Invalid log_every: {log_every}")
    if objective_func is not None and not callable(objective_func):
        raise TypeError("objective_func must be callable")
    if metrics_func is not None and not callable(metrics_func):
        raise TypeError("metrics_func must be callable")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")

    params_x = list(params_x)
    params_y = list(params_y)
    x_k = clone_vals(params_x)
    y_0 = clone_vals(params_y)

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
                max_iter=max_iter,
                verbose=verbose,
                log_every=log_every,
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
                max_iter=max_iter,
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
        ncwc_terminated = diff <= epsilon / (4 * lip_h)

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
                    "call_cumulative_scsc_inner_iters": num_inner_iters,
                    "objective": _evaluate_objective(objective_func, params_x, params_y),
                    "metrics": _evaluate_metrics(metrics_func, params_x, params_y),
                }
            )

        if verbose and (
            k % log_every == 0 or final_diff <= epsilon / (4 * lip_h) or num_outer_iters == max_iter
        ):
            print(
                f"optimize_NCWC outer={k} epsilon_k={epsilon_k:.3e} diff={final_diff:.3e} "
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
