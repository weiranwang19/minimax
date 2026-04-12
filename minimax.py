import math

import torch, tqdm
from torch.optim import Optimizer

from utils import clone_vals, scale_vals, add_vals, assign_vals, compute_prox, compute_norm


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
    def get_x_copy(self):
        return clone_vals(self.get_x())

    @torch.no_grad()
    def get_y_copy(self):
        return clone_vals(self.get_y())

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


    @torch.no_grad()
    def add_vals(self, vals1, vals2):
        return [a + b for a, b in zip(vals1, vals2)]

    def compute_h_bar_gradient(self, x_val, y_val):
        x_copy = self.get_x_copy()
        y_copy = self.get_y_copy()
        self.assign_x(x_val)
        self.assign_y(y_val)

        loss = self.h_bar()
        self.zero_grad()
        loss.backward()
        x_grad = self.get_x_grad()
        y_grad = self.get_y_grad()

        self.assign_x(x_copy)
        self.assign_y(y_copy)
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
        z_k = z_f_k = scale_vals(self.get_x_copy(), -self.sigma_x)
        y_k = y_f_k = self.get_y_copy()

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


def optimize_NCWC(params_x, params_y, h_func, lip_h, D_y, prox_x, prox_y, epsilon, epsilon_0, lr=1, max_iter=1000, verbose=False, log_every=1):
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

    x_k = [p.clone().detach() for p in params_x]
    y_0 = [p.clone().detach() for p in params_y]

    num_outer_iters = 0
    final_diff = None
    terminated = False

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
        solver_stats = solver.run()

        x_kp1 = [p.clone().detach() for p in params_x]
        diff = compute_norm(solver.add_vals(x_kp1, solver.scale_vals(x_k, -1)))
        final_diff = float(diff.item())
        num_outer_iters = k + 1

        if verbose and (
            k % log_every == 0 or final_diff <= epsilon / (4 * lip_h) or num_outer_iters == max_iter
        ):
            print(
                f"optimize_NCWC outer={k} epsilon_k={epsilon_k:.3e} diff={final_diff:.3e} "
                f"inner_terminated={solver_stats['terminated']}",
                flush=True,
            )

        if diff <= epsilon / (4 * lip_h):
            terminated = True
            break

        x_k = x_kp1

    # TODO: add metrics here and stop early.

    return {
        "num_outer_iters": num_outer_iters,
        "final_diff": final_diff,
        "terminated": terminated,
    }

