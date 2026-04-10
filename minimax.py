import math

import torch, tqdm
from torch.optim import Optimizer


def compute_norm(vals):
    s = 0
    for v in vals:
        s += torch.sum(torch.square(v))
    return torch.sqrt(s)


def _expand_prox_spec(prox_func, count):
    return [prox_func] * count


def _scale_prox_spec(prox_func, scale):

    def scaled(v, coeff):
        return prox_func(v, scale * coeff)
    
    return scaled


def _positive_part_norm_sq(v):
    return torch.sum(torch.square(torch.maximum(v, 0.0)))


def compute_prox(vals, prox_funcs, prox_coeff):
    prox_vals = []
    for p, prox in zip(vals, prox_funcs):
        prox_vals.append(prox(p, prox_coeff))
    return prox_vals


class Minimax_SCSC(Optimizer):
    """
    Algorithm 5 of "First-order penalty methods for bilevel optimization"
    for strongly-convex-strongly-concave minimax problems.
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
    def copy_vars(self, vars):
        return [p.clone().detach() for p in vars]

    @torch.no_grad()
    def get_x_copy(self):
        return self.copy_vars(self.get_x())

    @torch.no_grad()
    def get_y_copy(self):
        return self.copy_vars(self.get_y())

    @torch.no_grad()
    def get_x_grad(self):
        return [p.grad for p in self.get_x()]

    @torch.no_grad()
    def get_y_grad(self):
        return [p.grad for p in self.get_y()]

    @torch.no_grad()
    def assign_x(self, val):
        for p, v in zip(self.param_groups[0]["params"], val):
            p.data.copy_(v)

    @torch.no_grad()
    def assign_y(self, val):
        for p, v in zip(self.param_groups[1]["params"], val):
            p.data.copy_(v)

    @torch.no_grad()
    def scale_vals(self, vals, s):
        return [s * p for p in vals]

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
        x_grad_hat = self.add_vals(x_grad, self.scale_vals(x_val, -self.sigma_x))
        y_grad_hat = self.add_vals(y_grad, self.scale_vals(y_val, self.sigma_y))
        if return_h_hat_grad:
            return x_grad_hat, y_grad_hat

        x_tmp = self.add_vals(x_val, self.scale_vals(z_g_k, -1 / self.sigma_x))
        a_x_k = self.add_vals(x_grad_hat, self.scale_vals(x_tmp, self.sigma_x / 2))

        y_tmp = self.add_vals(y_val, self.scale_vals(y_g_k, -1))
        y_tmp = self.add_vals(
            self.scale_vals(y_val, self.sigma_y),
            self.scale_vals(y_tmp, self.sigma_x / 8),
        )
        a_y_k = self.add_vals(self.scale_vals(y_grad_hat, -1), y_tmp)
        return a_x_k, a_y_k

    def run(self):
        z_k = z_f_k = self.scale_vals(self.get_x_copy(), -self.sigma_x)
        y_k = y_f_k = self.get_y_copy()

        num_outer_iters = 0
        final_delta = None
        terminated = False

        for k in range(self.max_iter):
            z_g_k = self.add_vals(
                self.scale_vals(z_k, self.alpha_bar),
                self.scale_vals(z_f_k, 1 - self.alpha_bar),
            )
            y_g_k = self.add_vals(
                self.scale_vals(y_k, self.alpha_bar),
                self.scale_vals(y_f_k, 1 - self.alpha_bar),
            )

            x_k_m1 = self.scale_vals(z_g_k, -1 / self.sigma_x)
            y_k_m1 = self.copy_vars(y_g_k)

            a_x_k, a_y_k = self.compute_a_k(x_k_m1, y_k_m1, z_g_k, y_g_k)
            x_tmp = self.add_vals(x_k_m1, self.scale_vals(a_x_k, -self.zeta * self.gamma_x))
            y_tmp = self.add_vals(y_k_m1, self.scale_vals(a_y_k, -self.zeta * self.gamma_y))
            x_k_0 = x_k_t = compute_prox(x_tmp, self.prox_x, self.zeta * self.gamma_x)
            y_k_0 = y_k_t = compute_prox(y_tmp, self.prox_y, self.zeta * self.gamma_y)
            b_x_k_t = self.scale_vals(
                self.add_vals(x_tmp, self.scale_vals(x_k_t, -1)),
                1 / (self.zeta * self.gamma_x),
            )
            b_y_k_t = self.scale_vals(
                self.add_vals(y_tmp, self.scale_vals(y_k_t, -1)),
                1 / (self.zeta * self.gamma_y),
            )

            t = 0
            while True:
                beta_t = self.lr * 2 / (t + 3)

                a_x_t, a_y_t = self.compute_a_k(x_k_t, y_k_t, z_g_k, y_g_k)
                x_tmp = self.add_vals(a_x_t, b_x_k_t)
                y_tmp = self.add_vals(a_y_t, b_y_k_t)
                lhs = self.gamma_x * compute_norm(x_tmp) ** 2 + self.gamma_y * compute_norm(y_tmp) ** 2
                x_diff = self.add_vals(x_k_t, self.scale_vals(x_k_m1, -1))
                y_diff = self.add_vals(y_k_t, self.scale_vals(y_k_m1, -1))
                rhs = (1 / self.gamma_x) * compute_norm(x_diff) ** 2 + (1 / self.gamma_y) * compute_norm(y_diff) ** 2
                if lhs <= rhs + 1e-8 or t > 10000:
                    break

                x_diff = self.add_vals(x_k_0, self.scale_vals(x_k_t, -1))
                y_diff = self.add_vals(y_k_0, self.scale_vals(y_k_t, -1))
                x_k_t_half = self.add_vals(x_k_t, self.scale_vals(x_diff, beta_t))
                x_k_t_half = self.add_vals(x_k_t_half, self.scale_vals(x_tmp, -self.zeta * self.gamma_x))
                y_k_t_half = self.add_vals(y_k_t, self.scale_vals(y_diff, beta_t))
                y_k_t_half = self.add_vals(y_k_t_half, self.scale_vals(y_tmp, -self.zeta * self.gamma_y))

                a_x_t_half, a_y_t_half = self.compute_a_k(x_k_t_half, y_k_t_half, z_g_k, y_g_k)
                x_k_tp1_before_prox = self.add_vals(x_k_t, self.scale_vals(x_diff, beta_t))
                x_k_tp1_before_prox = self.add_vals(
                    x_k_tp1_before_prox,
                    self.scale_vals(a_x_t_half, -self.zeta * self.gamma_x),
                )
                x_k_tp1 = compute_prox(x_k_tp1_before_prox, self.prox_x, self.zeta * self.gamma_x)
                y_k_tp1_before_prox = self.add_vals(y_k_t, self.scale_vals(y_diff, beta_t))
                y_k_tp1_before_prox = self.add_vals(
                    y_k_tp1_before_prox,
                    self.scale_vals(a_y_t_half, -self.zeta * self.gamma_y),
                )
                y_k_tp1 = compute_prox(y_k_tp1_before_prox, self.prox_y, self.zeta * self.gamma_y)

                b_x_k_tp1 = self.scale_vals(
                    self.add_vals(x_k_tp1_before_prox, self.scale_vals(x_k_tp1, -1)),
                    1 / (self.zeta * self.gamma_x),
                )
                b_y_k_tp1 = self.scale_vals(
                    self.add_vals(y_k_tp1_before_prox, self.scale_vals(y_k_tp1, -1)),
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
            z_f_kp1 = self.add_vals(x_grad_hat, b_x_k_t)
            w_f_kp1 = self.add_vals(self.scale_vals(y_grad_hat, -1), b_y_k_t)

            z_diff = self.add_vals(z_f_kp1, self.scale_vals(z_k, -1))
            z_kp1 = self.add_vals(z_k, self.scale_vals(z_diff, self.eta_z / self.sigma_x))
            z_tmp = self.add_vals(x_f_kp1, self.scale_vals(z_f_kp1, 1 / self.sigma_x))
            z_kp1 = self.add_vals(z_kp1, self.scale_vals(z_tmp, -self.eta_z))

            y_diff = self.add_vals(y_f_kp1, self.scale_vals(y_k, -1))
            y_kp1 = self.add_vals(y_k, self.scale_vals(y_diff, self.eta_y * self.sigma_y))
            y_tmp = self.add_vals(w_f_kp1, self.scale_vals(y_f_kp1, self.sigma_y))
            y_kp1 = self.add_vals(y_kp1, self.scale_vals(y_tmp, -self.eta_y))

            x_kp1 = self.scale_vals(z_kp1, -1 / self.sigma_x)

            x_grad, y_grad = self.compute_h_bar_gradient(x_kp1, y_kp1)
            x_hat_kp1 = self.add_vals(x_kp1, self.scale_vals(x_grad, -self.zeta_hat))
            x_hat_kp1 = compute_prox(x_hat_kp1, self.prox_x, self.zeta_hat)
            y_hat_kp1 = self.add_vals(y_kp1, self.scale_vals(y_grad, self.zeta_hat))
            y_hat_kp1 = compute_prox(y_hat_kp1, self.prox_y, self.zeta_hat)

            self.assign_x(x_hat_kp1)
            self.assign_y(y_hat_kp1)

            x_grad_hat, y_grad_hat = self.compute_h_bar_gradient(x_hat_kp1, y_hat_kp1)
            grad_delta_x = self.add_vals(x_grad, self.scale_vals(x_grad_hat, -1))
            delta_x = self.add_vals(x_kp1, self.scale_vals(x_hat_kp1, -1))
            delta_x = self.add_vals(
                self.scale_vals(delta_x, 1 / self.zeta_hat),
                self.scale_vals(grad_delta_x, -1),
            )

            grad_delta_y = self.add_vals(y_grad, self.scale_vals(y_grad_hat, -1))
            delta_y = self.add_vals(y_hat_kp1, self.scale_vals(y_kp1, -1))
            delta_y = self.add_vals(
                self.scale_vals(delta_y, 1 / self.zeta_hat),
                self.scale_vals(grad_delta_y, -1),
            )

            delta = compute_norm(delta_x + delta_y)
            final_delta = float(delta.item())
            num_outer_iters = k + 1

            if self.verbose and (
                k % self.log_every == 0 or final_delta < self.tau or num_outer_iters == self.max_iter
            ):
                self._log(
                    f"Minimax_SCSC outer={k} inner_t={t} delta={final_delta:.3e} tau={self.tau:.3e}"
                )

            if delta < self.tau:
                terminated = True
                break

            z_k = z_kp1
            y_k = y_kp1
            z_f_k = z_f_kp1
            y_f_k = y_f_kp1

        return {
            "num_outer_iters": num_outer_iters,
            "final_delta": final_delta,
            "terminated": terminated,
        }


def optimize_NCWC(params_x, params_y, h_func, lip_h, D_y, prox_x, prox_y, epsilon, epsilon_0, lr=1, max_iter=1000, verbose=False, log_every=1):
    """
    Algorithm 6 of "First-order penalty methods for bilevel optimization"
    for non-convex-weakly-concave minimax problems.
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

    for k in tqdm.trange(max_iter, desc="optimize_NCWC", unit="outer iter"):

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

    return {
        "num_outer_iters": num_outer_iters,
        "final_diff": final_diff,
        "terminated": terminated,
    }


def optimize_bilevel_constrained_fop(
    params_x,
    params_y,
    upper_smooth,
    lower_smooth,
    lower_constraints,
    prox_x,
    prox_y,
    D_y,
    L_grad_f1,
    L_grad_ftilde1,
    L_grad_gtilde,
    L_gtilde,
    gtilde_hi,
    epsilon,
    lr=1,
    max_iter=1000,
    verbose=False,
    log_every=1,
):
    """
    Algorithm 4 for the constrained penalty reformulation.

    The minimization block is the concatenated tuple (x, y), and z is the
    internal maximization block initialized from y.

    h_alg4 computes the smooth part.
    """
    if epsilon <= 0 or epsilon > 0.25:
        raise ValueError(f"Algorithm 4 requires epsilon in (0, 1/4], got {epsilon}")
    if D_y <= 0:
        raise ValueError(f"Invalid D_y: {D_y}")
    if log_every <= 0:
        raise ValueError(f"Invalid log_every: {log_every}")

    params_x = list(params_x)
    params_y = list(params_y)
    if not params_x:
        raise ValueError("params_x must contain at least one tensor")
    if not params_y:
        raise ValueError("params_y must contain at least one tensor")

    rho = epsilon ** -1
    mu = epsilon ** -2
    epsilon_0 = epsilon ** (5 / 2)

    z_params = [p.clone().detach().requires_grad_(True) for p in params_y]
    # TODO: DOUBLE CHECK THAT Z HAS GRADIENT.
    
    def h_alg4():
        upper_term = upper_smooth(params_x, params_y)
        lower_y = lower_smooth(params_x, params_y)
        lower_z = lower_smooth(params_x, z_params)
        constraint_y = lower_constraints(params_x, params_y)
        constraint_z = lower_constraints(params_x, z_params)
        return (
            upper_term
            + rho * lower_y
            + rho * mu * _positive_part_norm_sq(constraint_y)
            - rho * lower_z
            - rho * mu * _positive_part_norm_sq(constraint_z)
        )

    prox_hat = _expand_prox_spec(prox_x, len(params_x)) + _expand_prox_spec(
        _scale_prox_spec(prox_y, rho),
        len(params_y),
    )
    prox_z = _expand_prox_spec(_scale_prox_spec(prox_y, rho), len(z_params))
    params_hat = params_x + params_y

    lip_h = (
        L_grad_f1
        + 2 * rho * L_grad_ftilde1
        + 4 * rho * mu * (gtilde_hi * L_grad_gtilde + L_gtilde ** 2)
    )

    solver_stats = optimize_NCWC(
        params_hat,
        z_params,
        h_alg4,
        lip_h,
        D_y,
        prox_hat,
        prox_z,
        epsilon,
        epsilon_0,
        lr=lr,
        max_iter=max_iter,
        verbose=verbose,
        log_every=log_every,
    )

    return {
        "z_eps": [p.clone().detach() for p in z_params],
        "rho": rho,
        "mu": mu,
        "epsilon_0": epsilon_0,
        "solver_stats": solver_stats,
    }


def optimize_bilevel_constrained_minimax(
        params_x,
        params_y1,
        upper_smooth,
        lower_smooth,
        lower_constraints,
        lagrange_bound,  # TODO: bound of Lagrange multipliers from strong Slater's condition.
        prox_x,
        prox_y1,
        D_y,
        L_grad_f1,
        L_grad_ftilde1,
        L_grad_gtilde,
        L_gtilde,
        gtilde_hi,
        epsilon,
        lr=1,
        max_iter=1000,
        verbose=False,
        log_every=1,
):
    """
    Our proposed method for the constrained penalty reformulation.

    Using the Lagrangian formulation of the lower level problem, we solve only one instance of NCWC problem.
    """

    if D_y <= 0:
        raise ValueError(f"Invalid D_y: {D_y}")
    if log_every <= 0:
        raise ValueError(f"Invalid log_every: {log_every}")

    params_x = list(params_x)
    params_y1 = list(params_y1)
    if not params_x:
        raise ValueError("params_x must contain at least one tensor")
    if not params_y1:
        raise ValueError("params_y must contain at least one tensor")

    # Note we are transforming the lower level problem into unconstrained minimax, so we use settings in Algorithm 2.
    rho = epsilon ** -1
    epsilon_0 = epsilon ** (3 / 2)

    # Apply lower constraint once to get the number of constraints.
    # The copy of y.
    params_z1 = [p.clone().detach().requires_grad_(True) for p in params_y1]
    tmp = lower_constraints(params_x, params_y1)
    num_constraints = tmp.shape[0]
    # Create and initialize the Lagrange multipliers.
    params_y2 = [torch.zeros_like(tmp).requires_grad_(True)]
    # The copy of y2.
    params_z2 = [torch.zeros_like(tmp).requires_grad_(True)]

    # prox operators for y2 and z2.
    def prox_lagrange_multipliers(v, coeff):
        del coeff
        # Lagrange multipliers are non-negative.
        v = torch.maximum(v, 0.0)
        # Box constrains from Strong Slater condition.
        v = torch.mininum(v, lagrange_bound)
        return v

    def h_lagrangian():
        upper_term = upper_smooth(params_x, params_y1)
        lower_z1_y2 = lower_smooth(params_x, params_z1) + torch.dot(params_y2[0], lower_constraints(params_x, params_z1))
        lower_y1_z2 = lower_smooth(params_x, params_y1) + torch.dot(params_z2[0], lower_constraints(params_x, params_y1))
        return (
                upper_term
                + rho * lower_y1_z2
                - rho * lower_z1_y2
        )

    # We minimize over x, y1, y2 and maximize over z1, z2.
    prox_x_y1_y2 = _expand_prox_spec(prox_x, len(params_x)) + _expand_prox_spec(
        _scale_prox_spec(prox_y1, rho),
        len(params_y1),
    ) + [_scale_prox_spec(prox_lagrange_multipliers, rho)]
    prox_z1_z2 = _expand_prox_spec(_scale_prox_spec(prox_y1, rho), len(params_z1)) + [_scale_prox_spec(prox_lagrange_multipliers, rho)]

    lip_h = (
            L_grad_f1
            + 2 * rho * L_grad_ftilde1
            # TODO: it is best to tune another parameter in place of torch.sqrt(num_constraints) * lagrange_bound * L_grad_gtilde
            + L_gtilde + torch.sqrt(num_constraints) * lagrange_bound * L_grad_gtilde
    )

    # TODO: given we do not have additional outside loop, we have to monitor the progress by extracting progress from
    # the optimize_NCWC procedure.
    solver_stats = optimize_NCWC(
        params_x + params_y1 + params_y2,
        params_z1 + params_z2,
        h_lagrangian,
        lip_h,
        D_y,
        prox_x_y1_y2,
        prox_z1_z2,
        epsilon,
        epsilon_0,
        lr=lr,
        max_iter=max_iter,
        verbose=verbose,
        log_every=log_every,
    )

    # TODO: Monitor feasibility of lower level constraint, suboptimality of lower level, obj value.

    return {
        "z_eps": [p.clone().detach() for p in z_params],
        "rho": rho,
        "epsilon_0": epsilon_0,
        "solver_stats": solver_stats,
    }
