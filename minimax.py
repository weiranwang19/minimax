import math

import torch
from torch.optim import Optimizer


def compute_norm(vals):
    s = 0
    for v in vals:
        s += torch.sum(torch.square(v))
    return torch.sqrt(s)


def _expand_prox_spec(prox_func, count):
    if isinstance(prox_func, (list, tuple)):
        if len(prox_func) != count:
            raise ValueError(f"Expected {count} prox operators, got {len(prox_func)}")
        return list(prox_func)
    return [prox_func] * count


def _scale_prox_operator(prox_func, scale):
    if prox_func is None:
        return None

    def scaled(v, coeff, prox_func=prox_func, scale=scale):
        return prox_func(v, scale * coeff)

    return scaled


def _scale_prox_spec(prox_func, scale):
    if isinstance(prox_func, (list, tuple)):
        return [_scale_prox_operator(p, scale) for p in prox_func]
    return _scale_prox_operator(prox_func, scale)


def _as_tensor_list(vals):
    if torch.is_tensor(vals):
        return [vals]
    return list(vals)


def _positive_part_norm_sq(vals):
    total = None
    for v in _as_tensor_list(vals):
        term = torch.sum(torch.square(torch.clamp(v, min=0.0)))
        total = term if total is None else total + term
    if total is None:
        raise ValueError("Expected at least one tensor when computing the positive-part norm")
    return total


def compute_prox(vals, prox_func, prox_coeff):
    # the prox_func shall solve min_x (1/2) |x - x_0|^2 + prox_coeff * fun(x)
    prox_funcs = _expand_prox_spec(prox_func, len(vals))
    prox_vals = []
    for p, prox in zip(vals, prox_funcs):
        if prox is None:
            prox_vals.append(p)
        else:
            prox_vals.append(prox(p, prox_coeff))
    return prox_vals


class Minimax_SCSC(Optimizer):
    """
    A custom PyTorch optimizer implementing the Algorithm 5 of
        First-order penalty methods for bilevel optimization.
    https://arxiv.org/pdf/2301.01716

    This method optimizes a strongly-convex-strongly-concave function.
    """

    def __init__(self, params_x, params_y, h_bar, sigma_x, sigma_y, lip, prox_x, prox_y, tol, lr=1, max_iter=1000):
        # 1. Validate your hyperparameters
        # lr will be the initial step size maintained by learning rate scheduler.
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        # 2. Define the default hyperparameters for all parameter groups
        defaults = dict(lr=lr)

        # 3. Initialize the base Optimizer class.
        # This will automatically create self.param_groups using the defaults.
        super().__init__([{'params':params_x}, {'params':params_y}], defaults)
        # We create two param groups, x and y.
        assert len(self.param_groups) == 2

        # Inputs to the algorithm.
        self.h_bar = h_bar  # function handle.
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.alpha_bar = min(1, math.sqrt(8 * sigma_y / sigma_x))
        self.eta_z = sigma_x / 2
        self.eta_y = min(1/(2 * sigma_y), 4/(self.alpha_bar * sigma_x))
        self.lip = lip
        self.zeta = 1 / (2 * math.sqrt(5) * (1 + 8 * lip / sigma_x))
        self.gamma_x = self.gamma_y = (8 / sigma_x)
        self.zeta_hat = min(sigma_x, sigma_y) / (lip ** 2)
        print(f"alpha_bar={self.alpha_bar}, sigma_x={self.sigma_x}, sigma_y={self.sigma_y}, eta_z={self.eta_z}, eta_y={self.eta_y}, zeta={self.zeta}, gamma_x={self.gamma_x}, gamma_y={self.gamma_y}, zeta_hat={self.zeta_hat}")

        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter

        self.prox_x = prox_x
        self.prox_y = prox_y

        assert len(self.param_groups) == 2
        print("Finished initialization!")

    def get_x(self):
        return self.param_groups[0]['params']

    def get_y(self):
        return self.param_groups[1]['params']

    @torch.no_grad()
    def copy_vars(self, vars):
        # Returned tensors have no gradient.
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
        for p, v in zip(self.param_groups[0]['params'], val):
            p.data.copy_(v)

    @torch.no_grad()
    def assign_y(self, val):
        for p, v in zip(self.param_groups[1]['params'], val):
            p.data.copy_(v)

    @torch.no_grad()
    def scale_vals(self, vals, s):
        return [s * p for p in vals]

    @torch.no_grad()
    def add_vals(self, vals1, vals2):
        return [a + b for a, b in zip(vals1, vals2)]

    def compute_h_bar_gradient(self, x_val, y_val):
        # We assume h_bar is already a function of x and y, e.g., h_bar forwards a model whose variables are x and y.
        # We temporarily assign x and y for computing the function value and gradient, and then restore the values back.
        x_copy = self.get_x_copy()
        y_copy = self.get_y_copy()
        self.assign_x(x_val)
        self.assign_y(y_val)

        # If the h_bar is different during training, e.g., we compute loss on minibatch,
        # then h_bar can take additional arguments.
        loss = self.h_bar()
        self.zero_grad()
        loss.backward()
        x_grad = self.get_x_grad()
        y_grad = self.get_y_grad()

        self.assign_x(x_copy)
        self.assign_y(y_copy)
        return x_grad, y_grad

    def compute_a_k(self, x_val, y_val, z_g_k, y_g_k, return_h_hat_grad=False):
        # Top of page 25 of https://arxiv.org/pdf/2301.01716.
        x_grad, y_grad = self.compute_h_bar_gradient(x_val, y_val)
        x_grad_hat = self.add_vals(x_grad, self.scale_vals(x_val, - self.sigma_x))
        y_grad_hat = self.add_vals(y_grad, self.scale_vals(y_val, self.sigma_y))
        if return_h_hat_grad:
            return x_grad_hat, y_grad_hat

        x_tmp = self.add_vals(x_val, self.scale_vals(z_g_k, - 1 / self.sigma_x))
        a_x_k = self.add_vals(x_grad_hat, self.scale_vals(x_tmp, self.sigma_x / 2))

        y_tmp = self.add_vals(y_val, self.scale_vals(y_g_k, -1))
        y_tmp = self.add_vals(self.scale_vals(y_val, self.sigma_y),
                              self.scale_vals(y_tmp, self.sigma_x / 8))
        a_y_k = self.add_vals(self.scale_vals(y_grad_hat, -1), y_tmp)
        return a_x_k, a_y_k

    def run(self, debug=False):

        z_k = z_f_k = self.scale_vals(self.get_x_copy(), - self.sigma_x)
        y_k = y_f_k = self.get_y_copy()

        # Main loop, line 1
        for k in range(self.max_iter):

            # line 2
            z_g_k = self.add_vals(self.scale_vals(z_k, self.alpha_bar),
                                  self.scale_vals(z_f_k, 1 - self.alpha_bar))
            y_g_k = self.add_vals(self.scale_vals(y_k, self.alpha_bar),
                                  self.scale_vals(y_f_k, 1 - self.alpha_bar))

            # line 3
            x_k_m1 = self.scale_vals(z_g_k, - 1 / self.sigma_x)
            y_k_m1 = self.copy_vars(y_g_k)

            # line 4-7
            a_x_k, a_y_k = self.compute_a_k(x_k_m1, y_k_m1, z_g_k, y_g_k)
            x_tmp = self.add_vals(x_k_m1, self.scale_vals(a_x_k, - self.zeta * self.gamma_x))
            y_tmp = self.add_vals(y_k_m1, self.scale_vals(a_y_k, - self.zeta * self.gamma_y))
            x_k_0 = x_k_t = compute_prox(x_tmp, self.prox_x, self.zeta * self.gamma_x)
            y_k_0 = y_k_t = compute_prox(y_tmp, self.prox_y, self.zeta * self.gamma_y)
            b_x_k_t = self.scale_vals(self.add_vals(x_tmp, self.scale_vals(x_k_t, -1)), 1 / (self.zeta * self.gamma_x))
            b_y_k_t = self.scale_vals(self.add_vals(y_tmp, self.scale_vals(y_k_t, -1)), 1 / (self.zeta * self.gamma_y))

            # line 8
            t = 0

            # the inner loop over t.
            while True:
                beta_t = self.lr * 2 / (t + 3)

                # line 9
                a_x_t, a_y_t = self.compute_a_k(x_k_t, y_k_t, z_g_k, y_g_k)
                x_tmp = self.add_vals(a_x_t, b_x_k_t)
                y_tmp = self.add_vals(a_y_t, b_y_k_t)
                lhs = self.gamma_x * compute_norm(x_tmp) ** 2 + self.gamma_y * compute_norm(y_tmp) ** 2
                x_diff = self.add_vals(x_k_t, self.scale_vals(x_k_m1, -1))
                y_diff = self.add_vals(y_k_t, self.scale_vals(y_k_m1, -1))
                rhs = (1 / self.gamma_x) * compute_norm(x_diff) ** 2 + (1 / self.gamma_y) * compute_norm(y_diff) ** 2
                if lhs <= rhs + 1e-8 or t>10000:  # Weiran: added slackness.
                    if debug:
                        print(f"k={k}: inner check passed at t={t}, lhs={lhs}, rhs={rhs}")
                    break

                x_diff = self.add_vals(x_k_0, self.scale_vals(x_k_t, -1))
                y_diff = self.add_vals(y_k_0, self.scale_vals(y_k_t, -1))
                # line 10
                x_k_t_half = self.add_vals(x_k_t, self.scale_vals(x_diff, beta_t))
                x_k_t_half = self.add_vals(x_k_t_half, self.scale_vals(x_tmp, - self.zeta * self.gamma_x))
                # line 11
                y_k_t_half = self.add_vals(y_k_t, self.scale_vals(y_diff, beta_t))
                y_k_t_half = self.add_vals(y_k_t_half, self.scale_vals(y_tmp, - self.zeta * self.gamma_y))

                a_x_t_half, a_y_t_half = self.compute_a_k(x_k_t_half, y_k_t_half, z_g_k, y_g_k)
                # line 12
                x_k_tp1_before_prox = self.add_vals(x_k_t, self.scale_vals(x_diff, beta_t))
                x_k_tp1_before_prox = self.add_vals(x_k_tp1_before_prox, self.scale_vals(a_x_t_half, - self.zeta * self.gamma_x))
                x_k_tp1 = compute_prox(x_k_tp1_before_prox, self.prox_x, self.zeta * self.gamma_x)
                # line 13
                y_k_tp1_before_prox = self.add_vals(y_k_t, self.scale_vals(y_diff, beta_t))
                y_k_tp1_before_prox = self.add_vals(y_k_tp1_before_prox, self.scale_vals(a_y_t_half, - self.zeta * self.gamma_y))
                y_k_tp1 = compute_prox(y_k_tp1_before_prox, self.prox_y, self.zeta * self.gamma_y)

                # line 14
                b_x_k_tp1 = self.scale_vals(self.add_vals(x_k_tp1_before_prox, self.scale_vals(x_k_tp1, -1)), 1/(self.zeta * self.gamma_x))
                # line 15
                b_y_k_tp1 = self.scale_vals(self.add_vals(y_k_tp1_before_prox, self.scale_vals(y_k_tp1, -1)), 1/(self.zeta * self.gamma_y))

                # line 16
                t += 1
                x_k_t = x_k_tp1
                y_k_t = y_k_tp1
                b_x_k_t = b_x_k_tp1
                b_y_k_t = b_y_k_tp1
                if t % 100 == 0 and debug:
                    print(f"inner loop, t={t}, beta_t={beta_t}")
                    print(f"inner check lhs={lhs}, rhs={rhs}")
                    print(f"x_k_t={x_k_t}, y_k_t={y_k_t}")

            # line 18
            x_f_kp1, y_f_kp1 = x_k_t, y_k_t

            x_grad_hat, y_grad_hat = self.compute_a_k(x_f_kp1, y_f_kp1, None, None, return_h_hat_grad=True)
            # line 19
            z_f_kp1 = self.add_vals(x_grad_hat, b_x_k_t)
            w_f_kp1 = self.add_vals(self.scale_vals(y_grad_hat, -1), b_y_k_t)

            z_diff = self.add_vals(z_f_kp1, self.scale_vals(z_k, -1))
            # line 20
            z_kp1 = self.add_vals(z_k, self.scale_vals(z_diff, self.eta_z / self.sigma_x))
            z_tmp = self.add_vals(x_f_kp1, self.scale_vals(z_f_kp1, 1 / self.sigma_x))
            z_kp1 = self.add_vals(z_kp1, self.scale_vals(z_tmp, - self.eta_z))

            y_diff = self.add_vals(y_f_kp1, self.scale_vals(y_k, -1))
            # line 21
            y_kp1 = self.add_vals(y_k, self.scale_vals(y_diff, self.eta_y * self.sigma_y))
            y_tmp = self.add_vals(w_f_kp1, self.scale_vals(y_f_kp1, self.sigma_y))
            y_kp1 = self.add_vals(y_kp1, self.scale_vals(y_tmp, - self.eta_y))

            # line 22
            x_kp1 = self.scale_vals(z_kp1, - 1 / self.sigma_x)

            x_grad, y_grad = self.compute_h_bar_gradient(x_kp1, y_kp1)
            # line 23
            x_hat_kp1 = self.add_vals(x_kp1, self.scale_vals(x_grad, - self.zeta_hat))
            x_hat_kp1 = compute_prox(x_hat_kp1, self.prox_x, self.zeta_hat)
            # line 24
            y_hat_kp1 = self.add_vals(y_kp1, self.scale_vals(y_grad, self.zeta_hat))
            y_hat_kp1 = compute_prox(y_hat_kp1, self.prox_y, self.zeta_hat)

            # The best x and y after outer loop k.
            self.assign_x(x_hat_kp1)
            self.assign_y(y_hat_kp1)

            # Final check
            x_grad_hat, y_grad_hat = self.compute_h_bar_gradient(x_hat_kp1, y_hat_kp1)

            grad_delta_x = self.add_vals(x_grad, self.scale_vals(x_grad_hat, -1))
            delta_x = self.add_vals(x_kp1, self.scale_vals(x_hat_kp1, -1))
            delta_x = self.add_vals(self.scale_vals(delta_x, 1 / self.zeta_hat), self.scale_vals(grad_delta_x, -1))

            grad_delta_y = self.add_vals(y_grad, self.scale_vals(y_grad_hat, -1))
            delta_y = self.add_vals(y_hat_kp1, self.scale_vals(y_kp1, -1))
            delta_y = self.add_vals(self.scale_vals(delta_y, 1 / self.zeta_hat), self.scale_vals(grad_delta_y, -1))

            # line 25
            delta = compute_norm(delta_x + delta_y)
            if debug:
                print(f"outer loop {k}, x_kp1={x_kp1}, y_kp1={y_kp1}")
                print(f"outer loop {k}, x_hat_kp1={x_hat_kp1}, y_hat_kp1={y_hat_kp1}")
                print(f"outer loop k={k}, x_grad={compute_norm(x_grad)}, y_grad={compute_norm(y_grad)}")
                print(f"outer loop k={k}, final check norm: {delta}")
            if delta < self.tol:
                break

            z_k = z_kp1
            y_k = y_kp1
            z_f_k = z_f_kp1
            y_f_k = y_f_kp1


def optimize_NCWC(params_x, params_y, h_func, lip_h, D_y, prox_x, prox_y, epsilon, epsilon_0=None, lr=1, max_iter=1000):
    """
    A custom PyTorch optimizer implementing the Algorithm 6 of
        First-order penalty methods for bilevel optimization.
    https://arxiv.org/pdf/2301.01716

    This method optimizes a non-convex-weakly-concave function.
    """
    x_k = [p.clone().detach() for p in params_x]
    y_0 = [p.clone().detach() for p in params_y]
    if epsilon_0 is None:
        epsilon_0 = epsilon / 2
    if epsilon_0 <= 0:
        raise ValueError(f"Invalid epsilon_0: {epsilon_0}")

    for k in range(max_iter):

        def h_alg6():
            """
            Defined in (102)
            """
            f = h_func()
            s_y = 0
            for p, p0 in zip(params_y, y_0):
                s_y += torch.sum(torch.square(p - p0))
            s_x = 0
            for p, p0 in zip(params_x, x_k):
                s_x += torch.sum(torch.square(p - p0))
            f = f - (epsilon * s_y) / (4 * D_y) + lip_h * s_x
            # print(f"h_k={f}")
            return f

        epsilon_k = epsilon_0 / (k + 1)
        sigma_y = epsilon / (2 * D_y)
        lip_k = 3 * lip_h + epsilon / (2 * D_y)
        # print(f"K={k}, epsilon_k={epsilon_k}")
        # Variables params_x and params_y are updated by solver.rum().
        solver = Minimax_SCSC(params_x, params_y, h_alg6, sigma_x=lip_h, sigma_y=sigma_y, lip=lip_k, prox_x=prox_x, prox_y=prox_y, tol=epsilon_k, lr=lr, max_iter=max_iter)
        solver.run(debug=False)
        # At the end of run(), params_x and params_y are updated.
        x_kp1 = [p.clone().detach() for p in params_x]
        y_kp1 = [p.clone().detach() for p in params_y]

        diff = compute_norm(solver.add_vals(x_kp1, solver.scale_vals(x_k, -1)))
        print(f"x_kp1={x_kp1}, y_kp1={y_kp1}, diff={diff}")
        if diff <= epsilon / (4 * lip_h):
            return
        # import pdb;pdb.set_trace()

        x_k = x_kp1


def optimize_bilevel_constrained(
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
):
    """
    Algorithm 4 of the FOP paper, building the constrained penalty minimax problem and solving it through optimize_NCWC.

    The optimization variables in the minimization block are the concatenated tuple (x, y),
    while the maximization block is an internal copy of z initialized from y.
    """
    if epsilon <= 0 or epsilon > 0.25:
        raise ValueError(f"Algorithm 4 requires epsilon in (0, 1/4], got {epsilon}")
    if D_y <= 0:
        raise ValueError(f"Invalid D_y: {D_y}")

    params_x = list(params_x)
    params_y = list(params_y)

    rho = epsilon ** -1
    mu = epsilon ** -2
    epsilon_0 = epsilon ** (5 / 2)

    z_params = [p.clone().detach().requires_grad_(True) for p in params_y]

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
        _scale_prox_spec(prox_y, rho), len(params_y)
    )
    prox_z = _expand_prox_spec(_scale_prox_spec(prox_y, rho), len(z_params))
    params_hat = params_x + params_y

    lip_h = (
        L_grad_f1
        + 2 * rho * L_grad_ftilde1
        + 4 * rho * mu * (gtilde_hi * L_grad_gtilde + L_gtilde ** 2)
    )

    optimize_NCWC(
        params_hat,
        z_params,
        h_alg4,
        lip_h,
        D_y,
        prox_hat,
        prox_z,
        epsilon,
        epsilon_0=epsilon_0,
        lr=lr,
        max_iter=max_iter,
    )
    return [p.clone().detach() for p in z_params]
