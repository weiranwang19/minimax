import math

import torch
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_scheduler


class MinimaxGD(Optimizer):
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
        self.gamma_x = self.gamma_y = 8 / sigma_x
        self.zeta_hat = min(sigma_x, sigma_y) / (lip ** 2)
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

    @torch.no_grad()
    def compute_prox(self, vars, prox_func, prox_coeff):
        # the prox_func shall solve min_x (1/2) |x - x_0|^2 + prox_coeff * fun(x)
        return [prox_func(p, prox_coeff) for p in vars]

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

    def compute_a_k(self, x_val, y_val, z_g_k, y_g_k):
        # Top of page 25 of https://arxiv.org/pdf/2301.01716.
        x_grad, y_grad = self.compute_h_bar_gradient(x_val, y_val)
        x_grad_hat = self.add_vals(x_grad, self.scale_vals(x_val, - self.sigma_x))
        y_grad_hat = self.add_vals(y_grad, self.scale_vals(y_val, self.sigma_y))

        x_tmp = self.add_vals(x_val, self.scale_vals(z_g_k, - 1 / self.sigma_x))
        a_x_k = self.add_vals(x_grad_hat, self.scale_vals(x_tmp, self.sigma_x / 2))

        y_tmp = self.add_vals(y_val, self.scale_vals(y_g_k, -1))
        y_tmp = self.add_vals(self.scale_vals(y_val, self.sigma_y),
                              self.scale_vals(y_tmp, self.sigma_x / 8))
        a_y_k = self.add_vals(self.scale_vals(y_grad_hat, -1), y_tmp)
        return a_x_k, a_y_k

    def run(self):
        z = z_f = self.scale_vals(self.get_x_copy(), - self.sigma_x)
        y = y_f = self.get_y_copy()

        # Main loop, line 1.
        for k in range(self.max_iter):

            # line 2.
            z_g_k = self.add_vals(self.scale_vals(z), self.alpha_bar,
                                  self.scale_vals(z_f, 1 - self.alpha_bar))
            y_g_k = self.add_vals(self.scale_vals(y), self.alpha_bar,
                                  self.scale_vals(y_f, 1 - self.alpha_bar))

            # line 3.
            x_k_m1 = self.scale_vals(z_g_k, - 1 / self.sigma_x)
            y_k_m1 = self.copy_vars(y_g_k)

            # line 4-7
            a_x_k, a_y_k = self.compute_a_k(x_k_m1, y_k_m1, z_g_k, y_g_k)
            x_k_0 = self.add_vals(x_k_m1, self.scale_vals(a_x_k, - self.zeta * self.gamma_x))
            x_tmp = self.compute_prox(self.prox_x, x_k_0, self.zeta * self.gamma_x)
            y_tmp = self.add_vals(y_k_m1, self.scale_vals(a_y_k, - self.zeta * self.gamma_y))

