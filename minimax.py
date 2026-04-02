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

    def __init__(self, params_x, params_y, h_bar, sigma_x, sigma_y, lip, prox_x, prox_y, tol, lr=1e-3):
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
        self.xi = 1 / (2 * math.sqrt(5) * (1 + 8 * lip / sigma_x))
        self.gamma_x = self.gamma_y = 8 / sigma_x
        self.hat_xi = min(sigma_x, sigma_y) / (lip ** 2)
        self.tol = tol

        self.prox_x = prox_x
        self.prox_y = prox_y

        # 4. Initialize the internal state 'z' for all 'x' parameters
        self.z = []
        self.zf = []
        # self.param_groups[0] corresponds to params_x
        for p in self.param_groups[0]['params']:
            # Use .clone().detach() to ensure z is a separate memory block,
            # doesn't track gradients, but shares the same shape/device as p.
            self.z.append(sigma_x * p.clone().detach())
            self.zf.append(sigma_x * p.clone().detach())
        print("Finished initialization!")

    def get_x(self):
        return self.param_groups[0]['params']

    def get_y(self):
        return self.param_groups[1]['params']

    @torch.no_grad()
    def assign_x(self, val):
        for p, v in zip(self.param_groups[0]['params'], val):
            p.data.copy_(v)

    @torch.no_grad()
    def assign_y(self, val):
        for p, v in zip(self.param_groups[1]['params'], val):
            p.data.copy_(v)

    @torch.no_grad()
    def scale_vars(self, vars, s):
        return [s * p for p in vars]

    @torch.no_grad()
    def add_vars(self, vars1, vars2):
        return [a + b for a, b in zip(vars1, vars2)]

    #
    # @torch.no_grad()
    # def step(self):
    #     """
    #     Performs a single outer loop.
    #     """
    #
    #     x = self.param_groups[0]
    #         lr = group['lr']
    #
    #         # Iterate over the parameters within this group
    #         for p in group['params']:
    #             # Skip parameters that have no gradient computed
    #             if p.grad is None:
    #                 continue
    #
    #             grad = p.grad
    #
    #             # ==========================================
    #             # State Initialization & Tracking
    #             # ==========================================
    #             state = self.state[p]
    #
    #             # If this is the first time we are updating this parameter,
    #             # initialize its state variables.
    #             if len(state) == 0:
    #                 state['step'] = 0
    #                 # Example of initializing a state tensor:
    #                 # state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
    #
    #             state['step'] += 1
    #
    #             # ==========================================
    #             # The Update Math
    #             # ==========================================
    #
    #             # 2. Compute the actual parameter update
    #             # For SignSGD, we take the sign of the gradient
    #             update = torch.sign(grad)
    #
    #             # 3. Apply the update to the parameter
    #             # p = p - lr * update
    #             p.sub_(update, alpha=lr)
    #
    #     return
    #


