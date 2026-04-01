import math

import torch
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_scheduler


class MinimaxGD(Optimizer):
    """
    A custom PyTorch optimizer implementing the Algorithm 5 of
        First-order penalty methods for bilevel optimization.
    https://arxiv.org/pdf/2301.01716
    """

    def __init__(self, params_x, params_y, sigma_x, sigma_y, lip, prox_x, prox_y, lr=1e-3):
        # 1. Validate your hyperparameters
        # lr will be the initial step size maintained by learning rate scheduler.
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        # 2. Define the default hyperparameters for all parameter groups
        defaults = dict(lr=lr)

        # 3. Initialize the base Optimizer class.
        # This will automatically create self.param_groups using the defaults.
        super().__init__([{'params':params_x}, {'params':params_y}], defaults)
        # Weiran: check what is in self.param_groups.
        # We create two param groups, x and y.
        assert len(self.param_groups) == 2

        # Inputs to the algorithm.
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.alpha_bar = min(1, math.sqrt(8 * sigma_y / sigma_x))
        self.eta_z = sigma_x / 2
        self.eta_y = min(1/(2 * sigma_y), 4/(self.alpha_bar * sigma_x))
        self.lip = lip
        self.xi = 1 / (2 * math.sqrt(5) * (1 + 8 * lip / sigma_x))
        self.gamma_x = self.gamma_y = 8 / sigma_x
        self.hat_xi = min(sigma_x, sigma_y) / (lip ** 2)

        self.prox_x = prox_x
        self.prox_y = prox_y

        # 4. Initialize the internal state 'z' for all 'x' parameters
        # self.param_groups[0] corresponds to params_x
        for p in self.param_groups[0]['params']:
            import pdb;pdb.set_trace()
            state = self.state[p]
            # Use .clone().detach() to ensure z is a separate memory block,
            # doesn't track gradients, but shares the same shape/device as p.
            state['z'] = -p.clone().detach()

        print("Finished initialization!")

#     @torch.no_grad()
#     def step(self):
#         """
#         Performs a single optimization step. Which corresponds to an outer loop of the algorithm.
#         """
#
#         # Iterate over each parameter group
#         for group in self.param_groups:
#             lr = group['lr']
#
#             # Iterate over the parameters within this group
#             for p in group['params']:
#                 # Skip parameters that have no gradient computed
#                 if p.grad is None:
#                     continue
#
#                 grad = p.grad
#
#                 # ==========================================
#                 # State Initialization & Tracking
#                 # ==========================================
#                 state = self.state[p]
#
#                 # If this is the first time we are updating this parameter,
#                 # initialize its state variables.
#                 if len(state) == 0:
#                     state['step'] = 0
#                     # Example of initializing a state tensor:
#                     # state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#
#                 state['step'] += 1
#
#                 # ==========================================
#                 # The Update Math
#                 # ==========================================
#
#                 # 2. Compute the actual parameter update
#                 # For SignSGD, we take the sign of the gradient
#                 update = torch.sign(grad)
#
#                 # 3. Apply the update to the parameter
#                 # p = p - lr * update
#                 p.sub_(update, alpha=lr)
#
#         return



