import math
import torch
from minimax import Minimax_SCSC, optimize_NCWC

# Setup simple test
x = torch.rand(10, requires_grad=True)
y = torch.rand(10, requires_grad=True)
print(x)
print(y)

sigma_x = 0.1
sigma_y = 0.1
lip=2

def h_bar():
    loss = torch.sum(x*y) + (sigma_x/2) * torch.sum(x**2) - (sigma_y/2) * torch.sum(y**2)
    # print(f"loss={loss}")
    return loss

def prox_bound(v, coeff):
    del coeff
    return torch.clamp(v, min=-1.0, max=1.0)


# opt = Minimax_SCSC([x],[y], h_bar, sigma_x, sigma_y, lip, prox_bound, prox_bound, tol=1e-2)
# opt.run(debug=True)


def h_ncwc():
    loss = torch.sum(x*y) - 1 * torch.sum(x**2) # - (sigma_y/2) * torch.sum(y**2)
    # print(f"loss={loss}")
    return loss

D_y = 10  # math.sqrt(40)
lip_h = 2
optimize_NCWC([x], [y], h_ncwc, lip_h, D_y, prox_bound, prox_bound, 1e-2)