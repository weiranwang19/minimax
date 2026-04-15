import math
import torch
from minimax import Minimax_SCSC, SAPD_SCSC, optimize_NCWC

# Setup simple test
x = torch.rand(10, requires_grad=True)
y = torch.rand(10, requires_grad=True)
print(x)
print(y)

sigma_x = 0.1
sigma_y = 0.1


def h_bar():
    loss = torch.sum(x*y) + (sigma_x/2) * torch.sum(x**2) - (sigma_y/2) * torch.sum(y**2)
    # print(f"loss={loss}")
    return loss

def prox_bound(v, coeff):
    del coeff
    return torch.clamp(v, min=-1.0, max=1.0)


# lip=2
# opt = Minimax_SCSC([x],[y], h_bar, sigma_x, sigma_y, lip, [prox_bound], [prox_bound], tau=1e-2)
# opt.run()
# print(x)
# print(y)

# lip=4
# opt = SAPD_SCSC([x],[y], h_bar, sigma_x, sigma_y, lip, [prox_bound], [prox_bound], max_iter=10000)
# opt.run()
# print(x)
# print(y)


def h_ncwc():
    loss = torch.sum(x*y) - 0.01 * torch.sum(x**2) # - (sigma_y/2) * torch.sum(y**2)
    # print(f"loss={loss}")
    return loss

D_y = 10  # math.sqrt(40)
lip_h = 10
optimize_NCWC([x], [y], h_ncwc, lip_h, D_y, [prox_bound], [prox_bound], 2e-2, 1e-2, sub_routine='stochastic')
print(x)
print(y)



# Test project_onto_simplex:
# Test 1: Single vector
v1 = torch.tensor([1.2, -0.5, 2.0, 0.8])
p1 = project_onto_simplex(v1)
print(f"Original: {v1}")
print(f"Projected: {p1}")
print(f"Sum: {p1.sum().item():.4f}\n")

# Test 2: Batched tensors (e.g., batch_size=3, classes=4)
v2 = torch.randn(3, 4)
p2 = project_onto_simplex(v2, dim=1)
print(f"Batched Original:\n{v2}")
print(f"Batched Projected:\n{p2}")
print(f"Batched Sums: {p2.sum(dim=1)}")
