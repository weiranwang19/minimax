import torch
from minimax import MinimaxGD

# Setup simple test
x = torch.rand(10, requires_grad=True)
y = torch.rand(10, requires_grad=True)
print(x)
print(y)

sigma_x = 0.1
sigma_y = 0.1
lip=10

def h_bar():
    loss = torch.sum(x*y) + (sigma_x/2) * torch.sum(x**2) - (sigma_y/2) * torch.sum(y**2)
    print(f"loss={loss}")
    return loss

def prox_bound(v, coeff):
    del coeff
    return torch.clamp(v, min=-2.0, max=2.0)
    # return v


opt = MinimaxGD([x],[y], h_bar, sigma_x, sigma_y, lip, prox_bound, prox_bound, tol=1e-2)

opt.run()
