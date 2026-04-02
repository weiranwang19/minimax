import torch
from minimax import MinimaxGD

# Setup simple test
x = torch.randn(10, requires_grad=True)
y = torch.randn(10, requires_grad=True)
print(x)
print(y)

opt = MinimaxGD([x],[y], None,0.01, 0.01, 1, None, None, 1e-3)

opt.scale_vals(opt.get_x(), 10)
print(opt.get_x())
print(opt.get_y())

opt.assign_x(opt.scale_vals(opt.get_x(), 10))
print(opt.get_x())
print(opt.get_y())

