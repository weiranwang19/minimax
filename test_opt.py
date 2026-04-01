import torch
from minimax import MinimaxGD

# Setup variables
x = torch.randn(10, requires_grad=True)
y = torch.randn(10, requires_grad=True)

opt = MinimaxGD([x],[y], 0.01, 0.01, 1, None, None)
