import torch
import torch.nn as nn
from minimax import MinimaxGD

# # Setup simple test
# x = torch.randn(10, requires_grad=True)
# y = torch.randn(10, requires_grad=True)
# print(x)
# print(y)
#
# opt = MinimaxGD([x],[y], None,0.01, 0.01, 1, None, None, 1e-3)
#
# opt.scale_vals(opt.get_x(), 10)
# print(opt.get_x())
# print(opt.get_y())
#
# opt.assign_x(opt.scale_vals(opt.get_x(), 10))
# print(opt.get_x())
# print(opt.get_y())

# prox onto simplex
# h_bar is function handle which may update values of a model before computing loss.
# quickly define a model and get parameters() for params_x and params_y.

# More complex test.

# 1. Generate synthetic regression data (y = 2x + noise)
inputs = torch.linspace(-10, 10, 100).view(-1, 1)
targets = 2 * inputs + torch.randn(inputs.size()) * 2

# 2. Define a small neural network using Sequential
model_x = nn.Sequential(
    nn.Linear(1, 16),
    nn.Tanh(),
    nn.Linear(16, 1)
)
model_y = nn.Sequential(
    nn.Linear(1, 16),
    nn.Tanh(),
    nn.Linear(16, 1)
)

# 3. Setup the regression loss and optimizer
criterion = nn.MSELoss()
predictions = model_x(inputs)
loss_x = criterion(predictions, targets)
print(f"Loss x: {loss_x.item()}")

predictions = model_y(inputs)
loss_y = criterion(predictions, targets)
print(f"Loss y: {loss_y.item()}")

def h_bar():
    predictions = model_x(inputs)
    loss_x = criterion(predictions, targets)

    predictions = model_y(inputs)
    loss_y = criterion(predictions, targets)
    return loss_x - loss_y

def prox_bound(v, coeff):
    del coeff
    return torch.clamp(v, min=0.0, max=1.0)

opt = MinimaxGD(model_x.parameters(), model_y.parameters(), h_bar,0.01, 0.01, 1, None, None, 1e-3)

print(f"x={opt.get_x()}")
print(f"y={opt.get_y()}")

# x is updated
opt.assign_x(opt.scale_vals(opt.get_x(), 10))
print(f"x={opt.get_x()}")
# but y is not updated
opt.scale_vals(opt.get_y(), 100)
print(f"y={opt.get_y()}")

# model_x is now changed
predictions = model_x(inputs)
loss_x = criterion(predictions, targets)
print(f"Loss: {loss_x.item()}")

# model_y is not changed
predictions = model_y(inputs)
loss_y = criterion(predictions, targets)
print(f"Loss y: {loss_y.item()}")

print(opt.get_x_copy())

x_grad, y_grad = opt.compute_h_bar_gradient(opt.get_x_copy(), opt.get_y_copy())
print(f"y_grad: {y_grad}")