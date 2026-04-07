import torch

from minimax import optimize_bilevel_constrained


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def prox_box(v, coeff):
    del coeff
    return torch.clamp(v, min=-1.0, max=1.0)


n = 2
m = 2
l = 2

x = torch.zeros(n, requires_grad=True)
y = torch.ones(m, requires_grad=True)

c = torch.tensor([1.0, 0.5])
d = torch.tensor([0.05, 0.05])
d_tilde = torch.tensor([-10.0, -8.0])

A = torch.tensor([
    [0.05, 0.025],
    [0.03, 0.02],
])
B = torch.tensor([
    [0.10, 0.08],
    [0.02, 0.03],
])
b = torch.tensor([0.175, 0.045])


def upper_smooth(x_params, y_params):
    x_vec = x_params[0]
    y_vec = y_params[0]
    return torch.dot(c, x_vec) + torch.dot(d, y_vec)


def lower_smooth(x_params, v_params):
    del x_params
    return torch.dot(d_tilde, v_params[0])


def lower_constraints(x_params, v_params):
    x_vec = x_params[0]
    v_vec = v_params[0]
    return A @ x_vec + B @ v_vec - b


def box_violation(x_vec, v_vec):
    violation = torch.clamp(A @ x_vec + B @ v_vec - b, min=0.0)
    return torch.linalg.vector_norm(violation).item()


def smooth_penalty_norm_sq(x_vec, v_vec):
    violation = torch.clamp(A @ x_vec + B @ v_vec - b, min=0.0)
    return torch.sum(torch.square(violation))


def upper_value(x_vec, y_vec):
    return (torch.dot(c, x_vec) + torch.dot(d, y_vec)).item()


def minimax_value(x_vec, y_vec, z_vec, rho, mu):
    return (
        torch.dot(c, x_vec)
        + torch.dot(d, y_vec)
        + rho * torch.dot(d_tilde, y_vec)
        + rho * mu * smooth_penalty_norm_sq(x_vec, y_vec)
        - rho * torch.dot(d_tilde, z_vec)
        - rho * mu * smooth_penalty_norm_sq(x_vec, z_vec)
    ).item()


row_bounds = torch.sum(torch.abs(A), dim=1) + torch.sum(torch.abs(B), dim=1) + torch.abs(b)
gtilde_hi = torch.linalg.vector_norm(row_bounds).item()
L_gtilde = torch.linalg.matrix_norm(torch.cat([A, B], dim=1), ord=2).item()
D_y = 2.0 * (m ** 0.5)

epsilon = 0.25
rho = epsilon ** -1
mu = epsilon ** -2

initial_x = x.detach().clone()
initial_y = y.detach().clone()
initial_z = y.detach().clone()
initial_upper = upper_value(initial_x, initial_y)
initial_z_violation = box_violation(initial_x, initial_z)
initial_value = minimax_value(initial_x, initial_y, initial_z, rho, mu)

z_eps = optimize_bilevel_constrained(
    [x],
    [y],
    upper_smooth,
    lower_smooth,
    lower_constraints,
    prox_box,
    prox_box,
    D_y=D_y,
    L_grad_f1=0.0,
    L_grad_ftilde1=0.0,
    L_grad_gtilde=0.0,
    L_gtilde=L_gtilde,
    gtilde_hi=gtilde_hi,
    epsilon=epsilon,
    max_iter=20,
)

final_x = x.detach().clone()
final_y = y.detach().clone()
final_z = z_eps[0]
final_upper = upper_value(final_x, final_y)
final_z_violation = box_violation(final_x, final_z)
final_value = minimax_value(final_x, final_y, final_z, rho, mu)

print(f"initial x={initial_x}")
print(f"initial y={initial_y}")
print(f"final x={final_x}")
print(f"final y={final_y}")
print(f"final z={final_z}")
print(f"initial upper objective={initial_upper:.6f}")
print(f"final upper objective={final_upper:.6f}")
print(f"initial z violation={initial_z_violation:.6f}")
print(f"final z violation={final_z_violation:.6f}")
print(f"initial minimax value={initial_value:.6f}")
print(f"final minimax value={final_value:.6f}")

assert torch.all(torch.abs(final_x) <= 1.0 + 1e-8)
assert torch.all(torch.abs(final_y) <= 1.0 + 1e-8)
assert torch.all(torch.abs(final_z) <= 1.0 + 1e-8)
assert torch.isfinite(torch.tensor(initial_value))
assert torch.isfinite(torch.tensor(final_value))
assert final_z_violation <= initial_z_violation - 1e-4
assert final_upper <= initial_upper - 1e-4
