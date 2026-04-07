import contextlib
import io
import math

import torch
from scipy.optimize import linprog

from minimax import optimize_bilevel_constrained


torch.set_default_dtype(torch.float64)


PROBLEM_SIZES = [(10 * k, 10 * k, 5 * k) for k in range(1, 11)]
NUM_INSTANCES = 10
BASE_RHO = 5.0
FINAL_EPS = 1e-4
FEAS_TOL = 1e-4
LOWER_GAP_TOL = 1e-4
SEED = 0
VERBOSE = False
MAX_OUTER_ITERS = 16
ALG4_MAX_ITERS = 200

N = 200
M = 200
L = 10
C = None
D = None
D_TILDE = None
A_MAT = None
B_MAT = None
B_VEC = None
Y_HAT = None

CURRENT_RHO = None
CURRENT_MU = None
CURRENT_EPS = None
CURRENT_L_G = None


def prox_box(v, coeff):
    del coeff
    return torch.clamp(v, min=-1.0, max=1.0)


def project_box(v):
    return torch.clamp(v, min=-1.0, max=1.0)


def positive_part(v):
    return torch.clamp(v, min=0.0)


def upper_smooth(x_params, y_params):
    return torch.dot(C, x_params[0]) + torch.dot(D, y_params[0])


def lower_smooth(x_params, z_params):
    del x_params
    return torch.dot(D_TILDE, z_params[0])


def lower_constraints(x_params, z_params):
    return A_MAT @ x_params[0] + B_MAT @ z_params[0] - B_VEC


def upper_objective(x_vec, y_vec):
    return (torch.dot(C, x_vec) + torch.dot(D, y_vec)).item()


def lower_objective(x_vec, z_vec):
    del x_vec
    return torch.dot(D_TILDE, z_vec).item()


def constraint_residual(x_vec, z_vec):
    return A_MAT @ x_vec + B_MAT @ z_vec - B_VEC


def feasibility_norm(x_vec, z_vec):
    return torch.linalg.vector_norm(positive_part(constraint_residual(x_vec, z_vec))).item()


def lower_penalty_value(x_vec, z_vec):
    residual = positive_part(constraint_residual(x_vec, z_vec))
    return torch.dot(D_TILDE, z_vec) + CURRENT_MU * torch.sum(torch.square(residual))


def compute_l_g():
    return torch.linalg.matrix_norm(torch.cat([A_MAT, B_MAT], dim=1), ord=2).item()


def compute_gtilde_hi():
    row_bounds = torch.sum(torch.abs(A_MAT), dim=1)
    row_bounds = row_bounds + torch.sum(torch.abs(B_MAT), dim=1) + torch.abs(B_VEC)
    return torch.linalg.vector_norm(row_bounds).item()


def compute_d_y():
    return 2.0 * math.sqrt(M)


def sample_interior_box_point(dim):
    while True:
        candidate = project_box(0.1 * torch.randn(dim))
        if torch.all(torch.abs(candidate) < 1.0):
            return candidate


def generate_instance(problem_size):
    global N, M, L
    global C, D, D_TILDE, A_MAT, B_MAT, B_VEC, Y_HAT

    N, M, L = problem_size
    C = torch.randn(N)
    D = torch.randn(M)
    A_MAT = 0.01 * torch.randn(L, N)
    B_MAT = 0.01 * torch.randn(L, M)
    Y_HAT = sample_interior_box_point(M)

    lam = torch.abs(torch.randn(L))
    B_VEC = B_MAT @ Y_HAT
    D_TILDE = -B_MAT.T @ lam


def apg_iteration_budget():
    if CURRENT_EPS <= 0:
        raise ValueError(f"Invalid CURRENT_EPS: {CURRENT_EPS}")
    if CURRENT_L_G is None:
        raise ValueError("CURRENT_L_G must be set before calling APG")
    l_apg = 2.0 * CURRENT_MU * CURRENT_L_G * CURRENT_L_G
    return max(1, math.ceil(math.sqrt(8.0 * l_apg * M / CURRENT_EPS)))


def apg_warm_start(x_vec, z_init):
    step_lip = max(2.0 * CURRENT_MU * CURRENT_L_G * CURRENT_L_G, 1.0)
    z_prev = project_box(z_init.clone())
    z_acc = z_prev.clone()
    t_prev = 1.0

    for _ in range(apg_iteration_budget()):
        residual = positive_part(constraint_residual(x_vec, z_acc))
        grad = D_TILDE + 2.0 * CURRENT_MU * (B_MAT.T @ residual)
        z_next = project_box(z_acc - grad / step_lip)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_prev * t_prev))
        z_acc = z_next + ((t_prev - 1.0) / t_next) * (z_next - z_prev)
        z_prev = z_next
        t_prev = t_next

    z_prev = project_box(z_prev)
    if torch.any(torch.abs(z_prev) > 1.0 + 1e-10):
        raise RuntimeError("APG warm start left the box constraints")
    return z_prev


def solve_lower_level_value(x_vec):
    result = linprog(
        c=D_TILDE.detach().cpu().numpy(),
        A_ub=B_MAT.detach().cpu().numpy(),
        b_ub=(B_VEC - A_MAT @ x_vec).detach().cpu().numpy(),
        bounds=[(-1.0, 1.0)] * M,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Lower-level LP solve failed: {result.message}")
    return float(result.fun)


def quiet_solver_output():
    if VERBOSE:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def run_single_instance():
    global CURRENT_RHO, CURRENT_MU, CURRENT_EPS, CURRENT_L_G

    x_prev = torch.zeros(N)
    y_prev = Y_HAT.clone()
    y_tilde_prev = Y_HAT.clone()
    initial_objective = upper_objective(x_prev, Y_HAT)
    gtilde_hi = compute_gtilde_hi()
    d_y = compute_d_y()

    for k in range(MAX_OUTER_ITERS):
        CURRENT_RHO = BASE_RHO ** (k + 1)
        CURRENT_MU = CURRENT_RHO ** 2
        CURRENT_EPS = 1.0 / CURRENT_RHO
        CURRENT_L_G = compute_l_g()

        y_tilde_prev = apg_warm_start(x_prev, y_tilde_prev)

        x_tensor = x_prev.clone().requires_grad_(True)
        y_tensor = y_tilde_prev.clone().requires_grad_(True)
        with quiet_solver_output():
            optimize_bilevel_constrained(
                [x_tensor],
                [y_tensor],
                upper_smooth,
                lower_smooth,
                lower_constraints,
                prox_box,
                prox_box,
                D_y=d_y,
                L_grad_f1=0.0,
                L_grad_ftilde1=0.0,
                L_grad_gtilde=0.0,
                L_gtilde=CURRENT_L_G,
                gtilde_hi=gtilde_hi,
                epsilon=CURRENT_EPS,
                max_iter=ALG4_MAX_ITERS,
            )

        x_prev = x_tensor.detach().clone()
        y_prev = y_tensor.detach().clone()

        if torch.any(torch.abs(x_prev) > 1.0 + 1e-10):
            raise RuntimeError("Outer iterate x left the box constraints")
        if torch.any(torch.abs(y_prev) > 1.0 + 1e-10):
            raise RuntimeError("Outer iterate y left the box constraints")

        feas = feasibility_norm(x_prev, y_prev)
        lower_star = solve_lower_level_value(x_prev)
        lower_gap = lower_objective(x_prev, y_prev) - lower_star

        if VERBOSE:
            penalty_value = lower_penalty_value(x_prev, y_prev).item()
            print(
                f"k={k} eps={CURRENT_EPS:.3e} feas={feas:.3e} "
                f"gap={lower_gap:.3e} penalty={penalty_value:.3e}"
            )

        if CURRENT_EPS <= FINAL_EPS and feas <= FEAS_TOL and lower_gap <= LOWER_GAP_TOL:
            return initial_objective, upper_objective(x_prev, y_prev)

    raise RuntimeError(
        f"Instance did not satisfy the stopping rule after {MAX_OUTER_ITERS} outer iterations"
    )


def run_problem_size(problem_size):
    initial_values = []
    final_values = []

    for _ in range(NUM_INSTANCES):
        generate_instance(problem_size)
        initial_value, final_value = run_single_instance()
        initial_values.append(initial_value)
        final_values.append(final_value)
        print(f"Instance completed with initial objective {initial_value:.2f} and final objective {final_value:.2f}")

    avg_initial = sum(initial_values) / len(initial_values)
    avg_final = sum(final_values) / len(final_values)
    return avg_initial, avg_final


def run_experiment():
    torch.manual_seed(SEED)
    print("n m l Initial objective value Final objective value")
    results = []
    for problem_size in PROBLEM_SIZES:
        avg_initial, avg_final = run_problem_size(problem_size)
        n_val, m_val, l_val = problem_size
        print(f"{n_val} {m_val} {l_val} {avg_initial:.2f} {avg_final:.2f}")
        results.append((problem_size, avg_initial, avg_final))
    return results


if __name__ == "__main__":
    run_experiment()
