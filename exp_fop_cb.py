import math
import time
import tqdm

import torch
from scipy.optimize import linprog

from minimax import optimize_bilevel_constrained_fop, optimize_bilevel_constrained_smo


torch.set_default_dtype(torch.float64)


# Section 4.2 / Table 2 experiment controls. `PROBLEM_SIZES` is the actual grid of (n, m, l) values; the rest are algorithmic parameters that can be tuned for better performance.
# PROBLEM_SIZES = [(100 * k, 100 * k, 5 * k) for k in range(1, 11)]
PROBLEM_SIZES = [(400, 400, 20)]
NUM_INSTANCES = 10
SOLVER_METHOD = "smo"
BASE_RHO = 5.0
FINAL_EPS = 0.04
SMO_EPS = 1e-2
SMO_ETA0 = 1.0
SMO_TAU = 0.8
FEAS_TOL = 1e-2
LOWER_GAP_TOL = 1e-2
SEED = 0
VERBOSE = False
MAX_OUTER_ITERS = 16
MAX_APG_ITERS = 100
ALG4_MAX_ITERS = 20
SMO_SUBPROBLEM_MAX_ITERS = 20
SOLVER_LOG_EVERY = 1
APG_LOG_EVERY = 250
APG_PG_TOL_FACTOR = 0.1
APG_OBJ_TOL_FACTOR = 0.1

# Global state placeholders for the current problem instance and iteration
N = None
M = None
L = None
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
CURRENT_B_NORM = None


def prox_box(v, coeff):
    """Projection onto the box [-1, 1]^d used as the prox of the indicator."""
    del coeff
    return torch.clamp(v, min=-1.0, max=1.0)


def project_box(v):
    return torch.clamp(v, min=-1.0, max=1.0)


def positive_part(v):
    return torch.clamp(v, min=0.0)


def upper_smooth(x_params, y_params):
    """Smooth upper objective term c^T x + d^T y."""
    return torch.dot(C, x_params[0]) + torch.dot(D, y_params[0])


def lower_smooth(x_params, z_params):
    """Smooth lower objective term d_tilde^T z."""
    del x_params
    return torch.dot(D_TILDE, z_params[0])


def lower_constraints(x_params, z_params):
    """Lower-level linear constraint map A x + B z - b."""
    return A_MAT @ x_params[0] + B_MAT @ z_params[0] - B_VEC


def upper_objective(x_vec, y_vec):
    return float((torch.dot(C, x_vec) + torch.dot(D, y_vec)).item())


def lower_objective(x_vec, z_vec):
    del x_vec
    return float(torch.dot(D_TILDE, z_vec).item())


def constraint_residual(x_vec, z_vec):
    return A_MAT @ x_vec + B_MAT @ z_vec - B_VEC


def feasibility_norm(x_vec, z_vec):
    residual = positive_part(constraint_residual(x_vec, z_vec))
    return float(torch.linalg.vector_norm(residual).item())


# Move these inside.
def lower_penalty_objective(x_vec, z_vec):
    """Penalized lower problem used to compute the APG warm start y_tilde."""
    residual = positive_part(constraint_residual(x_vec, z_vec))
    return torch.dot(D_TILDE, z_vec) + CURRENT_MU * torch.sum(torch.square(residual))

# Move inside fop.
def lower_penalty_gradient(x_vec, z_vec):
    residual = positive_part(constraint_residual(x_vec, z_vec))
    return D_TILDE + 2.0 * CURRENT_MU * (B_MAT.T @ residual)


def compute_joint_constraint_lipschitz():
    """Lipschitz constant L_gtilde for the joint constraint map g_tilde(x, y) = A x + B y - b."""
    return float(torch.linalg.matrix_norm(torch.cat([A_MAT, B_MAT], dim=1), ord=2).item())


def compute_b_matrix_norm():
    return float(torch.linalg.matrix_norm(B_MAT, ord=2).item())


def compute_gtilde_hi():
    row_bounds = torch.sum(torch.abs(A_MAT), dim=1)
    row_bounds = row_bounds + torch.sum(torch.abs(B_MAT), dim=1) + torch.abs(B_VEC)
    return float(torch.linalg.vector_norm(row_bounds).item())


def compute_d_y():
    return 2.0 * math.sqrt(M)


def sample_interior_box_point(dim):
    """Sample y_hat strictly inside the box so the lower KKT construction is simple."""
    while True:
        candidate = project_box(0.1 * torch.randn(dim))
        if torch.all(torch.abs(candidate) < 1.0):
            return candidate


def generate_instance(problem_size):
    """
    Generate one random constrained bilevel linear instance from Section 4.2.

    The construction of `B_VEC` and `D_TILDE` makes `Y_HAT` an optimal
    lower-level solution when x = 0.
    """
    global N, M, L
    global C, D, D_TILDE, A_MAT, B_MAT, B_VEC, Y_HAT
    global CURRENT_L_G, CURRENT_B_NORM
    
    # Page 12 top
    N, M, L = problem_size
    C = torch.randn(N)
    D = torch.randn(M)
    A_MAT = 0.01 * torch.randn(L, N)
    B_MAT = 0.01 * torch.randn(L, M)
    Y_HAT = sample_interior_box_point(M)

    # Pick nonnegative multipliers and back out (b, d_tilde) so Y_HAT solves the lower problem at x = 0.
    lam = torch.abs(torch.randn(L))
    B_VEC = B_MAT @ Y_HAT
    D_TILDE = -B_MAT.T @ lam

    CURRENT_L_G = compute_joint_constraint_lipschitz()
    CURRENT_B_NORM = compute_b_matrix_norm()


def projected_gradient_mapping_norm(x_vec, z_vec, step_lip):
    grad = lower_penalty_gradient(x_vec, z_vec)
    prox_point = project_box(z_vec - grad / step_lip)
    pg_norm = step_lip * torch.linalg.vector_norm(z_vec - prox_point)
    return float(pg_norm.item()), prox_point


def apg_warm_start(x_vec, z_init):
    step_lip = max(2.0 * CURRENT_MU * CURRENT_B_NORM * CURRENT_B_NORM, 1e-12)
    pg_tol = max(1e-8, APG_PG_TOL_FACTOR * CURRENT_EPS)
    obj_tol = APG_OBJ_TOL_FACTOR * CURRENT_EPS

    z_curr = project_box(z_init.clone())
    y_curr = z_curr.clone()
    t_curr = 1.0
    curr_obj = float(lower_penalty_objective(x_vec, z_curr).item())
    best_z = z_curr.clone()
    best_obj = curr_obj
    last_pg_norm = None

    for apg_iter in range(1, MAX_APG_ITERS + 1):
        grad = lower_penalty_gradient(x_vec, y_curr)
        z_next = project_box(y_curr - grad / step_lip)
        next_obj = float(lower_penalty_objective(x_vec, z_next).item())
        restarted = False

        # Restart when the accelerated point increases the objective.
        if next_obj > curr_obj + 1e-12:
            y_curr = z_curr.clone()
            t_curr = 1.0
            grad = lower_penalty_gradient(x_vec, y_curr)
            z_next = project_box(y_curr - grad / step_lip)
            next_obj = float(lower_penalty_objective(x_vec, z_next).item())
            restarted = True

        pg_norm = float((step_lip * torch.linalg.vector_norm(y_curr - z_next)).item())
        delta_obj = abs(next_obj - curr_obj)

        if next_obj < best_obj:
            best_obj = next_obj
            best_z = z_next.clone()

        if VERBOSE and (apg_iter == 1 or apg_iter % APG_LOG_EVERY == 0):
            print(
                f"    APG iter={apg_iter} obj={next_obj:.3e} pg={pg_norm:.3e} restart={restarted}",
                flush=True,
            )

        if pg_norm <= pg_tol and delta_obj <= obj_tol:
            return best_z, {
                "num_iters": apg_iter,
                "terminated": True,
                "best_obj": best_obj,
                "pg_norm": pg_norm,
            }

        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_curr * t_curr))
        y_next = z_next + ((t_curr - 1.0) / t_next) * (z_next - z_curr)
        y_next = project_box(y_next)

        z_curr = z_next
        y_curr = y_next
        t_curr = t_next
        curr_obj = next_obj
        last_pg_norm = pg_norm

    # TODO: monitor that this function finds initial y quickly to high accuracy. Pay attention to the results below.
    print(
        f"    APG warning: reached MAX_APG_ITERS={MAX_APG_ITERS} at epsilon={CURRENT_EPS:.3e}; using best iterate",
        flush=True,
    )
    return best_z, {
        "num_iters": MAX_APG_ITERS,
        "terminated": False,
        "best_obj": best_obj,
        "pg_norm": last_pg_norm,
    }


# TODO: replace with CVX if lower level problem is not linear programming.
def solve_lower_level_value(x_vec):
    """Exact lower-level optimal value f_tilde^*(x) via LP for the stop check."""
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


def run_single_instance_fop(instance_idx):
    """
    Run the practical outer loop for one random instance with the FOP solver.

    Each outer stage:
    1. updates (rho_k, mu_k, epsilon_k),
    2. computes a warm start y_tilde by APG,
    3. calls Algorithm 4 / 6 through `optimize_bilevel_constrained_fop`,
    4. checks feasibility and the lower-level optimality gap.
    """
    global CURRENT_RHO, CURRENT_MU, CURRENT_EPS

    x_prev = torch.zeros(N)
    y_prev = Y_HAT.clone()
    y_tilde_prev = Y_HAT.clone()
    initial_objective = upper_objective(x_prev, y_prev)
    gtilde_hi = compute_gtilde_hi()
    d_y = compute_d_y()

    # for k in range(MAX_OUTER_ITERS):
    for k in tqdm.trange(MAX_OUTER_ITERS, desc=f"Instance {instance_idx + 1}/{NUM_INSTANCES}", unit="outer iter"):
        CURRENT_RHO = BASE_RHO ** (k + 1)
        CURRENT_MU = CURRENT_RHO ** 2
        CURRENT_EPS = 1.0 / CURRENT_RHO

        if VERBOSE:
            print(
                f"  Instance {instance_idx + 1}: outer={k} rho={CURRENT_RHO:.3e} "
                f"mu={CURRENT_MU:.3e} epsilon={CURRENT_EPS:.3e}",
                flush=True,
            )

        y_tilde_prev, apg_stats = apg_warm_start(x_prev, y_tilde_prev)

        # The solver mutates these tensors in place to the next outer iterate.
        x_tensor = x_prev.clone().requires_grad_(True)
        y_tensor = y_tilde_prev.clone().requires_grad_(True)
        solver_result = optimize_bilevel_constrained_fop(
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
            verbose=VERBOSE,
            log_every=SOLVER_LOG_EVERY,
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
        current_upper = upper_objective(x_prev, y_prev)

        if VERBOSE:
            print(
                f"  Instance {instance_idx + 1}: outer={k} apg_iters={apg_stats['num_iters']} "
                f"solver_outer={solver_result['solver_stats']['num_outer_iters']} "
                f"feas={feas:.3e} gap={lower_gap:.3e} upper={current_upper:.3e}",
                flush=True,
            )

        if CURRENT_EPS <= FINAL_EPS and feas <= FEAS_TOL and lower_gap <= LOWER_GAP_TOL:
            return {
                "initial_objective": initial_objective,
                "final_objective": current_upper,
                "num_outer_iters": k + 1,
                "final_feas": feas,
                "final_lower_gap": lower_gap,
            }

    raise RuntimeError(
        f"Instance {instance_idx + 1} did not satisfy the stopping rule after {MAX_OUTER_ITERS} outer iterations"
    )


def run_single_instance_smo(instance_idx):
    """Run one SMO solve for the current random instance."""
    initial_x = torch.zeros(N)
    initial_y = Y_HAT.clone()
    initial_objective = upper_objective(initial_x, initial_y)
    gtilde_hi = compute_gtilde_hi()
    d_y = compute_d_y()

    x_tensor = initial_x.clone().requires_grad_(True)
    y_tensor = initial_y.clone().requires_grad_(True)
    solver_result = optimize_bilevel_constrained_smo(
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
        epsilon=SMO_EPS,
        tau=SMO_TAU,
        epsilon_0=SMO_ETA0,
        subproblem_max_iter=SMO_SUBPROBLEM_MAX_ITERS,
        verbose=VERBOSE,
        log_every=SOLVER_LOG_EVERY,
    )

    x_final = x_tensor.detach().clone()
    y_final = y_tensor.detach().clone()
    if torch.any(torch.abs(x_final) > 1.0 + 1e-10):
        raise RuntimeError("SMO iterate x left the box constraints")
    if torch.any(torch.abs(y_final) > 1.0 + 1e-10):
        raise RuntimeError("SMO iterate y left the box constraints")

    feas = feasibility_norm(x_final, y_final)
    lower_star = solve_lower_level_value(x_final)
    lower_gap = lower_objective(x_final, y_final) - lower_star
    current_upper = upper_objective(x_final, y_final)

    if VERBOSE:
        print(
            f"  Instance {instance_idx + 1}: smo_outer={solver_result['num_outer_iters']} "
            f"feas={feas:.3e} gap={lower_gap:.3e} upper={current_upper:.3e}",
            flush=True,
        )

    if feas > FEAS_TOL or lower_gap > LOWER_GAP_TOL:
        raise RuntimeError(
            f"Instance {instance_idx + 1} SMO solve failed the stop check: "
            f"feas={feas:.3e}, gap={lower_gap:.3e}"
        )

    return {
        "initial_objective": initial_objective,
        "final_objective": current_upper,
        "num_outer_iters": solver_result["num_outer_iters"],
        "final_feas": feas,
        "final_lower_gap": lower_gap,
    }


def run_single_instance(instance_idx):
    if SOLVER_METHOD == "fop":
        return run_single_instance_fop(instance_idx)
    if SOLVER_METHOD == "smo":
        return run_single_instance_smo(instance_idx)
    raise ValueError(f"Unknown SOLVER_METHOD: {SOLVER_METHOD}")


def run_problem_size(problem_size):
    """Average the paper's initial/final objective values over NUM_INSTANCES."""
    n_val, m_val, l_val = problem_size
    print(f"Starting triple ({n_val}, {m_val}, {l_val})", flush=True)

    initial_values = []
    final_values = []

    for instance_idx in range(NUM_INSTANCES):
        generate_instance(problem_size)
        start_time = time.perf_counter()
        result = run_single_instance(instance_idx)
        elapsed = time.perf_counter() - start_time

        initial_values.append(result["initial_objective"])
        final_values.append(result["final_objective"])

        print(
            f"Completed instance {instance_idx + 1}/{NUM_INSTANCES} for ({n_val}, {m_val}, {l_val}) "
            f"in {result['num_outer_iters']} outer iterations [{elapsed:.1f}s]: "
            f"initial={result['initial_objective']:.2f} final={result['final_objective']:.2f} "
            f"feas={result['final_feas']:.2e} gap={result['final_lower_gap']:.2e}",
            flush=True,
        )

    avg_initial = sum(initial_values) / len(initial_values)
    avg_final = sum(final_values) / len(final_values)
    return {
        "avg_initial": avg_initial,
        "avg_final": avg_final,
    }


def run_experiment():
    """Main Table 2 driver over the requested size grid."""
    torch.manual_seed(SEED)
    print("n m l Initial objective value Final objective value", flush=True)
    results = []

    for problem_size in PROBLEM_SIZES:
        stats = run_problem_size(problem_size)
        n_val, m_val, l_val = problem_size
        print(f"{n_val} {m_val} {l_val} {stats['avg_initial']:.2f} {stats['avg_final']:.2f}", flush=True)
        results.append((problem_size, stats["avg_initial"], stats["avg_final"]))

    return results


if __name__ == "__main__":
    run_experiment()
