"""Constrained bilevel linear optimization experiments from FOP 4.2 and SMO 3.1"""

import math
import time
import tqdm

import torch
from scipy.optimize import linprog

from utils import clone_vals, assign_vals
from agd import agd_convex
from bilevel_solvers import optimize_bilevel_constrained_fop, optimize_bilevel_constrained_smo, positive_part_norm_sq


torch.set_default_dtype(torch.float64)


# Section 4.2 / Table 2 experiment controls. `PROBLEM_SIZES` is the actual grid of (n, m, l) values; the rest are algorithmic parameters that can be tuned for better performance.
# PROBLEM_SIZES = [(100 * k, 100 * k, 5 * k) for k in range(1, 11)]
PROBLEM_SIZES = [(200, 200, 10)]
NUM_INSTANCES = 10
SOLVER_METHOD = "smo"
BASE_RHO = 5.0
FINAL_EPS = 1e-2
SMO_EPS = 1e-2
SMO_EPSILON_0 = 1.0
SMO_TAU = 0.8
FEAS_TOL = 1e-2
LOWER_GAP_TOL = 1e-2
SEED = 0
VERBOSE = True
MAX_OUTER_ITERS = 16
MAX_APG_ITERS = 100
ALG4_MAX_ITERS = 20
SMO_SUBPROBLEM_MAX_ITERS = 200
SOLVER_LOG_EVERY = 1
APG_LOG_EVERY = 250
APG_PG_TOL_FACTOR = 0.1
APG_OBJ_TOL_FACTOR = 0.1
WANDB_ENABLED = True
WANDB_PROJECT = "minimax"
WANDB_ENTITY = None
WANDB_MODE = "online"
WANDB_TAGS = ()

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


class _NoOpRun:
    def log(self, *args, **kwargs):
        del args, kwargs

    def finish(self, *args, **kwargs):
        del args, kwargs


def _resolve_wandb():
    if not WANDB_ENABLED:
        return None
    import wandb

    return wandb


def init_instance_run(problem_size, instance_idx):
    wandb = _resolve_wandb()
    if wandb is None:
        return _NoOpRun()

    n_val, m_val, l_val = problem_size
    config = {
        "solver_method": SOLVER_METHOD,
        "problem_size": {"n": n_val, "m": m_val, "l": l_val},
        "instance_idx": instance_idx,
        "seed": SEED,
        "base_rho": BASE_RHO,
        "final_eps": FINAL_EPS,
        "feas_tol": FEAS_TOL,
        "lower_gap_tol": LOWER_GAP_TOL,
        "max_outer_iters": MAX_OUTER_ITERS,
        "max_apg_iters": MAX_APG_ITERS,
        "alg4_max_iters": ALG4_MAX_ITERS,
        "smo_eps": SMO_EPS,
        "smo_epsilon_0": SMO_EPSILON_0,
        "smo_tau": SMO_TAU,
        "smo_subproblem_max_iters": SMO_SUBPROBLEM_MAX_ITERS,
    }
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        group=f"{SOLVER_METHOD}-n{n_val}-m{m_val}-l{l_val}",
        name=f"{SOLVER_METHOD}-n{n_val}-m{m_val}-l{l_val}-inst{instance_idx + 1}-seed{SEED}",
        tags=list(WANDB_TAGS),
        config=config,
        reinit=True,
    )


def log_stage(run, metrics, step):
    if run is None:
        return
    run.log(metrics, step=step)


def finish_instance_run(run, summary):
    if run is None:
        return
    for key, value in summary.items():
        if hasattr(run, "summary"):
            run.summary[key] = value
    run.finish()


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


# def lower_penalty_objective(x_vec, z_vec):
#     """Penalized lower problem used to compute the APG warm start y_tilde."""
#     residual = positive_part(constraint_residual(x_vec, z_vec))
#     return torch.dot(D_TILDE, z_vec) + CURRENT_MU * torch.sum(torch.square(residual))
#
# def lower_penalty_gradient(x_vec, z_vec):
#     residual = positive_part(constraint_residual(x_vec, z_vec))
#     return D_TILDE + 2.0 * CURRENT_MU * (B_MAT.T @ residual)


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

#
# def apg_warm_start(x_vec, z_init):
#     step_lip = max(2.0 * CURRENT_MU * CURRENT_B_NORM * CURRENT_B_NORM, 1e-12)
#     pg_tol = max(1e-8, APG_PG_TOL_FACTOR * CURRENT_EPS)
#     obj_tol = APG_OBJ_TOL_FACTOR * CURRENT_EPS
#
#     z_curr = project_box(z_init.clone())
#     y_curr = z_curr.clone()
#     t_curr = 1.0
#     curr_obj = float(lower_penalty_objective(x_vec, z_curr).item())
#     best_z = z_curr.clone()
#     best_obj = curr_obj
#     last_pg_norm = None
#
#     for apg_iter in range(1, MAX_APG_ITERS + 1):
#         grad = lower_penalty_gradient(x_vec, y_curr)
#         z_next = project_box(y_curr - grad / step_lip)
#         next_obj = float(lower_penalty_objective(x_vec, z_next).item())
#         restarted = False
#
#         # Restart when the accelerated point increases the objective.
#         if next_obj > curr_obj + 1e-12:
#             y_curr = z_curr.clone()
#             t_curr = 1.0
#             grad = lower_penalty_gradient(x_vec, y_curr)
#             z_next = project_box(y_curr - grad / step_lip)
#             next_obj = float(lower_penalty_objective(x_vec, z_next).item())
#             restarted = True
#
#         pg_norm = float((step_lip * torch.linalg.vector_norm(y_curr - z_next)).item())
#         delta_obj = abs(next_obj - curr_obj)
#
#         if next_obj < best_obj:
#             best_obj = next_obj
#             best_z = z_next.clone()
#
#         if VERBOSE and (apg_iter == 1 or apg_iter % APG_LOG_EVERY == 0):
#             print(
#                 f"    APG iter={apg_iter} obj={next_obj:.3e} pg={pg_norm:.3e} restart={restarted}",
#                 flush=True,
#             )
#
#         if pg_norm <= pg_tol and delta_obj <= obj_tol:
#             return best_z, {
#                 "num_iters": apg_iter,
#                 "terminated": True,
#                 "best_obj": best_obj,
#                 "pg_norm": pg_norm,
#             }
#
#         t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_curr * t_curr))
#         y_next = z_next + ((t_curr - 1.0) / t_next) * (z_next - z_curr)
#         y_next = project_box(y_next)
#
#         z_curr = z_next
#         y_curr = y_next
#         t_curr = t_next
#         curr_obj = next_obj
#         last_pg_norm = pg_norm
#
#     print(
#         f"    APG warning: reached MAX_APG_ITERS={MAX_APG_ITERS} at epsilon={CURRENT_EPS:.3e}; using best iterate",
#         flush=True,
#     )
#     return best_z, {
#         "num_iters": MAX_APG_ITERS,
#         "terminated": False,
#         "best_obj": best_obj,
#         "pg_norm": last_pg_norm,
#     }


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


def _vector_distance(u_vec, v_vec):
    return float(torch.linalg.vector_norm(u_vec - v_vec).item())


def _evaluate_lower_candidate(x_vec, candidate_vec, lower_star=None):
    lower_obj = lower_objective(x_vec, candidate_vec)
    if lower_star is None:
        lower_star = solve_lower_level_value(x_vec)
    return {
        "lower_obj": lower_obj,
        "lower_gap": lower_obj - lower_star,
        "feas": feasibility_norm(x_vec, candidate_vec),
    }


def build_smo_diagnostics(x_vec, y_vec, z_vec, solver_result):
    lower_star = solve_lower_level_value(x_vec)
    y_metrics = _evaluate_lower_candidate(x_vec, y_vec, lower_star)
    z_metrics = _evaluate_lower_candidate(x_vec, z_vec, lower_star)
    diagnostics = {
        "y_feas": y_metrics["feas"],
        "y_lower_obj": y_metrics["lower_obj"],
        "y_lower_gap": y_metrics["lower_gap"],
        "z_feas": z_metrics["feas"],
        "z_lower_obj": z_metrics["lower_obj"],
        "z_lower_gap": z_metrics["lower_gap"],
        "yz_distance": _vector_distance(y_vec, z_vec),
    }

    history = solver_result.get("history", [])
    if history:
        last_stage = history[-1]
        subproblem_stats = last_stage["subproblem_stats"]
        diagnostics.update(
            {
                "last_stage_index": last_stage["stage_index"],
                "last_stage_epsilon": last_stage["epsilon_k"],
                "last_stage_rho": last_stage["rho_k"],
                "last_stage_mu": last_stage["mu_k"],
                "last_stage_lambda_norm": last_stage["lambda_norm"],
                "last_stage_next_lambda_norm": last_stage["next_lambda_norm"],
                "last_warm_start_iters": last_stage["warm_start_iters"],
                "last_warm_start_target_iters": last_stage["warm_start_target_iters"],
                "last_warm_start_capped": float(last_stage["warm_start_capped"]),
                "last_subproblem_num_outer_iters": subproblem_stats["num_outer_iters"],
                "last_subproblem_final_diff": subproblem_stats["final_diff"],
                "last_subproblem_terminated": float(subproblem_stats["terminated"]),
            }
        )

    return diagnostics


def run_single_instance_fop(instance_idx, problem_size):
    """
    Run the practical outer loop for one random instance with the FOP solver.

    Each outer stage:
    1. updates (rho_k, mu_k, epsilon_k),
    2. computes a warm start y_tilde by APG,
    3. calls Algorithm 4 / 6 through `optimize_bilevel_constrained_fop`,
    4. checks feasibility and the lower-level optimality gap.
    """
    run = init_instance_run(problem_size, instance_idx)
    start_time = time.perf_counter()
    run_finished = False

    x_prev = torch.zeros(N)
    y_prev = Y_HAT.clone()
    y_tilde_prev = Y_HAT.clone()
    initial_objective = upper_objective(x_prev, y_prev)
    gtilde_hi = compute_gtilde_hi()
    d_y = compute_d_y()

    prox_x = prox_box
    prox_y = prox_box

    L_grad_f1 = 0.0
    L_grad_ftilde1 = 0.0
    L_grad_gtilde = 0.0
    L_gtilde = CURRENT_L_G

    try:
        for k in tqdm.trange(MAX_OUTER_ITERS, desc=f"Instance {instance_idx + 1}/{NUM_INSTANCES}", unit="outer iter"):
            rho_k = BASE_RHO ** (k + 1)   # Weiran: paper says k-1 instead of k+1.
            mu_k = rho_k ** 2
            epsilon_k = 1.0 / rho_k

            if VERBOSE:
                print(
                    f"  Instance {instance_idx + 1}: outer={k} rho={rho_k:.3e} "
                    f"mu={mu_k:.3e} epsilon={epsilon_k:.3e}",
                    flush=True,
                )

            def smooth_lower_penalty(z_vars):
                constraint_z = lower_constraints(x_prev, z_vars)
                penalty = positive_part_norm_sq(constraint_z) * mu_k
                return lower_smooth(x_prev, z_vars) + penalty

            # Estimating the gradient lipschitz constant for AGD.
            L_hat_k = (
                    L_grad_ftilde1
                    + 2 * mu_k * (L_gtilde ** 2 + gtilde_hi * L_grad_gtilde)
            )

            y_init, apg_stats = agd_convex(
                clone_vals(y_prev),
                smooth_lower_penalty,
                prox_y,
                L_hat_k,
                epsilon_k,
                d_y,
            )
            assign_vals(y_tilde_prev, y_init)

            # y_tilde_prev, apg_stats = apg_warm_start(x_prev, y_tilde_prev)

            # The solver mutates these tensors in place to the next outer iterate.
            x_tensor = x_prev.clone().requires_grad_(True)
            y_tensor = y_tilde_prev.clone().requires_grad_(True)
            solver_result = optimize_bilevel_constrained_fop(
                [x_tensor],
                [y_tensor],
                upper_smooth,
                lower_smooth,
                lower_constraints,
                prox_x,
                prox_y,
                D_y=d_y,
                L_grad_f1=L_grad_f1,
                L_grad_ftilde1=L_grad_ftilde1,
                L_grad_gtilde=L_grad_gtilde,
                L_gtilde=CURRENT_L_G,
                gtilde_hi=gtilde_hi,
                epsilon=epsilon_k,
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
            log_stage(
                run,
                {
                    "fop/stage/index": k,
                    "fop/stage/rho": rho_k,
                    "fop/stage/mu": mu_k,
                    "fop/stage/epsilon": epsilon_k,
                    # "fop/apg/num_iters": apg_stats["num_iters"],
                    # "fop/apg/terminated": float(apg_stats["terminated"]),
                    # "fop/apg/best_obj": apg_stats["best_obj"],
                    # "fop/apg/pg_norm": apg_stats["pg_norm"],
                    "fop/subproblem/num_outer_iters": solver_result["solver_stats"]["num_outer_iters"],
                    "fop/subproblem/final_diff": solver_result["solver_stats"]["final_diff"],
                    "fop/subproblem/terminated": float(solver_result["solver_stats"]["terminated"]),
                    "fop/post/upper_obj": current_upper,
                    "fop/post/feas": feas,
                    "fop/post/lower_gap": lower_gap,
                },
                step=k,
            )

            if VERBOSE:
                print(
                    f"  Instance {instance_idx + 1}: outer={k} apg_iters={apg_stats['num_iters']} "
                    f"solver_outer={solver_result['solver_stats']['num_outer_iters']} "
                    f"feas={feas:.3e} gap={lower_gap:.3e} upper={current_upper:.3e}",
                    flush=True,
                )

            if epsilon_k <= FINAL_EPS and feas <= FEAS_TOL and lower_gap <= LOWER_GAP_TOL:
                elapsed = time.perf_counter() - start_time
                result = {
                    "initial_objective": initial_objective,
                    "final_objective": current_upper,
                    "num_outer_iters": k + 1,
                    "final_feas": feas,
                    "final_lower_gap": lower_gap,
                }
                finish_instance_run(
                    run,
                    {
                        "instance/initial_objective": result["initial_objective"],
                        "instance/final_objective": result["final_objective"],
                        "instance/num_outer_iters": result["num_outer_iters"],
                        "instance/final_feas": result["final_feas"],
                        "instance/final_lower_gap": result["final_lower_gap"],
                        "instance/elapsed_sec": elapsed,
                    },
                )
                run_finished = True
                return result

        elapsed = time.perf_counter() - start_time
        finish_instance_run(
            run,
            {
                "instance/failed": 1.0,
                "instance/initial_objective": initial_objective,
                "instance/elapsed_sec": elapsed,
            },
        )
        run_finished = True
        raise RuntimeError(
            f"Instance {instance_idx + 1} did not satisfy the stopping rule after {MAX_OUTER_ITERS} outer iterations"
        )
    except Exception:
        if not run_finished:
            finish_instance_run(
                run,
                {
                    "instance/failed": 1.0,
                    "instance/initial_objective": initial_objective,
                    "instance/elapsed_sec": time.perf_counter() - start_time,
                },
            )
        raise


def run_single_instance_smo(instance_idx, problem_size):
    """Run one SMO solve for the current random instance."""
    run = init_instance_run(problem_size, instance_idx)
    start_time = time.perf_counter()
    run_finished = False
    initial_x = torch.zeros(N)
    initial_y = Y_HAT.clone()
    initial_objective = upper_objective(initial_x, initial_y)
    gtilde_hi = compute_gtilde_hi()
    d_y = compute_d_y()
    stage_diagnostics = []

    def stage_callback(payload):
        x_curr = payload["x"][0]
        y_curr = payload["y"][0]
        z_curr = payload["z"][0]
        lower_star = solve_lower_level_value(x_curr)
        y_metrics = _evaluate_lower_candidate(x_curr, y_curr, lower_star)
        z_metrics = _evaluate_lower_candidate(x_curr, z_curr, lower_star)
        stage_diag = {
            "stage_index": payload["stage_index"],
            "epsilon": payload["epsilon_k"],
            "rho": payload["rho_k"],
            "mu": payload["mu_k"],
            "lambda_norm": payload["lambda_norm"],
            "y_feas": y_metrics["feas"],
            "y_lower_gap": y_metrics["lower_gap"], # \tilde f(x,y) - \tilde f^*(x)
            "z_feas": z_metrics["feas"],
            "z_lower_gap": z_metrics["lower_gap"],
            "yz_distance": _vector_distance(y_curr, z_curr),
            "subproblem_num_outer_iters": payload["subproblem_stats"]["num_outer_iters"],
            "subproblem_final_diff": payload["subproblem_stats"]["final_diff"],
            "subproblem_terminated": float(payload["subproblem_stats"]["terminated"]),
        }
        stage_diagnostics.append(stage_diag)
        metrics = {
            "smo/stage/index": stage_diag["stage_index"],
            "smo/stage/epsilon": stage_diag["epsilon"],
            "smo/stage/rho": stage_diag["rho"],
            "smo/stage/mu": stage_diag["mu"],
            "smo/stage/lambda_norm": stage_diag["lambda_norm"],
            "smo/warm_start/num_iters": payload["warm_start_iters"],
            "smo/warm_start/target_num_iters": payload["warm_start_target_iters"],
            "smo/warm_start/capped": float(payload["warm_start_capped"]),
            "smo/subproblem/num_outer_iters": stage_diag["subproblem_num_outer_iters"],
            "smo/subproblem/final_diff": stage_diag["subproblem_final_diff"],
            "smo/subproblem/terminated": stage_diag["subproblem_terminated"],
            "smo/post/upper_obj": upper_objective(x_curr, y_curr),
            "smo/post/y_feas": stage_diag["y_feas"],
            "smo/post/y_lower_gap": stage_diag["y_lower_gap"],
            "smo/post/z_feas": stage_diag["z_feas"],
            "smo/post/z_lower_gap": stage_diag["z_lower_gap"],
            "smo/post/yz_distance": stage_diag["yz_distance"],
        }
        log_stage(run, metrics, step=payload["stage_index"])
        if VERBOSE:
            print(
                f"    SMO stage={stage_diag['stage_index']} eps={stage_diag['epsilon']:.3e} "
                f"y_feas={stage_diag['y_feas']:.3e} y_gap={stage_diag['y_lower_gap']:.3e} "
                f"z_feas={stage_diag['z_feas']:.3e} z_gap={stage_diag['z_lower_gap']:.3e} "
                f"yz_dist={stage_diag['yz_distance']:.3e} "
                f"sub_outer={stage_diag['subproblem_num_outer_iters']} "
                f"sub_diff={stage_diag['subproblem_final_diff']:.3e} "
                f"sub_done={int(stage_diag['subproblem_terminated'])}",
                flush=True,
            )

    try:
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
            epsilon_0=SMO_EPSILON_0,
            subproblem_max_iter=SMO_SUBPROBLEM_MAX_ITERS,
            stage_callback=stage_callback,
            verbose=VERBOSE,
            log_every=SOLVER_LOG_EVERY,
        )

        x_final = x_tensor.detach().clone()
        y_final = y_tensor.detach().clone()
        z_final = solver_result["z_eps"][0]
        if torch.any(torch.abs(x_final) > 1.0 + 1e-10):
            raise RuntimeError("SMO iterate x left the box constraints")
        if torch.any(torch.abs(y_final) > 1.0 + 1e-10):
            raise RuntimeError("SMO iterate y left the box constraints")
        if torch.any(torch.abs(z_final) > 1.0 + 1e-10):
            raise RuntimeError("SMO auxiliary iterate z left the box constraints")

        diagnostics = build_smo_diagnostics(x_final, y_final, z_final, solver_result)
        feas = diagnostics["y_feas"]
        lower_gap = diagnostics["y_lower_gap"]
        current_upper = upper_objective(x_final, y_final)

        if VERBOSE:
            print(
                f"  Instance {instance_idx + 1}: smo_outer={solver_result['num_outer_iters']} "
                f"y_feas={feas:.3e} y_gap={lower_gap:.3e} "
                f"z_feas={diagnostics['z_feas']:.3e} z_gap={diagnostics['z_lower_gap']:.3e} "
                f"yz_dist={diagnostics['yz_distance']:.3e} upper={current_upper:.3e}",
                flush=True,
            )

        elapsed = time.perf_counter() - start_time
        if feas > FEAS_TOL or lower_gap > LOWER_GAP_TOL:
            failure_summary = {
                "instance/failed": 1.0,
                "instance/initial_objective": initial_objective,
                "instance/final_objective": current_upper,
                "instance/num_outer_iters": solver_result["num_outer_iters"],
                "instance/final_feas": feas,
                "instance/final_lower_gap": lower_gap,
                "instance/elapsed_sec": elapsed,
                "instance/final_z_feas": diagnostics["z_feas"],
                "instance/final_z_lower_gap": diagnostics["z_lower_gap"],
                "instance/final_yz_distance": diagnostics["yz_distance"],
                "instance/last_subproblem_num_outer_iters": diagnostics.get("last_subproblem_num_outer_iters"),
                "instance/last_subproblem_final_diff": diagnostics.get("last_subproblem_final_diff"),
                "instance/last_subproblem_terminated": diagnostics.get("last_subproblem_terminated"),
            }
            finish_instance_run(
                run,
                failure_summary,
            )
            run_finished = True
            raise RuntimeError(
                f"Instance {instance_idx + 1} SMO solve failed the stop check: "
                f"y_feas={feas:.3e}, y_gap={lower_gap:.3e}, "
                f"z_feas={diagnostics['z_feas']:.3e}, z_gap={diagnostics['z_lower_gap']:.3e}, "
                f"yz_dist={diagnostics['yz_distance']:.3e}, "
                f"last_subproblem_terminated={int(diagnostics.get('last_subproblem_terminated', 0.0))}, "
                f"last_subproblem_outer={diagnostics.get('last_subproblem_num_outer_iters')}, "
                f"last_subproblem_diff={diagnostics.get('last_subproblem_final_diff')}"
            )

        result = {
            "initial_objective": initial_objective,
            "final_objective": current_upper,
            "num_outer_iters": solver_result["num_outer_iters"],
            "final_feas": feas,
            "final_lower_gap": lower_gap,
            "smo_diagnostics": diagnostics,
            "stage_diagnostics": stage_diagnostics,
        }
        finish_instance_run(
            run,
            {
                "instance/initial_objective": result["initial_objective"],
                "instance/final_objective": result["final_objective"],
                "instance/num_outer_iters": result["num_outer_iters"],
                "instance/final_feas": result["final_feas"],
                "instance/final_lower_gap": result["final_lower_gap"],
                "instance/elapsed_sec": elapsed,
                "instance/final_z_feas": diagnostics["z_feas"],
                "instance/final_z_lower_gap": diagnostics["z_lower_gap"],
                "instance/final_yz_distance": diagnostics["yz_distance"],
                "instance/last_subproblem_num_outer_iters": diagnostics.get("last_subproblem_num_outer_iters"),
                "instance/last_subproblem_final_diff": diagnostics.get("last_subproblem_final_diff"),
                "instance/last_subproblem_terminated": diagnostics.get("last_subproblem_terminated"),
            },
        )
        run_finished = True
        return result
    except Exception:
        if not run_finished:
            finish_instance_run(
                run,
                {
                    "instance/failed": 1.0,
                    "instance/initial_objective": initial_objective,
                    "instance/elapsed_sec": time.perf_counter() - start_time,
                },
            )
        raise


def run_single_instance(instance_idx, problem_size):
    if SOLVER_METHOD == "fop":
        return run_single_instance_fop(instance_idx, problem_size)
    if SOLVER_METHOD == "smo":
        return run_single_instance_smo(instance_idx, problem_size)
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
        result = run_single_instance(instance_idx, problem_size)
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
