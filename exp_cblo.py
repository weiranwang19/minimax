"""Constrained bilevel linear optimization experiments from FOP 4.2 and SMO 3.1"""

import math
import time
import tqdm

import torch
from scipy.optimize import linprog

from utils import clone_vals
from bilevel_solvers import (
    optimize_bilevel_constrained_minimax,
    optimize_bilevel_constrained_smo,
    optimize_bilevel_contrained_fop_practical,
)


torch.set_default_dtype(torch.float64)


# Section 4.2 / Table 2 experiment controls. `PROBLEM_SIZES` is the actual grid of (n, m, l) values; the rest are algorithmic parameters that can be tuned for better performance.
# PROBLEM_SIZES = [(100 * k, 100 * k, 5 * k) for k in range(1,4)]
PROBLEM_SIZES = [(500, 500, 25)]
# Zero-based instance indices. Edit this list to rerun only failed/interrupted instances
# while keeping the exact same sampled instance for each (n, m, l) across methods.
# NUM_INSTANCES = list(range(5))
NUM_INSTANCES = [0,2,3,4,6,7,8,9]
SOLVER_METHOD = "gcmo"
# FOP
BASE_RHO = 5.0
FINAL_EPS = 1e-2
MAX_OUTER_ITERS = 200
ALG4_MAX_ITERS = 200

# SMO
SMO_EPS = 1e-2
SMO_EPSILON_0 = 1.0
SMO_TAU = 0.8
SMO_SUBPROBLEM_MAX_ITERS = 200

# GCMO
GCMO_EPS = 1e-2
GCMO_MAX_ITERS = 200
GCMO_LAGRANGE_BOUND = 200
GCMO_LIP_OVERRIDE = None

# Common
FEAS_TOL = 1e-2
LOWER_GAP_TOL = 1e-2
SEED = 0
VERBOSE = True
SOLVER_LOG_EVERY = 1
WANDB_ENABLED = True
WANDB_PROJECT = "minimax"
WANDB_ENTITY = None
WANDB_MODE = "online"
WANDB_TAGS = ()
NCWC_HISTORY_METRICS = (
    "ncwc/upper_obj",
    "ncwc/feas",
    "ncwc/lower_gap",
    "ncwc/call_index",
    "ncwc/completed_outer_iters",
    "ncwc/scsc_inner_iters",
    "ncwc/final_diff",
    "ncwc/inner_terminated",
)
FOP_STAGE_METRICS = (
    "fop/stage/rho",
    "fop/stage/mu",
    "fop/stage/epsilon",
    "fop/warm_start/num_iters",
    "fop/warm_start/target_num_iters",
    "fop/warm_start/capped",
    "fop/subproblem/num_outer_iters",
    "fop/subproblem/num_inner_iters",
    "fop/subproblem/final_diff",
    "fop/subproblem/terminated",
    "fop/post/upper_obj",
    "fop/post/feas",
    "fop/post/lower_gap",
)
SMO_STAGE_METRICS = (
    "smo/stage/epsilon",
    "smo/stage/rho",
    "smo/stage/mu",
    "smo/stage/lambda_norm",
    "smo/warm_start/num_iters",
    "smo/warm_start/target_num_iters",
    "smo/warm_start/capped",
    "smo/subproblem/num_outer_iters",
    "smo/subproblem/num_inner_iters",
    "smo/subproblem/final_diff",
    "smo/subproblem/terminated",
    "smo/post/upper_obj",
    "smo/post/y_feas",
    "smo/post/y_lower_gap",
    "smo/post/z_feas",
    "smo/post/z_lower_gap",
    "smo/post/yz_distance",
)
GCMO_STAGE_METRICS = (
    "gcmo/stage/rho",
    "gcmo/stage/epsilon",
    "gcmo/stage/epsilon_0",
    "gcmo/stage/lip_h",
    "gcmo/stage/computed_lip_h",
    "gcmo/subproblem/num_outer_iters",
    "gcmo/subproblem/num_inner_iters",
    "gcmo/subproblem/final_diff",
    "gcmo/subproblem/terminated",
    "gcmo/post/upper_obj",
    "gcmo/post/y_feas",
    "gcmo/post/y_lower_gap",
    "gcmo/post/z_feas",
    "gcmo/post/z_lower_gap",
    "gcmo/post/yz_distance",
    "gcmo/post/lambda_norm",
    "gcmo/post/z_lambda_norm",
    "gcmo/post/lambda_distance",
    "gcmo/post/lambda_touched_bound",
    "gcmo/post/z_lambda_touched_bound",
)

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
CURRENT_L_G = None


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


def _compact_metrics(metrics):
    return {key: value for key, value in metrics.items() if value is not None}


def _configure_run_metrics(run):
    if not hasattr(run, "define_metric"):
        return
    step_metric = "ncwc/cumulative_scsc_inner_iters"
    run.define_metric(step_metric)
    for metric_name in NCWC_HISTORY_METRICS:
        run.define_metric(metric_name, step_metric=step_metric)
    run.define_metric("fop/stage/index")
    for metric_name in FOP_STAGE_METRICS:
        run.define_metric(metric_name, step_metric="fop/stage/index")
    run.define_metric("smo/stage/index")
    for metric_name in SMO_STAGE_METRICS:
        run.define_metric(metric_name, step_metric="smo/stage/index")
    run.define_metric("gcmo/stage/index")
    for metric_name in GCMO_STAGE_METRICS:
        run.define_metric(metric_name, step_metric="gcmo/stage/index")


def init_instance_run(problem_size, instance_idx, instance_position, num_instances):
    wandb = _resolve_wandb()
    if wandb is None:
        return _NoOpRun()

    n_val, m_val, l_val = problem_size
    config = {
        "solver_method": SOLVER_METHOD,
        "problem_size": {"n": n_val, "m": m_val, "l": l_val},
        "instance_idx": instance_idx,
        "instance_position": instance_position,
        "num_instances": num_instances,
        "seed": SEED,
        "base_rho": BASE_RHO,
        "final_eps": FINAL_EPS,
        "feas_tol": FEAS_TOL,
        "lower_gap_tol": LOWER_GAP_TOL,
        "max_outer_iters": MAX_OUTER_ITERS,
        "alg4_max_iters": ALG4_MAX_ITERS,
        "smo_eps": SMO_EPS,
        "smo_epsilon_0": SMO_EPSILON_0,
        "smo_tau": SMO_TAU,
        "smo_subproblem_max_iters": SMO_SUBPROBLEM_MAX_ITERS,
        "gcmo_eps": GCMO_EPS,
        "gcmo_max_iters": GCMO_MAX_ITERS,
        "gcmo_lagrange_bound": GCMO_LAGRANGE_BOUND,
        "gcmo_lip_override": GCMO_LIP_OVERRIDE,
    }
    lip_label = f"lip{GCMO_LIP_OVERRIDE:g}" if GCMO_LIP_OVERRIDE is not None else "lipauto"
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        group=f"{SOLVER_METHOD}-n{n_val}-m{m_val}-l{l_val}",
        name=f"{SOLVER_METHOD}-n{n_val}-m{m_val}-l{l_val}-idx{instance_idx}-seed{SEED}-{lip_label}",
        tags=list(WANDB_TAGS),
        config=config,
        reinit=True,
    )
    _configure_run_metrics(run)
    return run


def log_history(run, metrics):
    if run is None:
        return
    run.log(_compact_metrics(metrics))


def log_stage(run, metrics, step):
    if run is None:
        return
    del step
    run.log(_compact_metrics(metrics))


def build_ncwc_progress_logger(run, ncwc_state):
    def callback(payload):
        is_new_call = (
            payload["completed_outer_iters"] == 0
            and payload["call_cumulative_scsc_inner_iters"] == 0
        )
        if is_new_call:
            ncwc_state["ncwc_call_index"] += 1
            ncwc_state["current_call_index"] = ncwc_state["ncwc_call_index"]
            ncwc_state["current_call_base_scsc_inner_iters"] = ncwc_state["base_scsc_inner_iters"]

        call_index = ncwc_state["current_call_index"]
        base_scsc_inner_iters = ncwc_state["current_call_base_scsc_inner_iters"]
        cumulative_scsc_inner_iters = (
            base_scsc_inner_iters + payload["call_cumulative_scsc_inner_iters"]
        )
        metrics = payload.get("metrics") or {}
        log_history(
            run,
            {
                "ncwc/cumulative_scsc_inner_iters": cumulative_scsc_inner_iters,
                "ncwc/upper_obj": metrics.get("upper_obj", payload["objective"]),
                "ncwc/feas": metrics.get("feas"),
                "ncwc/lower_gap": metrics.get("lower_gap"),
                "ncwc/call_index": call_index,
                "ncwc/completed_outer_iters": payload["completed_outer_iters"],
                "ncwc/scsc_inner_iters": payload["scsc_num_inner_iters"],
                "ncwc/final_diff": payload["final_diff"],
                "ncwc/inner_terminated": float(payload["inner_terminated"]),
            },
        )
        ncwc_state["base_scsc_inner_iters"] = max(
            ncwc_state["base_scsc_inner_iters"],
            cumulative_scsc_inner_iters,
        )

    return callback

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


def compute_joint_constraint_lipschitz():
    """Lipschitz constant L_gtilde for the joint constraint map g_tilde(x, y) = A x + B y - b."""
    return float(torch.linalg.matrix_norm(torch.cat([A_MAT, B_MAT], dim=1), ord=2).item())


def compute_gtilde_hi():
    row_bounds = torch.sum(torch.abs(A_MAT), dim=1)
    row_bounds = row_bounds + torch.sum(torch.abs(B_MAT), dim=1) + torch.abs(B_VEC)
    return float(torch.linalg.vector_norm(row_bounds).item())


def compute_d_y():
    return 2.0 * math.sqrt(M)


def compute_gcmo_d_y(lagrange_bound):
    return math.sqrt(4.0 * M + (float(lagrange_bound) ** 2) * L)


def _instance_seed(problem_size, instance_idx):
    seed = int(SEED) % (2 ** 63 - 1)
    for value in (*problem_size, instance_idx):
        seed = (seed * 1000003 + int(value) + 0x9E3779B97F4A7C15) % (2 ** 63 - 1)
    return seed


def _make_instance_generator(problem_size, instance_idx):
    generator = torch.Generator()
    generator.manual_seed(_instance_seed(problem_size, instance_idx))
    return generator


def resolve_instance_indices():
    return [int(instance_idx) for instance_idx in NUM_INSTANCES]


def format_instance_label(instance_idx, instance_position, num_instances):
    return f"idx={instance_idx} ({instance_position}/{num_instances})"


def sample_interior_box_point(dim, generator=None):
    """Sample y_hat strictly inside the box so the lower KKT construction is simple."""
    while True:
        candidate = project_box(0.1 * torch.randn(dim, generator=generator))
        if torch.all(torch.abs(candidate) < 1.0):
            return candidate


def generate_instance(problem_size, instance_idx):
    """
    Generate one random constrained bilevel linear instance from Section 4.2.

    The construction of `B_VEC` and `D_TILDE` makes `Y_HAT` an optimal
    lower-level solution when x = 0.
    """
    global N, M, L
    global C, D, D_TILDE, A_MAT, B_MAT, B_VEC, Y_HAT
    global CURRENT_L_G

    # The seed depends only on (n, m, l, instance_idx), so the same explicit instance
    # index reproduces the same random problem across solver methods.
    # Page 12 top
    N, M, L = problem_size
    generator = _make_instance_generator(problem_size, instance_idx)
    C = torch.randn(N, generator=generator)
    D = torch.randn(M, generator=generator)
    A_MAT = 0.01 * torch.randn(L, N, generator=generator)
    B_MAT = 0.01 * torch.randn(L, M, generator=generator)
    Y_HAT = sample_interior_box_point(M, generator=generator)

    # Pick nonnegative multipliers and back out (b, d_tilde) so Y_HAT solves the lower problem at x = 0.
    lam = torch.abs(torch.randn(L, generator=generator))
    B_VEC = B_MAT @ Y_HAT
    D_TILDE = -B_MAT.T @ lam

    CURRENT_L_G = compute_joint_constraint_lipschitz()


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
                "last_subproblem_num_inner_iters": subproblem_stats["num_inner_iters"],
                "last_subproblem_final_diff": subproblem_stats["final_diff"],
                "last_subproblem_terminated": float(subproblem_stats["terminated"]),
            }
        )

    return diagnostics


def build_gcmo_diagnostics(x_vec, y_vec, z_vec, lambda_vec, z_lambda_vec, solver_result, lagrange_bound):
    lower_star = solve_lower_level_value(x_vec)
    y_metrics = _evaluate_lower_candidate(x_vec, y_vec, lower_star)
    z_metrics = _evaluate_lower_candidate(x_vec, z_vec, lower_star)
    solver_stats = solver_result["solver_stats"]
    bound_tol = 1e-10
    diagnostics = {
        "upper_obj": upper_objective(x_vec, y_vec),
        "y_feas": y_metrics["feas"],
        "y_lower_obj": y_metrics["lower_obj"],
        "y_lower_gap": y_metrics["lower_gap"],
        "z_feas": z_metrics["feas"],
        "z_lower_obj": z_metrics["lower_obj"],
        "z_lower_gap": z_metrics["lower_gap"],
        "yz_distance": _vector_distance(y_vec, z_vec),
        "lambda_norm": float(torch.linalg.vector_norm(lambda_vec).item()),
        "z_lambda_norm": float(torch.linalg.vector_norm(z_lambda_vec).item()),
        "lambda_distance": _vector_distance(lambda_vec, z_lambda_vec),
        "lambda_touched_bound": float(torch.any(lambda_vec >= lagrange_bound - bound_tol).item()),
        "z_lambda_touched_bound": float(torch.any(z_lambda_vec >= lagrange_bound - bound_tol).item()),
        "subproblem_num_outer_iters": solver_stats["num_outer_iters"],
        "subproblem_num_inner_iters": solver_stats["num_inner_iters"],
        "subproblem_final_diff": solver_stats["final_diff"],
        "subproblem_terminated": float(solver_stats["terminated"]),
    }
    return diagnostics


def run_single_instance_fop(instance_idx, problem_size, instance_position, num_instances):
    """
    Run the practical outer loop for one random instance with the FOP solver.

    Each outer stage:
    1. updates (rho_k, mu_k, epsilon_k),
    2. computes a warm start y_tilde,
    3. calls the practical FOP wrapper in `bilevel_solvers.py`,
    4. checks feasibility and the lower-level optimality gap.
    """
    instance_label = format_instance_label(instance_idx, instance_position, num_instances)
    run = init_instance_run(problem_size, instance_idx, instance_position, num_instances)
    start_time = time.perf_counter()
    run_finished = False

    x_prev = torch.zeros(N)
    y_prev = torch.zeros(M)
    initial_objective = upper_objective(x_prev, y_prev)
    log_history(run, {"instance/initial_objective": initial_objective})
    gtilde_hi = compute_gtilde_hi()
    d_y = compute_d_y()
    ncwc_state = {
        "base_scsc_inner_iters": 0,
        "ncwc_call_index": 0,
    }
    ncwc_progress_callback = build_ncwc_progress_logger(run, ncwc_state)

    prox_x = prox_box
    prox_y = prox_box

    L_grad_f1 = 0.0
    L_grad_ftilde1 = 0.0
    L_grad_gtilde = 0.0
    L_gtilde = CURRENT_L_G

    def evaluate_iterate(x_vals, y_vals):
        x_curr = x_vals[0]
        y_curr = y_vals[0]
        if torch.any(torch.abs(x_curr) > 1.0 + 1e-10):
            raise RuntimeError("Outer iterate x left the box constraints")
        if torch.any(torch.abs(y_curr) > 1.0 + 1e-10):
            raise RuntimeError("Outer iterate y left the box constraints")
        lower_star = solve_lower_level_value(x_curr)
        return {
            "upper_obj": upper_objective(x_curr, y_curr),
            "feas": feasibility_norm(x_curr, y_curr),
            "lower_gap": lower_objective(x_curr, y_curr) - lower_star,
        }

    def stage_callback(payload):
        metrics = payload["metrics"]
        log_stage(
            run,
            {
                "fop/stage/index": payload["stage_index"],
                "fop/stage/rho": payload["rho_k"],
                "fop/stage/mu": payload["mu_k"],
                "fop/stage/epsilon": payload["epsilon_k"],
                "fop/warm_start/num_iters": payload["warm_start_iters"],
                "fop/warm_start/target_num_iters": payload["warm_start_target_iters"],
                "fop/warm_start/capped": float(payload["warm_start_capped"]),
                "fop/subproblem/num_outer_iters": payload["subproblem_stats"]["num_outer_iters"],
                "fop/subproblem/num_inner_iters": payload["subproblem_stats"]["num_inner_iters"],
                "fop/subproblem/final_diff": payload["subproblem_stats"]["final_diff"],
                "fop/subproblem/terminated": float(payload["subproblem_stats"]["terminated"]),
                "fop/post/upper_obj": metrics["upper_obj"],
                "fop/post/feas": metrics["feas"],
                "fop/post/lower_gap": metrics["lower_gap"],
            },
            step=payload["stage_index"],
        )

        if VERBOSE:
            print(
                f"  Instance {instance_label}: outer={payload['stage_index']} "
                f"warm_start_iters={payload['warm_start_iters']}/{payload['warm_start_target_iters']} "
                f"warm_start_capped={int(payload['warm_start_capped'])} "
                f"solver_outer={payload['subproblem_stats']['num_outer_iters']} "
                f"feas={metrics['feas']:.3e} gap={metrics['lower_gap']:.3e} upper={metrics['upper_obj']:.3e}",
                flush=True,
            )

    try:
        x_tensor = x_prev.clone().requires_grad_(True)
        y_tensor = y_prev.clone().requires_grad_(True)
        solver_result = optimize_bilevel_contrained_fop_practical(
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
            L_gtilde=L_gtilde,
            gtilde_hi=gtilde_hi,
            base_rho=BASE_RHO,
            final_epsilon=FINAL_EPS,
            feas_tol=FEAS_TOL,
            lower_gap_tol=LOWER_GAP_TOL,
            subproblem_max_iter=ALG4_MAX_ITERS,
            max_outer_iters=MAX_OUTER_ITERS,
            objective_func=upper_smooth,
            metrics_func=evaluate_iterate,
            evaluate_iterate=evaluate_iterate,
            stage_callback=stage_callback,
            progress_callback=ncwc_progress_callback,
            verbose=VERBOSE,
            log_every=SOLVER_LOG_EVERY,
            outer_desc=f"Instance {instance_label}",
        )

        final_metrics = solver_result["final_metrics"]
        if solver_result["terminated"] and final_metrics is not None:
            elapsed = time.perf_counter() - start_time
            result = {
                "initial_objective": initial_objective,
                "final_objective": final_metrics["upper_obj"],
                "num_outer_iters": solver_result["num_outer_iters"],
                "final_feas": final_metrics["feas"],
                "final_lower_gap": final_metrics["lower_gap"],
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
            f"Instance {instance_label} did not satisfy the stopping rule after {MAX_OUTER_ITERS} outer iterations"
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


def run_single_instance_smo(instance_idx, problem_size, instance_position, num_instances):
    """Run one SMO solve for the current random instance."""
    instance_label = format_instance_label(instance_idx, instance_position, num_instances)
    run = init_instance_run(problem_size, instance_idx, instance_position, num_instances)
    start_time = time.perf_counter()
    run_finished = False
    initial_x = torch.zeros(N)
    initial_y = torch.zeros(M)
    initial_objective = upper_objective(initial_x, initial_y)
    log_history(run, {"instance/initial_objective": initial_objective})
    gtilde_hi = compute_gtilde_hi()
    d_y = compute_d_y()
    stage_diagnostics = []
    ncwc_state = {
        "base_scsc_inner_iters": 0,
        "ncwc_call_index": 0,
    }
    ncwc_progress_callback = build_ncwc_progress_logger(run, ncwc_state)

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
            "subproblem_num_inner_iters": payload["subproblem_stats"]["num_inner_iters"],
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
            "smo/subproblem/num_inner_iters": stage_diag["subproblem_num_inner_iters"],
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

    def evaluate_ncwc_iterate(x_vals, y_vals):
        x_curr = x_vals[0]
        y_curr = y_vals[0]
        lower_star = solve_lower_level_value(x_curr)
        return {
            "upper_obj": upper_objective(x_curr, y_curr),
            "feas": feasibility_norm(x_curr, y_curr),
            "lower_gap": lower_objective(x_curr, y_curr) - lower_star,
        }

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
            objective_func=upper_smooth,
            metrics_func=evaluate_ncwc_iterate,
            progress_callback=ncwc_progress_callback,
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
                f"  Instance {instance_label}: smo_outer={solver_result['num_outer_iters']} "
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
                "instance/last_subproblem_num_inner_iters": diagnostics.get("last_subproblem_num_inner_iters"),
                "instance/last_subproblem_final_diff": diagnostics.get("last_subproblem_final_diff"),
                "instance/last_subproblem_terminated": diagnostics.get("last_subproblem_terminated"),
            }
            finish_instance_run(
                run,
                failure_summary,
            )
            run_finished = True
            print(
                f"Instance {instance_label} SMO solve failed the stop check: "
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
                "instance/last_subproblem_num_inner_iters": diagnostics.get("last_subproblem_num_inner_iters"),
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


def run_single_instance_gcmo(instance_idx, problem_size, instance_position, num_instances):
    """Run one GCMO solve for the current random instance."""
    instance_label = format_instance_label(instance_idx, instance_position, num_instances)
    run = init_instance_run(problem_size, instance_idx, instance_position, num_instances)
    start_time = time.perf_counter()
    run_finished = False
    initial_x = torch.zeros(N)
    initial_y = torch.zeros(M)
    initial_objective = upper_objective(initial_x, initial_y)
    log_history(run, {"instance/initial_objective": initial_objective})
    gtilde_hi = compute_gtilde_hi()
    d_y = compute_gcmo_d_y(GCMO_LAGRANGE_BOUND)
    ncwc_state = {
        "base_scsc_inner_iters": 0,
        "ncwc_call_index": 0,
    }
    ncwc_progress_callback = build_ncwc_progress_logger(run, ncwc_state)

    def evaluate_ncwc_iterate(x_vals, y_vals):
        x_curr = x_vals[0]
        y_curr = y_vals[0]
        if torch.any(torch.abs(x_curr) > 1.0 + 1e-10):
            raise RuntimeError("GCMO iterate x left the box constraints")
        if torch.any(torch.abs(y_curr) > 1.0 + 1e-10):
            raise RuntimeError("GCMO iterate y left the box constraints")
        lower_star = solve_lower_level_value(x_curr)
        return {
            "upper_obj": upper_objective(x_curr, y_curr),
            "feas": feasibility_norm(x_curr, y_curr),
            "lower_gap": lower_objective(x_curr, y_curr) - lower_star,
        }

    try:
        x_tensor = initial_x.clone().requires_grad_(True)
        y_tensor = initial_y.clone().requires_grad_(True)
        solver_result = optimize_bilevel_constrained_minimax(
            [x_tensor],
            [y_tensor],
            upper_smooth,
            lower_smooth,
            lower_constraints,
            lagrange_bound=GCMO_LAGRANGE_BOUND,
            prox_x=prox_box,
            prox_y1=prox_box,
            D_y=d_y,
            L_grad_f1=0.0,
            L_grad_ftilde1=0.0,
            L_grad_gtilde=0.0,
            L_gtilde=CURRENT_L_G,
            gtilde_hi=gtilde_hi,
            epsilon=GCMO_EPS,
            max_iter=GCMO_MAX_ITERS,
            verbose=VERBOSE,
            log_every=SOLVER_LOG_EVERY,
            objective_func=upper_smooth,
            metrics_func=evaluate_ncwc_iterate,
            progress_callback=ncwc_progress_callback,
            lip_override=GCMO_LIP_OVERRIDE,
        )

        x_final = x_tensor.detach().clone()
        y_final = y_tensor.detach().clone()
        z_final = solver_result["z_eps"][0]
        lambda_final = solver_result["lambda_eps"]
        z_lambda_final = solver_result["z_lambda_eps"]
        if torch.any(torch.abs(x_final) > 1.0 + 1e-10):
            raise RuntimeError("GCMO iterate x left the box constraints")
        if torch.any(torch.abs(y_final) > 1.0 + 1e-10):
            raise RuntimeError("GCMO iterate y left the box constraints")
        if torch.any(torch.abs(z_final) > 1.0 + 1e-10):
            raise RuntimeError("GCMO auxiliary iterate z left the box constraints")
        if torch.any(lambda_final < -1e-10) or torch.any(lambda_final > GCMO_LAGRANGE_BOUND + 1e-10):
            raise RuntimeError("GCMO iterate lambda left the dual box constraints")
        if torch.any(z_lambda_final < -1e-10) or torch.any(z_lambda_final > GCMO_LAGRANGE_BOUND + 1e-10):
            raise RuntimeError("GCMO auxiliary iterate z_lambda left the dual box constraints")

        diagnostics = build_gcmo_diagnostics(
            x_final,
            y_final,
            z_final,
            lambda_final,
            z_lambda_final,
            solver_result,
            GCMO_LAGRANGE_BOUND,
        )
        log_stage(
            run,
            {
                "gcmo/stage/index": 0,
                "gcmo/stage/rho": solver_result["rho"],
                "gcmo/stage/epsilon": GCMO_EPS,
                "gcmo/stage/epsilon_0": solver_result["epsilon_0"],
                "gcmo/stage/lip_h": solver_result["lip_h"],
                "gcmo/stage/computed_lip_h": solver_result["computed_lip_h"],
                "gcmo/subproblem/num_outer_iters": diagnostics["subproblem_num_outer_iters"],
                "gcmo/subproblem/num_inner_iters": diagnostics["subproblem_num_inner_iters"],
                "gcmo/subproblem/final_diff": diagnostics["subproblem_final_diff"],
                "gcmo/subproblem/terminated": diagnostics["subproblem_terminated"],
                "gcmo/post/upper_obj": diagnostics["upper_obj"],
                "gcmo/post/y_feas": diagnostics["y_feas"],
                "gcmo/post/y_lower_gap": diagnostics["y_lower_gap"],
                "gcmo/post/z_feas": diagnostics["z_feas"],
                "gcmo/post/z_lower_gap": diagnostics["z_lower_gap"],
                "gcmo/post/yz_distance": diagnostics["yz_distance"],
                "gcmo/post/lambda_norm": diagnostics["lambda_norm"],
                "gcmo/post/z_lambda_norm": diagnostics["z_lambda_norm"],
                "gcmo/post/lambda_distance": diagnostics["lambda_distance"],
                "gcmo/post/lambda_touched_bound": diagnostics["lambda_touched_bound"],
                "gcmo/post/z_lambda_touched_bound": diagnostics["z_lambda_touched_bound"],
            },
            step=0,
        )

        if VERBOSE:
            print(
                f"  Instance {instance_label}: gcmo_outer={diagnostics['subproblem_num_outer_iters']} "
                f"y_feas={diagnostics['y_feas']:.3e} y_gap={diagnostics['y_lower_gap']:.3e} "
                f"z_feas={diagnostics['z_feas']:.3e} z_gap={diagnostics['z_lower_gap']:.3e} "
                f"yz_dist={diagnostics['yz_distance']:.3e} "
                f"lambda_norm={diagnostics['lambda_norm']:.3e} "
                f"z_lambda_norm={diagnostics['z_lambda_norm']:.3e} "
                f"lambda_dist={diagnostics['lambda_distance']:.3e} "
                f"upper={diagnostics['upper_obj']:.3e}",
                flush=True,
            )

        elapsed = time.perf_counter() - start_time
        if diagnostics["y_feas"] > FEAS_TOL or diagnostics["y_lower_gap"] > LOWER_GAP_TOL:
            failure_summary = {
                "instance/failed": 1.0,
                "instance/initial_objective": initial_objective,
                "instance/final_objective": diagnostics["upper_obj"],
                "instance/num_outer_iters": diagnostics["subproblem_num_outer_iters"],
                "instance/final_feas": diagnostics["y_feas"],
                "instance/final_lower_gap": diagnostics["y_lower_gap"],
                "instance/elapsed_sec": elapsed,
                "instance/final_z_feas": diagnostics["z_feas"],
                "instance/final_z_lower_gap": diagnostics["z_lower_gap"],
                "instance/final_yz_distance": diagnostics["yz_distance"],
                "instance/final_lambda_norm": diagnostics["lambda_norm"],
                "instance/final_z_lambda_norm": diagnostics["z_lambda_norm"],
                "instance/final_lambda_distance": diagnostics["lambda_distance"],
                "instance/final_lambda_touched_bound": diagnostics["lambda_touched_bound"],
                "instance/final_z_lambda_touched_bound": diagnostics["z_lambda_touched_bound"],
                "instance/subproblem_num_outer_iters": diagnostics["subproblem_num_outer_iters"],
                "instance/subproblem_num_inner_iters": diagnostics["subproblem_num_inner_iters"],
                "instance/subproblem_final_diff": diagnostics["subproblem_final_diff"],
                "instance/subproblem_terminated": diagnostics["subproblem_terminated"],
            }
            finish_instance_run(run, failure_summary)
            run_finished = True
            print(
                f"Instance {instance_label} GCMO solve failed the stop check: "
                f"y_feas={diagnostics['y_feas']:.3e}, y_gap={diagnostics['y_lower_gap']:.3e}, "
                f"z_feas={diagnostics['z_feas']:.3e}, z_gap={diagnostics['z_lower_gap']:.3e}, "
                f"lambda_norm={diagnostics['lambda_norm']:.3e}, "
                f"z_lambda_norm={diagnostics['z_lambda_norm']:.3e}, "
                f"lambda_dist={diagnostics['lambda_distance']:.3e}, "
                f"subproblem_terminated={int(diagnostics['subproblem_terminated'])}, "
                f"subproblem_outer={diagnostics['subproblem_num_outer_iters']}, "
                f"subproblem_diff={diagnostics['subproblem_final_diff']}"
            )

        result = {
            "initial_objective": initial_objective,
            "final_objective": diagnostics["upper_obj"],
            "num_outer_iters": diagnostics["subproblem_num_outer_iters"],
            "final_feas": diagnostics["y_feas"],
            "final_lower_gap": diagnostics["y_lower_gap"],
            "gcmo_diagnostics": diagnostics,
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
                "instance/final_lambda_norm": diagnostics["lambda_norm"],
                "instance/final_z_lambda_norm": diagnostics["z_lambda_norm"],
                "instance/final_lambda_distance": diagnostics["lambda_distance"],
                "instance/final_lambda_touched_bound": diagnostics["lambda_touched_bound"],
                "instance/final_z_lambda_touched_bound": diagnostics["z_lambda_touched_bound"],
                "instance/subproblem_num_outer_iters": diagnostics["subproblem_num_outer_iters"],
                "instance/subproblem_num_inner_iters": diagnostics["subproblem_num_inner_iters"],
                "instance/subproblem_final_diff": diagnostics["subproblem_final_diff"],
                "instance/subproblem_terminated": diagnostics["subproblem_terminated"],
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


def run_single_instance(instance_idx, problem_size, instance_position, num_instances):
    if SOLVER_METHOD == "fop":
        return run_single_instance_fop(instance_idx, problem_size, instance_position, num_instances)
    if SOLVER_METHOD == "smo":
        return run_single_instance_smo(instance_idx, problem_size, instance_position, num_instances)
    if SOLVER_METHOD == "gcmo":
        return run_single_instance_gcmo(instance_idx, problem_size, instance_position, num_instances)
    raise ValueError(f"Unknown SOLVER_METHOD: {SOLVER_METHOD}")


def run_problem_size(problem_size):
    """Run the selected instance indices for one problem size."""
    n_val, m_val, l_val = problem_size
    instance_indices = resolve_instance_indices()
    num_instances = len(instance_indices)
    print(
        f"Starting triple ({n_val}, {m_val}, {l_val}) with instance indices {instance_indices}",
        flush=True,
    )

    for instance_position, instance_idx in enumerate(instance_indices, start=1):
        start_time = time.perf_counter()
        try:
            generate_instance(problem_size, instance_idx)
            result = run_single_instance(
                instance_idx,
                problem_size,
                instance_position,
                num_instances,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            print(
                f"Failed instance {format_instance_label(instance_idx, instance_position, num_instances)} "
                f"for ({n_val}, {m_val}, {l_val}) after {elapsed:.1f}s: {exc}",
                flush=True,
            )
            continue

        elapsed = time.perf_counter() - start_time

        print(
            f"Completed instance {format_instance_label(instance_idx, instance_position, num_instances)} "
            f"for ({n_val}, {m_val}, {l_val}) "
            f"in {result['num_outer_iters']} outer iterations [{elapsed:.1f}s]: "
            f"initial={result['initial_objective']:.2f} final={result['final_objective']:.2f} "
            f"feas={result['final_feas']:.2e} gap={result['final_lower_gap']:.2e}",
            flush=True,
        )


def run_experiment():
    """Main driver over the requested size grid."""
    torch.manual_seed(SEED)

    for problem_size in PROBLEM_SIZES:
        run_problem_size(problem_size)


if __name__ == "__main__":
    run_experiment()
