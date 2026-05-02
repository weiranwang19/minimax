"""SVM hyperparameter tuning experiment from SMO Section 3.3.

This file intentionally keeps the editable experiment controls and solver
dispatch here, while implementation details live in the local ``svm`` package.
Generated raw data, processed splits, checkpoints, and summaries live under
svm/data, svm/models, and svm/results.
"""

import math
import time
from pathlib import Path

import torch

from bilevel_solvers import (
    optimize_bilevel_constrained_minimax,
    optimize_bilevel_constrained_smo,
    optimize_bilevel_contrained_fop_practical,
)
from svm.artifacts import save_artifacts
from svm.data import DEFAULT_DATASETS, prepare_dataset_split, set_seed
from svm.diagnostics import (
    build_solution_diagnostics,
    cheap_iterate_metrics as _cheap_iterate_metrics,
    reference_iterate_metrics as _reference_iterate_metrics,
)
from svm.problem import SVMProblem
from svm.reference import LowerReferenceSolver


torch.set_default_dtype(torch.float64)


REPO_ROOT = Path(__file__).resolve().parent
SVM_ROOT = REPO_ROOT / "svm"
DATA_ROOT = SVM_ROOT / "data"
MODEL_ROOT = SVM_ROOT / "models"
RESULT_ROOT = SVM_ROOT / "results"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"

# Paper/Table 3 datasets. Edit this tuple to rerun only selected datasets.
DATASETS = DEFAULT_DATASETS
# Zero-based deterministic train/validation split indices.
SPLIT_INDICES = [0]
VALIDATION_FRACTION = 0.25
SEED = 0
SAVE_PROCESSED_DATA = True

# Solver selection: "fop", "smo", or "minimax".
SOLVER_METHOD = "fop"

# FOP practical variant.
BASE_RHO = 5.0
FINAL_EPS = 1e-2
MAX_OUTER_ITERS = 1000
ALG4_MAX_ITERS = 1000
FOP_WARM_START_MAX_ITERS = 1000

# SMO.
SMO_EPS = 1e-2
SMO_EPSILON_0 = 1.0
SMO_TAU = 0.9
SMO_SUBPROBLEM_MAX_ITERS = 200
SMO_WARM_START_MAX_ITERS = 200

# Deterministic minimax method.
MINIMAX_EPS = 1e-2
MINIMAX_MAX_ITERS = 1000
MINIMAX_LAGRANGE_BOUND = 1000.0
MINIMAX_LIP_OVERRIDE = None

# Common stopping/reporting controls.
FEAS_TOL = 1e-2
LOWER_GAP_TOL = 1e-2
LOWER_REFERENCE_SOLVER = "cvxpy"  # "scipy" or "cvxpy" if cvxpy is installed.
LOWER_REFERENCE_MAXITER = 1000
LOWER_REFERENCE_FTOL = 1e-9
VERBOSE = True
SOLVER_LOG_EVERY = 1

# W&B controls.
WANDB_ENABLED = True
WANDB_PROJECT = "minimax"
WANDB_ENTITY = None
WANDB_MODE = "online"
WANDB_TAGS = ("svm",)


# Current problem globals are kept for lightweight interactive debugging and
# backwards compatibility with earlier local scripts.
PROBLEM = None
REFERENCE_SOLVER = None
CURRENT_DATASET = None
CURRENT_SPLIT_IDX = None
X_TRAIN = None
Y_TRAIN = None
X_VAL = None
Y_VAL = None
TRAIN_INDICES = None
VAL_INDICES = None
N_TRAIN = None
N_VAL = None
Q = None
Y_DIM = None
LOWER_CONSTRAINT_JAC_NP = None
CURRENT_L_G = None
CURRENT_G_HI = None
CURRENT_L_GRAD_F = None
CURRENT_L_GRAD_LOWER = None


class _NoOpRun:
    def __init__(self):
        self.summary = {}

    def log(self, *args, **kwargs):
        del args, kwargs

    def finish(self, *args, **kwargs):
        del args, kwargs


def _require_problem():
    if PROBLEM is None:
        raise RuntimeError("No SVM problem is loaded. Call load_problem(dataset_name, split_idx) first.")
    return PROBLEM


def _require_reference_solver():
    if REFERENCE_SOLVER is None:
        raise RuntimeError("No lower reference solver is configured. Call load_problem first.")
    return REFERENCE_SOLVER


def _resolve_wandb():
    if not WANDB_ENABLED:
        return None
    import wandb

    return wandb


def _compact_metrics(metrics):
    return {key: value for key, value in metrics.items() if value is not None}


def _serialize_config():
    return {
        "datasets": list(DATASETS),
        "split_indices": list(SPLIT_INDICES),
        "validation_fraction": VALIDATION_FRACTION,
        "seed": SEED,
        "solver_method": SOLVER_METHOD,
        "base_rho": BASE_RHO,
        "final_eps": FINAL_EPS,
        "max_outer_iters": MAX_OUTER_ITERS,
        "alg4_max_iters": ALG4_MAX_ITERS,
        "fop_warm_start_max_iters": FOP_WARM_START_MAX_ITERS,
        "smo_eps": SMO_EPS,
        "smo_epsilon_0": SMO_EPSILON_0,
        "smo_tau": SMO_TAU,
        "smo_subproblem_max_iters": SMO_SUBPROBLEM_MAX_ITERS,
        "smo_warm_start_max_iters": SMO_WARM_START_MAX_ITERS,
        "minimax_eps": MINIMAX_EPS,
        "minimax_max_iters": MINIMAX_MAX_ITERS,
        "minimax_lagrange_bound": MINIMAX_LAGRANGE_BOUND,
        "minimax_lip_override": MINIMAX_LIP_OVERRIDE,
        "feas_tol": FEAS_TOL,
        "lower_gap_tol": LOWER_GAP_TOL,
        "lower_reference_solver": LOWER_REFERENCE_SOLVER,
        "lower_reference_maxiter": LOWER_REFERENCE_MAXITER,
        "lower_reference_ftol": LOWER_REFERENCE_FTOL,
    }


def init_run(dataset_name, split_idx):
    problem = _require_problem()
    wandb = _resolve_wandb()
    if wandb is None:
        return _NoOpRun()

    config = _serialize_config()
    config.update(
        {
            "dataset_name": dataset_name,
            "split_idx": split_idx,
            "n_train": problem.n_train,
            "n_val": problem.n_val,
            "num_features": problem.q,
        }
    )
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        group=f"svm-{SOLVER_METHOD}-{dataset_name}",
        name=f"svm-{SOLVER_METHOD}-{dataset_name}-split{split_idx}-seed{SEED}",
        tags=list(WANDB_TAGS) + [SOLVER_METHOD, dataset_name],
        config=config,
        reinit=True,
    )
    if hasattr(run, "define_metric"):
        run.define_metric("ncwc/cumulative_inner_iters")
        for metric_name in (
            "ncwc/upper_obj",
            "ncwc/feas",
            "ncwc/val_accuracy",
            "ncwc/final_diff",
            "ncwc/completed_outer_iters",
            "ncwc/inner_terminated",
        ):
            run.define_metric(metric_name, step_metric="ncwc/cumulative_inner_iters")
        run.define_metric("svm/stage/index")
        run.define_metric("svm/stage/*", step_metric="svm/stage/index")
    return run


def log_metrics(run, metrics):
    if run is not None:
        run.log(_compact_metrics(metrics))


def finish_run(run, summary):
    if run is None:
        return
    for key, value in summary.items():
        if hasattr(run, "summary"):
            run.summary[key] = value
    run.finish()


def load_problem(dataset_name, split_idx):
    """Load one deterministic local dataset split and expose it as globals."""
    global PROBLEM, REFERENCE_SOLVER
    global CURRENT_DATASET, CURRENT_SPLIT_IDX
    global X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, TRAIN_INDICES, VAL_INDICES
    global N_TRAIN, N_VAL, Q, Y_DIM, LOWER_CONSTRAINT_JAC_NP
    global CURRENT_L_G, CURRENT_G_HI, CURRENT_L_GRAD_F, CURRENT_L_GRAD_LOWER

    split = prepare_dataset_split(
        dataset_name,
        split_idx,
        DATA_ROOT,
        PROCESSED_DATA_ROOT,
        SEED,
        VALIDATION_FRACTION,
        save_processed_data=SAVE_PROCESSED_DATA,
    )
    PROBLEM = SVMProblem(split, seed=SEED)
    REFERENCE_SOLVER = LowerReferenceSolver(
        method=LOWER_REFERENCE_SOLVER,
        maxiter=LOWER_REFERENCE_MAXITER,
        ftol=LOWER_REFERENCE_FTOL,
    )

    CURRENT_DATASET = PROBLEM.dataset_name
    CURRENT_SPLIT_IDX = PROBLEM.split_idx
    X_TRAIN = PROBLEM.x_train
    Y_TRAIN = PROBLEM.y_train
    X_VAL = PROBLEM.x_val
    Y_VAL = PROBLEM.y_val
    TRAIN_INDICES = PROBLEM.train_indices
    VAL_INDICES = PROBLEM.val_indices
    N_TRAIN = PROBLEM.n_train
    N_VAL = PROBLEM.n_val
    Q = PROBLEM.q
    Y_DIM = PROBLEM.y_dim
    LOWER_CONSTRAINT_JAC_NP = PROBLEM.lower_constraint_jac_np
    CURRENT_L_G = PROBLEM.l_g
    CURRENT_G_HI = PROBLEM.g_hi
    CURRENT_L_GRAD_F = PROBLEM.l_grad_f
    CURRENT_L_GRAD_LOWER = PROBLEM.l_grad_lower
    return PROBLEM


# Compatibility wrappers for interactive notebooks/scripts that imported the
# original monolithic exp_svm.py helpers.
def unpack_lower(y_vec):
    return _require_problem().unpack_lower(y_vec)


def pack_lower(w, b, xi):
    return _require_problem().pack_lower(w, b, xi)


def prox_c(v, coeff):
    return _require_problem().prox_c(v, coeff)


def prox_lower(v, coeff):
    return _require_problem().prox_lower(v, coeff)


def project_lower(v):
    return _require_problem().project_lower(v)


def upper_smooth(x_params, y_params):
    return _require_problem().upper_smooth(x_params, y_params)


def lower_smooth(x_params, z_params):
    return _require_problem().lower_smooth(x_params, z_params)


def lower_constraints(x_params, z_params):
    return _require_problem().lower_constraints(x_params, z_params)


def upper_objective(c_vec, y_vec):
    return _require_problem().upper_objective(c_vec, y_vec)


def lower_objective(c_vec, y_vec):
    return _require_problem().lower_objective(c_vec, y_vec)


def feasibility_norm(c_vec, y_vec):
    return _require_problem().feasibility_norm(c_vec, y_vec)


def validation_accuracy(y_vec):
    return _require_problem().validation_accuracy(y_vec)


def train_accuracy(y_vec):
    return _require_problem().train_accuracy(y_vec)


def classification_counts(y_vec, x_mat, labels):
    return _require_problem().classification_counts(y_vec, x_mat, labels)


def compute_d_y():
    return _require_problem().compute_d_y()


def compute_minimax_d_y(lagrange_bound):
    return _require_problem().compute_minimax_d_y(lagrange_bound)


def make_initial_iterates(dataset_name=None, split_idx=None):
    del dataset_name, split_idx
    return _require_problem().make_initial_iterates()


def solve_lower_reference(c_vec, y_init=None):
    return _require_reference_solver().solve(_require_problem(), c_vec, y_init=y_init)


def safe_lower_reference(c_vec, y_init=None):
    return _require_reference_solver().safe_solve(_require_problem(), c_vec, y_init=y_init)


def cheap_iterate_metrics(x_vals, y_vals):
    return _cheap_iterate_metrics(_require_problem(), x_vals, y_vals)


def reference_iterate_metrics(x_vals, y_vals):
    return _reference_iterate_metrics(_require_problem(), _require_reference_solver(), x_vals, y_vals)


def build_ncwc_progress_logger(run):
    state = {
        "call_index": 0,
        "current_call": 0,
        "base_inner_iters": 0,
        "current_call_base": 0,
    }

    def callback(payload):
        is_new_call = (
            payload["completed_outer_iters"] == 0
            and payload["call_cumulative_scsc_inner_iters"] == 0
        )
        if is_new_call:
            state["call_index"] += 1
            state["current_call"] = state["call_index"]
            state["current_call_base"] = state["base_inner_iters"]

        cumulative_inner_iters = state["current_call_base"] + payload["call_cumulative_scsc_inner_iters"]
        metrics = payload.get("metrics") or {}
        log_metrics(
            run,
            {
                "ncwc/cumulative_inner_iters": cumulative_inner_iters,
                "ncwc/upper_obj": metrics.get("upper_obj", payload.get("objective")),
                "ncwc/feas": metrics.get("feas"),
                "ncwc/val_accuracy": metrics.get("val_accuracy"),
                "ncwc/final_diff": payload.get("final_diff"),
                "ncwc/completed_outer_iters": payload.get("completed_outer_iters"),
                "ncwc/inner_terminated": float(payload.get("inner_terminated", False)),
                "ncwc/call_index": state["current_call"],
            },
        )
        state["base_inner_iters"] = max(state["base_inner_iters"], cumulative_inner_iters)

    return callback


def log_stage_diagnostics(run, stage_index, method, diagnostics, extra=None):
    metrics = {
        "svm/stage/index": stage_index,
        "svm/stage/method": {"fop": 0.0, "smo": 1.0, "minimax": 2.0}.get(method, -1.0),
        "svm/stage/upper_obj": diagnostics.get("upper_obj"),
        "svm/stage/validation_accuracy": diagnostics.get("validation_accuracy"),
        "svm/stage/validation_error_count": diagnostics.get("validation_error_count"),
        "svm/stage/validation_num_samples": diagnostics.get("validation_num_samples"),
        "svm/stage/train_accuracy": diagnostics.get("train_accuracy"),
        "svm/stage/train_error_count": diagnostics.get("train_error_count"),
        "svm/stage/train_num_samples": diagnostics.get("train_num_samples"),
        "svm/stage/y_feas": diagnostics.get("y_feas"),
        "svm/stage/y_lower_gap": diagnostics.get("y_lower_gap"),
        "svm/stage/z_feas": diagnostics.get("z_feas"),
        "svm/stage/z_lower_gap": diagnostics.get("z_lower_gap"),
        "svm/stage/yz_distance": diagnostics.get("yz_distance"),
        "svm/stage/reference_success": diagnostics.get("reference_success"),
        "svm/stage/subproblem_num_outer_iters": diagnostics.get("subproblem_num_outer_iters"),
        "svm/stage/subproblem_num_inner_iters": diagnostics.get("subproblem_num_inner_iters"),
        "svm/stage/subproblem_final_diff": diagnostics.get("subproblem_final_diff"),
        "svm/stage/subproblem_terminated": diagnostics.get("subproblem_terminated"),
    }
    if extra:
        metrics.update(extra)
    log_metrics(run, metrics)


def _print_stage(prefix, diagnostics):
    if not VERBOSE:
        return
    print(
        f"{prefix} upper={diagnostics.get('upper_obj', float('nan')):.3e} "
        f"acc={diagnostics.get('validation_accuracy', float('nan')):.3f} "
        f"feas={diagnostics.get('y_feas', float('nan')):.3e} "
        f"gap={diagnostics.get('y_lower_gap', float('nan')):.3e}",
        flush=True,
    )


def _initial_reference_objective(c0):
    problem = _require_problem()
    lower_ref = safe_lower_reference(c0)
    if lower_ref["success"] and lower_ref["solution"] is not None:
        return problem.upper_objective(c0, lower_ref["solution"]), lower_ref
    return float("nan"), lower_ref


def _summary_from_diagnostics(initial_objective, initial_iterate_objective, diagnostics, elapsed):
    passed = (
        diagnostics["y_feas"] <= FEAS_TOL
        and diagnostics["y_lower_gap"] <= LOWER_GAP_TOL
        and bool(diagnostics["reference_success"])
    )
    return {
        "instance/initial_objective": initial_objective,
        "instance/initial_iterate_objective": initial_iterate_objective,
        "instance/final_objective": diagnostics["upper_obj"],
        "instance/validation_accuracy": diagnostics["validation_accuracy"],
        "instance/validation_error_count": diagnostics["validation_error_count"],
        "instance/validation_num_samples": diagnostics["validation_num_samples"],
        "instance/train_accuracy": diagnostics["train_accuracy"],
        "instance/train_error_count": diagnostics["train_error_count"],
        "instance/train_num_samples": diagnostics["train_num_samples"],
        "instance/final_feas": diagnostics["y_feas"],
        "instance/final_lower_gap": diagnostics["y_lower_gap"],
        "instance/reference_success": diagnostics["reference_success"],
        "instance/passed_stop_check": float(passed),
        "instance/elapsed_sec": elapsed,
    }


def _finalize_instance(
    method,
    c_final,
    y_final,
    diagnostics,
    solver_result,
    elapsed,
    initial_objective,
    initial_iterate_objective,
    run,
    extra_tensors=None,
):
    problem = _require_problem()
    model_path, result_path = save_artifacts(
        MODEL_ROOT,
        RESULT_ROOT,
        _serialize_config(),
        problem,
        method,
        c_final,
        y_final,
        diagnostics,
        solver_result,
        elapsed,
        extra_tensors=extra_tensors,
    )
    summary = _summary_from_diagnostics(initial_objective, initial_iterate_objective, diagnostics, elapsed)
    summary.update({"instance/model_path": str(model_path), "instance/result_path": str(result_path)})
    finish_run(run, summary)
    return {
        "initial_objective": initial_objective,
        "initial_iterate_objective": initial_iterate_objective,
        "final_objective": diagnostics["upper_obj"],
        "validation_accuracy": diagnostics["validation_accuracy"],
        "final_feas": diagnostics["y_feas"],
        "final_lower_gap": diagnostics["y_lower_gap"],
        "num_outer_iters": diagnostics["num_outer_iters"],
        "elapsed_sec": elapsed,
        "model_path": model_path,
        "result_path": result_path,
    }


def _initial_state(run):
    problem = _require_problem()
    c0, y0 = problem.make_initial_iterates()
    initial_iterate_objective = problem.upper_objective(c0, y0)
    initial_objective, initial_ref = _initial_reference_objective(c0)
    if not math.isfinite(initial_objective):
        initial_objective = initial_iterate_objective
    log_metrics(
        run,
        {
            "instance/initial_objective": initial_objective,
            "instance/initial_iterate_objective": initial_iterate_objective,
            "instance/initial_reference_success": float(initial_ref["success"]),
        },
    )
    return c0, y0, initial_objective, initial_iterate_objective


def run_single_instance_fop(dataset_name, split_idx):
    problem = _require_problem()
    reference_solver = _require_reference_solver()
    run = init_run(dataset_name, split_idx)
    start_time = time.perf_counter()
    ncwc_progress_callback = build_ncwc_progress_logger(run)
    c0, y0, initial_objective, initial_iterate_objective = _initial_state(run)
    stage_diagnostics = []

    def stage_callback(payload):
        y_curr = payload["y"][0]
        metrics = payload["metrics"] or reference_iterate_metrics(payload["x"], payload["y"])
        diagnostics = {
            "upper_obj": metrics["upper_obj"],
            "validation_accuracy": metrics.get("val_accuracy", problem.validation_accuracy(y_curr)),
            "validation_error_count": metrics.get("validation_error_count"),
            "validation_num_samples": metrics.get("validation_num_samples"),
            "train_accuracy": metrics.get("train_accuracy", problem.train_accuracy(y_curr)),
            "train_error_count": metrics.get("train_error_count"),
            "train_num_samples": metrics.get("train_num_samples"),
            "y_feas": metrics["feas"],
            "y_lower_obj": metrics.get("lower_obj"),
            "y_lower_star": metrics.get("lower_star"),
            "y_lower_gap": metrics.get("lower_gap"),
            "reference_success": metrics.get("reference_success"),
            "subproblem_num_outer_iters": payload["subproblem_stats"]["num_outer_iters"],
            "subproblem_num_inner_iters": payload["subproblem_stats"]["num_inner_iters"],
            "subproblem_final_diff": payload["subproblem_stats"]["final_diff"],
            "subproblem_terminated": float(payload["subproblem_stats"]["terminated"]),
        }
        stage_diagnostics.append(diagnostics)
        log_stage_diagnostics(
            run,
            payload["stage_index"],
            "fop",
            diagnostics,
            extra={
                "svm/stage/rho": payload["rho_k"],
                "svm/stage/mu": payload["mu_k"],
                "svm/stage/epsilon": payload["epsilon_k"],
                "svm/stage/warm_start_iters": payload["warm_start_iters"],
                "svm/stage/warm_start_target_iters": payload["warm_start_target_iters"],
                "svm/stage/warm_start_capped": float(payload["warm_start_capped"]),
            },
        )
        _print_stage(f"    FOP stage={payload['stage_index']}", diagnostics)

    c_tensor = c0.clone().requires_grad_(True)
    y_tensor = y0.clone().requires_grad_(True)
    solver_result = optimize_bilevel_contrained_fop_practical(
        [c_tensor],
        [y_tensor],
        problem.upper_smooth,
        problem.lower_smooth,
        problem.lower_constraints,
        problem.prox_c,
        problem.prox_lower,
        D_y=problem.compute_d_y(),
        L_grad_f1=problem.l_grad_f,
        L_grad_ftilde1=problem.l_grad_lower,
        L_grad_gtilde=0.0,
        L_gtilde=problem.l_g,
        gtilde_hi=problem.g_hi,
        base_rho=BASE_RHO,
        final_epsilon=FINAL_EPS,
        feas_tol=FEAS_TOL,
        lower_gap_tol=LOWER_GAP_TOL,
        warm_start_max_iter=FOP_WARM_START_MAX_ITERS,
        subproblem_max_iter=ALG4_MAX_ITERS,
        max_outer_iters=MAX_OUTER_ITERS,
        objective_func=problem.upper_smooth,
        metrics_func=lambda x_vals, y_vals: _cheap_iterate_metrics(problem, x_vals, y_vals),
        evaluate_iterate=lambda x_vals, y_vals: _reference_iterate_metrics(
            problem,
            reference_solver,
            x_vals,
            y_vals,
        ),
        stage_callback=stage_callback,
        progress_callback=ncwc_progress_callback,
        verbose=VERBOSE,
        log_every=SOLVER_LOG_EVERY,
        outer_desc=f"SVM {dataset_name} split={split_idx} FOP" if VERBOSE else None,
    )

    c_final = c_tensor.detach().clone()
    y_final = y_tensor.detach().clone()
    last_subproblem = solver_result.get("last_subproblem") or {}
    z_final = last_subproblem["z_eps"][0] if last_subproblem.get("z_eps") else None
    diagnostics = build_solution_diagnostics(
        problem,
        reference_solver,
        c_final,
        y_final,
        z_vec=z_final,
        solver_stats=(last_subproblem.get("solver_stats") if last_subproblem else None),
    )
    diagnostics["num_outer_iters"] = solver_result["num_outer_iters"]
    diagnostics["terminated"] = float(solver_result["terminated"])
    elapsed = time.perf_counter() - start_time
    result = _finalize_instance(
        "fop",
        c_final,
        y_final,
        diagnostics,
        solver_result,
        elapsed,
        initial_objective,
        initial_iterate_objective,
        run,
        extra_tensors={"z": z_final} if z_final is not None else None,
    )
    result["stage_diagnostics"] = stage_diagnostics
    return result


def run_single_instance_smo(dataset_name, split_idx):
    problem = _require_problem()
    reference_solver = _require_reference_solver()
    run = init_run(dataset_name, split_idx)
    start_time = time.perf_counter()
    ncwc_progress_callback = build_ncwc_progress_logger(run)
    c0, y0, initial_objective, initial_iterate_objective = _initial_state(run)
    stage_diagnostics = []

    def stage_callback(payload):
        diagnostics = build_solution_diagnostics(
            problem,
            reference_solver,
            payload["x"][0],
            payload["y"][0],
            z_vec=payload["z"][0],
            lambda_vec=payload["lambda"],
            solver_stats=payload["subproblem_stats"],
        )
        stage_diagnostics.append(diagnostics)
        log_stage_diagnostics(
            run,
            payload["stage_index"],
            "smo",
            diagnostics,
            extra={
                "svm/stage/epsilon": payload["epsilon_k"],
                "svm/stage/rho": payload["rho_k"],
                "svm/stage/mu": payload["mu_k"],
                "svm/stage/lambda_norm": payload["lambda_norm"],
                "svm/stage/next_lambda_norm": payload["next_lambda_norm"],
                "svm/stage/warm_start_iters": payload["warm_start_iters"],
                "svm/stage/warm_start_target_iters": payload["warm_start_target_iters"],
                "svm/stage/warm_start_capped": float(payload["warm_start_capped"]),
            },
        )
        _print_stage(f"    SMO stage={payload['stage_index']}", diagnostics)

    c_tensor = c0.clone().requires_grad_(True)
    y_tensor = y0.clone().requires_grad_(True)
    solver_result = optimize_bilevel_constrained_smo(
        [c_tensor],
        [y_tensor],
        problem.upper_smooth,
        problem.lower_smooth,
        problem.lower_constraints,
        problem.prox_c,
        problem.prox_lower,
        D_y=problem.compute_d_y(),
        L_grad_f1=problem.l_grad_f,
        L_grad_ftilde1=problem.l_grad_lower,
        L_grad_gtilde=0.0,
        L_gtilde=problem.l_g,
        gtilde_hi=problem.g_hi,
        epsilon=SMO_EPS,
        tau=SMO_TAU,
        epsilon_0=SMO_EPSILON_0,
        warm_start_max_iter=SMO_WARM_START_MAX_ITERS,
        subproblem_max_iter=SMO_SUBPROBLEM_MAX_ITERS,
        stage_callback=stage_callback,
        verbose=VERBOSE,
        log_every=SOLVER_LOG_EVERY,
        objective_func=problem.upper_smooth,
        metrics_func=lambda x_vals, y_vals: _cheap_iterate_metrics(problem, x_vals, y_vals),
        progress_callback=ncwc_progress_callback,
    )

    c_final = c_tensor.detach().clone()
    y_final = y_tensor.detach().clone()
    z_final = solver_result["z_eps"][0]
    lambda_final = solver_result["lambda"]
    diagnostics = build_solution_diagnostics(
        problem,
        reference_solver,
        c_final,
        y_final,
        z_vec=z_final,
        lambda_vec=lambda_final,
        solver_stats=(solver_result["history"][-1]["subproblem_stats"] if solver_result["history"] else None),
    )
    diagnostics["num_outer_iters"] = solver_result["num_outer_iters"]
    diagnostics["terminated"] = float(solver_result["terminated"])
    elapsed = time.perf_counter() - start_time
    result = _finalize_instance(
        "smo",
        c_final,
        y_final,
        diagnostics,
        solver_result,
        elapsed,
        initial_objective,
        initial_iterate_objective,
        run,
        extra_tensors={"z": z_final, "lambda": lambda_final},
    )
    result["stage_diagnostics"] = stage_diagnostics
    return result


def run_single_instance_minimax(dataset_name, split_idx):
    problem = _require_problem()
    reference_solver = _require_reference_solver()
    run = init_run(dataset_name, split_idx)
    start_time = time.perf_counter()
    ncwc_progress_callback = build_ncwc_progress_logger(run)
    c0, y0, initial_objective, initial_iterate_objective = _initial_state(run)

    c_tensor = c0.clone().requires_grad_(True)
    y_tensor = y0.clone().requires_grad_(True)
    solver_result = optimize_bilevel_constrained_minimax(
        [c_tensor],
        [y_tensor],
        problem.upper_smooth,
        problem.lower_smooth,
        problem.lower_constraints,
        lagrange_bound=MINIMAX_LAGRANGE_BOUND,
        prox_x=problem.prox_c,
        prox_y1=problem.prox_lower,
        D_y=problem.compute_minimax_d_y(MINIMAX_LAGRANGE_BOUND),
        L_grad_f1=problem.l_grad_f,
        L_grad_ftilde1=problem.l_grad_lower,
        L_grad_gtilde=0.0,
        L_gtilde=problem.l_g,
        gtilde_hi=problem.g_hi,
        epsilon=MINIMAX_EPS,
        max_iter=MINIMAX_MAX_ITERS,
        verbose=VERBOSE,
        log_every=SOLVER_LOG_EVERY,
        objective_func=problem.upper_smooth,
        metrics_func=lambda x_vals, y_vals: _cheap_iterate_metrics(problem, x_vals, y_vals),
        progress_callback=ncwc_progress_callback,
        lip_override=MINIMAX_LIP_OVERRIDE,
    )

    c_final = c_tensor.detach().clone()
    y_final = y_tensor.detach().clone()
    z_final = solver_result["z_eps"][0]
    lambda_final = solver_result["lambda_eps"]
    z_lambda_final = solver_result["z_lambda_eps"]
    diagnostics = build_solution_diagnostics(
        problem,
        reference_solver,
        c_final,
        y_final,
        z_vec=z_final,
        lambda_vec=lambda_final,
        z_lambda_vec=z_lambda_final,
        solver_stats=solver_result["solver_stats"],
    )
    diagnostics["num_outer_iters"] = solver_result["solver_stats"]["num_outer_iters"]
    diagnostics["terminated"] = float(solver_result["solver_stats"]["terminated"])
    log_stage_diagnostics(
        run,
        0,
        "minimax",
        diagnostics,
        extra={
            "svm/stage/rho": solver_result["rho"],
            "svm/stage/epsilon": MINIMAX_EPS,
            "svm/stage/epsilon_0": solver_result["epsilon_0"],
            "svm/stage/lip_h": solver_result["lip_h"],
            "svm/stage/computed_lip_h": solver_result["computed_lip_h"],
            "svm/stage/lambda_norm": diagnostics.get("lambda_norm"),
            "svm/stage/z_lambda_norm": diagnostics.get("z_lambda_norm"),
            "svm/stage/lambda_distance": diagnostics.get("lambda_distance"),
        },
    )
    _print_stage("    Minimax final", diagnostics)

    elapsed = time.perf_counter() - start_time
    return _finalize_instance(
        "minimax",
        c_final,
        y_final,
        diagnostics,
        solver_result,
        elapsed,
        initial_objective,
        initial_iterate_objective,
        run,
        extra_tensors={"z": z_final, "lambda": lambda_final, "z_lambda": z_lambda_final},
    )


def run_single_instance(dataset_name, split_idx):
    if SOLVER_METHOD == "fop":
        return run_single_instance_fop(dataset_name, split_idx)
    if SOLVER_METHOD == "smo":
        return run_single_instance_smo(dataset_name, split_idx)
    if SOLVER_METHOD == "minimax":
        return run_single_instance_minimax(dataset_name, split_idx)
    raise ValueError(f"Unknown SOLVER_METHOD: {SOLVER_METHOD}")


def run_dataset(dataset_name):
    split_indices = [int(split_idx) for split_idx in SPLIT_INDICES]
    print(f"Starting SVM dataset {dataset_name} with split indices {split_indices}", flush=True)
    for split_position, split_idx in enumerate(split_indices, start=1):
        start_time = time.perf_counter()
        try:
            load_problem(dataset_name, split_idx)
            result = run_single_instance(dataset_name, split_idx)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            print(
                f"Failed dataset={dataset_name} split={split_idx} "
                f"({split_position}/{len(split_indices)}) after {elapsed:.1f}s: {exc}",
                flush=True,
            )
            continue

        elapsed = time.perf_counter() - start_time
        print(
            f"Completed dataset={dataset_name} split={split_idx} "
            f"({split_position}/{len(split_indices)}) in {result['num_outer_iters']} outer iterations "
            f"[{elapsed:.1f}s]: initial={result['initial_objective']:.3f} "
            f"final={result['final_objective']:.3f} acc={100.0 * result['validation_accuracy']:.1f}% "
            f"feas={result['final_feas']:.2e} gap={result['final_lower_gap']:.2e}",
            flush=True,
        )


def run_experiment():
    set_seed(SEED)
    for dataset_name in DATASETS:
        run_dataset(dataset_name)


if __name__ == "__main__":
    run_experiment()
