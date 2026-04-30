"""Diagnostics and metric adapters for solver callbacks."""

import torch


def vector_distance(u_vec, v_vec):
    return float(torch.linalg.vector_norm(u_vec - v_vec).item())


def candidate_lower_metrics(problem, reference_solver, c_vec, y_vec, lower_ref=None):
    if lower_ref is None:
        lower_ref = reference_solver.safe_solve(problem, c_vec, y_init=y_vec)
    lower_obj = problem.lower_objective(c_vec, y_vec)
    lower_star = lower_ref["objective"]
    lower_gap = lower_obj - lower_star if lower_ref["success"] else float("inf")
    return {
        "lower_obj": lower_obj,
        "lower_star": lower_star,
        "lower_gap": lower_gap,
        "feas": problem.feasibility_norm(c_vec, y_vec),
        "reference_success": float(lower_ref["success"]),
        "reference_status": lower_ref["status"],
        "reference_message": lower_ref["message"],
        "reference_nit": lower_ref["nit"],
    }


def build_solution_diagnostics(
    problem,
    reference_solver,
    c_vec,
    y_vec,
    z_vec=None,
    lambda_vec=None,
    z_lambda_vec=None,
    solver_stats=None,
):
    lower_ref = reference_solver.safe_solve(problem, c_vec, y_init=y_vec)
    y_metrics = candidate_lower_metrics(problem, reference_solver, c_vec, y_vec, lower_ref=lower_ref)
    val_counts = problem.classification_counts(y_vec, problem.x_val, problem.y_val)
    train_counts = problem.classification_counts(y_vec, problem.x_train, problem.y_train)
    diagnostics = {
        "upper_obj": problem.upper_objective(c_vec, y_vec),
        "validation_accuracy": val_counts["accuracy"],
        "validation_error_count": val_counts["num_errors"],
        "validation_correct_count": val_counts["num_correct"],
        "validation_num_samples": val_counts["num_samples"],
        "train_accuracy": train_counts["accuracy"],
        "train_error_count": train_counts["num_errors"],
        "train_correct_count": train_counts["num_correct"],
        "train_num_samples": train_counts["num_samples"],
        "y_feas": y_metrics["feas"],
        "y_lower_obj": y_metrics["lower_obj"],
        "y_lower_star": y_metrics["lower_star"],
        "y_lower_gap": y_metrics["lower_gap"],
        "reference_success": y_metrics["reference_success"],
        "reference_status": y_metrics["reference_status"],
        "reference_message": y_metrics["reference_message"],
        "reference_nit": y_metrics["reference_nit"],
        "c_min": float(torch.min(c_vec).item()),
        "c_max": float(torch.max(c_vec).item()),
        "c_mean": float(torch.mean(c_vec).item()),
        "w_norm": float(torch.linalg.vector_norm(problem.unpack_lower(y_vec)[0]).item()),
        "xi_min": float(torch.min(problem.unpack_lower(y_vec)[2]).item()),
        "xi_max": float(torch.max(problem.unpack_lower(y_vec)[2]).item()),
        "xi_mean": float(torch.mean(problem.unpack_lower(y_vec)[2]).item()),
    }
    if z_vec is not None:
        z_metrics = candidate_lower_metrics(problem, reference_solver, c_vec, z_vec, lower_ref=lower_ref)
        diagnostics.update(
            {
                "z_feas": z_metrics["feas"],
                "z_lower_obj": z_metrics["lower_obj"],
                "z_lower_gap": z_metrics["lower_gap"],
                "yz_distance": vector_distance(y_vec, z_vec),
            }
        )
    if lambda_vec is not None:
        diagnostics["lambda_norm"] = float(torch.linalg.vector_norm(lambda_vec).item())
    if z_lambda_vec is not None:
        diagnostics["z_lambda_norm"] = float(torch.linalg.vector_norm(z_lambda_vec).item())
        if lambda_vec is not None:
            diagnostics["lambda_distance"] = vector_distance(lambda_vec, z_lambda_vec)
    if solver_stats is not None:
        diagnostics.update(
            {
                "subproblem_num_outer_iters": solver_stats.get("num_outer_iters"),
                "subproblem_num_inner_iters": solver_stats.get("num_inner_iters"),
                "subproblem_final_diff": solver_stats.get("final_diff"),
                "subproblem_terminated": float(solver_stats.get("terminated", False)),
            }
        )
    return diagnostics


def cheap_iterate_metrics(problem, x_vals, y_vals):
    c_curr = x_vals[0]
    y_curr = y_vals[0]
    val_counts = problem.classification_counts(y_curr, problem.x_val, problem.y_val)
    train_counts = problem.classification_counts(y_curr, problem.x_train, problem.y_train)
    return {
        "upper_obj": problem.upper_objective(c_curr, y_curr),
        "feas": problem.feasibility_norm(c_curr, y_curr),
        "val_accuracy": val_counts["accuracy"],
        "validation_error_count": val_counts["num_errors"],
        "validation_num_samples": val_counts["num_samples"],
        "train_accuracy": train_counts["accuracy"],
        "train_error_count": train_counts["num_errors"],
        "train_num_samples": train_counts["num_samples"],
    }


def reference_iterate_metrics(problem, reference_solver, x_vals, y_vals):
    c_curr = x_vals[0]
    y_curr = y_vals[0]
    metrics = cheap_iterate_metrics(problem, x_vals, y_vals)
    lower_ref = reference_solver.safe_solve(problem, c_curr, y_init=y_curr)
    lower_obj = problem.lower_objective(c_curr, y_curr)
    metrics.update(
        {
            "lower_obj": lower_obj,
            "lower_star": lower_ref["objective"],
            "lower_gap": lower_obj - lower_ref["objective"] if lower_ref["success"] else float("inf"),
            "reference_success": float(lower_ref["success"]),
        }
    )
    return metrics
