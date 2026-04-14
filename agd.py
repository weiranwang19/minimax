import math
import torch

from utils import clone_vals, add_vals, scale_vals, blend_vals, compute_prox, compute_value_and_grad


def _iteration_bound(D_y, L_grad_phi, epsilon):
    """
    SMO Paper Theorem A.1.
    """
    if D_y <= 0:
        raise ValueError(f"Invalid D_y: {D_y}")
    if L_grad_phi < 0:
        raise ValueError(f"Invalid L_grad_phi: {L_grad_phi}")
    if epsilon <= 0:
        raise ValueError(f"Invalid epsilon: {epsilon}")
    if L_grad_phi == 0:
        return 0
    return int(math.ceil(D_y * math.sqrt((2.0 * L_grad_phi) / epsilon)))


def agd_convex(
    init_vals,
    phi_func,
    prox_func,
    L_grad_phi,
    epsilon,
    D_y,
    max_iter=None,
    objective_func=None,
):
    """
    Algorithm A.1 of SMO for the convex composite lower warm-start subproblem.

    The implementation uses the theorem-backed fixed iteration count from
    Lemma 7 instead of the paper's explicit lower-bound stopping certificate.
    """
    if max_iter is not None and max_iter < 0:
        raise ValueError(f"Invalid max_iter: {max_iter}")

    init_vals = list(init_vals)
    if not init_vals:
        raise ValueError("init_vals must contain at least one tensor")

    target_iters = _iteration_bound(D_y, L_grad_phi, epsilon)
    num_iters = target_iters if max_iter is None else min(target_iters, max_iter)

    x_k = clone_vals(init_vals)
    z_k = clone_vals(init_vals)

    if L_grad_phi > 0:
        for k in range(num_iters):
            y_k = blend_vals(x_k, z_k, k / (k + 2), 2.0 / (k + 2))
            _, grad_y = compute_value_and_grad(y_k, phi_func)
            prox_coeff = (k + 2) / (2.0 * L_grad_phi)
            # essentially prox_arg = z_k - prox_coeff * grad_y
            prox_arg = add_vals(z_k, scale_vals(grad_y, -prox_coeff))
            z_kp1 = compute_prox(prox_arg, prox_func, prox_coeff)
            x_kp1 = blend_vals(x_k, z_kp1, k / (k + 2), 2.0 / (k + 2))
            x_k = x_kp1
            z_k = z_kp1

    final_vals = clone_vals(x_k)
    stats = {
        "num_iters": num_iters,
        "target_num_iters": target_iters,
        "capped": num_iters != target_iters,
        "terminated": num_iters == target_iters,
    }
    if objective_func is not None:
        with torch.no_grad():
            stats["final_objective"] = float(objective_func(final_vals).item())
    return final_vals, stats
