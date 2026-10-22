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


def _dot(values, coefficients):
    return sum(torch.sum(v * a) for v, a in zip(values, coefficients))


def _require_finite(values, label):
    if any(not torch.isfinite(value).all().item() for value in values):
        raise ValueError(f"Nonfinite AGD {label}")


@torch.no_grad()
def _explicit_value_and_grad(values, value_and_grad_func):
    result = value_and_grad_func(values)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("value_and_grad_func must return (scalar_tensor, gradient_list)")
    value, grads = result
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ValueError("value_and_grad_func must return a scalar tensor value")
    if not isinstance(grads, (tuple, list)) or len(grads) != len(values) or any(
        not isinstance(g, torch.Tensor) or g.shape != v.shape
        or g.dtype != v.dtype or g.device != v.device for g, v in zip(grads, values)
    ):
        raise ValueError("value_and_grad_func gradients must match the iterate tensors")
    _require_finite([value], "objective")
    _require_finite(grads, "gradient")
    return value.detach(), [g.detach() for g in grads]


def _certificate_value_and_grad(values, phi_func, value_and_grad_func=None):
    if value_and_grad_func is not None:
        return _explicit_value_and_grad(values, value_and_grad_func)
    working = clone_vals(values, requires_grad=True)
    value = phi_func(working)
    _require_finite([value], "objective")
    grads = (
        torch.autograd.grad(value, working, allow_unused=True)
        if value.requires_grad else [None] * len(working)
    )
    grads = [torch.zeros_like(v) if g is None else g.detach() for v, g in zip(working, grads)]
    _require_finite(grads, "gradient")
    return value.detach(), grads


def _linear_solution(linear_minimizer, coefficients):
    with torch.no_grad():
        solution = list(linear_minimizer(clone_vals(coefficients)))
    if len(solution) != len(coefficients) or any(
        v.shape != a.shape for v, a in zip(solution, coefficients)
    ):
        raise ValueError("linear_minimizer must return tensors matching the coefficient shapes")
    _require_finite(solution, "linear minimizer")
    return clone_vals(solution)


def _certificate_stats(values, phi_func, avg_gradient, avg_offset, linear_minimizer):
    minimizer = _linear_solution(linear_minimizer, avg_gradient)
    with torch.no_grad():
        objective = phi_func(values).detach()
        lower_bound = avg_offset + _dot(minimizer, avg_gradient)
    _require_finite([objective, lower_bound], "certificate")
    gap = float((objective - lower_bound).item())
    roundoff = 100 * torch.finfo(objective.dtype).eps * max(
        1.0, abs(objective.item()), abs(lower_bound.item()),
    )
    if gap < -roundoff:
        raise ValueError(f"Negative AGD certificate gap: {gap}; check convexity and linear_minimizer")
    return {
        "objective": float(objective.item()),
        "lower_bound": float(lower_bound.item()),
        "certificate_gap": max(0.0, gap),
    }


def _agd_certified(init_vals, phi_func, prox_func, L_grad_phi, epsilon, max_iter,
                   linear_minimizer, value_and_grad_func=None):
    """Algorithm 2 of SMO for the convex composite lower warm-start subproblem, with a certificate of optimality."""
    x_k = clone_vals(init_vals)
    z_k = clone_vals(init_vals)
    _require_finite(x_k, "initial iterate")

    # With no updates allowed, a single tangent still certifies the input.
    # For L=0 the objective is affine, so the linear oracle solves it exactly.
    if max_iter == 0 or L_grad_phi == 0:
        value, gradient = _certificate_value_and_grad(x_k, phi_func, value_and_grad_func)
        offset = value - _dot(x_k, gradient)
        num_iters = 0
        if max_iter != 0:
            x_k = _linear_solution(linear_minimizer, gradient)
            num_iters = 1
        stats = _certificate_stats(x_k, phi_func, gradient, offset, linear_minimizer)
    else:
        gradient_sum = [torch.zeros_like(v) for v in x_k]
        offset_sum = x_k[0].new_zeros(())
        weight_sum = 0.0
        num_iters = 0
        while max_iter is None or num_iters < max_iter:
            k = num_iters
            y_k = blend_vals(x_k, z_k, k / (k + 2), 2.0 / (k + 2))
            value, grad_y = _certificate_value_and_grad(y_k, phi_func, value_and_grad_func)
            weight = (k + 2) / 2.0
            gradient_sum = add_vals(gradient_sum, scale_vals(grad_y, weight))
            offset_sum = offset_sum + weight * (value - _dot(y_k, grad_y))
            # Sum_{i=0}^k (i+2)/2 = (k+1)(k+4)/4, correcting Algorithm 2, line 5.
            weight_sum += weight

            prox_coeff = (k + 2) / (2.0 * L_grad_phi)
            prox_arg = add_vals(z_k, scale_vals(grad_y, -prox_coeff))
            z_kp1 = clone_vals(compute_prox(prox_arg, prox_func, prox_coeff))
            x_k = blend_vals(x_k, z_kp1, k / (k + 2), 2.0 / (k + 2))
            z_k = z_kp1
            _require_finite(x_k + z_k, "iterate")
            num_iters += 1
            stats = _certificate_stats(
                x_k, phi_func, scale_vals(gradient_sum, 1.0 / weight_sum),
                offset_sum / weight_sum, linear_minimizer,
            )
            if stats["certificate_gap"] <= epsilon:
                break

    terminated = stats["certificate_gap"] <= epsilon
    # A failure here means the L=0 declaration or oracle is inconsistent.
    if not terminated and L_grad_phi == 0 and max_iter != 0:
        raise ValueError("Affine AGD solve failed its certificate; check L_grad_phi and linear_minimizer")
    stats.update({
        "num_iters": num_iters,
        "terminated": terminated,
        "capped": not terminated and max_iter is not None and num_iters >= max_iter,
        "epsilon": epsilon,
        "stopping_rule": "certificate",
        "stopping_reason": "certificate" if terminated else "max_iter",
    })
    return clone_vals(x_k), stats


def agd_convex(
    init_vals,
    phi_func,
    prox_func,
    L_grad_phi,
    epsilon,
    D_y,
    max_iter=None,
    objective_func=None,
    linear_minimizer=None,
    value_and_grad_func=None,
):
    """
    Algorithm A.1 (Published) or 2 (ArXiv) of SMO for the convex composite lower warm-start subproblem.

    If linear_minimizer is supplied, the nonsmooth term MUST be an indicator
    of a compact convex domain. The callback takes coefficient tensors and
    returns an exact linear minimizer over that domain. Stop using Algorithm
    2's averaged affine lower bound, normalized by the actual weight sum.
    max_iter=None means no cap; target_num_iters is diagnostic only in this mode.

    Without the oracle, retain the fixed iteration count for general composite
    problems. objective_func remains an optional reporting function.
    Optional value_and_grad_func(values) returns (scalar_tensor, gradient_list)
    for phi_func without autograd. It must describe the same smooth objective;
    phi_func is still used for certificate values and reporting.
    """
    if max_iter is not None and max_iter < 0:
        raise ValueError(f"Invalid max_iter: {max_iter}")
    if value_and_grad_func is not None and not callable(value_and_grad_func):
        raise TypeError("value_and_grad_func must be callable or None")

    init_vals = list(init_vals)
    if not init_vals:
        raise ValueError("init_vals must contain at least one tensor")

    if linear_minimizer is not None:
        if not callable(linear_minimizer):
            raise TypeError("linear_minimizer must be callable")
        if not all(math.isfinite(v) for v in (D_y, L_grad_phi, epsilon)):
            raise ValueError("AGD parameters must be finite")
        if max_iter is not None and (not isinstance(max_iter, int) or isinstance(max_iter, bool)):
            raise ValueError("max_iter must be a nonnegative integer or None")
    target_iters = _iteration_bound(D_y, L_grad_phi, epsilon)
    if linear_minimizer is not None:
        final_vals, stats = _agd_certified(
            init_vals, phi_func, prox_func, L_grad_phi, epsilon, max_iter, linear_minimizer,
            value_and_grad_func,
        )
        stats["target_num_iters"] = target_iters
        if objective_func is not None:
            with torch.no_grad():
                stats["final_objective"] = float(objective_func(final_vals).item())
        return final_vals, stats

    num_iters = target_iters if max_iter is None else min(target_iters, max_iter)

    x_k = clone_vals(init_vals)
    z_k = clone_vals(init_vals)

    if L_grad_phi > 0:
        for k in range(num_iters):
            y_k = blend_vals(x_k, z_k, k / (k + 2), 2.0 / (k + 2))
            if value_and_grad_func is None:
                _, grad_y = compute_value_and_grad(y_k, phi_func)
            else:
                _, grad_y = _explicit_value_and_grad(y_k, value_and_grad_func)
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
