from minimax import optimize_NCWC
from utils import clone_vals, assign_vals


def _expand_prox_spec(prox_func, count):
    return [prox_func] * count


def _scale_prox_spec(prox_func, scale):

    def scaled(v, coeff):
        return prox_func(v, scale * coeff)

    return scaled


def _positive_part(v):
    return torch.clamp(v, min=0.0)


def _positive_part_norm_sq(v):
    return torch.sum(torch.square(_positive_part(v)))


def optimize_bilevel_constrained_fop(
    params_x,
    params_y,
    upper_smooth,
    lower_smooth,
    lower_constraints,
    prox_x,
    prox_y,
    D_y,
    L_grad_f1,
    L_grad_ftilde1,
    L_grad_gtilde,
    L_gtilde,
    gtilde_hi,
    epsilon,
    lr=1,
    max_iter=1000,
    verbose=False,
    log_every=1,
):
    """
    Algorithm 4 for the constrained penalty reformulation.

    The minimization block is the concatenated tuple (x, y), and z is the
    internal maximization block initialized from y.

    h_alg4 computes the smooth part.
    """
    if epsilon <= 0 or epsilon > 0.25:
        raise ValueError(f"Algorithm 4 requires epsilon in (0, 1/4], got {epsilon}")
    if D_y <= 0:
        raise ValueError(f"Invalid D_y: {D_y}")
    if log_every <= 0:
        raise ValueError(f"Invalid log_every: {log_every}")

    params_x = list(params_x)
    params_y = list(params_y)
    if not params_x:
        raise ValueError("params_x must contain at least one tensor")
    if not params_y:
        raise ValueError("params_y must contain at least one tensor")

    rho = epsilon ** -1
    mu = epsilon ** -2
    epsilon_0 = epsilon ** (5 / 2)

    z_params = clone_vals(params_y, requires_grad=True)

    def h_alg4():
        upper_term = upper_smooth(params_x, params_y)
        lower_y = lower_smooth(params_x, params_y)
        lower_z = lower_smooth(params_x, z_params)
        constraint_y = lower_constraints(params_x, params_y)
        constraint_z = lower_constraints(params_x, z_params)
        return (
            upper_term
            + rho * lower_y
            + rho * mu * _positive_part_norm_sq(constraint_y)
            - rho * lower_z
            - rho * mu * _positive_part_norm_sq(constraint_z)
        )

    prox_hat = _expand_prox_spec(prox_x, len(params_x)) + _expand_prox_spec(
        _scale_prox_spec(prox_y, rho),
        len(params_y),
    )
    prox_z = _expand_prox_spec(_scale_prox_spec(prox_y, rho), len(z_params))
    params_hat = params_x + params_y

    lip_h = (
        L_grad_f1
        + 2 * rho * L_grad_ftilde1
        + 4 * rho * mu * (gtilde_hi * L_grad_gtilde + L_gtilde ** 2)
    )

    solver_stats = optimize_NCWC(
        params_hat,
        z_params,
        h_alg4,
        lip_h,
        D_y,
        prox_hat,
        prox_z,
        epsilon,
        epsilon_0,
        lr=lr,
        max_iter=max_iter,
        verbose=verbose,
        log_every=log_every,
    )

    return {
        "z_eps": clone_vals(z_params),
        "rho": rho,
        "mu": mu,
        "epsilon_0": epsilon_0,
        "solver_stats": solver_stats,
    }


def optimize_bilevel_constrained_smo(
    params_x,
    params_y,
    upper_smooth,
    lower_smooth,
    lower_constraints,
    prox_x,
    prox_y,
    D_y,
    L_grad_f1,
    L_grad_ftilde1,
    L_grad_gtilde,
    L_gtilde,
    gtilde_hi,
    epsilon,
    tau=0.8,
    epsilon_0=1.0,
    lambda0=None,
    z0=None,
    warm_start_max_iter=None,
    subproblem_max_iter=1000,
    stage_callback=None,
    verbose=False,
    log_every=1,
):
    """
    Algorithm 1 for the constrained bilevel reformulation in the sigma = 0 case.

    The solver owns the full SMO outer loop. It warm-starts each lower subproblem
    with Algorithm A.1 and solves each minimax stage via the existing B.2-style
    NCWC wrapper.
    """
    if epsilon <= 0:
        raise ValueError(f"Invalid epsilon: {epsilon}")
    if not 0 < tau < 1:
        raise ValueError(f"Invalid tau: {tau}")
    if epsilon_0 <= 0:
        raise ValueError(f"Invalid epsilon0: {epsilon_0}")
    if D_y <= 0:
        raise ValueError(f"Invalid D_y: {D_y}")
    if subproblem_max_iter <= 0:
        raise ValueError(f"Invalid subproblem_max_iter: {subproblem_max_iter}")
    if warm_start_max_iter is not None and warm_start_max_iter < 0:
        raise ValueError(f"Invalid warm_start_max_iter: {warm_start_max_iter}")
    if log_every <= 0:
        raise ValueError(f"Invalid log_every: {log_every}")
    if stage_callback is not None and not callable(stage_callback):
        raise TypeError("stage_callback must be callable")

    params_x = list(params_x)
    params_y = list(params_y)
    if not params_x:
        raise ValueError("params_x must contain at least one tensor")
    if not params_y:
        raise ValueError("params_y must contain at least one tensor")

    prox_x_funcs = list(prox_x) if isinstance(prox_x, (list, tuple)) else _expand_prox_spec(prox_x, len(params_x))
    prox_y_funcs = list(prox_y) if isinstance(prox_y, (list, tuple)) else _expand_prox_spec(prox_y, len(params_y))
    if len(prox_x_funcs) != len(params_x):
        raise ValueError(f"Expected {len(params_x)} proximal operators for x, got {len(prox_x_funcs)}")
    if len(prox_y_funcs) != len(params_y):
        raise ValueError(f"Expected {len(params_y)} proximal operators for y, got {len(prox_y_funcs)}")

    if z0 is None:
        z_state = clone_vals(params_y)
    else:
        if torch.is_tensor(z0):
            z0 = [z0]
        else:
            z0 = list(z0)
        if len(z0) != len(params_y):
            raise ValueError(f"z0 must contain exactly {len(params_y)} tensors")
        z_state = clone_vals(z0)

    with torch.no_grad():
        constraint_template = lower_constraints(params_x, z_state)
    if lambda0 is None:
        lambda_k = torch.zeros_like(constraint_template)
    else:
        if isinstance(lambda0, (list, tuple)):
            if len(lambda0) != 1:
                raise ValueError("lambda0 must be a tensor or a single-tensor sequence")
            lambda0 = lambda0[0]
        if not torch.is_tensor(lambda0):
            raise TypeError("lambda0 must be a torch.Tensor")
        if lambda0.shape != constraint_template.shape:
            raise ValueError(
                f"lambda0 shape {tuple(lambda0.shape)} does not match constraints {tuple(constraint_template.shape)}"
            )
        lambda_k = lambda0.clone().detach()

    history = []
    num_outer_iters = 0
    epsilon_k = epsilon_0
    while True:
        rho_k = epsilon_k ** -1
        mu_k = epsilon_k ** -3
        lambda_norm = float(torch.linalg.vector_norm(lambda_k).item())

        def smooth_lower_aug(z_vars):
            constraint_z = lower_constraints(params_x, z_vars)
            penalty = _positive_part_norm_sq(lambda_k + mu_k * constraint_z) / (2.0 * rho_k * mu_k)
            return lower_smooth(params_x, z_vars) + penalty

        L_hat_k = (
            L_grad_ftilde1
            + (mu_k * (L_gtilde ** 2 + gtilde_hi * L_grad_gtilde) + lambda_norm * L_grad_gtilde) / rho_k
        )

        y_init, warm_start_stats = _solve_alg_a1_convex(
            clone_vals(params_y),
            smooth_lower_aug,
            prox_y_funcs,
            L_hat_k,
            epsilon_k,
            D_y,
            max_iter=warm_start_max_iter,
        )
        assign_vals(params_y, y_init)

        z_params = clone_vals(z_state, requires_grad=True)

        def h_smo():
            # Eq (8)
            upper_term = upper_smooth(params_x, params_y)
            lower_y = lower_smooth(params_x, params_y)
            lower_z = lower_smooth(params_x, z_params)
            constraint_y = lower_constraints(params_x, params_y)
            constraint_z = lower_constraints(params_x, z_params)
            penalty_y = _positive_part_norm_sq(lambda_k + mu_k * constraint_y) / (2.0 * mu_k)
            penalty_z = _positive_part_norm_sq(lambda_k + mu_k * constraint_z) / (2.0 * mu_k)
            return upper_term + rho_k * lower_y + penalty_y - rho_k * lower_z - penalty_z

        prox_hat = prox_x_funcs + [_scale_prox_spec(p, rho_k) for p in prox_y_funcs]
        prox_z = [_scale_prox_spec(p, rho_k) for p in prox_y_funcs]
        # Eq (11)
        L_k = (
            L_grad_f1
            + 2.0 * rho_k * L_grad_ftilde1
            + 2.0 * mu_k * (L_gtilde ** 2 + gtilde_hi * L_grad_gtilde)
            + 2.0 * lambda_norm * L_grad_gtilde
        )
        subproblem_epsilon_0 = epsilon_k / (2.0 * math.sqrt(mu_k))

        subproblem_stats = optimize_NCWC(
            params_x + params_y,
            z_params,
            h_smo,
            L_k,
            D_y,
            prox_hat,
            prox_z,
            epsilon_k,
            subproblem_epsilon_0,
            max_iter=subproblem_max_iter,
            verbose=verbose,
            log_every=log_every,
        )

        z_state = clone_vals(z_params)
        with torch.no_grad():
            lambda_next = _positive_part(lambda_k + mu_k * lower_constraints(params_x, z_state)).detach()
        next_lambda_norm = float(torch.linalg.vector_norm(lambda_next).item())

        stage_summary = {
            "stage_index": num_outer_iters,
            "epsilon_k": epsilon_k,
            "rho_k": rho_k,
            "mu_k": mu_k,
            "lambda_norm": lambda_norm,
            "warm_start_iters": warm_start_stats["num_iters"],
            "warm_start_target_iters": warm_start_stats["target_num_iters"],
            "warm_start_capped": warm_start_stats["capped"],
            "subproblem_stats": subproblem_stats,
            "next_lambda_norm": next_lambda_norm,
        }
        history.append(stage_summary)
        if stage_callback is not None:
            stage_callback(
                {
                    **stage_summary,
                    "x": clone_vals(params_x),
                    "y": clone_vals(params_y),
                    "z": clone_vals(z_state),
                    "lambda": lambda_k.clone().detach(),
                    "next_lambda": lambda_next.clone().detach(),
                }
            )

        num_outer_iters += 1
        lambda_k = lambda_next
        if epsilon_k <= epsilon:
            break
        epsilon_k *= tau

    return {
        "z_eps": clone_vals(z_state),
        "lambda": lambda_k.clone().detach(),
        "num_outer_iters": num_outer_iters,
        "terminated": True,
        "history": history,
    }


def optimize_bilevel_constrained_minimax(
        params_x,
        params_y1,
        upper_smooth,
        lower_smooth,
        lower_constraints,
        lagrange_bound,  # TODO: bound of Lagrange multipliers from strong Slater's condition.
        prox_x,
        prox_y1,
        D_y,
        L_grad_f1,
        L_grad_ftilde1,
        L_grad_gtilde,
        L_gtilde,
        gtilde_hi,
        epsilon,
        lr=1,
        max_iter=1000,
        verbose=False,
        log_every=1,
):
    """
    Our proposed method for the constrained penalty reformulation.

    Using the Lagrangian formulation of the lower level problem, we solve only one instance of NCWC problem.
    """

    if D_y <= 0:
        raise ValueError(f"Invalid D_y: {D_y}")
    if log_every <= 0:
        raise ValueError(f"Invalid log_every: {log_every}")

    params_x = list(params_x)
    params_y1 = list(params_y1)
    if not params_x:
        raise ValueError("params_x must contain at least one tensor")
    if not params_y1:
        raise ValueError("params_y must contain at least one tensor")

    # Note we are transforming the lower level problem into unconstrained minimax, so we use settings in Algorithm 2.
    rho = epsilon ** -1
    epsilon_0 = epsilon ** (3 / 2)

    # Apply lower constraint once to get the number of constraints.
    # The copy of y.
    params_z1 = clone_vals(params_y1, requires_grad=True)
    tmp = lower_constraints(params_x, params_y1)
    num_constraints = tmp.shape[0]
    # Create and initialize the Lagrange multipliers.
    params_y2 = [torch.zeros_like(tmp).requires_grad_(True)]
    # The copy of y2.
    params_z2 = [torch.zeros_like(tmp).requires_grad_(True)]

    # prox operators for y2 and z2.
    def prox_lagrange_multipliers(v, coeff):
        del coeff
        # Lagrange multipliers are non-negative.
        v = torch.maximum(v, 0.0)
        # Box constrains from Strong Slater condition.
        v = torch.minimum(v, lagrange_bound)
        return v

    def h_lagrangian():
        upper_term = upper_smooth(params_x, params_y1)
        lower_z1_y2 = lower_smooth(params_x, params_z1) + torch.dot(params_y2[0], lower_constraints(params_x, params_z1))
        lower_y1_z2 = lower_smooth(params_x, params_y1) + torch.dot(params_z2[0], lower_constraints(params_x, params_y1))
        return (
                upper_term
                - rho * lower_z1_y2
                + rho * lower_y1_z2
        )

    # We minimize over x, y1, y2 and maximize over z1, z2.
    prox_x_y1_y2 = _expand_prox_spec(prox_x, len(params_x)) + _expand_prox_spec(
        _scale_prox_spec(prox_y1, rho),
        len(params_y1),
    ) + [_scale_prox_spec(prox_lagrange_multipliers, rho)]
    prox_z1_z2 = _expand_prox_spec(_scale_prox_spec(prox_y1, rho), len(params_z1)) + [_scale_prox_spec(prox_lagrange_multipliers, rho)]

    lip_h = (
            L_grad_f1
            + 2 * rho * L_grad_ftilde1
            # TODO: it is best to tune another parameter in place of torch.sqrt(num_constraints) * lagrange_bound * L_grad_gtilde
            + L_gtilde + torch.sqrt(num_constraints) * lagrange_bound * L_grad_gtilde
    )

    # TODO: given we do not have additional outside loop, we have to monitor the progress by extracting progress from
    # the optimize_NCWC procedure.
    # Insert metric function, f(x,y_1), ~g(x,y1), ~f(x,y1)-~f*(x)
    solver_stats = optimize_NCWC(
        params_x + params_y1 + params_y2,
        params_z1 + params_z2,
        h_lagrangian,
        lip_h,
        D_y,
        prox_x_y1_y2,
        prox_z1_z2,
        epsilon,
        epsilon_0,
        lr=lr,
        max_iter=max_iter,
        verbose=verbose,
        log_every=log_every,
    )

    # TODO: Monitor feasibility of lower level constraint, suboptimality of lower level, obj value.

    return {
        "z_eps": clone_vals(z_params),
        "rho": rho,
        "epsilon_0": epsilon_0,
        "solver_stats": solver_stats,
    }
