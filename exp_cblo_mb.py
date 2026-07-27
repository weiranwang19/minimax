"""Solve the minimum uniform constraint-slack LP for CBLO instances."""

import numpy as np
from scipy.optimize import linprog

import exp_cblo


# Keep these as explicit, editable lists of problem sizes and zero-based instance
# indices. The defaults match the current experiment selection in exp_cblo.py.
PROBLEM_SIZES = [(300,300,15)]
NUM_INSTANCES = [0, 1, 3]


def solve_instance(problem_size, instance_idx):
    """
    Generate one exp_cblo instance and solve its minimum-slack linear program.

    The optimization variables are ordered as (a, z), with scalar a followed by
    the m-dimensional vector z.
    """
    exp_cblo.generate_instance(problem_size, instance_idx)

    m_val = exp_cblo.M
    l_val = exp_cblo.L
    b_mat = exp_cblo.B_MAT.detach().cpu().numpy()
    b_vec = exp_cblo.B_VEC.detach().cpu().numpy()

    objective = np.zeros(m_val + 1, dtype=np.float64)
    objective[0] = 1.0

    # B z - b <= 1 * a is equivalent to [-1, B] [a; z] <= b.
    inequality_matrix = np.empty((l_val, m_val + 1), dtype=np.float64)
    inequality_matrix[:, 0] = -1.0
    inequality_matrix[:, 1:] = b_mat

    result = linprog(
        c=objective,
        A_ub=inequality_matrix,
        b_ub=b_vec,
        bounds=[(None, None)] + [(-1.0, 1.0)] * m_val,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"LP solve failed for problem_size={tuple(problem_size)}, "
            f"instance_idx={instance_idx}: {result.message}"
        )

    a_star = float(result.x[0])
    z_star = result.x[1:].copy()
    return a_star, z_star


def compute_theoretical_lagrange_bound(a_star):
    """
    Compute the equation (23) multiplier bound for the current instance at x=0.

    Equation (23) uses B = 2 * L_f * D_Y / G. Here L_f is the Euclidean
    norm of the linear lower-objective coefficient D_TILDE, D_Y is the
    diameter of [-1, 1]^m, and G = -a_star is the Slater margin.
    """
    slater_margin = -float(a_star)
    if slater_margin <= 0.0:
        raise RuntimeError(
            "The minimum-slack LP did not certify a positive Slater margin: "
            f"a_star={a_star}"
        )

    lower_lipschitz = float(
        np.linalg.norm(exp_cblo.D_TILDE.detach().cpu().numpy())
    )
    lower_domain_diameter = 2.0 * np.sqrt(exp_cblo.M)
    theoretical_bound = (
        2.0 * lower_lipschitz * lower_domain_diameter / slater_margin
    )
    return {
        "G": slater_margin,
        "L": lower_lipschitz,
        "D": lower_domain_diameter,
        "B": theoretical_bound,
    }


def run_experiment():
    """Solve all requested problem-size and instance-index combinations."""
    for problem_size in PROBLEM_SIZES:
        for instance_idx in NUM_INSTANCES:
            try:
                a_star, z_star = solve_instance(problem_size, instance_idx)
                bound_terms = compute_theoretical_lagrange_bound(a_star)
            except Exception as exc:
                print(
                    f"Failed problem_size={tuple(problem_size)}, "
                    f"instance_idx={instance_idx}: {exc}",
                    flush=True,
                )
                continue

            print(
                f"problem_size={tuple(problem_size)}, instance_idx={instance_idx}",
                flush=True,
            )
            print(f"a_star={a_star:.17g}", flush=True)
            print(f"slater_margin_G={bound_terms['G']:.17g}", flush=True)
            print(f"lower_objective_lipschitz_L={bound_terms['L']:.17g}", flush=True)
            print(f"lower_domain_diameter_D={bound_terms['D']:.17g}", flush=True)
            print(f"theoretical_lagrange_bound_B={bound_terms['B']:.17g}", flush=True)
            print(
                f"configured_lagrange_bound={exp_cblo.GCMO_LAGRANGE_BOUND:.17g}",
                flush=True,
            )
            print(
                "configured_over_theoretical="
                f"{exp_cblo.GCMO_LAGRANGE_BOUND / bound_terms['B']:.17g}",
                flush=True,
            )
            print(f"z_star={z_star.tolist()}", flush=True)


if __name__ == "__main__":
    run_experiment()
