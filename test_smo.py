import math
import os

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from scipy.optimize import linprog

from minimax import _solve_alg_a1_convex, optimize_bilevel_constrained_smo


torch.set_default_dtype(torch.float64)


def prox_box(v, coeff):
    del coeff
    return torch.clamp(v, min=-1.0, max=1.0)


def test_alg_a1_convex_quadratic_box():
    target = torch.tensor([0.25, -0.4])
    init = [torch.tensor([1.0, -1.0])]
    d_y = 2.0 * math.sqrt(target.numel())
    eta = 1e-3

    def phi(vals):
        return 0.5 * torch.sum(torch.square(vals[0] - target))

    def psi(vals):
        return phi(vals)

    solution, stats = _solve_alg_a1_convex(
        init,
        phi,
        prox_box,
        L_phi=1.0,
        eta=eta,
        D_y=d_y,
        objective_func=psi,
    )

    optimum = [target.clone()]
    objective_gap = float((psi(solution) - psi(optimum)).item())
    assert objective_gap <= eta + 1e-8
    assert stats["num_iters"] == stats["target_num_iters"]


def build_small_linear_instance():
    c_vec = torch.tensor([0.2, -0.1])
    d_vec = torch.tensor([-0.15, 0.25])
    a_mat = torch.tensor([[0.02, -0.03]])
    b_mat = torch.tensor([[0.05, -0.04]])
    y_hat = torch.tensor([-0.2, 0.1])
    lam = torch.tensor([0.7])
    b_vec = b_mat @ y_hat
    d_tilde = -(b_mat.T @ lam)

    joint_mat = torch.cat([a_mat, b_mat], dim=1)
    l_gtilde = float(torch.linalg.matrix_norm(joint_mat, ord=2).item())
    gtilde_hi = float(
        torch.linalg.vector_norm(
            torch.sum(torch.abs(a_mat), dim=1)
            + torch.sum(torch.abs(b_mat), dim=1)
            + torch.abs(b_vec)
        ).item()
    )

    return {
        "C": c_vec,
        "D": d_vec,
        "A": a_mat,
        "B": b_mat,
        "B_VEC": b_vec,
        "D_TILDE": d_tilde,
        "Y_HAT": y_hat,
        "L_GTILDE": l_gtilde,
        "GTILDE_HI": gtilde_hi,
    }


def solve_lower_level_value(instance, x_vec):
    result = linprog(
        c=instance["D_TILDE"].detach().cpu().numpy(),
        A_ub=instance["B"].detach().cpu().numpy(),
        b_ub=(instance["B_VEC"] - instance["A"] @ x_vec).detach().cpu().numpy(),
        bounds=[(-1.0, 1.0)] * x_vec.numel(),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun)


def test_constrained_smo_smoke():
    instance = build_small_linear_instance()
    d_y = 2.0 * math.sqrt(instance["Y_HAT"].numel())

    def upper_smooth(x_params, y_params):
        return torch.dot(instance["C"], x_params[0]) + torch.dot(instance["D"], y_params[0])

    def lower_smooth(x_params, z_params):
        del x_params
        return torch.dot(instance["D_TILDE"], z_params[0])

    def lower_constraints(x_params, z_params):
        return instance["A"] @ x_params[0] + instance["B"] @ z_params[0] - instance["B_VEC"]

    x_tensor = torch.zeros_like(instance["C"]).requires_grad_(True)
    y_tensor = torch.zeros_like(instance["Y_HAT"]).requires_grad_(True)
    z0 = [torch.zeros_like(instance["Y_HAT"])]

    initial_feas = float(torch.linalg.vector_norm(torch.clamp(lower_constraints([x_tensor.detach()], z0), min=0.0)).item())

    result = optimize_bilevel_constrained_smo(
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
        L_gtilde=instance["L_GTILDE"],
        gtilde_hi=instance["GTILDE_HI"],
        epsilon=0.1,
        tau=0.8,
        epsilon_0=1.0,
        z0=z0,
        subproblem_max_iter=8,
        verbose=False,
        log_every=1,
    )

    x_final = x_tensor.detach().clone()
    y_final = y_tensor.detach().clone()
    z_final = result["z_eps"][0]
    lambda_final = result["lambda"]

    assert torch.all(torch.isfinite(x_final))
    assert torch.all(torch.isfinite(y_final))
    assert torch.all(torch.isfinite(z_final))
    assert torch.all(torch.isfinite(lambda_final))
    assert torch.all(torch.abs(x_final) <= 1.0 + 1e-10)
    assert torch.all(torch.abs(y_final) <= 1.0 + 1e-10)
    assert torch.all(torch.abs(z_final) <= 1.0 + 1e-10)
    assert torch.all(lambda_final >= -1e-12)
    assert result["num_outer_iters"] == len(result["history"])

    for stage in result["history"]:
        assert math.isclose(stage["rho"], stage["eta"] ** -1, rel_tol=1e-12)
        assert math.isclose(stage["pi"], stage["eta"] ** -3, rel_tol=1e-12)

    final_residual = torch.clamp(lower_constraints([x_final], [y_final]), min=0.0)
    final_feas = float(torch.linalg.vector_norm(final_residual).item())
    final_gap = lower_smooth([x_final], [y_final]).item() - solve_lower_level_value(instance, x_final)

    assert final_feas <= initial_feas + 1e-8
    assert math.isfinite(final_gap)


if __name__ == "__main__":
    test_alg_a1_convex_quadratic_box()
    test_constrained_smo_smoke()
    print("test_smo.py: PASS")
