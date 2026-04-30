"""Lower-level reference solves for SVM stopping diagnostics."""

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit


class LowerReferenceSolver:
    def __init__(self, method="scipy", maxiter=1000, ftol=1e-9):
        self.method = method
        self.maxiter = maxiter
        self.ftol = ftol

    def solve(self, problem, c_vec, y_init=None):
        if self.method == "scipy":
            return self._solve_scipy(problem, c_vec, y_init=y_init)
        if self.method == "cvxpy":
            return self._solve_cvxpy(problem, c_vec)
        raise ValueError(f"Unknown lower reference solver: {self.method}")

    def safe_solve(self, problem, c_vec, y_init=None):
        try:
            return self.solve(problem, c_vec, y_init=y_init)
        except Exception as exc:
            return {
                "success": False,
                "objective": float("nan"),
                "solution": None,
                "status": None,
                "message": str(exc),
                "nit": None,
            }

    def _reference_initial_point(self, problem, y_init=None):
        if y_init is not None:
            y0 = problem.project_lower(y_init.detach()).cpu().numpy().astype(np.float64, copy=True)
            w = y0[: problem.q]
            b = y0[problem.q]
            xi = y0[problem.q + 1 :]
            margins = problem.y_train.detach().cpu().numpy() * (
                problem.x_train.detach().cpu().numpy() @ w + b
            )
            xi = np.maximum(xi, 1.0 - margins)
            xi = np.clip(xi, 0.0, 20.0)
            candidate = np.concatenate([w, np.array([b]), xi])
            cons = self._scipy_constraints(problem, candidate)
            if np.min(cons) >= -1e-8:
                return candidate

        x0 = np.zeros(problem.y_dim, dtype=np.float64)
        x0[problem.q + 1 :] = 1.0
        return x0

    def _scipy_objective_factory(self, problem, c_np):
        x_train_np = problem.x_train.detach().cpu().numpy()
        y_train_np = problem.y_train.detach().cpu().numpy()

        def objective(v):
            w = v[: problem.q]
            b = v[problem.q]
            xi = v[problem.q + 1 :]
            scores = x_train_np @ w + b
            return float(np.sum(np.logaddexp(0.0, -y_train_np * scores)) + c_np @ xi)

        def gradient(v):
            w = v[: problem.q]
            b = v[problem.q]
            scores = x_train_np @ w + b
            coeff = -y_train_np * expit(-y_train_np * scores)
            grad_w = x_train_np.T @ coeff
            grad_b = np.array([np.sum(coeff)])
            grad_xi = c_np.copy()
            return np.concatenate([grad_w, grad_b, grad_xi])

        return objective, gradient

    def _scipy_constraints(self, problem, v):
        x_train_np = problem.x_train.detach().cpu().numpy()
        y_train_np = problem.y_train.detach().cpu().numpy()
        w = v[: problem.q]
        b = v[problem.q]
        xi = v[problem.q + 1 :]
        return y_train_np * (x_train_np @ w + b) + xi - 1.0

    def _solve_scipy(self, problem, c_vec, y_init=None):
        c_np = c_vec.detach().cpu().numpy().astype(np.float64, copy=False)
        objective, gradient = self._scipy_objective_factory(problem, c_np)
        bounds = [(-1.0, 1.0)] * (problem.q + 1) + [(0.0, 20.0)] * problem.n_train
        result = minimize(
            objective,
            self._reference_initial_point(problem, y_init),
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=(
                {
                    "type": "ineq",
                    "fun": lambda v: self._scipy_constraints(problem, v),
                    "jac": lambda _v: problem.lower_constraint_jac_np,
                },
            ),
            options={
                "maxiter": self.maxiter,
                "ftol": self.ftol,
                "disp": False,
            },
        )
        solution = torch.as_tensor(result.x, dtype=torch.get_default_dtype())
        return {
            "success": bool(result.success),
            "objective": float(result.fun),
            "solution": problem.project_lower(solution),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(getattr(result, "nit", -1)),
        }

    def _solve_cvxpy(self, problem, c_vec):
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise RuntimeError("LOWER_REFERENCE_SOLVER='cvxpy' requires cvxpy to be installed") from exc

        x_train_np = problem.x_train.detach().cpu().numpy()
        y_train_np = problem.y_train.detach().cpu().numpy()
        c_np = c_vec.detach().cpu().numpy().astype(np.float64, copy=False)
        w_var = cp.Variable(problem.q)
        b_var = cp.Variable()
        xi_var = cp.Variable(problem.n_train)
        scores = x_train_np @ w_var + b_var
        objective = cp.Minimize(cp.sum(cp.logistic(cp.multiply(-y_train_np, scores))) + c_np @ xi_var)
        constraints = [
            w_var >= -1.0,
            w_var <= 1.0,
            b_var >= -1.0,
            b_var <= 1.0,
            xi_var >= 0.0,
            xi_var <= 20.0,
            cp.multiply(y_train_np, scores) + xi_var >= 1.0,
        ]
        cvx_problem = cp.Problem(objective, constraints)
        value = cvx_problem.solve()
        success = cvx_problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        if w_var.value is None or b_var.value is None or xi_var.value is None:
            solution = torch.zeros(problem.y_dim)
        else:
            solution = problem.pack_lower(
                torch.as_tensor(w_var.value, dtype=torch.get_default_dtype()),
                torch.as_tensor(float(b_var.value), dtype=torch.get_default_dtype()),
                torch.as_tensor(xi_var.value, dtype=torch.get_default_dtype()),
            )
        return {
            "success": bool(success),
            "objective": float(value),
            "solution": problem.project_lower(solution),
            "status": cvx_problem.status,
            "message": cvx_problem.status,
            "nit": None,
        }
