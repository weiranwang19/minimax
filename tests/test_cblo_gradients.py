"""Analytic CBLO oracles and complete solver paths, with independent references."""
import contextlib
import io
import itertools
import unittest
from functools import partial
from unittest import mock

import torch

import bilevel_solvers as solvers
import exp_cblo as cblo
import minimax
from agd import agd_convex
from utils import clone_vals


def forbid_autograd():
    stack = contextlib.ExitStack()
    for target in ('torch.autograd.grad', 'torch.autograd.backward', 'torch.Tensor.backward'):
        stack.enter_context(mock.patch(target, side_effect=AssertionError('Unexpected autograd')))
    return stack


def reference(oracle, method, min_vals, max_vals, rho, mu, lam, anchors, lip, sigma):
    mins, maxs = clone_vals(min_vals, True), clone_vals(max_vals, True)
    x, y = mins[:2]
    z = maxs[0]
    sy = oracle.a @ x + oracle.b @ y - oracle.rhs
    sz = oracle.a @ x + oracle.b @ z - oracle.rhs
    value = oracle.c @ x + oracle.d @ y + rho * (oracle.d_tilde @ y - oracle.d_tilde @ z)
    if method == 'fop':
        value = value + rho * mu * (sy.relu().square().sum() - sz.relu().square().sum())
    elif method == 'smo':
        value = value + ((lam + mu*sy).relu().square().sum()
                         - (lam + mu*sz).relu().square().sum()) / (2*mu)
    else:
        value = value + rho * (maxs[1] @ sy - mins[2] @ sz)
    value = value + lip * sum((v-a).square().sum() for v, a in zip(mins, anchors[0]))
    value = value - sigma/2 * sum((v-a).square().sum() for v, a in zip(maxs, anchors[1]))
    return list(torch.autograd.grad(value, mins+maxs))


class LinearOracleTests(unittest.TestCase):
    def test_saddle_gradients_and_differences(self):
        for dtype, method, rho, boundary in itertools.product(
            (torch.float32, torch.float64), ('fop', 'smo', 'gcmo'),
            (1., 5., 25., 125.), (False, True),
        ):
            with self.subTest(dtype=dtype, method=method, rho=rho, boundary=boundary):
                rng = torch.Generator().manual_seed(22)
                rand = lambda *shape: torch.randn(*shape, dtype=dtype, generator=rng)
                oracle = cblo.LinearCBLOOracle(rand(3, 2), torch.eye(3, dtype=dtype),
                                              torch.zeros(3, dtype=dtype), rand(2), rand(3), rand(3))
                mu = rho**3 if method == 'smo' else rho**2
                lam = rand(3).abs()
                old_min, old_max = [rand(2), rand(3)], [rand(3)]
                new_min, new_max = [rand(2), rand(3)], [rand(3)]
                if boundary:
                    shift = -lam/mu if method == 'smo' else torch.zeros_like(lam)
                    step = 1e-4 if dtype == torch.float32 else 1e-8
                    old_min = [torch.zeros(2, dtype=dtype), shift.clone()]
                    old_max = [shift.clone()]
                    new_min = [old_min[0]+step, shift-step]
                    new_max = [shift+step]
                if method == 'gcmo':
                    old_min.append(rand(3)); old_max.append(rand(3))
                    new_min.append(rand(3)); new_max.append(rand(3))
                anchors = ([rand(*v.shape) for v in old_min], [rand(*v.shape) for v in old_max])
                lip, sigma = .3*rho, .2/rho
                gradient, difference = minimax._regularize_gradient_callbacks(
                    *oracle.saddle_gradients(method, rho, mu, lam), *anchors, lip, sigma,
                )
                old_ref = reference(oracle, method, old_min, old_max, rho, mu, lam, anchors, lip, sigma)
                new_ref = reference(oracle, method, new_min, new_max, rho, mu, lam, anchors, lip, sigma)
                before = clone_vals(old_min+old_max+new_min+new_max)
                with forbid_autograd():
                    gmin, gmax = gradient(new_min, new_max)
                    dmin, dmax = difference(old_min, old_max, new_min, new_max)
                for got, want, delta, old in zip(gmin+gmax, new_ref, dmin+dmax, old_ref):
                    scale = max(1., want.abs().max().item(), old.abs().max().item())
                    atol = 40*torch.finfo(dtype).eps*scale
                    torch.testing.assert_close(got, want, rtol=0, atol=atol)
                    torch.testing.assert_close(delta, want-old, rtol=0, atol=atol)
                    self.assertEqual(got.shape, want.shape)
                    self.assertIsNone(got.grad_fn)
                    self.assertFalse(got.requires_grad or delta.requires_grad)
                for a, b in zip(before, old_min+old_max+new_min+new_max):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_callbacks_snapshot_anchors_and_multiplier_and_detach_outputs(self):
        cblo.generate_instance((2, 3, 2), 1)
        oracle = cblo.LinearCBLOOracle.from_current_instance()
        x, y, z = torch.zeros(2, requires_grad=True), torch.ones(3, requires_grad=True), torch.zeros(3, requires_grad=True)
        lam = torch.ones(2, requires_grad=True)
        for method in ('fop', 'smo', 'gcmo'):
            mins, maxs = [x, y], [z]
            if method == 'gcmo':
                mins, maxs = mins+[lam], maxs+[lam]
            gradient, _ = oracle.saddle_gradients(method, 5., 25., lam)
            warm = oracle.warm_start_value_and_grad(method, [x], 5., 25., lam)
            old_grad, old_warm = gradient(mins, maxs), warm([z])
            with torch.no_grad():
                x.add_(1)
                lam.add_(2)
            # The warm-up fixes x and lambda, whereas GCMO saddle duals are variables.
            new_warm = warm([z])
            for a, b in zip([old_warm[0]]+old_warm[1], [new_warm[0]]+new_warm[1]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertFalse(b.requires_grad)
            for g in old_grad[0]+old_grad[1]:
                self.assertFalse(g.requires_grad)
            if method == 'smo':
                with torch.no_grad():
                    x.sub_(1)
                new_grad = gradient(mins, maxs)
                for a, b in zip(old_grad[0]+old_grad[1], new_grad[0]+new_grad[1]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_warm_values_gradients_and_agd_certificates(self):
        for dtype, method in itertools.product((torch.float32, torch.float64), ('fop', 'smo', 'gcmo')):
            with self.subTest(dtype=dtype, method=method):
                a = torch.tensor([[.2, -.1], [.1, .2]], dtype=dtype)
                b = torch.eye(2, dtype=dtype)
                oracle = cblo.LinearCBLOOracle(a, b, torch.zeros(2, dtype=dtype),
                                              torch.ones(2, dtype=dtype), torch.ones(2, dtype=dtype),
                                              torch.tensor([.2, -.1], dtype=dtype))
                x = torch.tensor([.2, -.3], dtype=dtype)
                lam = torch.tensor([.1, .2], dtype=dtype)
                for rho in (1., 5., 25.):
                    mu = rho**2
                    warm = oracle.warm_start_value_and_grad(method, [x], rho, mu, lam)

                    def phi(values):
                        z, = values
                        residual = a@x+b@z
                        penalty = ((lam+mu*residual).relu().square().sum()/(2*rho*mu)
                                   if method == 'smo' else mu*residual.relu().square().sum())
                        return oracle.d_tilde@z+penalty

                    probe = [torch.tensor([.1, -.4], dtype=dtype, requires_grad=True)]
                    ref_value = phi(probe)
                    ref_grad = torch.autograd.grad(ref_value, probe)[0]
                    value, grads = warm(probe)
                    torch.testing.assert_close(value, ref_value)
                    torch.testing.assert_close(grads[0], ref_grad)
                    self.assertIsNone(value.grad_fn)
                    for cap in (0, 3, None):
                        options = dict(L_grad_phi=2*mu, epsilon=.05, D_y=3., max_iter=cap,
                                       linear_minimizer=cblo.box_linear_minimizer)
                        expected, expected_stats = agd_convex(probe, phi, [cblo.prox_box], **options)
                        with forbid_autograd():
                            actual, stats = agd_convex(probe, phi, [cblo.prox_box],
                                                      value_and_grad_func=warm, **options)
                        torch.testing.assert_close(actual[0], expected[0])
                        for field in ('num_iters', 'terminated', 'capped'):
                            self.assertEqual(stats[field], expected_stats[field])
                        self.assertAlmostEqual(stats['certificate_gap'], expected_stats['certificate_gap'],
                                               delta=2e-5 if dtype == torch.float32 else 1e-10)


class SolverIntegrationTests(unittest.TestCase):
    def test_all_wrappers_run_without_autograd_and_assign_projected_outputs(self):
        for method, bounds_options in (
            ('fop', {'box_bounds': (-1., 1.)}),
            ('smo', {'box_bounds': (-1., 1.)}),
            ('smo', {}),  # Generic callers omit bounds and retain the old residual.
            ('gcmo', {}),
        ):
            with self.subTest(method=method, bounds_options=bounds_options):
                cblo.generate_instance((2, 3, 2), 1)
                oracle = cblo.LinearCBLOOracle.from_current_instance()
                x, y = torch.zeros(2), cblo.Y_HAT.clone()
                observed, warm_stats, factory_args = [], [], []
                original_run = minimax.Minimax_SCSC.run

                def run(solver):
                    self.assertIsNotNone(solver.gradient_func)
                    self.assertIsNotNone(solver.gradient_difference_func)
                    self.assertEqual(solver.box_bounds, bounds_options.get('box_bounds'))
                    self.assertFalse(any(v.requires_grad for v in solver.get_x()+solver.get_y()))
                    assignments = {}
                    assign_x, assign_y = solver.assign_x, solver.assign_y

                    def record_x(values):
                        assignments['x'] = clone_vals(values)
                        assign_x(values)

                    def record_y(values):
                        assignments['y'] = clone_vals(values)
                        assign_y(values)

                    with mock.patch.object(solver, 'assign_x', side_effect=record_x), mock.patch.object(
                        solver, 'assign_y', side_effect=record_y,
                    ):
                        stats = original_run(solver)
                    for got, want in zip(solver.get_x()+solver.get_y(), assignments['x']+assignments['y']):
                        torch.testing.assert_close(got, want)
                    observed.append((clone_vals(solver.get_x()), clone_vals(solver.get_y())))
                    return stats

                def warm(*args, **kwargs):
                    self.assertIsNotNone(kwargs['value_and_grad_func'])
                    values, stats = agd_convex(*args, **kwargs)
                    warm_stats.append(stats)
                    return values, stats

                def saddle_factory(*args):
                    factory_args.append(args)
                    return oracle.saddle_gradients(method, *args)

                kwargs = dict(
                    upper_smooth=cblo.upper_smooth, lower_smooth=cblo.lower_smooth,
                    lower_constraints=cblo.lower_constraints, prox_x=cblo.prox_box,
                    D_y=4., L_grad_f1=10., L_grad_ftilde1=0., L_grad_gtilde=0.,
                    L_gtilde=cblo.CURRENT_L_G, gtilde_hi=cblo.compute_gtilde_hi(),
                    warm_start_linear_minimizer=cblo.box_linear_minimizer,
                    saddle_gradient_factory=saddle_factory,
                    warm_start_value_and_grad_factory=partial(oracle.warm_start_value_and_grad, method),
                    **bounds_options,
                )
                with forbid_autograd(), mock.patch.object(minimax.Minimax_SCSC, 'run', run), mock.patch.object(
                    solvers, 'agd_convex', side_effect=warm,
                ), contextlib.redirect_stdout(io.StringIO()):
                    if method == 'fop':
                        result = solvers.optimize_bilevel_contrained_fop_practical(
                            [x], [y], prox_y=cblo.prox_box, base_rho=2., final_epsilon=.01,
                            max_outer_iters=2, subproblem_max_iter=2, **kwargs,
                        )
                    elif method == 'smo':
                        result = solvers.optimize_bilevel_constrained_smo(
                            [x], [y], prox_y=cblo.prox_box, epsilon=.5, epsilon_0=1., tau=.5,
                            subproblem_max_iter=2, **kwargs,
                        )
                    else:
                        result = solvers.optimize_bilevel_constrained_minimax(
                            [x], [y], prox_y1=cblo.prox_box, lagrange_bound=3., epsilon=1.,
                            max_iter=2, warm_start_lower=True, **kwargs,
                        )
                self.assertTrue(observed)
                self.assertTrue(all(s['num_iters'] > 0 and s['terminated'] for s in warm_stats))
                self.assertEqual(len(warm_stats), 1 if method == 'gcmo' else 2)
                if method != 'gcmo':
                    self.assertNotEqual(factory_args[0][0], factory_args[1][0])
                mins, maxs = observed[-1]
                torch.testing.assert_close(x, mins[0])
                torch.testing.assert_close(y, mins[1])
                for value in (x, y, maxs[0]):
                    self.assertTrue(torch.all(value.abs() <= 1))
                if method == 'gcmo':
                    torch.testing.assert_close(result['lambda_eps'], mins[2])
                    torch.testing.assert_close(result['z_lambda_eps'], maxs[1])
                    self.assertTrue(torch.all((mins[2] >= 0) & (mins[2] <= 3)))
                    self.assertTrue(torch.all((maxs[1] >= 0) & (maxs[1] <= 3)))
                if method != 'fop':
                    torch.testing.assert_close(result['z_eps'][0], maxs[0])

    def test_nonlinear_ncwc_keeps_autograd_fallback(self):
        x = torch.tensor([.4], requires_grad=True)
        y = torch.tensor([-.3], requires_grad=True)

        def h():
            return (x.square()/2 + x.pow(4)/10 + x*y/10 - y.square()/2 - y.pow(4)/10).sum()

        with mock.patch('torch.autograd.backward', wraps=torch.autograd.backward) as backward:
            stats = minimax.optimize_NCWC([x], [y], h, 3., 2., [cblo.prox_box], [cblo.prox_box],
                                          .1, .01, max_iter=2)
        self.assertGreater(backward.call_count, 0)
        self.assertGreater(stats['num_inner_iters'], 0)
        self.assertTrue(torch.isfinite(x).all() and torch.isfinite(y).all())


if __name__ == '__main__':
    unittest.main()
