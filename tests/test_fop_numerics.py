import itertools
import unittest
from decimal import Decimal, localcontext
from unittest import mock

import torch

import bilevel_solvers
import exp_cblo as experiment
import minimax
from utils import clone_vals, compute_norm


def box(value, coeff):
    return value.clamp(-1, 1)


def scalar_solver(dtype=torch.float64, **kwargs):
    x = torch.tensor([0.5], dtype=dtype, requires_grad=True)
    y = torch.tensor([0.5], dtype=dtype, requires_grad=True)
    return minimax.Minimax_SCSC(
        [x], [y], lambda: (x+y).sum(), 1., 1., 2., [box], [box],
        tau=1e-6, max_iter=2, **kwargs,
    )


class BoxResidualTests(unittest.TestCase):
    def test_quotient_against_decimal_reference(self):
        # Independent 80-digit projection and subtraction, not the clip-gradient identity.
        with localcontext() as ctx:
            ctx.prec = 80
            for dtype, bounds, maximize, step in itertools.product(
                (torch.float32, torch.float64), ((-1., 1.), (-2., 3.)),
                (False, True), (0.3, 2.9127111101169064e-16),
            ):
                lo, hi = bounds
                values = torch.tensor([lo-1, lo, lo+0.25, 0., hi-0.25, hi, hi+1], dtype=dtype)
                for slope in (-3., 0., 1., 4.):
                    gradients = torch.full_like(values, slope)
                    actual = minimax._box_projection_quotient(
                        [values], [gradients], step, bounds, maximize,
                    )[0]
                    expected = []
                    eta = Decimal.from_float(step)
                    for value in values.tolist():
                        v, g = Decimal.from_float(value), Decimal.from_float(slope)
                        trial = v + eta*g if maximize else v - eta*g
                        projected = min(max(trial, Decimal(lo)), Decimal(hi))
                        expected.append(float((projected-v if maximize else v-projected)/eta))
                    torch.testing.assert_close(
                        actual, torch.tensor(expected, dtype=dtype),
                        rtol=3e-6 if dtype == torch.float32 else 3e-15, atol=1e-7,
                    )

    def test_tiny_step_unit_gradient_residual(self):
        for dtype, eta in itertools.product(
            (torch.float32, torch.float64), (2.2755550701037816e-11, 2.9127111101169064e-16),
        ):
            solver = scalar_solver(dtype, box_bounds=(-1., 1.))
            solver.zeta_hat = eta
            x, y = clone_vals(solver.get_x()), clone_vals(solver.get_y())
            gx, gy = solver.compute_h_bar_gradient(x, y)
            xh = [box(x[0]-eta*gx[0], eta)]
            yh = [box(y[0]+eta*gy[0], eta)]
            rx, ry = solver.compute_stationarity_residual(x, y, xh, yh, gx, gy)
            self.assertEqual(rx[0].item(), 1.)
            self.assertEqual(ry[0].item(), 1.)
            if dtype == torch.float32 or eta < 1e-15:
                self.assertNotEqual(((x[0]-xh[0])/eta).item(), 1.)

    def test_full_quadratic_residual_against_decimal(self):
        # h(u,v) = u² + 0.5uv - 1.5v²; the reference computes both projected
        # gradients as well as quotients in high precision.
        for u, v, eta in itertools.product((-1.2, -1., .5, 1., 1.2),
                                          (-1.2, -.5, 1.2), (.2, 3e-16)):
            x = torch.tensor([u], requires_grad=True)
            y = torch.tensor([v], requires_grad=True)
            solver = minimax.Minimax_SCSC(
                [x], [y], lambda: (x*x+.5*x*y-1.5*y*y).sum(),
                1., 1., 4., [box], [box], tau=1e-6, box_bounds=(-1., 1.),
            )
            solver.zeta_hat = eta
            xx, yy = clone_vals([x]), clone_vals([y])
            gx, gy = solver.compute_h_bar_gradient(xx, yy)
            xh, yh = [(xx[0]-eta*gx[0]).clamp(-1, 1)], [(yy[0]+eta*gy[0]).clamp(-1, 1)]
            actual = solver.compute_stationarity_residual(xx, yy, xh, yh, gx, gy)
            with localcontext() as ctx:
                ctx.prec = 80
                du, dv, de = map(Decimal.from_float, (u, v, eta))
                half = Decimal('.5')
                gu, gv = 2*du+half*dv, half*du-3*dv
                uh, vh = min(max(du-de*gu, -1), 1), min(max(dv+de*gv, -1), 1)
                expected = ((du-uh)/de + (2*uh+half*vh)-gu,
                            (vh-dv)/de + (half*uh-3*vh)-gv)
            for block, reference in zip(actual, expected):
                torch.testing.assert_close(block[0], torch.tensor([float(reference)]), rtol=1e-14, atol=1e-14)

    def test_generic_prox_keeps_original_residual(self):
        solver = scalar_solver()
        x, y = clone_vals(solver.get_x()), clone_vals(solver.get_y())
        gx, gy = solver.compute_h_bar_gradient(x, y)
        # A generic soft-threshold prox, deliberately not the declared box.
        xh = [torch.sign(x[0]) * (x[0].abs()-0.2).clamp_min(0)]
        yh = [torch.sign(y[0]) * (y[0].abs()-0.1).clamp_min(0)]
        rx, ry = solver.compute_stationarity_residual(x, y, xh, yh, gx, gy)
        torch.testing.assert_close(rx[0], (x[0]-xh[0])/solver.zeta_hat)
        torch.testing.assert_close(ry[0], (yh[0]-y[0])/solver.zeta_hat)

    def test_invalid_bounds(self):
        for bounds in ((1, -1), (1, 1), (-1, float('inf'))):
            with self.assertRaises(ValueError):
                scalar_solver(box_bounds=bounds)


class ExplicitGradientTests(unittest.TestCase):
    def setUp(self):
        experiment.generate_instance((8, 7, 3), 1)
        self.generator = torch.Generator().manual_seed(123)

    def points(self):
        return ([torch.randn(8, generator=self.generator), torch.randn(7, generator=self.generator)],
                [torch.randn(7, generator=self.generator)])

    def reference(self, min_vals, max_vals, rho, mu, anchors, lip, sigma):
        x, y = [v.detach().clone().requires_grad_() for v in min_vals]
        z, = [v.detach().clone().requires_grad_() for v in max_vals]
        ry = experiment.lower_constraints([x], [y]).clamp_min(0)
        rz = experiment.lower_constraints([x], [z]).clamp_min(0)
        value = (experiment.C@x + experiment.D@y + rho*(experiment.D_TILDE@y-experiment.D_TILDE@z)
                 + rho*mu*(ry.square().sum()-rz.square().sum())
                 + lip*((x-anchors[0][0]).square().sum()+(y-anchors[0][1]).square().sum())
                 - sigma/2*(z-anchors[1][0]).square().sum())
        return torch.autograd.grad(value, (x, y, z))

    def test_gradients_and_differences_with_regularization(self):
        for rho, near_boundary in itertools.product((1., 5., 25., 125.), (False, True)):
            mu, lip, sigma = rho**2, .1*rho**3, 1/(rho*80)
            anchors = self.points()
            old_min, old_max = self.points()
            new_min, new_max = self.points()
            if near_boundary:
                old_min = [torch.zeros(8), experiment.Y_HAT.clone()]
                old_max = [experiment.Y_HAT.clone()]
                new_min = [old_min[0]+1e-8, old_min[1]-1e-8]
                new_max = [old_max[0]+1e-8]
            grad, diff = minimax._regularize_gradient_callbacks(
                *experiment.make_fop_gradients(rho, mu), *anchors, lip, sigma,
            )
            old_ref = self.reference(old_min, old_max, rho, mu, anchors, lip, sigma)
            new_ref = self.reference(new_min, new_max, rho, mu, anchors, lip, sigma)
            gx, gz = grad(new_min, new_max)
            dx, dz = diff(old_min, old_max, new_min, new_max)
            for actual, expected in zip(gx+gz, new_ref):
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-8)
                self.assertFalse(actual.requires_grad)
            for actual, old, new in zip(dx+dz, old_ref, new_ref):
                torch.testing.assert_close(actual, new-old, rtol=1e-9, atol=1e-8)
                self.assertFalse(actual.requires_grad)

    def test_factory_captures_instance(self):
        grad, _ = experiment.make_fop_gradients(5., 25.)
        points = self.points()
        expected = grad(*points)
        experiment.generate_instance((8, 7, 3), 2)
        actual = grad(*points)
        for a, b in zip(actual[0]+actual[1], expected[0]+expected[1]):
            torch.testing.assert_close(a, b)

    def test_explicit_callback_is_pure_and_detached(self):
        solver = scalar_solver(gradient_func=lambda x, y: ([2*x[0]], [-3*y[0]]))
        before = clone_vals(solver.get_x()+solver.get_y())
        values = [torch.tensor([.8], requires_grad=True)]
        with mock.patch.object(solver, 'assign_x', side_effect=AssertionError('parameter copy')):
            gx, gy = solver.compute_h_bar_gradient(values, values)
        self.assertFalse(gx[0].requires_grad or gy[0].requires_grad)
        for a, b in zip(before, solver.get_x()+solver.get_y()):
            torch.testing.assert_close(a, b)

    def test_callback_contract_errors(self):
        invalid = (
            lambda x, y: [x],
            lambda x, y: ([], y),
            lambda x, y: ([torch.ones(2, dtype=x[0].dtype)], y),
            lambda x, y: ([x[0].float()], y),
        )
        for callback in invalid:
            solver = scalar_solver(gradient_func=callback)
            with self.assertRaises(ValueError):
                solver.compute_h_bar_gradient(solver.get_x(), solver.get_y())

    def test_stochastic_rejects_options(self):
        for option in ({'box_bounds': (-1, 1)}, {'gradient_func': lambda x, y: (x, y)},
                       {'gradient_difference_func': lambda *args: None}):
            with self.assertRaisesRegex(ValueError, 'deterministic'):
                minimax.optimize_NCWC(
                    [torch.zeros(1)], [torch.zeros(1)], lambda: None, 1., 1.,
                    [box], [box], 1., .1, sub_routine='stochastic', **option,
                )

    def test_practical_fop_integration_and_final_parameter_assignment(self):
        original_run = minimax.Minimax_SCSC.run
        observed = []

        def record_run(solver):
            self.assertIsNotNone(solver.gradient_func)
            self.assertEqual(solver.box_bounds, (-1., 1.))
            original_assign_x, original_assign_y = solver.assign_x, solver.assign_y
            assignments = {}

            def assign_x(values):
                assignments['x'] = clone_vals(values)
                original_assign_x(values)

            def assign_y(values):
                assignments['y'] = clone_vals(values)
                original_assign_y(values)

            with mock.patch.object(solver, 'assign_x', side_effect=assign_x), mock.patch.object(
                solver, 'assign_y', side_effect=assign_y,
            ):
                result = original_run(solver)
            for actual, expected in zip(solver.get_x()+solver.get_y(), assignments['x']+assignments['y']):
                torch.testing.assert_close(actual, expected)
                self.assertTrue(torch.all(actual.abs() <= 1))
            observed.append(clone_vals(solver.get_x()))
            return result

        x = torch.zeros(8, requires_grad=True)
        y = experiment.Y_HAT.clone().requires_grad_()
        with mock.patch.object(minimax.Minimax_SCSC, 'run', record_run):
            result = bilevel_solvers.optimize_bilevel_contrained_fop_practical(
                [x], [y], experiment.upper_smooth, experiment.lower_smooth,
                experiment.lower_constraints, box, box, D_y=2*7**.5,
                # A conservative valid upper bound keeps this integration test
                # away from the poorly conditioned production penalty schedule.
                L_grad_f1=10., L_grad_ftilde1=0., L_grad_gtilde=0.,
                L_gtilde=experiment.CURRENT_L_G, gtilde_hi=experiment.compute_gtilde_hi(),
                base_rho=1., final_epsilon=.01, warm_start_max_iter=0,
                subproblem_max_iter=2, max_outer_iters=1,
                saddle_gradient_factory=experiment.make_fop_gradients, box_bounds=(-1., 1.),
            )
        self.assertTrue(observed)
        self.assertEqual(result['num_outer_iters'], 1)
        torch.testing.assert_close(x, observed[-1][0])
        torch.testing.assert_close(y, observed[-1][1])


if __name__ == '__main__':
    unittest.main()
