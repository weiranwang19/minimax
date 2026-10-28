"""Stationarity certificates checked against analytic saddle/KKT solutions."""
import itertools
import unittest
from unittest import mock

import torch

import minimax


def box(value, step):
    return value.clamp(-1, 1)


def scalar_solver(x, y, h, **kwargs):
    return minimax.Minimax_SCSC(
        [x], [y], h, 1., 1., 2., [box], [box],
        **{"tau": 1e-10, "max_iter": 100, **kwargs},
    )


class BoxCertificateTests(unittest.TestCase):
    def test_boundary_signs_and_adjacent_interior_points(self):
        for dtype, bounds, maximize in itertools.product(
            (torch.float32, torch.float64), ((-1., 1.), (0., 200.)), (False, True),
        ):
            with self.subTest(dtype=dtype, bounds=bounds, maximize=maximize):
                lo, hi = (torch.tensor(v, dtype=dtype) for v in bounds)
                values = torch.stack((lo, lo, (lo+hi)/2, hi, hi,
                                      torch.nextafter(lo, hi), torch.nextafter(hi, lo)))
                gradients = torch.tensor([2, -3, .5, -4, 5, 6, -7], dtype=dtype)
                expected = [-2, 0, -.5, 4, 0, -6, 7] if maximize else [0, -3, .5, 0, 5, 6, -7]
                actual = minimax._box_stationarity_residual(
                    [values], [gradients], [bounds], maximize,
                )[0]
                torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype), rtol=0, atol=0)

    def test_infeasible_or_nonfinite_candidates_are_not_certified(self):
        for value, gradient in ((-1.01, 1.), (1.01, -1.), (float('nan'), 0.),
                                (float('inf'), 0.), (-1., float('inf')), (1., float('nan'))):
            with self.subTest(value=value, gradient=gradient), self.assertRaises(RuntimeError):
                minimax._box_stationarity_residual(
                    [torch.tensor([value])], [torch.tensor([gradient])], [(-1., 1.)],
                )

    def test_bounds_validation_and_common_pair_compatibility(self):
        self.assertEqual(minimax._normalize_box_bounds((-1, 1), 2, 1),
                         ([(-1, 1), (-1, 1)], [(-1, 1)]))
        for bounds in (None, (-1, 1), {'min': [(-1, 1)], 'max': [(0, 200)]}):
            x, y = torch.zeros(1), torch.zeros(1)
            solver = scalar_solver(x, y, lambda: None, box_bounds=bounds)
            self.assertEqual(solver.box_bounds, bounds)
        for bounds in (0, (-1,), (1, 1), (2, 1), (-1, float('inf')),
                       {'min': [(-1, 1)]}, {'min': [], 'max': [(-1, 1)]},
                       {'min': [(-1, 1)], 'max': [(-1, 1), (-1, 1)]},
                       {'min': [(-1, 1)], 'max': [(0, float('nan'))]}):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                minimax._normalize_box_bounds(bounds, 1, 1)

    def test_mixed_boxes_use_correct_projected_residual(self):
        # Independent projection/subtraction at a step large enough to resolve.
        x = [torch.tensor([-2.]), torch.tensor([4.])]
        y = [torch.tensor([2.]), torch.tensor([-1.])]
        bounds = {'min': [(-1, 1), (0, 3)], 'max': [(-1, 1), (0, 3)]}
        gx = [torch.tensor([2.]), torch.tensor([-2.])]
        gy = [torch.tensor([-3.]), torch.tensor([3.])]
        solver = minimax.Minimax_SCSC(
            x, y, lambda: None, 1., 1., 2., [box, box], [box, box], 1e-6,
            gradient_difference_func=lambda a, b, c, d: ([torch.zeros_like(v) for v in c],
                                                        [torch.zeros_like(v) for v in d]),
            box_bounds=bounds,
        )
        solver.zeta_hat = step = .5
        xh = [(v-step*g).clamp(*b) for v, g, b in zip(x, gx, bounds['min'])]
        yh = [(v+step*g).clamp(*b) for v, g, b in zip(y, gy, bounds['max'])]
        rx, ry = solver.compute_stationarity_residual(x, y, xh, yh, gx, gy)
        for got, expected in zip(rx+ry, [(v-w)/step for v, w in zip(x, xh)]
                                + [(w-v)/step for v, w in zip(y, yh)]):
            torch.testing.assert_close(got, expected)


class SolverCertificateTests(unittest.TestCase):
    def test_immediate_return_assigns_candidate_without_extra_gradient_calls(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                x = torch.zeros(1, dtype=torch.float64, requires_grad=not explicit)
                y = torch.zeros(1, dtype=torch.float64, requires_grad=not explicit)
                gradient = mock.Mock(side_effect=lambda u, v: ([u[0]+100], [100-v[0]]))
                solver = scalar_solver(
                    x, y, lambda: (x.square()/2+100*x-y.square()/2+100*y).sum(),
                    box_bounds=(-1, 1), gradient_func=gradient if explicit else None,
                )
                with mock.patch('torch.autograd.backward', wraps=torch.autograd.backward) as backward:
                    stats = solver.run()
                self.assertEqual(stats, dict(num_outer_iters=1, num_inner_iters=0,
                                             final_delta=0., terminated=True,
                                             termination_reason='inner_certificate'))
                self.assertEqual(gradient.call_count, 2 if explicit else 0)
                self.assertEqual(backward.call_count, 0 if explicit else 2)
                self.assertEqual(x.item(), -1.)
                self.assertEqual(y.item(), 1.)

    def test_generic_soft_threshold_prox_has_valid_subgradient_certificate(self):
        # min_x max_y .5(x-2)^2 - .5(y+3)^2 + |x| - |y| has saddle (1, -2).
        x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        y = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        prox = lambda v, step: v.sign() * (v.abs()-step).clamp_min(0)
        solver = minimax.Minimax_SCSC(
            [x], [y], lambda: ((x-2).square()/2-(y+3).square()/2).sum(),
            1., 1., 2., [prox], [prox], 1e-7, max_iter=100,
        )
        stats = solver.run()
        self.assertTrue(stats['terminated'])
        self.assertLessEqual(stats['final_delta'], solver.tau)
        # At the nonzero returned coordinates the absolute-value subgradients
        # are uniquely +1 and -1, independent of the solver's proximal normals.
        independent_delta = torch.linalg.vector_norm(torch.cat((x-1, y+2))).item()
        self.assertLessEqual(independent_delta, solver.tau)
        self.assertGreater(stats['num_inner_iters'], 0)

    def test_outer_certificate_remains_available(self):
        x = torch.tensor([.7], dtype=torch.float64)
        y = torch.tensor([-.6], dtype=torch.float64)
        solver = minimax.Minimax_SCSC(
            [x], [y], lambda: None, 1., 1., 1., [box], [box], .01,
            max_iter=10, gradient_func=lambda u, v: ([u[0]], [-v[0]]), box_bounds=(-1, 1),
        )
        stats = solver.run()
        self.assertTrue(stats['terminated'])
        self.assertEqual(stats['termination_reason'], 'outer_certificate')
        self.assertLessEqual(torch.linalg.vector_norm(torch.cat((x, y))).item(), solver.tau)

    def test_later_inner_return_counts_all_work_and_reuses_gradients(self):
        x = torch.tensor([.7], dtype=torch.float64)
        y = torch.tensor([-.6], dtype=torch.float64)
        gradient = mock.Mock(side_effect=lambda u, v: ([u[0]+.5*v[0]], [.5*u[0]-v[0]]))
        difference = lambda a, b, c, d: ([(c[0]-a[0])+.5*(d[0]-b[0])],
                                         [.5*(c[0]-a[0])-(d[0]-b[0])])
        solver = minimax.Minimax_SCSC(
            [x], [y], lambda: None, 1., 1., 1.5, [box], [box], .1,
            max_iter=10, gradient_func=gradient, gradient_difference_func=difference, box_bounds=(-1, 1),
        )
        stats = solver.run()
        self.assertEqual(stats['termination_reason'], 'inner_certificate')
        self.assertGreater(stats['num_outer_iters'], 1)
        self.assertGreater(stats['num_inner_iters'], 0)
        # Each inner update costs two gradients; completed outer iterations
        # cost four more, while the final early return costs only two.
        self.assertEqual(gradient.call_count, 2*stats['num_inner_iters']+4*stats['num_outer_iters']-2)
        independent_delta = torch.linalg.vector_norm(torch.cat((x+.5*y, .5*x-y))).item()
        self.assertAlmostEqual(independent_delta, stats['final_delta'], delta=1e-14)
        self.assertLessEqual(independent_delta, solver.tau)

    def test_no_absolute_slack_in_either_certificate(self):
        x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        y = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        solver = scalar_solver(
            x, y, lambda: ((x+2e-9).square()/2-(y-2e-9).square()/2).sum(),
            box_bounds=(-1, 1),
        )
        solver.max_iter = 1
        solver.tau = 1e-12
        stats = solver.run()
        self.assertFalse(stats['terminated'])
        self.assertEqual(stats['termination_reason'], 'max_iter')
        self.assertGreater(stats['final_delta'], solver.tau)
        self.assertLess(stats['final_delta'], 1e-8)
        self.assertGreater(stats['num_inner_iters'], 0)

    def test_ncwc_reports_immediate_certificate(self):
        # At zero the base gradient vanishes; after regularization it also
        # vanishes, allowing an immediate, independently verifiable exit.
        x = torch.zeros(1, dtype=torch.float64)
        y = torch.zeros(1, dtype=torch.float64)
        progress = []
        stats = minimax.optimize_NCWC(
            [x], [y], lambda: None, 2., 2., [box], [box], .1, .01,
            max_iter=2, box_bounds=(-1, 1),
            gradient_func=lambda u, v: ([u[0]], [-v[0]]),
            progress_callback=progress.append,
        )
        self.assertTrue(stats['terminated'])
        self.assertIsNone(progress[0]['scsc_final_delta'])
        self.assertIsNone(progress[0]['scsc_termination_reason'])
        self.assertEqual(progress[-1]['scsc_final_delta'], 0.)
        self.assertEqual(progress[-1]['scsc_termination_reason'], 'inner_certificate')

    def test_ncwc_certifies_regularized_objective(self):
        # h=2x+3y is never stationary in the box interior. With anchor x=.4
        # and lip_h=2, the regularized minimizer is x=.4-2/4=-.1.
        x = torch.tensor([.4], dtype=torch.float64)
        y = torch.zeros(1, dtype=torch.float64)
        progress = []
        minimax.optimize_NCWC(
            [x], [y], lambda: None, 2., 2., [box], [box], .1, .001,
            max_iter=1, subproblem_max_iter=200, box_bounds=(-1, 1),
            gradient_func=lambda u, v: ([torch.full_like(u[0], 2)], [torch.full_like(v[0], 3)]),
            progress_callback=progress.append,
        )
        self.assertTrue(progress[-1]['inner_terminated'])
        self.assertLessEqual(progress[-1]['scsc_final_delta'], .001)
        self.assertLessEqual(abs(2+4*(x.item()-.4)), .001)
        self.assertEqual(y.item(), 1.)


if __name__ == '__main__':
    unittest.main()
