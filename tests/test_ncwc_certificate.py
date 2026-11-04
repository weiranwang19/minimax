"""NCWC certificates against analytic residuals and controlled inner endpoints."""
import unittest
from unittest import mock

import torch

import minimax


def box(value, step):
    return value.clamp(-1, 1)


def tensor(value, requires_grad=False):
    return torch.tensor([value], dtype=torch.float64, requires_grad=requires_grad)


def certified_stats():
    return dict(terminated=True, final_delta=0., num_outer_iters=2,
                num_inner_iters=3, termination_reason='inner_certificate')


class NCWCCertificateTests(unittest.TestCase):
    def solve(self, x, y, h, **options):
        defaults = dict(epsilon=.1, epsilon_0=.001, max_iter=1,
                        subproblem_max_iter=1000, box_bounds=(-1., 1.))
        defaults.update(options)
        return minimax.optimize_NCWC([x], [y], h, 1., 2., [box], [box], **defaults)

    def test_direct_exit_before_old_step_threshold_with_both_gradient_paths(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                x, y = tensor(1., not explicit), tensor(0., not explicit)
                gradient = lambda u, v: ([torch.full_like(u[0], .075)], [-v[0]])
                progress = []
                stats = self.solve(x, y, lambda: (.075*x-.5*y.square()).sum(),
                                   gradient_func=gradient if explicit else None,
                                   progress_callback=progress.append)
                self.assertTrue(stats['terminated'])
                self.assertEqual(stats['termination_reason'], 'direct_certificate')
                self.assertEqual(stats['residual_type'], 'box_kkt')
                self.assertEqual(stats['num_outer_iters'], 1)
                self.assertGreater(stats['num_inner_iters'], 0)
                self.assertGreater(stats['final_diff'], .1/4)
                self.assertAlmostEqual(stats['final_residual_min'], .075)
                self.assertEqual(stats['final_residual_max'], 0.)
                self.assertGreater(x.item(), .9)
                self.assertEqual(y.item(), 0.)
                for key in ('final_residual_min', 'final_residual_max', 'residual_type',
                            'termination_reason'):
                    self.assertIsNone(progress[0][key])
                    self.assertEqual(progress[-1][key], stats[key])
                self.assertEqual(progress[0]['epsilon'], .1)
                self.assertEqual(progress[-1]['epsilon_k'], .001)
                self.assertTrue(progress[-1]['ncwc_terminated'])
                self.assertEqual(progress[-1]['call_cumulative_scsc_inner_iters'], stats['num_inner_iters'])

    def test_boundary_optimum_accepts_large_step(self):
        x, y = tensor(0.), tensor(0.)
        stats = self.solve(x, y, lambda: None,
                           gradient_func=lambda u, v: ([torch.full_like(u[0], -10)], [-v[0]]))
        self.assertEqual(x.item(), 1.)
        self.assertEqual(stats['final_diff'], 1.)
        self.assertEqual(stats['final_residual_min'], 0.)
        self.assertTrue(stats['terminated'])

    def test_regularized_stationarity_does_not_certify_original_objective(self):
        x, y = tensor(.4), tensor(0.)
        progress = []
        stats = self.solve(x, y, lambda: None,
                           gradient_func=lambda u, v: ([torch.full_like(u[0], .3)], [-v[0]]),
                           progress_callback=progress.append)
        self.assertTrue(progress[-1]['inner_terminated'])
        self.assertLessEqual(progress[-1]['scsc_final_delta'], .001)
        self.assertAlmostEqual(stats['final_residual_min'], .3)
        self.assertFalse(stats['terminated'])
        self.assertEqual(stats['termination_reason'], 'max_iter')

    def test_blockwise_tolerance_is_strict_and_not_a_joint_tolerance(self):
        for gx, gy, accepted in ((.1, -.1, True), (.1+1e-10, 0., False),
                                 (0., .1+1e-10, False)):
            with self.subTest(gx=gx, gy=gy):
                x, y = tensor(0.), tensor(0.)
                with mock.patch.object(minimax.Minimax_SCSC, 'run', return_value=certified_stats()):
                    stats = self.solve(x, y, lambda: None, gradient_func=lambda u, v: (
                        [torch.full_like(u[0], gx)], [torch.full_like(v[0], gy)]))
                self.assertEqual(stats['terminated'], accepted)
                self.assertEqual(stats['final_diff'], 0.)  # Old test passes in all cases.

    def test_mixed_primal_multiplier_bounds_and_adjacent_interiors(self):
        for maximize in (False, True):
            for adjacent in (False, True):
                with self.subTest(maximize=maximize, adjacent=adjacent):
                    mins = [tensor(-1.), tensor(0.)]
                    maxs = [tensor(1.), tensor(3.)]
                    if adjacent:
                        values = maxs if maximize else mins
                        values[1] = torch.nextafter(values[1], tensor(1.))
                    bounds = {'min': [(-1., 1.), (0., 3.)], 'max': [(-1., 1.), (0., 3.)]}
                    with mock.patch.object(minimax.Minimax_SCSC, 'run', return_value=certified_stats()):
                        stats = minimax.optimize_NCWC(
                            mins, maxs, lambda: None, 1., 2., [box, box], [box, box], .1, .001,
                            max_iter=1, box_bounds=bounds,
                            gradient_func=lambda u, v: ([torch.ones_like(p) for p in u],
                                                        [torch.ones_like(p) for p in v]),
                        )
                    self.assertEqual(stats['terminated'], not adjacent)
                    residual = stats['final_residual_max' if maximize else 'final_residual_min']
                    self.assertEqual(residual, 1. if adjacent else 0.)

    def test_nonfinite_or_infeasible_candidates_cannot_pass(self):
        for value, gradient in ((1.01, 0.), (float('nan'), 0.),
                                (float('inf'), 0.), (0., float('nan')), (0., float('inf'))):
            with self.subTest(value=value, gradient=gradient):
                x, y = tensor(value), tensor(0.)
                with mock.patch.object(minimax.Minimax_SCSC, 'run', return_value=certified_stats()):
                    with self.assertRaises(RuntimeError):
                        self.solve(x, y, lambda: None, gradient_func=lambda u, v: (
                            [torch.full_like(u[0], gradient)], [torch.zeros_like(v[0])]))

    def test_uncertified_subsolve_never_accepts_even_at_zero_step(self):
        for terminated, delta in ((False, 0.), (True, None), (True, float('nan')),
                                  (True, float('inf')), (True, .001+1e-12), (True, -1.)):
            for bounds in ((-1., 1.), None):
                with self.subTest(terminated=terminated, delta=delta, bounds=bounds):
                    inner = {**certified_stats(), 'terminated': terminated, 'final_delta': delta}
                    gradient = mock.Mock(side_effect=AssertionError('Uncertified endpoint evaluated'))
                    progress = []
                    with mock.patch.object(minimax.Minimax_SCSC, 'run', return_value=inner) as run:
                        stats = self.solve(tensor(0.), tensor(0.), lambda: None,
                                           max_iter=2, box_bounds=bounds, gradient_func=gradient,
                                           progress_callback=progress.append)
                    self.assertEqual(run.call_count, 2)
                    self.assertEqual(stats['num_inner_iters'], 6)
                    self.assertFalse(stats['terminated'])
                    self.assertEqual(stats['termination_reason'], 'max_iter')
                    self.assertIsNone(stats['residual_type'])
                    self.assertIsNone(stats['final_residual_min'])
                    self.assertFalse(progress[-1]['ncwc_terminated'])
                    self.assertEqual(progress[-1]['inner_terminated'], terminated)

    def test_diagnostics_reset_after_a_subsequent_uncertified_solve(self):
        progress = []
        bad = {**certified_stats(), 'terminated': False, 'final_delta': 1.}
        with mock.patch.object(minimax.Minimax_SCSC, 'run', side_effect=[certified_stats(), bad]):
            stats = self.solve(tensor(0.), tensor(0.), lambda: None, max_iter=2,
                               gradient_func=lambda u, v: ([torch.ones_like(u[0])], [-v[0]]),
                               progress_callback=progress.append)
        self.assertEqual(progress[1]['final_residual_min'], 1.)
        self.assertIsNone(stats['final_residual_min'])
        self.assertIsNone(stats['residual_type'])
        self.assertEqual(stats['termination_reason'], 'max_iter')

    def test_generic_prox_bound_matches_analytic_residual(self):
        # Saddle at (1,-2) for .5(x-2)^2-.5(y+3)^2+|x|-|y|.
        x, y = tensor(1.04, True), tensor(-2., True)
        prox = lambda v, step: v.sign() * (v.abs()-step).clamp_min(0)
        x0, y0 = x.item(), y.item()
        stats = minimax.optimize_NCWC(
            [x], [y], lambda: ((x-2).square()/2-(y+3).square()/2).sum(),
            1., 4., [prox], [prox], .1, .001, max_iter=1, subproblem_max_iter=1000,
        )
        self.assertTrue(stats['terminated'])
        self.assertEqual(stats['termination_reason'], 'bound_certificate')
        self.assertEqual(stats['residual_type'], 'bound')
        self.assertAlmostEqual(stats['final_residual_min'], .001+2*abs(x.item()-x0))
        self.assertAlmostEqual(stats['final_residual_max'], .001+.1/8*abs(y.item()-y0))
        self.assertLessEqual(abs(x.item()-1), stats['final_residual_min'])
        self.assertLessEqual(abs(y.item()+2), stats['final_residual_max'])

    def test_generic_bound_uses_fixed_max_anchor_and_scheduled_inner_tolerance(self):
        x, y = tensor(0.), tensor(0.)
        progress = []

        def move(solver):
            solver.assign_y([solver.get_y()[0]+1])
            return certified_stats()

        with mock.patch.object(minimax.Minimax_SCSC, 'run', autospec=True, side_effect=move):
            stats = self.solve(x, y, lambda: None, box_bounds=None, epsilon=.01,
                               epsilon_0=.02, max_iter=3, progress_callback=progress.append)
        self.assertFalse(stats['terminated'])
        self.assertAlmostEqual(stats['final_residual_min'], .02/3)
        self.assertAlmostEqual(stats['final_residual_max'], .02/3+.01/4*3)
        self.assertEqual([p['termination_reason'] for p in progress], [None, None, None, 'max_iter'])

    def test_one_extra_gradient_for_boxes_and_none_for_generic_bound(self):
        for bounds in ((-1., 1.), None):
            for explicit in (False, True):
                with self.subTest(bounds=bounds, explicit=explicit):
                    x, y = tensor(0., not explicit), tensor(0., not explicit)
                    h = mock.Mock(side_effect=lambda: (x.square()-y.square()).sum()/2)
                    gradient = mock.Mock(side_effect=lambda u, v: ([u[0]], [-v[0]]))
                    with mock.patch.object(minimax.Minimax_SCSC, 'run', return_value=certified_stats()):
                        stats = self.solve(x, y, h, box_bounds=bounds,
                                           gradient_func=gradient if explicit else None)
                    self.assertTrue(stats['terminated'])
                    self.assertEqual(gradient.call_count, int(explicit and bounds is not None))
                    self.assertEqual(h.call_count, int(not explicit and bounds is not None))

    def test_autograd_preserves_values_buffers_flags_and_handles_unused_parameters(self):
        x, y = tensor(.25, True), tensor(.5)
        x.grad, y.grad = tensor(7.), tensor(8.)
        xgrad, ygrad = x.grad, y.grad
        with torch.no_grad():
            gx, gy = minimax._ncwc_objective_gradient(lambda: x.square().sum()/2, None, [x], [y])
        self.assertEqual(gx[0].item(), .25)
        self.assertEqual(gy[0].item(), 0.)
        self.assertEqual((x.item(), y.item()), (.25, .5))
        self.assertTrue(x.requires_grad)
        self.assertFalse(y.requires_grad)
        self.assertIs(x.grad, xgrad)
        self.assertIs(y.grad, ygrad)
        self.assertEqual((x.grad.item(), y.grad.item()), (7., 8.))
        for constant in (tensor(5.).sum(), 5.):
            gx, gy = minimax._ncwc_objective_gradient(lambda: constant, None, [x], [y])
            self.assertEqual((gx[0].item(), gy[0].item()), (0., 0.))
        with self.assertRaisesRegex(RuntimeError, 'objective failed'):
            minimax._ncwc_objective_gradient(
                mock.Mock(side_effect=RuntimeError('objective failed')), None, [x], [y],
            )
        self.assertTrue(x.requires_grad)
        self.assertFalse(y.requires_grad)

    def test_explicit_gradient_contract_is_validated(self):
        with mock.patch.object(minimax.Minimax_SCSC, 'run', return_value=certified_stats()):
            with self.assertRaises(ValueError):
                self.solve(tensor(0.), tensor(0.), lambda: None, gradient_func=lambda u, v: ([], v))

    def test_stochastic_retains_step_threshold_and_no_certificate(self):
        inner = {**certified_stats(), 'terminated': False, 'final_delta': None}
        with mock.patch.object(minimax.SAPD_SCSC, 'run', return_value=inner):
            stats = self.solve(tensor(0.), tensor(0.), lambda: None,
                               sub_routine='stochastic', box_bounds=None)
        self.assertTrue(stats['terminated'])
        self.assertEqual(stats['termination_reason'], 'step_threshold')
        self.assertIsNone(stats['residual_type'])
        self.assertIsNone(stats['final_residual_max'])


if __name__ == '__main__':
    unittest.main()
