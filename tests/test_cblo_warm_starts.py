import contextlib
import io
import unittest
from unittest import mock

import numpy as np
import torch
from scipy.optimize import linprog

import bilevel_solvers as solvers
import exp_cblo as cblo
from agd import agd_convex
from utils import clone_vals, compute_value_and_grad


def upper(xs, ys):
    return xs[0].sum() + ys[0].sum()


def lower(xs, ys):
    return 0.3 * ys[0].sum() + 0.0 * xs[0].sum()


def ncwc_stats():
    return {'num_outer_iters': 1, 'num_inner_iters': 1, 'final_diff': 0.0, 'terminated': True}


def common_args(constraints):
    return dict(
        upper_smooth=upper, lower_smooth=lower, lower_constraints=constraints,
        prox_x=cblo.prox_box, D_y=2.0, L_grad_f1=0.0, L_grad_ftilde1=0.0,
        L_grad_gtilde=0.0, L_gtilde=1.0, gtilde_hi=2.0,
        warm_start_linear_minimizer=cblo.box_linear_minimizer,
    )


class SolverWarmStartTests(unittest.TestCase):
    def test_fop_and_smo_warm_every_stage_from_current_primal(self):
        for method in ('fop', 'smo'):
            with self.subTest(method=method):
                seeds = []
                x = torch.zeros(1, requires_grad=True)
                y = torch.tensor([0.2], requires_grad=True)

                def warmup(values, *args, **kwargs):
                    seeds.append(clone_vals(values))
                    self.assertIs(kwargs['linear_minimizer'], cblo.box_linear_minimizer)
                    return clone_vals(values), {'num_iters': 0, 'target_num_iters': 0, 'capped': False}

                def minimax(min_block, max_block, *args, **kwargs):
                    # FOP/SMO must not acquire GCMO's dual blocks.
                    self.assertEqual((len(min_block), len(max_block)), (2, 1))
                    with torch.no_grad():
                        min_block[1].add_(0.1)
                    return ncwc_stats()

                with mock.patch.object(solvers, 'agd_convex', side_effect=warmup), mock.patch.object(
                    solvers, 'optimize_NCWC', side_effect=minimax,
                ):
                    kwargs = common_args(lambda xs, ys: 0.0 * ys[0])
                    if method == 'fop':
                        result = solvers.optimize_bilevel_contrained_fop_practical(
                            [x], [y], prox_y=cblo.prox_box, base_rho=2.0,
                            final_epsilon=0.5, max_outer_iters=3, **kwargs,
                        )
                    else:
                        result = solvers.optimize_bilevel_constrained_smo(
                            [x], [y], prox_y=cblo.prox_box, epsilon=0.25,
                            epsilon_0=1.0, tau=0.5, **kwargs,
                        )
                self.assertEqual(result['num_outer_iters'], 3)
                self.assertEqual(len(seeds), 3)
                for i, seed in enumerate(seeds):
                    self.assertAlmostEqual(seed[0].item(), 0.2 + 0.1 * i)

    def test_smo_keeps_multiplier_fixed_in_agd_and_updates_after_minimax(self):
        records, stages = [], []
        initial_lambda = torch.tensor([0.7])

        def warmup(values, phi, *args, **kwargs):
            probe = [torch.tensor([0.1])]
            value, gradient = compute_value_and_grad(probe, phi)
            records.append((value.item(), gradient[0].item()))
            self.assertEqual(value.item(), phi(probe).item())
            return clone_vals(values), {'num_iters': 0, 'target_num_iters': 0, 'capped': False}

        def minimax(min_block, max_block, *args, **kwargs):
            with torch.no_grad():
                max_block[0].fill_(0.2)
            return ncwc_stats()

        with mock.patch.object(solvers, 'agd_convex', side_effect=warmup), mock.patch.object(
            solvers, 'optimize_NCWC', side_effect=minimax,
        ):
            solvers.optimize_bilevel_constrained_smo(
                [torch.zeros(1, requires_grad=True)], [torch.zeros(1, requires_grad=True)],
                prox_y=cblo.prox_box, epsilon=0.5, epsilon_0=1.0, tau=0.5,
                lambda0=initial_lambda, stage_callback=stages.append,
                **common_args(lambda xs, ys: ys[0]),
            )
        self.assertEqual(len(records), 2)
        for (value, grad), stage in zip(records, stages):
            lam, mu, rho = stage['lambda'].item(), stage['mu_k'], stage['rho_k']
            self.assertAlmostEqual(value, 0.03 + (lam + 0.1 * mu) ** 2 / (2.0 * rho * mu))
            self.assertAlmostEqual(grad, 0.3 + (lam + 0.1 * mu) / rho)
            self.assertAlmostEqual(stage['next_lambda'].item(), lam + mu * 0.2)
        self.assertAlmostEqual(initial_lambda.item(), 0.7)
        torch.testing.assert_close(stages[1]['lambda'], stages[0]['next_lambda'])

    def test_gcmo_warms_once_and_copies_clipped_dual_with_factor_two(self):
        events, progress_metrics = [], []
        x = torch.zeros(1, requires_grad=True)
        y = torch.tensor([0.4], requires_grad=True)

        def gap(xs, ys, multiplier):
            self.assertEqual((len(xs), len(ys)), (1, 1))
            return ys[0].sum() + multiplier.sum()

        def minimax(min_block, max_block, *args, **kwargs):
            self.assertEqual(events[0][0], 'warmup')
            self.assertEqual((len(min_block), len(max_block)), (3, 2))
            torch.testing.assert_close(min_block[1], max_block[0])
            torch.testing.assert_close(min_block[2], torch.tensor([0.0, 1.0, 3.0]))
            torch.testing.assert_close(min_block[2], max_block[1])
            self.assertNotEqual(min_block[1].data_ptr(), max_block[0].data_ptr())
            self.assertNotEqual(min_block[2].data_ptr(), max_block[1].data_ptr())
            progress_metrics.append(kwargs['metrics_func'](min_block, max_block))
            with torch.no_grad():
                min_block[1].fill_(0.25)
                min_block[2].copy_(torch.tensor([0.0, 0.5, 2.0]))
            progress_metrics.append(kwargs['metrics_func'](min_block, max_block))
            return ncwc_stats()

        def primal_metrics(xs, ys):
            self.assertEqual((len(xs), len(ys)), (1, 1))
            return {'existing_metric': 9.0}

        kwargs = common_args(lambda xs, ys: 0.0 * ys[0][0] + torch.tensor([-1.0, 0.125, 1.0]))
        kwargs['lower_smooth'] = lambda xs, ys: 0.0 * ys[0].sum()
        with mock.patch.object(solvers, 'agd_convex', wraps=agd_convex) as agd, mock.patch.object(
            solvers, 'optimize_NCWC', side_effect=minimax,
        ), contextlib.redirect_stdout(io.StringIO()):
            result = solvers.optimize_bilevel_constrained_minimax(
                [x], [y], prox_y1=cblo.prox_box, lagrange_bound=3.0, epsilon=0.5,
                warm_start_lower=True, primal_dual_gap_func=gap, metrics_func=primal_metrics,
                warm_start_callback=lambda stats: events.append(('warmup', stats)), **kwargs,
            )
        self.assertEqual(agd.call_count, 1)
        self.assertAlmostEqual(result['initial_primal_dual_gap'], 4.4)
        self.assertAlmostEqual(events[0][1]['primal_dual_gap'], 4.4)
        self.assertEqual([m['existing_metric'] for m in progress_metrics], [9.0, 9.0])
        self.assertAlmostEqual(progress_metrics[0]['primal_dual_gap'], 4.4)
        self.assertAlmostEqual(progress_metrics[1]['primal_dual_gap'], 2.75)
        self.assertTrue(result['warm_start_stats']['terminated'])

    def test_gcmo_cap_exit_reports_uncertified_warmup_and_still_solves(self):
        for cap in (0, 1):
            with self.subTest(cap=cap):
                reports = []
                kwargs = common_args(lambda xs, ys: ys[0] - 0.4)
                with mock.patch.object(solvers, 'optimize_NCWC', return_value=ncwc_stats()) as minimax:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = solvers.optimize_bilevel_constrained_minimax(
                            [torch.zeros(1, requires_grad=True)], [torch.tensor([0.4], requires_grad=True)],
                            prox_y1=cblo.prox_box, lagrange_bound=3.0, epsilon=0.01,
                            warm_start_lower=True, warm_start_max_iter=cap,
                            primal_dual_gap_func=lambda xs, ys, lam: (
                                0.3 * ys[0].sum() + 3.0 * (ys[0] - 0.4).clamp_min(0).sum()
                                + 0.4 * lam.sum() + (0.3 + lam).abs().sum()
                            ),
                            warm_start_callback=reports.append, **kwargs,
                        )
                minimax.assert_called_once()
                self.assertEqual(result['warm_start_stats']['num_iters'], cap)
                self.assertTrue(reports[0]['capped'])
                self.assertFalse(reports[0]['terminated'])
                self.assertEqual(reports[0]['stopping_reason'], 'max_iter')
                self.assertGreater(reports[0]['certificate_gap'], 0.01)
                self.assertGreater(reports[0]['primal_dual_gap'], 0.01)


class FakeRun:
    def __init__(self):
        self.rows = []
        self.summary = {}
        self.definitions = {}

    def log(self, metrics):
        self.rows.append(dict(metrics))

    def define_metric(self, name, **kwargs):
        self.definitions[name] = kwargs

    def finish(self):
        pass


class FakeWandb:
    def __init__(self):
        self.run = FakeRun()
        self.config = None

    def init(self, **kwargs):
        self.config = kwargs['config']
        return self.run


class CBLODiagnosticsTests(unittest.TestCase):
    def test_analytic_gap_matches_independent_box_lps(self):
        cblo.generate_instance((2, 3, 2), 0)
        rng = np.random.default_rng(21)
        for _ in range(5):
            x = torch.tensor(rng.uniform(-1, 1, 2))
            y = torch.tensor(rng.uniform(-1, 1, 3))
            lam = torch.tensor(rng.uniform(0, 3, 2))
            residual = cblo.constraint_residual(x, y).numpy()
            max_lp = linprog(-residual, bounds=[(0, 3)] * 2, method='highs')
            min_lp = linprog((cblo.D_TILDE + cblo.B_MAT.T @ lam).numpy(),
                             bounds=[(-1, 1)] * 3, method='highs')
            self.assertTrue(max_lp.success and min_lp.success)
            expected = (
                (cblo.D_TILDE @ y).item() - max_lp.fun - min_lp.fun
                - (lam @ (cblo.A_MAT @ x - cblo.B_VEC)).item()
            )
            self.assertAlmostEqual(cblo.lower_primal_dual_gap(x, y, lam, 3.0), expected)

    def test_planted_pair_has_zero_gap(self):
        cblo.generate_instance((2, 3, 2), 1)
        lam = torch.tensor([0.2, 0.7])
        with mock.patch.object(cblo, 'D_TILDE', -cblo.B_MAT.T @ lam):
            gap = cblo.lower_primal_dual_gap(torch.zeros(2), cblo.Y_HAT, lam, 3.0)
        self.assertAlmostEqual(gap, 0.0, places=12)

    def test_experiments_seed_once_and_log_without_external_wandb(self):
        for method, smo_stable in ((m, s) for m in ('fop', 'smo', 'gcmo') for s in (False, True)):
            with self.subTest(method=method, smo_stable=smo_stable):
                cblo.generate_instance((2, 3, 1), 2)
                seed = cblo.Y_HAT.clone()
                wandb = FakeWandb()
                agd_inputs = []

                def warmup(values, *args, **kwargs):
                    agd_inputs.append(clone_vals(values))
                    self.assertIsNone(kwargs['max_iter'])
                    self.assertIs(kwargs['linear_minimizer'], cblo.box_linear_minimizer)
                    self.assertIsNotNone(kwargs['value_and_grad_func'])
                    return agd_convex(values, *args, **kwargs)

                def minimax(min_block, max_block, *args, **kwargs):
                    self.assertFalse(any(v.requires_grad for v in min_block+max_block))
                    self.assertTrue(callable(kwargs['gradient_func']))
                    self.assertTrue(callable(kwargs['gradient_difference_func']))
                    stable = ((method == 'fop' and cblo.FOP_STABLE_BOX_RESIDUAL)
                              or (method == 'smo' and smo_stable))
                    self.assertEqual(kwargs.get('box_bounds'), (-1., 1.) if stable else None)
                    for step in (0, 1):
                        if step:
                            with torch.no_grad():
                                min_block[1].add_(0.01).clamp_(-1, 1)
                        kwargs['progress_callback']({
                            'completed_outer_iters': step,
                            'call_cumulative_scsc_inner_iters': step,
                            'scsc_num_inner_iters': step,
                            'final_diff': 0.0, 'inner_terminated': True,
                            'objective': float(kwargs['objective_func'](min_block, max_block)),
                            'metrics': kwargs['metrics_func'](min_block, max_block),
                        })
                    return ncwc_stats()

                settings = dict(
                    SOLVER_METHOD=method, VERBOSE=False, FINAL_EPS=2.0, MAX_OUTER_ITERS=1,
                    BASE_RHO=2.0, SMO_EPS=0.5, SMO_EPSILON_0=0.5, GCMO_EPS=0.5,
                    GCMO_LAGRANGE_BOUND=3.0, FEAS_TOL=1e6, LOWER_GAP_TOL=1e6,
                    SMO_STABLE_BOX_RESIDUAL=smo_stable,
                )
                with contextlib.ExitStack() as stack:
                    for key, value in settings.items():
                        stack.enter_context(mock.patch.object(cblo, key, value))
                    stack.enter_context(mock.patch.object(cblo, '_resolve_wandb', return_value=wandb))
                    stack.enter_context(mock.patch.object(solvers, 'agd_convex', side_effect=warmup))
                    stack.enter_context(mock.patch.object(solvers, 'optimize_NCWC', side_effect=minimax))
                    for target in ('torch.autograd.grad', 'torch.autograd.backward', 'torch.Tensor.backward'):
                        stack.enter_context(mock.patch(target, side_effect=AssertionError('Unexpected autograd')))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    result = cblo.run_single_instance(2, (2, 3, 1), 1, 1)
                self.assertEqual(len(agd_inputs), 1)
                torch.testing.assert_close(agd_inputs[0][0], seed)
                torch.testing.assert_close(cblo.Y_HAT, seed)
                self.assertAlmostEqual(result['initial_objective'], (cblo.D @ seed).item())
                self.assertEqual(wandb.config['gradient_backend'], 'explicit')
                self.assertEqual(wandb.config['smo_stable_box_residual'], smo_stable)
                self.assertNotIn('fop_explicit_gradients', wandb.config)
                for name in ('fop_agd_max_iters', 'smo_agd_max_iters', 'gcmo_agd_max_iters'):
                    self.assertIsNone(wandb.config[name])
                warm_rows = [row for row in wandb.run.rows if f'{method}/warm_start/certificate_gap' in row]
                self.assertEqual(len(warm_rows), 1)
                self.assertEqual(warm_rows[0][f'{method}/warm_start/terminated'], 1.0)
                self.assertIn(f'{method}/warm_start/certificate_gap', wandb.run.definitions)
                if method == 'gcmo':
                    warm_key = 'gcmo/warm_start/primal_dual_gap'
                    progress_rows = [row for row in wandb.run.rows if 'ncwc/primal_dual_gap' in row]
                    self.assertEqual(len(progress_rows), 2)
                    self.assertAlmostEqual(warm_rows[0][warm_key], progress_rows[0]['ncwc/primal_dual_gap'])
                    post = next(row for row in wandb.run.rows if 'gcmo/post/primal_dual_gap' in row)
                    self.assertAlmostEqual(post['gcmo/post/primal_dual_gap'], progress_rows[-1]['ncwc/primal_dual_gap'])
                    self.assertIn('ncwc/lower_gap', progress_rows[0])
                    self.assertLess(wandb.run.rows.index(warm_rows[0]), wandb.run.rows.index(progress_rows[0]))


if __name__ == '__main__':
    unittest.main()
