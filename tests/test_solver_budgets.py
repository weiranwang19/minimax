"""Independent nested limits and end-to-end propagation of incomplete solves."""
import contextlib
import io
import unittest
from unittest import mock

import torch

import bilevel_solvers as solvers
import exp_cblo as cblo
import minimax


def tensor(value):
    return torch.tensor([value], dtype=torch.float64)


def ncwc_stats(terminated=True, budget=None):
    inner = dict(num_outer_iters=3, num_inner_iters=4, final_delta=.02,
                 terminated=terminated, termination_reason='inner_max_iter' if budget == 'scsc_inner' else 'max_iter',
                 budget_exhausted=budget if budget in ('scsc_inner', 'scsc_outer') else None)
    return dict(num_outer_iters=2, num_inner_iters=4, final_diff=.25, terminated=terminated,
                termination_reason='direct_certificate' if terminated else (
                    'max_iter' if budget == 'ncwc' else 'inner_solver_failed'),
                budget_exhausted=budget, final_residual_min=.2, final_residual_max=.01,
                last_scsc_stats=inner)


def call_method(method, **options):
    x, y = tensor(0.).requires_grad_(), tensor(.2).requires_grad_()
    common = dict(upper_smooth=lambda x, y: (x[0]+y[0]).sum(),
                  lower_smooth=lambda x, y: (.3*y[0]+0*x[0]).sum(),
                  lower_constraints=lambda x, y: y[0]-.1, prox_x=cblo.prox_box,
                  D_y=2., L_grad_f1=1., L_grad_ftilde1=0., L_grad_gtilde=0.,
                  L_gtilde=1., gtilde_hi=1.)
    if method == 'fop_direct':
        common.update(prox_y=cblo.prox_box, epsilon=.5)
        return solvers.optimize_bilevel_constrained_fop([x], [y], **{**common, **options})
    if method == 'fop':
        common.update(prox_y=cblo.prox_box, base_rho=2., final_epsilon=2., max_outer_iters=1,
                      warm_start_max_iter=0)
        return solvers.optimize_bilevel_contrained_fop_practical([x], [y], **{**common, **options})
    if method == 'smo':
        common.update(prox_y=cblo.prox_box, epsilon=.5, epsilon_0=.5, warm_start_max_iter=0)
        return solvers.optimize_bilevel_constrained_smo([x], [y], **{**common, **options})
    common.update(prox_y1=cblo.prox_box, lagrange_bound=3., epsilon=.5)
    return solvers.optimize_bilevel_constrained_minimax([x], [y], **{**common, **options})


class InnerBudgetTests(unittest.TestCase):
    def solver(self, cap=None, outer=10, tau=.01):
        x, y = tensor(.7), tensor(-.6)
        return minimax.Minimax_SCSC(
            [x], [y], lambda: None, 1., 1., 1., [cblo.prox_box], [cblo.prox_box], tau,
            max_iter=outer, max_inner_iter=cap, gradient_func=lambda u, v: ([u[0]], [-v[0]]),
            box_bounds=(-1., 1.),
        )

    def test_inner_caps_return_current_feasible_candidate_and_exact_update_counts(self):
        for cap in (0, 1, 2):
            with self.subTest(cap=cap):
                solver = self.solver(cap)
                stats = solver.run()
                self.assertFalse(stats['terminated'])
                self.assertEqual(stats['termination_reason'], 'inner_max_iter')
                self.assertEqual(stats['budget_exhausted'], 'scsc_inner')
                self.assertEqual(stats['num_outer_iters'], 1)
                self.assertEqual(stats['num_inner_iters'], cap)
                independent = torch.linalg.vector_norm(torch.cat(solver.get_x()+solver.get_y())).item()
                self.assertAlmostEqual(stats['final_delta'], independent)
                self.assertTrue(all((p.abs() <= 1).all() for p in solver.get_x()+solver.get_y()))

    def test_certificate_is_allowed_at_exact_inner_limit(self):
        delta = self.solver(1).run()['final_delta']
        solver = self.solver(1, tau=delta)
        stats = solver.run()
        self.assertTrue(stats['terminated'])
        self.assertEqual(stats['num_inner_iters'], 1)
        self.assertEqual(stats['final_delta'], delta)
        self.assertEqual(stats['termination_reason'], 'inner_certificate')
        self.assertIsNone(stats['budget_exhausted'])
        immediate = self.solver(0, tau=1.).run()
        self.assertTrue(immediate['terminated'])
        self.assertEqual(immediate['num_inner_iters'], 0)

    def test_relative_completion_and_outer_certificate_are_allowed_at_exact_limit(self):
        # This quadratic reaches its outer certificate after a relative-test
        # completion. Setting the cap exactly there must reproduce its output.
        baseline = self.solver(outer=1)
        expected = baseline.run()
        self.assertEqual(expected['termination_reason'], 'outer_certificate')
        cap = expected['num_inner_iters']
        actual = self.solver(cap, outer=1)
        self.assertEqual(actual.run(), expected)
        for a, b in zip(actual.get_x()+actual.get_y(), baseline.get_x()+baseline.get_y()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual(self.solver(cap-1, outer=1).run()['budget_exhausted'], 'scsc_inner')

    def test_outer_exhaustion_remains_distinct(self):
        solver = self.solver(outer=1, tau=1e-12)
        # Coupling prevents the one-outer-step exact solution of the decoupled case.
        solver.gradient_func = lambda u, v: ([u[0]+.5*v[0]], [.5*u[0]-v[0]])
        stats = solver.run()
        self.assertFalse(stats['terminated'])
        self.assertEqual(stats['termination_reason'], 'max_iter')
        self.assertEqual(stats['budget_exhausted'], 'scsc_outer')

    def test_ncwc_independent_limits_and_legacy_fallback(self):
        for outer_override in (None, 3):
            for inner_cap in (None, 0, 4):
                observed = []
                def run(solver):
                    observed.append((solver.max_iter, solver.max_inner_iter))
                    return dict(num_outer_iters=1, num_inner_iters=0, final_delta=0., terminated=True)
                with mock.patch.object(minimax.Minimax_SCSC, 'run', autospec=True, side_effect=run):
                    stats = minimax.optimize_NCWC(
                        [tensor(0.)], [tensor(0.)], lambda: None, 1., 2.,
                        [cblo.prox_box], [cblo.prox_box], .1, .001, max_iter=2,
                        subproblem_max_iter=outer_override, subproblem_max_inner_iter=inner_cap,
                        box_bounds=(-1., 1.), gradient_func=lambda u, v: ([u[0]], [-v[0]]),
                    )
                self.assertTrue(stats['terminated'])
                self.assertEqual(observed, [(2 if outer_override is None else 3, inner_cap)])

    def test_nonbinding_caps_reproduce_analytic_outputs_and_work_counts(self):
        for case, count in (('linear', (1, 52)), ('boundary', (1, 1)), ('quadratic', (6, 1222))):
            outputs = []
            for finite_caps in (False, True):
                x, y = tensor(0. if case == 'boundary' else 1.), tensor(0.)
                if case == 'quadratic':
                    gradient = lambda u, v: ([u[0]], [-v[0]])
                else:
                    c = -10. if case == 'boundary' else .075
                    gradient = lambda u, v: ([torch.full_like(u[0], c)], [-v[0]])
                def difference(a, b, u, v):
                    old, old_y = gradient(a, b)
                    new, new_y = gradient(u, v)
                    return [new[0]-old[0]], [new_y[0]-old_y[0]]
                stats = minimax.optimize_NCWC(
                    [x], [y], lambda: None, 1., 2., [cblo.prox_box], [cblo.prox_box], .1, .001,
                    max_iter=1000, gradient_func=gradient, gradient_difference_func=difference,
                    box_bounds=(-1., 1.),
                    **({'subproblem_max_iter': 100, 'subproblem_max_inner_iter': 10000} if finite_caps else {}),
                )
                self.assertTrue(stats['terminated'])
                self.assertEqual((stats['num_outer_iters'], stats['num_inner_iters']), count)
                outputs.append((stats, x.item(), y.item()))
            self.assertEqual(outputs[0], outputs[1])

    def test_ncwc_exhaustion_and_child_exhaustion_callbacks(self):
        for budget in ('ncwc', 'scsc_outer', 'scsc_inner'):
            with self.subTest(budget=budget):
                child = dict(num_outer_iters=3, num_inner_iters=4, terminated=budget == 'ncwc',
                             final_delta=0. if budget == 'ncwc' else 1.,
                             termination_reason='inner_certificate' if budget == 'ncwc' else 'max_iter',
                             budget_exhausted=None if budget == 'ncwc' else budget)
                progress = []
                run_log = FakeRun()
                logger = cblo.build_ncwc_progress_logger(run_log, {
                    'base_scsc_inner_iters': 0, 'ncwc_call_index': 0,
                })
                with mock.patch.object(minimax.Minimax_SCSC, 'run', return_value=child) as run:
                    stats = minimax.optimize_NCWC(
                        [tensor(0.)], [tensor(0.)], lambda: None, 1., 2.,
                        [cblo.prox_box], [cblo.prox_box], .1, .001, max_iter=2,
                        box_bounds=(-1., 1.), gradient_func=lambda u, v: ([torch.ones_like(u[0])], [-v[0]]),
                        progress_callback=lambda p: (progress.append(p), logger(p)),
                    )
                self.assertFalse(stats['terminated'])
                self.assertEqual(stats['budget_exhausted'], budget)
                self.assertEqual(run.call_count, 2 if budget == 'ncwc' else 1)
                self.assertEqual(stats['num_inner_iters'], 4*run.call_count)
                self.assertEqual(stats['last_scsc_stats'], child)
                self.assertEqual(progress[-1]['budget_exhausted'], budget)
                self.assertEqual(run_log.rows[-1]['ncwc/budget_exhausted'], budget)
                self.assertEqual(run_log.rows[-1]['ncwc/scsc_outer_iters'], 3)
                self.assertEqual(run_log.rows[-1]['ncwc/terminated'], 0.)

    def test_invalid_limits_and_stochastic_inner_cap(self):
        for value in (-1, .5, 1., True, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.solver(value)
        for value in (0, -1, 1.5, False, None):
            with self.subTest(outer=value), self.assertRaises(ValueError):
                self.solver(outer=value)
        with self.assertRaisesRegex(ValueError, 'deterministic'):
            minimax.optimize_NCWC([tensor(0.)], [tensor(0.)], lambda: None, 1., 2.,
                                  [cblo.prox_box], [cblo.prox_box], .1, .01,
                                  sub_routine='stochastic', subproblem_max_inner_iter=0)


class MethodBudgetTests(unittest.TestCase):
    def test_real_nested_solvers_propagate_small_caps(self):
        for method in ('fop_direct', 'fop', 'smo', 'gcmo'):
            with self.subTest(method=method), contextlib.redirect_stdout(io.StringIO()):
                options = dict(scsc_max_iter=3, scsc_max_inner_iter=0)
                options['subproblem_max_iter' if method in ('fop', 'smo') else 'max_iter'] = 2
                if method == 'fop':
                    options.update(final_epsilon=.01, max_outer_iters=2)
                result = call_method(method, **options)
                self.assertFalse(result['terminated'])
                self.assertIn(result['budget_exhausted'], ('stages', 'ncwc', 'scsc_outer', 'scsc_inner'))
                stats = result['history'][-1]['subproblem_stats'] if method in ('fop', 'smo') else result['solver_stats']
                self.assertLessEqual(stats['num_outer_iters'], 2)
                self.assertLessEqual(stats['last_scsc_stats']['num_outer_iters'], 3)
                self.assertEqual(stats['num_inner_iters'], 0)

    def test_all_wrappers_forward_unequal_limits(self):
        for method in ('fop_direct', 'fop', 'smo', 'gcmo'):
            with self.subTest(method=method), contextlib.redirect_stdout(io.StringIO()):
                options = dict(scsc_max_iter=3, scsc_max_inner_iter=4)
                options['subproblem_max_iter' if method in ('fop', 'smo') else 'max_iter'] = 2
                with mock.patch.object(solvers, 'optimize_NCWC', return_value=ncwc_stats()) as ncwc:
                    result = call_method(method, **options)
                self.assertTrue(result['terminated'])
                self.assertEqual(ncwc.call_count, 1)
                self.assertEqual(ncwc.call_args.kwargs['max_iter'], 2)
                self.assertEqual(ncwc.call_args.kwargs['subproblem_max_iter'], 3)
                self.assertEqual(ncwc.call_args.kwargs['subproblem_max_inner_iter'], 4)
                if method in ('fop', 'smo'):
                    self.assertTrue(result['history'][0]['warm_start_stats']['capped'])

    def test_failed_stages_stop_and_smo_does_not_update_multipliers(self):
        for method in ('fop_direct', 'fop', 'smo', 'gcmo'):
            for budget in ('ncwc', 'scsc_outer', 'scsc_inner'):
                with self.subTest(method=method, budget=budget), contextlib.redirect_stdout(io.StringIO()):
                    stages = []
                    options = {}
                    if method in ('fop', 'smo'):
                        options.update(max_outer_iters=5, stage_callback=stages.append)
                    if method == 'smo':
                        options.update(lambda0=tensor(.7), epsilon=.01)
                    with mock.patch.object(solvers, 'optimize_NCWC', return_value=ncwc_stats(False, budget)) as ncwc:
                        result = call_method(method, **options)
                    self.assertFalse(result['terminated'])
                    self.assertEqual(result['budget_exhausted'], budget)
                    self.assertEqual(ncwc.call_count, 1)
                    if stages:
                        self.assertEqual(len(stages), 1)
                        self.assertEqual(stages[0]['budget_exhausted'], budget)
                    if method == 'smo':
                        torch.testing.assert_close(result['lambda'], tensor(.7))
                        torch.testing.assert_close(stages[0]['lambda'], stages[0]['next_lambda'])
                        self.assertEqual(stages[0]['epsilon_k'], .5)

    def test_stage_cap_and_success_on_last_permitted_stage(self):
        for method in ('fop', 'smo'):
            for success in (False, True):
                with self.subTest(method=method, success=success):
                    stages = []
                    options = dict(max_outer_iters=1, stage_callback=stages.append)
                    options['final_epsilon' if method == 'fop' else 'epsilon'] = (2. if method == 'fop' else .5) if success else .01
                    with mock.patch.object(solvers, 'optimize_NCWC', return_value=ncwc_stats()):
                        result = call_method(method, **options)
                    self.assertEqual(result['terminated'], success)
                    self.assertEqual(result['budget_exhausted'], None if success else 'stages')
                    self.assertEqual(result['termination_reason'], 'converged' if success else 'max_stages')
                    self.assertEqual(result['num_outer_iters'], 1)
                    self.assertEqual(stages[0]['budget_exhausted'], None if success else 'stages')

    def test_public_budget_validation_precedes_solver_execution(self):
        for method in ('fop_direct', 'fop', 'smo', 'gcmo'):
            cases = [('scsc_max_iter', 0), ('scsc_max_iter', 1.5), ('scsc_max_inner_iter', -1),
                     ('scsc_max_inner_iter', True)]
            if method in ('fop', 'smo', 'gcmo'):
                cases.append(('warm_start_max_iter', .5))
            if method in ('fop', 'smo'):
                cases.append(('max_outer_iters', 0))
            for name, value in cases:
                with self.subTest(method=method, name=name, value=value), self.assertRaises(ValueError):
                    call_method(method, **{name: value})


class FakeRun:
    def __init__(self):
        self.rows, self.summary, self.definitions = [], {}, {}
        self.finish_count = 0

    def log(self, metrics):
        self.rows.append(dict(metrics))

    def define_metric(self, name, **kwargs):
        self.definitions[name] = kwargs

    def finish(self):
        self.finish_count += 1


class FakeWandb:
    def __init__(self):
        self.run = FakeRun()

    def init(self, **kwargs):
        self.config = kwargs['config']
        return self.run


class ExperimentBudgetTests(unittest.TestCase):
    def settings(self, method):
        return dict(SOLVER_METHOD=method, VERBOSE=False, FOP_MAX_STAGES=1, SMO_MAX_STAGES=1,
                    FINAL_EPS=.01, SMO_EPS=.01, SMO_EPSILON_0=.5, GCMO_EPS=.5,
                    FOP_AGD_MAX_ITERS=0, SMO_AGD_MAX_ITERS=0, GCMO_AGD_MAX_ITERS=0,
                    **{f'{m}_{suffix}': value for m in ('FOP', 'SMO', 'GCMO')
                       for suffix, value in (('NCWC_MAX_ITERS', 2), ('SCSC_MAX_OUTER_ITERS', 3),
                                             ('SCSC_MAX_INNER_ITERS', 4))})

    def run_failure(self, method, budget):
        cblo.generate_instance((2, 3, 1), 1)
        wandb = FakeWandb()
        child = ncwc_stats(budget == 'stages', None if budget == 'stages' else budget)
        with contextlib.ExitStack() as stack:
            for key, value in self.settings(method).items():
                stack.enter_context(mock.patch.object(cblo, key, value))
            stack.enter_context(mock.patch.object(cblo, '_resolve_wandb', return_value=wandb))
            ncwc = stack.enter_context(mock.patch.object(solvers, 'optimize_NCWC', return_value=child))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            with self.assertRaisesRegex(RuntimeError, budget):
                cblo.run_single_instance(1, (2, 3, 1), 1, 1)
        self.assertEqual(wandb.run.finish_count, 1)
        self.assertEqual(wandb.run.summary['instance/failed'], 1.)
        self.assertEqual(wandb.run.summary['instance/terminated'], 0.)
        self.assertEqual(wandb.run.summary['instance/budget_exhausted'], budget)
        self.assertEqual(ncwc.call_args.kwargs['max_iter'], 2)
        self.assertEqual(ncwc.call_args.kwargs['subproblem_max_iter'], 3)
        self.assertEqual(ncwc.call_args.kwargs['subproblem_max_inner_iter'], 4)
        for prefix in ('fop', 'smo', 'gcmo'):
            for suffix, value in (('ncwc_max_iters', 2), ('scsc_max_outer_iters', 3),
                                  ('scsc_max_inner_iters', 4), ('agd_max_iters', 0)):
                self.assertEqual(wandb.config[f'{prefix}_{suffix}'], value)
        for key, value in (('max_outer_iters', 1), ('alg4_max_iters', 2),
                           ('smo_subproblem_max_iters', 2), ('gcmo_max_iters', 2)):
            self.assertEqual(wandb.config[key], value)
        self.assertIn('ncwc/budget_exhausted', wandb.run.definitions)
        self.assertIn(f'{method}/subproblem/budget_exhausted', wandb.run.definitions)
        if budget != 'stages':
            self.assertEqual(wandb.run.summary['instance/subproblem_num_inner_iters'], 4)
            self.assertEqual(wandb.run.summary['instance/scsc_final_delta'], .02)
        if budget == 'stages':
            self.assertTrue(any(row.get(f'{method}/stage/budget_exhausted') == 'stages'
                                for row in wandb.run.rows))

    def test_each_method_reports_solver_budget_failures_and_finishes_once(self):
        for method in ('fop', 'smo', 'gcmo'):
            for budget in ('ncwc', 'scsc_outer', 'scsc_inner'):
                with self.subTest(method=method, budget=budget):
                    self.run_failure(method, budget)

    def test_stage_exhaustion_reports_incomplete(self):
        for method in ('fop', 'smo'):
            with self.subTest(method=method):
                self.run_failure(method, 'stages')

    def test_final_feasibility_failure_raises_and_finishes_once(self):
        for method in ('smo', 'gcmo'):
            with self.subTest(method=method):
                cblo.generate_instance((2, 3, 1), 1)
                wandb = FakeWandb()
                settings = {**self.settings(method), 'SMO_EPS': .5, 'FEAS_TOL': 0.}
                with contextlib.ExitStack() as stack:
                    for key, value in settings.items():
                        stack.enter_context(mock.patch.object(cblo, key, value))
                    stack.enter_context(mock.patch.object(cblo, '_resolve_wandb', return_value=wandb))
                    stack.enter_context(mock.patch.object(solvers, 'optimize_NCWC', return_value=ncwc_stats()))
                    stack.enter_context(mock.patch.object(cblo, 'feasibility_norm', return_value=1.))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    with self.assertRaisesRegex(RuntimeError, 'stop check'):
                        cblo.run_single_instance(1, (2, 3, 1), 1, 1)
                self.assertEqual(wandb.run.finish_count, 1)
                self.assertEqual(wandb.run.summary['instance/failed'], 1.)
                self.assertEqual(wandb.run.summary['instance/terminated'], 0.)

    def test_grid_continues_after_incomplete_instance(self):
        success = dict(num_outer_iters=1, initial_objective=0., final_objective=0., final_feas=0., final_lower_gap=0.)
        output = io.StringIO()
        with mock.patch.object(cblo, 'NUM_INSTANCES', [1, 2]), mock.patch.object(
            cblo, 'run_single_instance', side_effect=[RuntimeError('budget exhausted'), success],
        ) as run, contextlib.redirect_stdout(output):
            cblo.run_problem_size((2, 3, 1))
        self.assertEqual(run.call_count, 2)
        self.assertIn('Failed instance', output.getvalue())
        self.assertIn('Completed instance', output.getvalue())


if __name__ == '__main__':
    unittest.main()
