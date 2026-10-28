"""Bounded certificate regressions; no W&B or changes to experiment settings.

Run with the repository's torch environment. The GCMO replay reconstructs
the historical first four calls, then runs the current solver on call five.
FOP/SMO check the saved inner candidates and restart from them at tau=1e-6;
their deliberately sub-roundoff saved tau=1e-14 is checked, not relaxed
silently or reported as solved.
"""
import ast
import json
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import exp_cblo as experiment
import minimax
from utils import clone_vals, compute_norm

OUT = Path(__file__).resolve().parent
HISTORICAL_COMMIT = 'ed38430f7cad0b0f38deee469313d2d7d82f1af5'


def independent_kkt(solver):
    # Autograd on the original h_bar closure, independently of both the
    # production analytic oracle and the certificate implementation.
    params = solver.get_x() + solver.get_y()
    with torch.enable_grad():
        for p in params:
            p.requires_grad_(True)
        try:
            gradients = torch.autograd.grad(solver.h_bar(), params)
        finally:
            for p in params:
                p.requires_grad_(False)
    nx = len(solver.get_x())
    residuals = []
    bounds = solver._min_box_bounds + solver._max_box_bounds
    for i, (p, gradient, (lo, hi)) in enumerate(zip(params, gradients, bounds)):
        assert torch.all((p >= lo) & (p <= hi))
        r = gradient.detach().clone() * (1 if i < nx else -1)
        r[(p == lo) & (r >= 0)] = 0
        r[(p == hi) & (r <= 0)] = 0
        residuals.append(r.flatten())
    return torch.linalg.vector_norm(torch.cat(residuals)).item()


def frozen_candidate(method):
    data = torch.load(ROOT / 'reports/scsc_roundoff' / f'{method}_snapshot.pt', weights_only=False)
    state, p = data['state'], data['params']
    oracle = experiment.LinearCBLOOracle(**data['oracle'])
    grad, diff = oracle.saddle_gradients(method, data['rho'], data['mu'], data['lam'])
    grad, diff = minimax._regularize_gradient_callbacks(
        grad, diff, state['min_anchor'], state['max_anchor'], p['sigma_x'], p['sigma_y'],
    )
    x, y = clone_vals(state['x_k_t']), clone_vals(state['y_k_t'])

    def objective():
        u, v = x
        z = y[0]
        rho, mu = data['rho'], data['mu']
        sy = oracle.a @ u + oracle.b @ v - oracle.rhs
        sz = oracle.a @ u + oracle.b @ z - oracle.rhs
        h = oracle.c @ u + oracle.d @ v + rho * (oracle.d_tilde @ v - oracle.d_tilde @ z)
        if method == 'fop':
            h = h + rho * mu * (sy.relu().square().sum() - sz.relu().square().sum())
        else:
            h = h + ((data['lam'] + mu*sy).relu().square().sum()
                     - (data['lam'] + mu*sz).relu().square().sum()) / (2*mu)
        return (h + p['sigma_x'] * sum((v-a).square().sum() for v, a in zip(x, state['min_anchor']))
                - p['sigma_y']/2 * sum((v-a).square().sum() for v, a in zip(y, state['max_anchor'])))

    solver = minimax.Minimax_SCSC(
        x, y, objective, p['sigma_x'], p['sigma_y'], p['lip'],
        [experiment.prox_box] * len(x), [experiment.prox_box] * len(y), p['tau'],
        max_iter=10, gradient_func=grad, gradient_difference_func=diff, box_bounds=(-1, 1),
    )
    gx, gy = solver.compute_h_bar_gradient(x, y)
    rx, ry = solver.compute_inner_stationarity_residual(x, y, gx, gy, state['b_x_k_t'], state['b_y_k_t'])
    frozen_delta = compute_norm(rx+ry).item()
    frozen_autograd_delta = independent_kkt(solver)
    # This is an explicit second tolerance, not a claim about saved tau.
    solver.tau = 1e-6
    stats = solver.run()
    checked_delta = independent_kkt(solver)
    assert stats['terminated'] and stats['final_delta'] <= solver.tau and checked_delta <= solver.tau
    return dict(method=method, saved_tau=p['tau'], frozen_delta=frozen_delta,
                frozen_autograd_delta=frozen_autograd_delta, passes_saved_tau=frozen_delta <= p['tau'],
                restart_tau=solver.tau, stats=stats, returned_autograd_delta=checked_delta)


def historical_run():
    source = subprocess.check_output(
        ['git', 'show', f'{HISTORICAL_COMMIT}:minimax.py'], cwd=ROOT, text=True,
    )
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Minimax_SCSC')
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    source = textwrap.dedent('\n'.join(source.splitlines()[run.lineno-1:run.end_lineno]))
    # Reproduce the condition documented in gcmo_live/report.md. That edit
    # was in the working tree, so is reconstructed explicitly from this commit.
    old = 'if lhs <= rhs + 1e-8:'
    assert old in source
    source = source.replace(old, 'if lhs <= rhs or lhs + rhs <= 1e-12:')
    namespace = dict(vars(minimax))
    exec(compile(source, '<historical GCMO warmup>', 'exec'), namespace)
    return namespace['run']


def gcmo_fifth_call():
    original, historical = minimax.Minimax_SCSC.run, historical_run()
    preceding, result = [], {}

    class Finished(Exception):
        pass

    def run(solver):
        if len(preceding) < 4:
            bounds = solver.box_bounds
            solver.box_bounds = None  # Historical GCMO used the generic outer residual.
            try:
                stats = historical(solver)
            finally:
                solver.box_bounds = bounds
            preceding.append(stats)
            print('historical preceding call', len(preceding), stats, flush=True)
            assert stats['num_outer_iters'] == [105, 114, 132, 196][len(preceding)-1]
            return stats
        solver.max_iter = 1000
        start = time.monotonic()
        stats = original(solver)
        checked_delta = independent_kkt(solver)
        assert stats['terminated'] and stats['final_delta'] <= solver.tau and checked_delta <= solver.tau
        result.update(method='gcmo', tau=solver.tau, stats=stats, returned_autograd_delta=checked_delta,
                      seconds=time.monotonic()-start, preceding_calls=preceding)
        raise Finished

    with mock.patch.multiple(experiment, WANDB_ENABLED=False, VERBOSE=False, SOLVER_METHOD='gcmo',
                             GCMO_LIP_OVERRIDE=None, GCMO_MAX_ITERS=100000), \
            mock.patch.object(minimax.Minimax_SCSC, 'run', run):
        experiment.generate_instance((100, 100, 5), 1)
        try:
            experiment.run_single_instance_gcmo(1, (100, 100, 5), 1, 1)
        except Finished:
            pass
    assert result
    return result


def main():
    torch.set_num_threads(1)

    def timeout(*args):
        raise TimeoutError('Certificate replay exceeded its 180 second budget')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    results = [frozen_candidate(method) for method in ('fop', 'smo')]
    print(json.dumps(results, indent=2), flush=True)
    results.append(gcmo_fifth_call())
    signal.alarm(0)
    (OUT / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results[-1], indent=2), flush=True)


if __name__ == '__main__':
    main()
