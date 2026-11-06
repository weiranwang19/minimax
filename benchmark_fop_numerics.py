"""Bounded four-way FOP/SCSC benchmark; no W&B or full experiment.

Run from this directory with the experiment's Python environment:
    python benchmark_fop_numerics.py --output-dir reports/fop_numerics
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import signal
import statistics
import sys
import time

import torch

import exp_cblo as experiment
from minimax import Minimax_SCSC, _box_projection_quotient, _regularize_gradient_callbacks
from utils import clone_vals, compute_norm


VARIANTS = (
    ('baseline', False, False),
    ('residual_only', True, False),
    ('gradients_only', False, True),
    ('combined', True, True),
)
RHO = 5.
LIP_H = .1 * RHO**3
EPSILON = 1 / RHO
SIGMA_Y = EPSILON / 80
TAU = 1e-6
MAX_ITERS = 100
TIMEOUT = 30.


@contextmanager
def deadline(seconds):
    """Bound even the solver's uncapped inner while loop (Unix main thread)."""
    def expired(signum, frame):
        raise TimeoutError(f'Solve exceeded {seconds:g} seconds')

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def make_solver(stable, explicit):
    # All trials clone the one generated instance; generation is outside timers.
    x = torch.zeros(experiment.N, requires_grad=True)
    y = experiment.Y_HAT.detach().clone().requires_grad_(True)
    z = experiment.Y_HAT.detach().clone().requires_grad_(True)
    min_anchor, max_anchor = clone_vals([x, y]), clone_vals([z])
    grad, diff = _regularize_gradient_callbacks(
        *experiment.LinearCBLOOracle.from_current_instance().saddle_gradients('fop', RHO, RHO**2),
        min_anchor, max_anchor, LIP_H, SIGMA_Y,
    )

    def h_bar():
        # The same FOP objective and equation-(102) regularization as the wrappers.
        ry = experiment.lower_constraints([x], [y]).clamp_min(0)
        rz = experiment.lower_constraints([x], [z]).clamp_min(0)
        return (
            experiment.upper_smooth([x], [y])
            + RHO * experiment.lower_smooth([x], [y])
            - RHO * experiment.lower_smooth([x], [z])
            + RHO**3 * (ry.square().sum() - rz.square().sum())
            + LIP_H * sum((v-a).square().sum() for v, a in zip([x, y], min_anchor))
            - SIGMA_Y/2 * (z-max_anchor[0]).square().sum()
        )

    solver = Minimax_SCSC(
        [x, y], [z], h_bar, LIP_H, SIGMA_Y, 3*LIP_H+SIGMA_Y,
        [experiment.prox_box]*2, [experiment.prox_box], tau=TAU,
        max_iter=MAX_ITERS, verbose=False,
        gradient_func=grad if explicit else None,
        gradient_difference_func=diff if explicit else None,
        box_bounds=(-1., 1.) if stable else None,
    )
    return solver, grad, diff


@torch.no_grad()
def common_accuracy(solver, gradient, difference):
    """Same stable certificate for every output, plus its direct box KKT norm."""
    u, v = clone_vals(solver.get_x()), clone_vals(solver.get_y())
    gu, gv = gradient(u, v)
    step = solver.zeta_hat
    uh = [(p-step*g).clamp(-1, 1) for p, g in zip(u, gu)]
    vh = [(p+step*g).clamp(-1, 1) for p, g in zip(v, gv)]
    du, dv = difference(u, v, uh, vh)
    qu = _box_projection_quotient(u, gu, step, (-1., 1.))
    qv = _box_projection_quotient(v, gv, step, (-1., 1.), maximize=True)
    residual = compute_norm([q+d for q, d in zip(qu+qv, du+dv)]).item()
    # Distance to box stationarity at the returned point, with exact bound tests.
    # For maximization, negate the gradient to apply minimization sign rules.
    kkt = []
    for p, g in zip(u+v, gu+[-g for g in gv]):
        kkt.append(torch.where(p <= -1, g.clamp_max(0),
                               torch.where(p >= 1, g.clamp_min(0), g)))
    return residual, compute_norm(kkt).item()


def solve_trial(stable, explicit):
    solver, grad, diff = make_solver(stable, explicit)
    started = time.perf_counter()
    try:
        with deadline(TIMEOUT):
            stats = solver.run()
    except TimeoutError:
        return {'seconds': time.perf_counter()-started, 'status': 'timeout'}, None
    elapsed = time.perf_counter()-started
    residual, kkt = common_accuracy(solver, grad, diff)
    solution = torch.cat([p.detach().flatten() for p in solver.get_x()+solver.get_y()])
    return {
        'seconds': elapsed, 'status': 'terminated' if stats['terminated'] else 'capped',
        **stats, 'common_stable_residual': residual, 'box_kkt_residual': kkt,
    }, solution


def median_call_us(func, count=200):
    for _ in range(10):
        func()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(count):
            func()
        samples.append((time.perf_counter()-start)*1e6/count)
    return statistics.median(samples)


def microbench(stable, explicit):
    solver, _, _ = make_solver(stable, explicit)
    generator = torch.Generator().manual_seed(12345)
    u = [2*torch.rand(p.shape, generator=generator)-1 for p in solver.get_x()]
    v = [2*torch.rand(p.shape, generator=generator)-1 for p in solver.get_y()]
    gu, gv = solver.compute_h_bar_gradient(u, v)
    uh = [(p-solver.zeta_hat*g).clamp(-1, 1) for p, g in zip(u, gu)]
    vh = [(p+solver.zeta_hat*g).clamp(-1, 1) for p, g in zip(v, gv)]

    def residual_call():
        ru, rv = solver.compute_stationarity_residual(u, v, uh, vh, gu, gv)
        return compute_norm(ru+rv)

    return {
        'gradient_us': median_call_us(lambda: solver.compute_h_bar_gradient(u, v)),
        # Includes gradient difference / second gradient evaluation and the norm;
        # excludes the first gradient and projection, common to all variants.
        'residual_us': median_call_us(residual_call),
    }


def high_penalty_checks():
    checks = []
    for rho in (25., 125.):
        lip, sigma = .1*rho**3, 1/(80*rho)
        step = sigma/(3*lip+sigma)**2
        for dtype in (torch.float32, torch.float64):
            p, g = torch.tensor([.5], dtype=dtype), torch.ones(1, dtype=dtype)
            ph = (p-step*g).clamp(-1, 1)
            checks.append({
                'rho': rho, 'dtype': str(dtype), 'zeta_hat': step,
                'original_unit_gradient_residual': ((p-ph)/step).item(),
                'stable_unit_gradient_residual': _box_projection_quotient(
                    [p], [g], step, (-1., 1.),
                )[0].item(),
                'reference': 1.,
            })
    return checks


def render_report(data):
    results = data['results']
    baseline = statistics.median([r['seconds'] for r in results['baseline']['trials']])
    combined = statistics.median([r['seconds'] for r in results['combined']['trials']])
    residual = statistics.median([r['seconds'] for r in results['residual_only']['trials']])
    accurate = all(r['status'] == 'terminated' and r['common_stable_residual'] <= TAU
                   and r['box_kkt_residual'] <= TAU
                   for row in results.values() for r in row['trials'])
    conclusion = (
        f'All measured trials terminated and passed the common accuracy checks. '
        f'Combined median runtime was {combined:.3f} s versus {baseline:.3f} s for baseline '
        f'({baseline/combined:.2f}x). Residual-only median runtime changed by '
        f'{100*(residual/baseline-1):+.2f}%; this small sample does not establish a speed '
        f'improvement from the residual reformulation.'
        if accurate else
        'Some trials were incomplete or did not pass the common accuracy checks. '
        'Inspect the statuses and residuals before interpreting runtime differences.'
    )
    lines = [
        '# FOP numerical changes: bounded benchmark', '',
        'This report measures isolated regularized SCSC solves, not complete bilevel experiments. '
        'The baseline uses the original autograd and projection-subtraction path through the updated solver.', '',
        conclusion, '',
        '## Implemented changes', '',
        '- `Minimax_SCSC` now annotates the input coefficients, gradient definitions, and update blocks '
        'against [Algorithm 5, page 25](https://arxiv.org/pdf/2301.01716#page=25). '
        'Existing deviations such as finite caps and the beta multiplier are documented.',
        '- Its optional box residual evaluates the line-25 projection quotients with algebraically equivalent '
        'clipped gradients. The step size, projected iterates, norm, and tolerance are unchanged.',
        '- The shared `LinearCBLOOracle` supplies explicit gradient/difference callbacks for CBLO FOP, SMO, and GCMO, '
        'with regularization supplied by NCWC. AGD warm-ups also use explicit value/gradient callbacks. '
        'Generic callers retain autograd; this benchmark intentionally retains autograd reference variants.',
        '- CBLO always uses explicit gradients and records `gradient_backend="explicit"`. '
        '`FOP_STABLE_BOX_RESIDUAL` remains independently configurable; this benchmark varies both mechanisms explicitly.', '',
        '## Setup', '',
        '- One `(400, 400, 20)` instance, index `1`, seed `0`, generated once and cloned for every trial.',
        '- CPU float64; one Torch thread; initial `x=0`, `y=z=Y_HAT`; regularization anchors equal these initial values.',
        '- `rho=5`, `mu=25`, `L_h=12.5`, `sigma_y=0.0025`, `tau=1e-6`; 100 SCSC outer iterations maximum.',
        '- One untimed warm-up plus three measured solves per variant, each with a 30-second timeout.',
        '- Instance generation and solver construction are outside solve timings. Accuracy checks run afterward; '
        'diagnostic LPs, logging, and W&B are disabled.',
        '- Explicit-gradient variants also use direct new-minus-old gradient differences.',
        '- The inner while loop is unchanged; the benchmark timeout bounds it externally.', '',
        'Environment: `' + json.dumps(data['environment'], sort_keys=True) + '`', '',
        '## Solve runtime', '',
        '| Variant | Median seconds | Range seconds | Baseline / median | Statuses | SCSC outer counts | Inner counts |',
        '|---|---:|---|---:|---|---|---|',
    ]
    for name, _, _ in VARIANTS:
        trials = results[name]['trials']
        times = [r['seconds'] for r in trials]
        valid = all(r['status'] == 'terminated' for r in trials)
        speedup = f'{baseline/statistics.median(times):.2f}x' if valid and all(
            r['status'] == 'terminated' for r in results['baseline']['trials']
        ) else 'n/a (incomplete)'
        lines.append(f"| {name} | {statistics.median(times):.4f} | {min(times):.4f}–{max(times):.4f} | {speedup} | "
                     f"{', '.join(r['status'] for r in trials)} | {', '.join(str(r.get('num_outer_iters', '—')) for r in trials)} | "
                     f"{', '.join(str(r.get('num_inner_iters', '—')) for r in trials)} |")
    lines += ['', '## Accuracy and individual evaluation costs', '',
              'Accuracy columns show the maximum across measured trials. Solution differences are infinity norms '
              'against the first completed baseline trial.', '',
              '| Variant | Reported final residual | Common stable residual | Direct box KKT norm | Solution difference | Gradient µs | Residual µs |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, _, _ in VARIANTS:
        row = results[name]
        def worst(key):
            values = [r[key] for r in row['trials'] if key in r]
            return f'{max(values):.3e}' if values else '—'
        lines.append(f"| {name} | {worst('final_delta')} | {worst('common_stable_residual')} | "
                     f"{worst('box_kkt_residual')} | {worst('solution_linf_difference')} | "
                     f"{row['microbench']['gradient_us']:.2f} | {row['microbench']['residual_us']:.2f} |")
    lines += ['', 'The common stable residual applies one certificate projection from each returned point using '
              'the same explicit gradient/difference oracle and the same step size. The direct box KKT norm is '
              'evaluated at the returned point itself. Neither metric depends on the variant’s termination flag.', '',
              'Evaluation costs are medians of three batches of 200 calls after ten warm-up calls. Residual cost '
              'includes the second gradient or gradient difference and the norm, but excludes the first gradient '
              'and projection. These timings separate evaluation overhead from iteration-count changes.', '',
              '## Higher-penalty cancellation checks', '',
              'For a constant unit gradient at the interior coordinate `0.5`, the exact residual is `1`.', '',
              '| rho | dtype | zeta_hat | Original residual | Stable residual |',
              '|---:|---|---:|---:|---:|']
    for r in data['high_penalty_checks']:
        lines.append(f"| {r['rho']:g} | {r['dtype']} | {r['zeta_hat']:.3e} | "
                     f"{r['original_unit_gradient_residual']:.9g} | {r['stable_unit_gradient_residual']:.9g} |")
    lines += ['', '## Interpretation and limits', '',
              '- The stable residual addresses cancellation; it is not inherently a speed optimization. '
              'Different stopping decisions can change both runtime and iteration counts.',
              '- Explicit gradients reduce evaluation work; the four variants isolate their effect and their interaction with the residual change.',
              '- A short isolated solve from the initial point does not represent the full continuation schedule. '
              'No end-to-end speedup is established here.',
              '- Penalty schedules, Lipschitz overrides, tolerances, AGD, and handling of failed inner solves were not tuned. '
              'The configured curvature override and later-stage conditioning remain separate concerns.',
              '- Timeouts and caps are incomplete solves, not evidence of a speedup. A premature baseline stop must be assessed using the common accuracy metrics.', '',
              '## Reproduction', '',
              'From the repository directory:', '', '```bash',
              f"{data['environment']['python_executable']} benchmark_fop_numerics.py --output-dir reports/fop_numerics", '```', '',
              'Raw repetitions, warm-ups, environment information, and the generated-data checksum are in `results.json`.', '']
    lines += ['## Numerical validation', '',
              'The targeted tests cover 80-digit scalar references, both projection signs, non-unit bounds, '
              'interior/boundary/extrapolated points, tiny steps, explicit gradients and differences near '
              'constraint activation, callback purity and shape validation, FOP integration, and final parameter assignment.', '',
              'Run the new and existing tests separately from benchmarking:', '',
              '```bash', f"{data['environment']['python_executable']} -m unittest discover -s tests -p 'test_*.py'", '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/fop_numerics'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    experiment.WANDB_ENABLED = False
    experiment.SEED = 0
    experiment.generate_instance((400, 400, 20), 1)
    digest = hashlib.sha256()
    for tensor in (experiment.C, experiment.D, experiment.D_TILDE, experiment.A_MAT,
                   experiment.B_MAT, experiment.B_VEC, experiment.Y_HAT):
        digest.update(tensor.numpy().tobytes())
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.processor())
    data = {
        'environment': {'python': platform.python_version(), 'python_executable': sys.executable,
                        'torch': torch.__version__, 'platform': platform.platform(), 'cpu': cpu,
                        'threads': torch.get_num_threads(), 'dtype': 'float64'},
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'instance_sha256': digest.hexdigest(), 'results': {},
        'high_penalty_checks': high_penalty_checks(),
    }
    baseline_solution = None
    for name, stable, explicit in VARIANTS:
        print(f'{name}: warm-up', flush=True)
        warmup, _ = solve_trial(stable, explicit)
        trials = []
        for repetition in range(3):
            trial, solution = solve_trial(stable, explicit)
            if name == 'baseline' and baseline_solution is None and solution is not None:
                baseline_solution = solution.clone()
            if solution is not None and baseline_solution is not None:
                trial['solution_linf_difference'] = (solution-baseline_solution).abs().max().item()
            trials.append(trial)
            print(f"{name} #{repetition+1}: {trial['seconds']:.3f}s {trial['status']}", flush=True)
        data['results'][name] = {'warmup': warmup, 'trials': trials, 'microbench': microbench(stable, explicit)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'results.json').write_text(json.dumps(data, indent=2)+'\n')
    (args.output_dir/'report.md').write_text(render_report(data))
    print(f'Report: {args.output_dir / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
