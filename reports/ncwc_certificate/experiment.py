"""Compare NCWC stopping rules while retaining the production SCSC solver.

Only an in-memory copy of optimize_NCWC changes. No W&B runs are created.
Every returned stage is independently checked with autograd. A timeout or
uncertified inner solve is reported as incomplete, never as convergence.
"""
import argparse
import inspect
import json
import math
from pathlib import Path
import signal
import sys
import time
from functools import partial
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bilevel_solvers as bilevel
import exp_cblo as exp
import minimax as mm

OUT = Path(__file__).resolve().parent


def box_kkt(values, gradients, bounds, maximize=False):
    residual = []
    for value, grad, (lo, hi) in zip(values, gradients, bounds):
        assert torch.isfinite(value).all() and torch.isfinite(grad).all()
        assert ((value >= lo) & (value <= hi)).all()
        r = grad.detach().clone() * (-1 if maximize else 1)
        r[(value == lo) & (r >= 0)] = 0
        r[(value == hi) & (r <= 0)] = 0
        residual.append(r.flatten())
    return torch.linalg.vector_norm(torch.cat(residual)).item()


def autograd_kkt(h, mins, maxs, bounds):
    params = mins + maxs
    flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(True)
        gradients = torch.autograd.grad(h(), params)
        n = len(mins)
        return (box_kkt(mins, gradients[:n], bounds[0]),
                box_kkt(maxs, gradients[n:], bounds[1], True))
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


class Audit:
    def __init__(self, rule):
        self.rule = rule
        self.calls = []
        source = inspect.getsource(mm.optimize_NCWC)
        old = 'ncwc_terminated = exit_criteria_gap <= 0'
        assert source.count(old) == 1
        namespace = dict(vars(mm), probe=self.probe)
        exec(compile(source.replace(old, 'ncwc_terminated = probe(locals())'),
                     '<NCWC certificate experiment>', 'exec'), namespace)
        self.instrumented = namespace['optimize_NCWC']

    def run(self, *args, **kwargs):
        self.calls.append({'iterations': [], 'complete': False})
        result = self.instrumented(*args, **kwargs)
        self.calls[-1]['complete'] = bool(result['terminated'])
        if not result['terminated']:
            raise RuntimeError('NCWC iteration budget exhausted')
        return result

    def probe(self, s):
        stats = s['solver_stats']
        if not stats['terminated']:
            raise RuntimeError('Uncertified SCSC subproblem: refusing to advance the experiment')
        u, v = s['params_x'], s['params_y']
        bounds = mm._normalize_box_bounds(s['box_bounds'], len(u), len(v))
        assert bounds[0] is not None
        gu, gv = s['gradient_func'](u, v)
        ru, rv = box_kkt(u, gu, bounds[0]), box_kkt(v, gv, bounds[1], True)
        epsilon, tau, lip, step = s['epsilon'], s['epsilon_k'], s['lip_h'], s['final_diff']
        bu, bv = tau + 2*lip*step, tau + epsilon/2
        old_pass = bool(s['exit_criteria_gap'] <= 0)
        bound_pass = max(bu, bv) <= epsilon
        direct_pass = max(ru, rv) <= epsilon
        # Allow arithmetic roundoff only in this diagnostic inequality check,
        # not in any acceptance threshold.
        assert ru <= bu + 1e-9*max(1., bu)
        assert rv <= bv + 1e-9*max(1., bv)
        passed = {'baseline': old_pass, 'bound': bound_pass,
                  'direct': direct_pass, 'joint': math.hypot(ru, rv) <= epsilon}[self.rule]
        row = dict(iteration=s['k']+1, epsilon=epsilon, tau=tau, lip=lip,
                   step=step, threshold=epsilon/(4*lip), step_ratio=step/(epsilon/(4*lip)),
                   r_min=ru, r_max=rv, r_joint=math.hypot(ru, rv),
                   bound_min=bu, bound_max=bv, baseline_pass=old_pass,
                   bound_pass=bound_pass, direct_pass=direct_pass,
                   scsc_delta=stats['final_delta'], scsc_outer=stats['num_outer_iters'],
                   scsc_inner=stats['num_inner_iters'], stop=passed)
        self.calls[-1]['iterations'].append(row)
        if passed:
            independent = autograd_kkt(s['h_func'], u, v, bounds)
            row['autograd_min'], row['autograd_max'] = independent
            assert max(independent) <= epsilon, (independent, epsilon)
            assert abs(ru-independent[0]) <= 1e-8*max(1., ru)
            assert abs(rv-independent[1]) <= 1e-8*max(1., rv)
        return passed

    def summary(self):
        rows = [r for c in self.calls for r in c['iterations']]
        return dict(ncwc_iterations=len(rows), scsc_inner=sum(r['scsc_inner'] for r in rows),
                    scsc_outer=sum(r['scsc_outer'] for r in rows),
                    completed_calls=sum(c['complete'] for c in self.calls))


def analytic(audit, case):
    u, v = torch.tensor([1.]), torch.tensor([0.])
    if case == 'linear':
        h = lambda: (.075*u-.5*v.square()).sum()
        grad = lambda x, y: ([torch.full_like(x[0], .075)], [-y[0]])
    elif case == 'boundary':
        u.zero_()
        h = lambda: (-10*u-.5*v.square()).sum()
        grad = lambda x, y: ([torch.full_like(x[0], -10)], [-y[0]])
    else:
        h = lambda: (.5*u.square()-.5*v.square()).sum()
        grad = lambda x, y: ([x[0]], [-y[0]])
    def difference(x, y, xx, yy):
        g, gy = grad(x, y)
        gg, ggy = grad(xx, yy)
        return [gg[0]-g[0]], [ggy[0]-gy[0]]
    result = audit.run([u], [v], h, 1., 2., [exp.prox_box], [exp.prox_box],
                       .1, .001, max_iter=1000, gradient_func=grad,
                       gradient_difference_func=difference, box_bounds=(-1., 1.))
    return dict(stats=result, u=u.item(), v=v.item(), objective=h().item())


def smo(audit, size, index, epsilon, stages):
    exp.generate_instance(tuple(size), index)
    oracle = exp.LinearCBLOOracle.from_current_instance()
    x, y = torch.zeros(size[0]), exp.Y_HAT.clone()

    def stage(p):
        xval, yval = p['x'][0], p['y'][0]
        lower_star = exp.solve_lower_level_value(xval)
        row = dict(stage=p['stage_index'], epsilon=p['epsilon_k'],
                   upper=exp.upper_objective(xval, yval), feas=exp.feasibility_norm(xval, yval),
                   lower_gap=exp.lower_objective(xval, yval)-lower_star,
                   ncwc=p['subproblem_stats']['num_outer_iters'],
                   scsc_inner=p['subproblem_stats']['num_inner_iters'],
                   lambda_norm=p['lambda_norm'], next_lambda_norm=p['next_lambda_norm'],
                   certificate=audit.calls[-1]['iterations'][-1])
        stages.append(row)
        print(json.dumps({'rule': audit.rule, **{k: row[k] for k in ('stage','epsilon','ncwc','scsc_inner','upper','feas','lower_gap')}}), flush=True)

    with mock.patch.object(bilevel, 'optimize_NCWC', audit.run):
        result = bilevel.optimize_bilevel_constrained_smo(
            [x], [y], exp.upper_smooth, exp.lower_smooth, exp.lower_constraints,
            exp.prox_box, exp.prox_box, D_y=exp.compute_d_y(),
            L_grad_f1=0., L_grad_ftilde1=0., L_grad_gtilde=0.,
            L_gtilde=exp.CURRENT_L_G, gtilde_hi=exp.compute_gtilde_hi(),
            epsilon=epsilon, epsilon_0=1., tau=.8, subproblem_max_iter=10000,
            warm_start_max_iter=None, warm_start_linear_minimizer=exp.box_linear_minimizer,
            saddle_gradient_factory=partial(oracle.saddle_gradients, 'smo'),
            warm_start_value_and_grad_factory=partial(oracle.warm_start_value_and_grad, 'smo'),
            box_bounds=(-1., 1.), stage_callback=stage,
        )
    return dict(terminated=result['terminated'], x=x.tolist(), y=y.tolist(),
                z=result['z_eps'][0].tolist(), multipliers=result['lambda'].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['analytic','smo'], default='analytic')
    parser.add_argument('--case', choices=['linear','boundary','quadratic'], default='linear')
    parser.add_argument('--rule', choices=['baseline','bound','direct','joint'], default='baseline')
    parser.add_argument('--size', type=int, nargs=3, default=[4,4,2])
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=.05)
    parser.add_argument('--seconds', type=int, default=180)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    audit, stages = Audit(args.rule), []
    start = time.monotonic()
    def timeout(*_):
        raise TimeoutError(f'{args.seconds} second experiment budget exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    data = dict(config=vars(args), status='complete')
    try:
        data['result'] = (analytic(audit, args.case) if args.kind == 'analytic'
                          else smo(audit, args.size, args.index, args.epsilon, stages))
    except (TimeoutError, RuntimeError) as error:
        data.update(status='incomplete', error=str(error))
    finally:
        signal.alarm(0)
    data.update(seconds=time.monotonic()-start, summary=audit.summary(), stages=stages, calls=audit.calls)
    label = args.case if args.kind == 'analytic' else '-'.join(map(str,args.size))+f'-idx{args.index}-eps{args.epsilon}'
    path = OUT/f'{args.kind}-{label}-{args.rule}.json'
    path.write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({k: v for k,v in data.items() if k not in ('stages','calls','result')}), flush=True)


if __name__ == '__main__':
    main()
