"""Capture the exact small FOP stage-4 input and check an independent reference.

No W&B runs or production-setting edits. Reuses a saved checkpoint on later
invocations. The prefix is verified against the recorded production history.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
from functools import partial
from unittest import mock

import numpy as np
from scipy.optimize import root, least_squares
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import exp_cblo as exp
import bilevel_solvers as bilevel
import minimax as mm
from utils import clone_vals
sys.path.insert(0, str(ROOT/'reports/latest_four'))
from analyze import history

OUT = Path(__file__).resolve().parent


class Captured(Exception):
    pass


def capture():
    checkpoint = OUT/'fop_stage4.pt'
    if checkpoint.exists():
        return torch.load(checkpoint, weights_only=False)
    exp.generate_instance((5, 5, 2), 1)
    oracle = exp.LinearCBLOOracle.from_current_instance()
    recorded = [r for r in history(ROOT/'wandb/run-20261104_150801-7jn3nd4d')
                if r.get('ncwc/completed_outer_iters', 0) > 0 and r['ncwc/inner_terminated']]
    original = mm.Minimax_SCSC.run
    matched = []
    start = time.monotonic()

    def run(solver):
        if solver.sigma_x > 1000:
            data = dict(mins=clone_vals(solver.get_x()), maxs=clone_vals(solver.get_y()),
                        params={name: getattr(solver, name) for name in ('sigma_x','sigma_y','lip','tau')},
                        oracle={name: getattr(oracle, name) for name in ('a','b','rhs','c','d','d_tilde')},
                        rho=125., mu=15625., matched_prefix=matched,
                        prefix_seconds=time.monotonic()-start)
            torch.save(data, checkpoint)
            print(json.dumps({'captured': True, 'matched_prefix': len(matched),
                              'seconds': data['prefix_seconds'], 'params': data['params']}), flush=True)
            raise Captured
        return original(solver)

    def progress(p):
        if not p['completed_outer_iters']:
            return
        old = recorded[len(matched)]
        assert p['final_diff'] == old['ncwc/final_diff']
        assert p['scsc_num_inner_iters'] == old['ncwc/scsc_inner_iters']
        assert p['final_residual_min'] == old['ncwc/final_residual_min']
        matched.append(dict(step=p['final_diff'], inner=p['scsc_num_inner_iters']))
        if len(matched) % 10 == 0:
            print(json.dumps({'matched': len(matched), 'seconds': time.monotonic()-start,
                              'residual': p['final_residual_min']}), flush=True)

    x, y = torch.zeros(5), exp.Y_HAT.clone()
    with mock.patch.object(mm.Minimax_SCSC, 'run', run):
        try:
            bilevel.optimize_bilevel_contrained_fop_practical(
                [x], [y], exp.upper_smooth, exp.lower_smooth, exp.lower_constraints,
                exp.prox_box, exp.prox_box, D_y=exp.compute_d_y(), L_grad_f1=0.,
                L_grad_ftilde1=0., L_grad_gtilde=0., L_gtilde=exp.CURRENT_L_G,
                gtilde_hi=exp.compute_gtilde_hi(), base_rho=5., final_epsilon=.01,
                subproblem_max_iter=1000, max_outer_iters=200, scsc_max_iter=5000,
                scsc_max_inner_iter=10000, warm_start_max_iter=None,
                warm_start_linear_minimizer=exp.box_linear_minimizer,
                saddle_gradient_factory=partial(oracle.saddle_gradients, 'fop'),
                warm_start_value_and_grad_factory=partial(oracle.warm_start_value_and_grad, 'fop'),
                box_bounds=(-1., 1.), progress_callback=progress,
            )
        except Captured:
            pass
    assert checkpoint.exists()
    return torch.load(checkpoint, weights_only=False)


def objective(data, mins, maxs):
    o, rho, mu, p = data['oracle'], data['rho'], data['mu'], data['params']
    x, y = mins; z, = maxs
    ry, rz = o['a']@x+o['b']@y-o['rhs'], o['a']@x+o['b']@z-o['rhs']
    return (o['c']@x+o['d']@y+rho*(o['d_tilde']@y-o['d_tilde']@z)
            +rho*mu*(ry.relu().square().sum()-rz.relu().square().sum())
            +p['sigma_x']*sum((v-a).square().sum() for v,a in zip(mins,data['mins']))
            -p['sigma_y']/2*sum((v-a).square().sum() for v,a in zip(maxs,data['maxs'])))


def independent_kkt(data, mins, maxs, regularized=True):
    u, v = clone_vals(mins, True), clone_vals(maxs, True)
    value = objective(data, u, v)
    if not regularized:
        value = (value-data['params']['sigma_x']*sum((p-a).square().sum() for p,a in zip(u,data['mins']))
                 +data['params']['sigma_y']/2*sum((p-a).square().sum() for p,a in zip(v,data['maxs'])))
    gradients = torch.autograd.grad(value, u+v)
    residuals = []
    for i, (value, grad) in enumerate(zip(u+v, gradients)):
        assert ((value >= -1) & (value <= 1)).all()
        r = grad.detach().clone() * (-1 if i >= len(u) else 1)
        r[(value == -1) & (r >= 0)] = 0
        r[(value == 1) & (r <= 0)] = 0
        residuals.append(r)
    return dict(min=torch.linalg.vector_norm(torch.cat(residuals[:len(u)])).item(),
                max=torch.linalg.vector_norm(torch.cat(residuals[len(u):])).item(),
                joint=torch.linalg.vector_norm(torch.cat(residuals)).item())


def reference(data):
    # Eliminate primal coordinates via their exact box minimizers/maximizers.
    # The remaining four dual weights obey alpha=2c[ry]+, beta=2c[rz]+.
    o = {k:v.numpy() for k,v in data['oracle'].items()}
    a,b,rhs = o['a'],o['b'],o['rhs']
    x0,y0 = (v.numpy() for v in data['mins']); z0 = data['maxs'][0].numpy()
    rho,c = data['rho'],data['rho']*data['mu']
    lip,sigma = data['params']['sigma_x'],data['params']['sigma_y']
    m = len(rhs)

    def coordinates(w):
        alpha,beta = w[:m],w[m:]
        x = np.clip(x0-(o['c']+a.T@(alpha-beta))/(2*lip),-1,1)
        y = np.clip(y0-(o['d']+rho*o['d_tilde']+b.T@alpha)/(2*lip),-1,1)
        z = np.clip(z0+(-rho*o['d_tilde']-b.T@beta)/sigma,-1,1)
        return x,y,z

    def fun(w):
        x,y,z = coordinates(w)
        return w/(2*c)-np.maximum(np.r_[a@x+b@y-rhs,a@x+b@z-rhs],0)

    def jac(w):
        x,y,z = coordinates(w)
        ax = (a*((x > -1)&(x < 1)))@a.T/(2*lip)
        by = (b*((y > -1)&(y < 1)))@b.T/(2*lip)
        bz = (b*((z > -1)&(z < 1)))@b.T/sigma
        masks = np.r_[a@x+b@y-rhs,a@x+b@z-rhs] > 0
        derivative = np.block([[-ax-by,ax],[-ax,ax-bz]])
        return np.eye(2*m)/(2*c)-masks[:,None]*derivative

    start = time.monotonic()
    candidates = []
    initials = [np.zeros(2*m), 2*c*np.maximum(np.r_[a@x0+b@y0-rhs,a@x0+b@z0-rhs],0)]
    if 'warm_dual' in data:
        initials.insert(0,np.asarray(data['warm_dual']))
    for initial in initials:
        sol = root(fun, initial, jac=jac, options={'xtol':1e-11,'maxfev':2000})
        candidates.append(sol.x)
    best = min(candidates,key=lambda w:np.linalg.norm(fun(w)))
    if np.linalg.norm(fun(best)) > 1e-12:
        sol = least_squares(fun,best,jac=jac,xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=10000)
        best = sol.x
    x,y,z = (torch.from_numpy(v.copy()) for v in coordinates(best))
    kkt = independent_kkt(data,[x,y],[z])
    return dict(seconds=time.monotonic()-start,dual_equation_residual=float(np.linalg.norm(fun(best))),
                kkt=kkt,passes=kkt['joint'] <= data['params']['tau'],
                original_kkt=independent_kkt(data,[x,y],[z],regularized=False),
                dual_weights=best.tolist(),
                mins=[x.tolist(),y.tolist()],maxs=[z.tolist()])


def probe(data, count, initial=None):
    mins,maxs = (clone_vals(data['mins']),clone_vals(data['maxs'])) if initial is None else (
        [torch.tensor(v) for v in initial['mins']], [torch.tensor(v) for v in initial['maxs']])
    oracle = exp.LinearCBLOOracle(**data['oracle'])
    grad,diff = oracle.saddle_gradients('fop',data['rho'],data['mu'])
    p = data['params']
    grad,diff = mm._regularize_gradient_callbacks(grad,diff,data['mins'],data['maxs'],p['sigma_x'],p['sigma_y'])
    solver = mm.Minimax_SCSC(mins,maxs,lambda:objective(data,mins,maxs),
                             **p,prox_x=[exp.prox_box]*2,prox_y=[exp.prox_box],max_iter=count,
                             max_inner_iter=10000,gradient_func=grad,gradient_difference_func=diff,
                             box_bounds=(-1.,1.))
    trace = []
    original_residual = solver.compute_stationarity_residual
    def residual(xval,yval,xhat,yhat,gx,gy):
        result = original_residual(xval,yval,xhat,yhat,gx,gy)
        kkt = independent_kkt(data,xhat,yhat)
        outside = torch.linalg.vector_norm(torch.cat([
            v-v.clamp(-1,1) for v in xval+yval])).item()
        trace.append(dict(outer=len(trace), projected_residual=mm.compute_norm(result[0]+result[1]).item(),
                          feasible_kkt=kkt,outside_box_distance=outside))
        if len(trace) in (1,101,501,1000):
            print(json.dumps({'probe':trace[-1]}),flush=True)
        return result
    solver.compute_stationarity_residual = residual
    start = time.monotonic(); stats = solver.run()
    return dict(stats=stats,seconds=time.monotonic()-start,kkt=independent_kkt(data,mins,maxs),trace=trace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-outers',type=int,default=1000)
    parser.add_argument('--output',default='results.json')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    def timeout(*_):
        raise TimeoutError('Diagnostic phase exceeded its wall-time budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(650)
    data = capture(); signal.alarm(180)
    result = dict(params=data['params'],matched_prefix=len(data['matched_prefix']),
                  reference=reference(data))
    print(json.dumps({'reference':result['reference']}),flush=True)
    if result['reference']['passes']:
        result['restart_from_reference'] = probe(data,10,result['reference'])
    result[f'production_{args.probe_outers}'] = probe(data,args.probe_outers)
    signal.alarm(0)
    (OUT/args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'saved':args.output, 'reference_passes':result['reference']['passes']}),flush=True)


if __name__ == '__main__':
    main()
