"""Replay the latest stalled subproblem with controlled inner stopping slack.

Run from any directory with the experiment's Python environment. W&B is disabled.
The solver implementation is instrumented in memory; source files are unchanged.
Each variant starts from the same captured subproblem, including its anchors.
"""
import sys, inspect, textwrap, json, time
from pathlib import Path
output_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(output_dir.parents[1]))
import torch
import exp_cblo as e
import minimax as mm
from utils import clone_vals, assign_vals, compute_norm

torch.set_num_threads(1)
e.WANDB_ENABLED=False
e.VERBOSE=False
original = mm.Minimax_SCSC.run
records=[]
source=textwrap.dedent(inspect.getsource(original))
assert source.count('if lhs <= rhs + 1e-8:') == 1
source=source.replace('if lhs <= rhs + 1e-8:', 'if lhs <= rhs + self.debug_slack:')
source=source.replace('final_delta = float(delta.item())', '''final_delta = float(delta.item())
        if k < 3 or k % 100 == 0 or final_delta <= self.tau + 1e-8:
            self.debug_records.append(dict(k=k, t=t, lhs=lhs.item(), rhs=rhs.item(), delta=final_delta,
                dx=compute_norm(delta_x).item(), dy=compute_norm(delta_y).item(),
                max_outside=max((v.abs()-1).clamp_min(0).max().item() for v in y_kp1),
                gradient_y=compute_norm(y_grad).item()))''')
ns=dict(vars(mm)); exec(source,ns); diagnostic_run=ns['run']
class Finished(Exception): pass

def intercept(solver):
    if abs(solver.tau - .2**2.5/5)>1e-12:
        result=original(solver)
        print('preceding',solver.sigma_x,solver.tau,result,flush=True)
        return result
    init=clone_vals(solver.get_x()+solver.get_y())
    print('TARGET',dict(sigma_x=solver.sigma_x,sigma_y=solver.sigma_y,lip=solver.lip,tau=solver.tau,alpha=solver.alpha_bar,zeta_hat=solver.zeta_hat),flush=True)
    print('CURVATURE_BOUNDS',dict(A_norm_sq=torch.linalg.matrix_norm(e.A_MAT,ord=2).square().item(),AB_norm_sq=e.CURRENT_L_G**2, min_curvature_bound=2*solver.sigma_x-250*torch.linalg.matrix_norm(e.A_MAT,ord=2).square().item(),computed_lip=500*e.CURRENT_L_G**2),flush=True)
    for slack in (1e-8, 1e-10, 1e-12, 0.):
        assign_vals(solver.get_x()+solver.get_y(),init)
        solver.max_iter=1500
        solver.debug_slack=slack
        solver.debug_records=[]
        start=time.monotonic()
        stats=diagnostic_run(solver)
        row=dict(slack=slack,stats=stats,seconds=time.monotonic()-start,trace=solver.debug_records)
        records.append(row)
        print('VARIANT',json.dumps(row),flush=True)
        with open(output_dir / 'results.json','w') as f:json.dump(records,f,indent=2)
    raise Finished
mm.Minimax_SCSC.run=intercept
e.generate_instance((400,400,20),1)
try:e.run_single_instance_fop(1,(400,400,20),1,1)
except Finished:pass
