"""Controlled replay of the fifth SCSC call in run aez6mm3i; no source edits/W&B."""
import sys,inspect,textwrap,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import minimax as mm
import exp_cblo as e
from utils import clone_vals,assign_vals,compute_norm
OUT=Path(__file__).resolve().parent
torch.set_num_threads(1)
e.WANDB_ENABLED=False;e.VERBOSE=False;e.SOLVER_METHOD='gcmo';e.GCMO_LIP_OVERRIDE=None
original=mm.Minimax_SCSC.run
records=[];count=0
class Finished(Exception):pass
class InnerStalled(Exception):pass

def kkt(solver,x,y):
    gx,gy=solver.compute_h_bar_gradient(x,y);res=[]
    for vals,gs in [(x,gx),(y,[-g for g in gy])]:
        for i,(v,g) in enumerate(zip(vals,gs)):
            lo,hi=(0.,200.) if i==len(vals)-1 else (-1.,1.)
            res.append(torch.where(v<=lo,g.clamp_max(0),torch.where(v>=hi,g.clamp_min(0),g)))
    return compute_norm(res).item()

def early(solver,frame):
    if frame['t']>3000:raise InnerStalled
    if solver.mode=='certified' and frame['t']%20==0:
        delta=kkt(solver,frame['x_k_t'],frame['y_k_t'])
        if delta<=solver.tau:
            solver.assign_x(frame['x_k_t']);solver.assign_y(frame['y_k_t'])
            return dict(num_outer_iters=frame['k']+1,num_inner_iters=frame['num_inner_iters']+frame['t'],final_delta=delta,terminated=True,certificate='inner_box_kkt')
    return None

def record(solver,frame):
    if frame['k']%100==0 or frame['final_delta']<=solver.tau:
        solver.trace.append(dict(k=frame['k'],t=frame['t'],lhs=frame['lhs'].item(),rhs=frame['rhs'].item(),outer_delta=frame['final_delta'],inner_kkt=kkt(solver,frame['x_k_t'],frame['y_k_t'])))

src=textwrap.dedent(inspect.getsource(original))
old='if lhs <= rhs or lhs + rhs <= 1e-12:'
assert old in src
src=src.replace(old,"result = early(self, locals())\n            if result is not None:\n                return result\n            if lhs <= rhs or (self.mode == 'baseline' and lhs + rhs <= 1e-12):")
src=src.replace('num_outer_iters = k + 1','record(self, locals())\n        num_outer_iters = k + 1')
ns=dict(vars(mm),early=early,record=record);exec(src,ns);instrumented=ns['run']

def wrapped(solver):
    global count
    count+=1
    if count<5:
        stats=original(solver);print('PRECEDING',count,stats,flush=True);return stats
    init=clone_vals(solver.get_x()+solver.get_y())
    for mode in ['baseline','strict','certified']:
        assign_vals(solver.get_x()+solver.get_y(),init)
        solver.max_iter=1000;solver.mode=mode;solver.trace=[]
        start=time.monotonic()
        try:stats=instrumented(solver)
        except InnerStalled:stats={'status':'inner_stalled_at_3001'}
        row=dict(mode=mode,stats=stats,seconds=time.monotonic()-start,params={k:getattr(solver,k) for k in ['tau','sigma_x','sigma_y','eta_y','zeta_hat']},trace=solver.trace)
        records.append(row)
        print('VARIANT',json.dumps(row),flush=True)
        (OUT/'results.json').write_text(json.dumps(records,indent=2))
    raise Finished
mm.Minimax_SCSC.run=wrapped
e.generate_instance((100,100,5),1)
signal.signal(signal.SIGALRM,lambda *args: (_ for _ in ()).throw(TimeoutError('300 s budget')));signal.alarm(300)
try:e.run_single_instance_gcmo(1,(100,100,5),1,1)
except Finished:pass
