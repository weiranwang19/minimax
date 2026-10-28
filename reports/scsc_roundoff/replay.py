"""Bounded, in-memory diagnostics; does not modify solver sources or use W&B."""
import sys, inspect, textwrap, json, signal
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import minimax as mm
import exp_cblo as e
from utils import clone_vals, compute_norm
OUT=Path(__file__).resolve().parent
method=sys.argv[1] if len(sys.argv)>1 else 'gcmo'
torch.set_num_threads(1)
e.WANDB_ENABLED=False
e.VERBOSE=False
e.FOP_LIP_SCALE=e.SMO_LIP_SCALE=e.GCMO_LIP_OVERRIDE=None
e.SOLVER_METHOD=method
count=0
class StopDiagnostic(Exception):pass

def save_state(solver, state):
    names=['x_k_t','y_k_t','x_k_m1','y_k_m1','z_g_k','y_g_k','b_x_k_t','b_y_k_t','x_k_0','y_k_0','z_k','y_k']
    tensors={n:clone_vals(state[n]) for n in names}
    closure=inspect.getclosurevars(inspect.unwrap(solver.gradient_func)).nonlocals
    base=inspect.getclosurevars(inspect.unwrap(closure['gradient_func'])).nonlocals
    weights=inspect.getclosurevars(base['weights']).nonlocals
    params={k:getattr(solver,k) for k in ['sigma_x','sigma_y','lip','tau','gamma_x','gamma_y','zeta','zeta_hat','eta_y','alpha_bar']}
    tensors.update(min_anchor=clone_vals(closure['min_anchor']),max_anchor=clone_vals(closure['max_anchor']))
    data=dict(method=method,params=params,state=tensors,oracle={k:getattr(weights['self'],k) for k in ['a','b','rhs','c','d','d_tilde']},rho=weights['rho'],mu=weights.get('mu'),lam=weights.get('lam'))
    torch.save(data,OUT/f'{method}_snapshot.pt')
    return params

def inspect_state(solver,state):
    t,k=state['t'],state['k']
    lhs,rhs=state['lhs'].item(),state['rhs'].item()
    if t==2000 and lhs>rhs and lhs<1e-18:
        params=save_state(solver,state)
        gx,gy=solver.compute_h_bar_gradient(state['x_k_t'],state['y_k_t'])
        norm=lambda a:compute_norm(a).item()
        residual=[]
        for values,grads,ismax in [(state['x_k_t'],gx,False),(state['y_k_t'],gy,True)]:
            for i,(v,g) in enumerate(zip(values,grads)):
                lo,hi=(0.,200.) if method=='gcmo' and i==len(values)-1 else (-1.,1.)
                g=-g if ismax else g
                residual.append(torch.where(v<=lo,g.clamp_max(0),torch.where(v>=hi,g.clamp_min(0),g)))
        row=dict(method=method,call=count,k=k,t=t,lhs=lhs,rhs=rhs,params=params,distance=norm(state['x_diff']+state['y_diff']),a_plus_b=norm(state['x_tmp']+state['y_tmp']),box_kkt=norm(residual),previous_delta=state['final_delta'])
        (OUT/f'{method}_replay.json').write_text(json.dumps(row,indent=2))
        print('STALLED',json.dumps(row),flush=True)
        raise StopDiagnostic
    if t>10000:
        raise RuntimeError(f'Unexpected non-roundoff slow inner solve: {k=} {t=} {lhs=} {rhs=}')

source=textwrap.dedent(inspect.getsource(mm.Minimax_SCSC.run))
source=source.replace('if lhs <= rhs or t > 10000:', 'inspect_state(self, locals())\n            if lhs <= rhs:')
assert 'inspect_state(self, locals())' in source
if len(sys.argv)>2 and sys.argv[2]=='isolated':
    source=source.replace('if delta <= self.tau + 1e-8:', 'if delta <= self.tau:')
ns=dict(vars(mm),inspect_state=inspect_state);exec(source,ns)
run=ns['run']
def wrapped(solver):
    global count
    count+=1
    if count%10==0:print('CALL',count,solver.tau,solver.sigma_x,flush=True)
    return run(solver)
mm.Minimax_SCSC.run=wrapped
size=(400,400,20)
e.generate_instance(size,1)
signal.signal(signal.SIGALRM,lambda *args: (_ for _ in ()).throw(TimeoutError('Replay exceeded 300 s')))
signal.alarm(300)
try:
    if len(sys.argv)>2 and sys.argv[2]=='isolated':
        # Controlled isolated stress test, NOT an exact continuation replay.
        rho=5. if method=='fop' else (100. if method=='gcmo' else .8**-10)
        mu=rho**2 if method=='fop' else rho**3
        sx=(4*rho*mu if method=='fop' else 2*mu)*e.CURRENT_L_G**2
        sy=1/(rho*80)
        if method=='gcmo':
            sx=2*rho*e.CURRENT_L_G
            sy=.01/(2*e.compute_gcmo_d_y(200))
        u=[torch.zeros(400),e.Y_HAT.clone()];v=[e.Y_HAT.clone()]
        if method=='gcmo':
            u.append(torch.zeros(20));v.append(torch.zeros(20))
        oracle=e.LinearCBLOOracle.from_current_instance()
        g,d=oracle.saddle_gradients(method,rho,mu,torch.zeros(20))
        g,d=mm._regularize_gradient_callbacks(g,d,clone_vals(u),clone_vals(v),sx,sy)
        px=[e.prox_box]*2;py=[e.prox_box]
        if method=='gcmo':
            dual=lambda x,c:x.clamp(0,200)
            px.append(dual);py.append(dual)
        solver=mm.Minimax_SCSC(u,v,lambda:None,sx,sy,3*sx+sy,px,py,1e-14,max_iter=2000,gradient_func=g,gradient_difference_func=d,box_bounds=None if method=='gcmo' else (-1.,1.))
        print('ISOLATED',method,flush=True)
        print(wrapped(solver),flush=True)
    else:
        getattr(e,'run_single_instance_'+method)(1,size,1,1)
except StopDiagnostic:pass
