"""Convert logged squared residuals to distances and stationarity bounds."""
from pathlib import Path
import sys,json,re,math
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import exp_cblo as e
OUT=Path(__file__).resolve().parent
torch.set_num_threads(1)
rows=json.loads((OUT/'log_evidence.json').read_text());results=[]
for r in rows:
 c=r['config'];method=c['solver_method'];size=tuple(c['problem_size'][k] for k in ('n','m','l'))
 e.generate_instance(size,c['instance_idx'])
 summary=json.loads((ROOT/'wandb'/r['run']/'files/wandb-summary.json').read_text())
 if method=='fop':
  stage=int(summary['fop/stage/index'])+1;rho=c['base_rho']**(stage-1);eps=1/rho
  sx=4*rho**3*e.CURRENT_L_G**2;sy=eps/(4*math.sqrt(size[1]))
 elif method=='smo':
  stage=int(summary['smo/stage/index'])+1;eps=c['smo_epsilon_0']*c['smo_tau']**stage
  sx=2*eps**-3*e.CURRENT_L_G**2;sy=eps/(4*math.sqrt(size[1]))
 else:
  eps=c['gcmo_eps'];sx=2/eps*e.CURRENT_L_G;sy=eps/(2*e.compute_gcmo_d_y(c['gcmo_lagrange_bound']))
 gamma=8/sx
 lhs,rhs=map(float,re.search(r'lhs=(\S+) rhs=(\S+)',r['last_inner']).groups())
 tau=float(re.search(r'tau=(\S+)',r['last_outer'])[1])
 dist=math.sqrt(gamma*rhs);res=math.sqrt(lhs/gamma)
 results.append(dict(run=r['run'],method=method,size=size,lhs=lhs,rhs=rhs,ratio=lhs/rhs,sigma_x=sx,sigma_y=sy,gamma=gamma,distance=dist,inner_residual=res,stationarity_upper_bound=res+sx/2*dist,tau=tau,certificate_step=sy/(3*sx+sy)**2))
(OUT/'log_analysis.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
