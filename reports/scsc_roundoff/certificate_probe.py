"""Compare inner-point stationarity with the extrapolated output certificate."""
from pathlib import Path
import sys,json
import torch
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parents[1]))
import minimax as mm
from exp_cblo import LinearCBLOOracle
from utils import compute_norm
method=sys.argv[1]
d=torch.load(OUT/f'{method}_snapshot.pt',weights_only=False);s=d['state'];p=d['params']
o=d['oracle'];oracle=LinearCBLOOracle(**o)
g,diff=oracle.saddle_gradients(method,d['rho'],d['mu'],d['lam'])
g,diff=mm._regularize_gradient_callbacks(g,diff,s['min_anchor'],s['max_anchor'],p['sigma_x'],p['sigma_y'])

def bounds(vals):return [(0,200) if method=='gcmo' and i==len(vals)-1 else (-1,1) for i in range(len(vals))]
def proj(vals):return [v.clamp(*b) for v,b in zip(vals,bounds(vals))]
def norm(vals):return compute_norm(vals).item()
def check(u,v):
    gx,gy=g(u,v);eta=p['zeta_hat']
    uh=proj([x-eta*a for x,a in zip(u,gx)]);vh=proj([x+eta*a for x,a in zip(v,gy)])
    dx,dy=diff(u,v,uh,vh)
    generic=[(x-y)/eta+a for x,y,a in zip(u,uh,dx)]+[(y-x)/eta+a for x,y,a in zip(v,vh,dy)]
    stable=[]
    for vals,gs,ds,ismax in [(u,gx,dx,False),(v,gy,dy,True)]:
        for x,a,da,b in zip(vals,gs,ds,bounds(vals)):
            stable+=mm._box_projection_quotient([x],[a],eta,b,ismax)
            stable[-1]+=da
    gx,gy=g(uh,vh);kkt=[]
    for vals,gs in [(uh,gx),(vh,[-a for a in gy])]:
        for x,a,(lo,hi) in zip(vals,gs,bounds(vals)):
            kkt.append(torch.where(x<=lo,a.clamp_max(0),torch.where(x>=hi,a.clamp_min(0),a)))
    return dict(generic_certificate=norm(generic),stable_certificate=norm(stable),projected_point_kkt=norm(kkt),max_infeasibility=max(max((lo-x).clamp_min(0).max().item(),(x-hi).clamp_min(0).max().item()) for vals in (u,v) for x,(lo,hi) in zip(vals,bounds(vals))))
xf,yf=s['x_k_t'],s['y_k_t'];gx,gy=g(xf,yf);sx,sy,eta=p['sigma_x'],p['sigma_y'],p['eta_y']
zf=[a-sx*x+b for a,x,b in zip(gx,xf,s['b_x_k_t'])];wf=[-(a+sy*y)+b for a,y,b in zip(gy,yf,s['b_y_k_t'])]
z=[z+.5*(f-z)-sx/2*(x+f/sx) for z,f,x in zip(s['z_k'],zf,xf)]
x=[-z/sx for z in z];y=[y+eta*sy*(f-y)-eta*(w+sy*f) for y,f,w in zip(s['y_k'],yf,wf)]
# Same outer updates after algebraic cancellation.
xs=[.5*(-z/sx+f) for z,f in zip(s['z_k'],xf)]
ys=[y+eta*sy*(f-y)+eta*(a-b) for y,f,a,b in zip(s['y_k'],yf,gy,s['b_y_k_t'])]
result=dict(method=method,inner=check(xf,yf),outer_original=check(x,y),outer_simplified=check(xs,ys))
(OUT/f'{method}_certificates.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
