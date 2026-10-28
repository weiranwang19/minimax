"""Continue captured fixed inner problems in float64 and extended precision."""
from pathlib import Path
import json, sys
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
method=sys.argv[1]
data=torch.load(OUT/f'{method}_snapshot.pt',weights_only=False)
p=data['params']

def probe(dtype, stable_a=False):
    cast=lambda t: np.asarray(t.numpy(),dtype=dtype)
    s={k:[cast(x).copy() for x in v] for k,v in data['state'].items()}
    o={k:cast(v) for k,v in data['oracle'].items()}
    sx,sy,step=dtype(p['sigma_x']),dtype(p['sigma_y']),dtype(p['zeta'])*dtype(p['gamma_x'])
    gamma=dtype(p['gamma_x']);rho=dtype(data['rho'])
    mu=None if data['mu'] is None else dtype(data['mu'])
    lam=None if data['lam'] is None else cast(data['lam'])
    def grad(u,v):
        x,y=u[:2];z=v[0]
        if method=='gcmo':wy,wz=rho*v[1],rho*u[2]
        else:
            ry=o['a']@x+o['b']@y-o['rhs'];rz=o['a']@x+o['b']@z-o['rhs']
            if method=='fop':wy,wz=2*rho*mu*np.maximum(ry,0),2*rho*mu*np.maximum(rz,0)
            else:wy,wz=np.maximum(lam+mu*ry,0),np.maximum(lam+mu*rz,0)
        gx=[o['c']+o['a'].T@(wy-wz),o['d']+rho*o['d_tilde']+o['b'].T@wy]
        gy=[-rho*o['d_tilde']-o['b'].T@wz]
        if method=='gcmo':
            gx.append(-rho*(o['a']@x+o['b']@z-o['rhs']))
            gy.append(rho*(o['a']@x+o['b']@y-o['rhs']))
        return ([g+2*sx*(x-a) for g,x,a in zip(gx,u,s['min_anchor'])],
                [g-sy*(y-a) for g,y,a in zip(gy,v,s['max_anchor'])])
    def operator(u,v):
        gx,gy=grad(u,v)
        if stable_a:
            # Algebraically equivalent to the paper with x_m1=-z_g/sigma_x.
            return ([g-sx/2*(x-a) for g,x,a in zip(gx,u,s['x_k_m1'])],
                    [-g+sx/8*(y-a) for g,y,a in zip(gy,v,s['y_g_k'])])
        return ([g-sx*x+sx/2*(x-z/sx) for g,x,z in zip(gx,u,s['z_g_k'])],
                [-(g+sy*y)+(sy*y+sx/8*(y-a)) for g,y,a in zip(gy,v,s['y_g_k'])])
    def bounds(vals):return [(0,200) if method=='gcmo' and i==len(vals)-1 else (-1,1) for i in range(len(vals))]
    def project(vals):return [np.clip(v,*b) for v,b in zip(vals,bounds(vals))]
    def sq(vals):return sum(np.sum(v*v) for v in vals)
    def kkt(u,v):
        gx,gy=grad(u,v);r=[]
        for vals,gs in [(u,gx),(v,[-g for g in gy])]:
            for x,g,(lo,hi) in zip(vals,gs,bounds(vals)):
                r.append(np.where(x<=lo,np.minimum(g,0),np.where(x>=hi,np.maximum(g,0),g)))
        return float(np.sqrt(sq(r)))
    u,v=s['x_k_t'],s['y_k_t'];bx,by=s['b_x_k_t'],s['b_y_k_t']
    rows=[]
    for j in range(10001):
        ax,ay=operator(u,v);rx=[a+b for a,b in zip(ax,bx)];ry=[a+b for a,b in zip(ay,by)]
        lhs=gamma*sq(rx+ry);rhs=(sq([x-a for x,a in zip(u,s['x_k_m1'])])+sq([x-a for x,a in zip(v,s['y_k_m1'])]))/gamma
        if j in (0,1,10,100,1000,10000) or lhs<=rhs:
            rows.append(dict(extra_steps=j,lhs=float(lhs),rhs=float(rhs),kkt=kkt(u,v),passed=bool(lhs<=rhs)))
        if lhs<=rhs:break
        beta=dtype(2)/dtype(2000+j+3)
        bu=[x+beta*(x0-x) for x,x0 in zip(u,s['x_k_0'])];bv=[x+beta*(x0-x) for x,x0 in zip(v,s['y_k_0'])]
        uh=[x-step*r for x,r in zip(bu,rx)];vh=[x-step*r for x,r in zip(bv,ry)]
        ahx,ahy=operator(uh,vh)
        pu=[x-step*a for x,a in zip(bu,ahx)];pv=[x-step*a for x,a in zip(bv,ahy)]
        un,vn=project(pu),project(pv)
        bx=[(a-b)/step for a,b in zip(pu,un)];by=[(a-b)/step for a,b in zip(pv,vn)]
        u,v=un,vn
    return dict(dtype=str(dtype),epsilon=float(np.finfo(dtype).eps),stable_a=stable_a,trace=rows)
results=[probe(np.float64),probe(np.longdouble),probe(np.float64,True)]
(OUT/f'{method}_precision.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
