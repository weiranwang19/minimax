# Minimax_SCSC stopping investigation

Investigated local run `run-20261022_184357-tqdvit92`, instance `(400,400,20)`,
index 1, seed 0. The log ends with an interrupt at SCSC outer iteration 8377,
`delta=6.077e-3`, `tau=3.578e-3`. This is practical FOP stage 2 (`rho=5`,
`mu=25`), NCWC iteration 4 (zero based).

## Reproduced cause

The fixed absolute slack in `minimax.py`'s **inner** stopping test,
`lhs <= rhs + 1e-8`, allows inaccurate subproblem solutions near convergence.
This is distinct from the final `delta <= tau + 1e-8` test.

Replaying all preceding solves reproduces the logged trajectory. At SCSC
iteration 1000, `lhs=9.977836257e-9`, `rhs=1.305420465e-11`, and
`delta=6.105142179e-3`. The code accepts a weighted squared residual roughly
764 times the required bound. The slack dominates the shrinking right-hand
side, so increasing the iteration cap does not restore the required accuracy.

Controlled runs restart the SAME captured solver inputs and regularization
anchors. Only the inner inequality's absolute slack changes; the final
stationarity test, gradients, step sizes, and curvature parameters are fixed.

| Inner slack | SCSC outer iterations | Inner updates | Final delta | Outcome |
|---|---:|---:|---:|---|
| `1e-8` | 1500 | 8592 | 0.0058870944 | Capped |
| `1e-10` | 24 | 2012 | 0.0035409980 | Converged |
| `1e-12` | 24 | 2692 | 0.0035163231 | Converged |
| `0` | 24 | 3740 | 0.0035161931 | Converged |

Target: `tau=0.003577708764`. CPU float64, one Torch thread. The baseline
residual is mostly in the maximization block (at iteration 1000, its norm is
0.00583408 versus 0.00179897 in the minimization block).

Algorithm 5, line 9 in the [FOP paper](https://arxiv.org/pdf/2301.01716v2#page=25)
requires `lhs <= rhs`, with no fixed absolute slack. Removing this slack restores
that condition and fixes this particular replay. Merely replacing it with a
smaller fixed number can postpone the same problem at tighter tolerances.
At machine precision, any numerical allowance needs separate scale-aware
analysis; an unbounded inner loop also needs explicit failure handling.

## The objective is strongly convex–strongly concave

Let `u=(x,y)` be the minimization block and `z` the maximization block. With
`r_y=Ax+By-b`, `r_z=Ax+Bz-b`, the regularized objective is

```
linear terms + rho*mu*(||[r_y]+||² - ||[r_z]+||²)
             + L_h*||u-u_anchor||² - sigma_z/2*||z-z_anchor||².
```

On each smooth region, let `D_y,D_z` be diagonal constraint-activity masks.
The minimization Hessian is

```
2*L_h*I + 2*rho*mu*[A B]^T*D_y*[A B]
        - 2*rho*mu*diag(A^T*D_z*A, 0).
```

It is bounded below by `(2*L_h - 2*rho*mu*||A||²)*I`. The maximization
Hessian is `-2*rho*mu*B^T*D_z*B - sigma_z*I`. The squared positive-part
penalty is continuously differentiable, so these region-wise bounds establish
global strong convexity/concavity, including across activation boundaries.

For this instance, `||A||²=0.0578662755583`, `L_h=12.5`, `rho*mu=125`,
and `sigma_z=0.0025`. Therefore the minimization curvature is at least
**10.5334311104 > 0**, and maximization curvature is at most **-0.0025**.
Strong convexity/concavity is not the cause of the reproduced plateau.

## Separate mismatch in declared constants

`FOP_LIP_SCALE=0.1` overrides the analytic `L_h=50.8750664716` with `12.5`.
The solver then declares `sigma_x=12.5` and `lip=37.5025`. Both declarations
fail globally for this regularized objective, even though it remains SCSC:

- When all y constraints are inactive and all z constraints are active, the
  smallest minimization Hessian eigenvalue is **10.5334311104**, below 12.5.
- Reversing those activity masks gives a largest minimization Hessian
  eigenvalue of **50.4375332358**, above the claimed full smoothness 37.5025.

Both masks are attainable strictly inside the boxes: take `x=0` and
`y,z=Y_HAT ± B^T(B B^T)^(-1)*(0.01*ones)`, swapping signs for the two
cases. The largest absolute coordinate is 0.3306795692 and the respective
constraint residuals are ±0.01. Thus these are actual violations of the
declared constants, not just pessimistic bounds.

For theoretical guarantees, use a valid smoothness bound (the existing
`FOP_LIP_SCALE=None` path is conservative) or derive and pass valid curvature
and smoothness constants separately. This is independent of the immediate
stall: eliminating the inner slack succeeds even with the current constants.

## Reproduction and scope

```
/data/scratch/yshen48/conda-envs/minimax/bin/python reports/scsc_stall/reproduce.py
```

The script disables W&B, replays the preceding solves unchanged, instruments
the solver in memory, caps each variant at 1500 SCSC outer iterations, and
writes `results.json`. The inner loop remains uncapped, as in the source.
It intentionally stops after the controlled comparison. This is a diagnosis
of the observed subproblem, not a validation of the complete continuation run.
No production solver or experiment settings were changed.
