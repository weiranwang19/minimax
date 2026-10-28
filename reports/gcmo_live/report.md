# Ongoing GCMO run: avoidable SCSC iterations

Investigated October 23, 2026. Primary run:
`wandb/run-20261023_123028-aez6mm3i`, instance `(100,100,5)`, index 1,
seed 0, analytic Lipschitz constant, GCMO epsilon 0.01, multiplier bound 200.
No running process, solver source, or experiment configuration was changed.

## What is actually repeating

At the last inspection, seven NCWC iterations had completed and the eighth
SCSC invocation was still running. `Minimax_SCSC outer=...` counts outer
iterations inside one SCSC invocation; it is not the number of SCSC calls.
The first seven invocation lengths were:

| SCSC invocation, one based | tau | SCSC outer iterations |
|---|---:|---:|
| 1 | 1.000e-3 | 105 |
| 2 | 5.000e-4 | 114 |
| 3 | 3.333e-4 | 132 |
| 4 | 2.500e-4 | 196 |
| 5 | 2.000e-4 | 8210 |
| 6 | 1.667e-4 | 8210 |
| 7 | 1.429e-4 | 9189 |

Invocation 8 had exceeded 16,000 SCSC iterations when sampled; delta continued
oscillating around 1e-3 while tau was 1.25e-4.

## Confirmed cause in the current stopping shortcut

The current inner condition is:

```
if lhs <= rhs or lhs + rhs <= 1e-12:
    break
```

The second condition again permits inaccurate relative solves. At outer index
100 in a replay of invocation 5, the inner loop accepts
`lhs=9.463126857e-13`, `rhs=3.880508738e-14`: lhs is about 24 times rhs.
The following outer certificate is 0.0038642, above tau=0.0002. The feasible
inner candidate itself has a direct box KKT residual of only 7.3001e-7.

Here gamma=0.2487423 and eta_y=74.61639. A squared-residual allowance near
1e-12 permits residual-vector errors around 2e-6. Such errors can be magnified
by the outer extrapolation. Also, line 25 examines a different point and uses
a particular normal-vector certificate; its value need not equal the minimum
KKT residual at the feasible inner candidate. The certificate step is
1.19977e-9, so feasibility and subtraction errors can be amplified there.
These observations explain why a superficially small absolute threshold can
produce a much larger final-delta plateau.

## Controlled replay

The first four SCSC calls were replayed unchanged and matched all four logged
iteration counts and final deltas. Invocation 5 was then restarted from the
same tensors and regularization anchors for every variant. Its tau remained
0.0002 and all curvature parameters remained unchanged.

| Variant | Outer iterations | Inner updates | Final certificate | Outcome |
|---|---:|---:|---:|---|
| Current shortcut | 1000 | 45793 | 0.0017038618 | Capped, not converged |
| Strict lhs <= rhs | 105 | 26014 | 0.0001348022 | Converged |
| Strict test + direct inner-candidate box KKT check | 32 | 7615 | 0.0001898659 | Certified early success |

The third variant periodically checks the same regularized SCSC objective at
the feasible inner candidate, using [-1,1] for primal variables and [0,200]
for multipliers, with the maximization gradient sign reversed. It returns the
candidate only when the joint KKT norm is <= tau. It does not pretend the
relative criterion passed or accept an iteration cap as success. Its reported
certificate differs from the original extrapolated-point certificate, but
certifies the requested stationarity goal directly.

This validates the cause and remedies for the actual fifth subproblem. It is
not an end-to-end convergence test. Strict comparison alone can still hit the
float64 stagnation documented in `../scsc_roundoff/report.md` later.

## Checks against other possible bugs

The GCMO objective is quadratic/bilinear for this generated linear instance,
so its full regularized Hessian can be assembled directly. Eigenvalue checks:

| Quantity | Actual | Supplied |
|---|---:|---:|
| Full gradient Lipschitz constant | 77.88710270 | 96.48542555 |
| Minimum minimization curvature | 52.51572782 | 32.16180479 |
| Maximization strong concavity | 1.116917629e-5 | 1.116917629e-5 |

The declarations are valid. The five existing tests in
`tests/test_cblo_gradients.py` also passed, including independent autograd
comparisons of the saddle gradients. These checks found no gradient-sign or
curvature-bound explanation for this episode; they are not a proof that the
entire solver is bug-free.

## Other consequential control-flow issues

- `GCMO_MAX_ITERS` becomes both the NCWC call cap and the per-call SCSC outer
  cap because `inner_max_iter=max_iter` when `subproblem_max_iter` is omitted.
  Thus 100000 is not a total-work budget. Separate the two controls.
- The companion run `run-20261023_123005-g84z8xxt` uses 200 SCSC outer
  iterations per call and explicitly logs `inner_terminated=False` from NCWC
  iteration 4 onward, yet continues to later NCWC iterations. `optimize_NCWC`
  should handle a failed SCSC solve explicitly and require successful inner
  accuracy before declaring outer convergence.
- The observed NCWC step gap is initially about 0.1943, whereas its stopping
  step threshold is only 7.7732e-5. Even accurate, fast SCSC calls do not imply
  that NCWC is already converged. The small maximization curvature and the
  tightening tau=0.001/(k+1) also make later calls more demanding.

Recommended next implementation: retain the relative test, remove the fixed
absolute shortcut, add certified early success at a feasible inner candidate,
report numerical stagnation explicitly if certification fails, and separate
the iteration budgets. Preserve correct per-block bounds for GCMO.

## Reproduction

```
/data/scratch/yshen48/conda-envs/minimax/bin/python reports/gcmo_live/reproduce.py
/data/scratch/yshen48/conda-envs/minimax/bin/python -m unittest discover -s tests -p test_cblo_gradients.py
```

The replay disables W&B and instruments a copy of the solver's run method in
memory. Every comparison has a 1000-outer-iteration cap and a diagnostic
3001-inner-update failure guard; the whole replay has a 300-second timeout.
The guards did not trigger for the two successful variants. Raw results are
in `results.json`.
