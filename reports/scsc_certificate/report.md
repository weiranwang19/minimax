# Always-active SCSC stationarity certificate

Implemented October 23, 2026. The deterministic SCSC solver now checks every
feasible inner candidate, including t=0, using the gradients of its supplied
regularized objective. A residual norm at most `tau` returns that candidate
immediately to the caller. It does not continue the accelerated updates after
acceptance. There is no certificate enable flag or additional gradient call.

Declared box projections use the minimum-norm box KKT residual, with exact
boundary comparisons and the negative gradient for maximization. Other
proximal operators use `(grad_x h_bar + b_x, -grad_y h_bar + b_y)`. This generic
path assumes the supplied proximal operators and their recovered subgradients
are valid, as the original algorithm does. Box metadata must describe the
actual projection operators.

Both absolute `1e-8` allowances were removed: auxiliary updates require
`lhs <= rhs`, and both stationarity exits require `delta <= tau`.

## Interfaces and method integration

- The existing `box_bounds=(lo, hi)` remains supported. SCSC/NCWC also accept
  `{'min': [pairs...], 'max': [pairs...]}`, one scalar bounds pair per tensor.
  The same normalized bounds are used for the stable outer residual.
- FOP/SMO already forward bounds. GCMO accepts shared primal `box_bounds`,
  appends `(0, lagrange_bound)` for both multiplier blocks, and passes the
  complete specification to NCWC. Its experiment now supplies `(-1, 1)`.
- Omitting bounds still runs the generic certificate. Existing stable-box
  experiment options cannot disable certification.
- SCSC results add `termination_reason`: `inner_certificate`,
  `outer_certificate`, or `max_iter`. `final_delta` describes the reported
  candidate's certificate; the two successful routes can use different
  normal vectors. Inner-update counts retain their existing meaning and
  include work performed before an early return.
- NCWC progress payloads add `scsc_final_delta` and
  `scsc_termination_reason`, initially `None`.

Tolerance schedules, iteration budgets, warm starts, NCWC failure propagation,
and the user's experiment settings were preserved. Existing Python processes
must be restarted to load the updated solver.

## Validation

All 52 discovered tests passed using:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/scratch/yshen48/conda-envs/minimax/bin/python -m unittest discover -s tests -q
```

Coverage includes both gradient paths for FOP, SMO, and GCMO; primal and dual
boundary signs; adjacent interior points; finite/feasible candidates; mixed
box bounds; generic soft-threshold proximal operators with an analytic saddle;
strict tolerances; all termination reasons; returned parameter assignment;
counters; regularized-objective stationarity; and gradient-call counts.

`replay.py` runs bounded, local diagnostics with W&B disabled and writes
`results.json`. Returned candidates are checked independently by differentiating
the scalar objective with autograd and evaluating its box KKT residual.

### Historical GCMO fifth call

The first four calls were reconstructed with their historical stopping rule
and generic outer residual, matching their recorded SCSC outer iteration
counts: 105, 114, 132, and 196. The updated production solver then solved the
same fifth subproblem, at the unchanged tolerance `tau=0.0002`:

| Measurement | Result |
|---|---:|
| SCSC outer iterations | 32 |
| Inner updates | 7605 |
| Reported certificate | 0.0001972461437568446 |
| Independent autograd KKT norm | 0.00019724614375656168 |
| Termination reason | inner_certificate |

The earlier shortcut replay exhausted 1000 outer iterations without
certification; those historical comparison results remain in
`../gcmo_live/results.json`. This is a controlled subproblem regression, not
an end-to-end convergence test of the full large experiment.

### Frozen FOP and SMO candidates

The saved snapshots were deliberately generated with `tau=1e-14`. Evaluating
their frozen candidates produces residuals above that target, so they are
correctly not certified at the saved tolerance. Separately, restarting the
same regularized objectives from those candidates at the explicitly chosen
test tolerance `1e-6` returns at t=0:

| Method | Frozen certificate | Saved tau accepted? | Restart autograd KKT norm |
|---|---:|---|---:|
| FOP | 1.1204994e-13 | No | 1.0890792e-13 |
| SMO | 4.4609845e-13 | No | 4.1901981e-13 |

The restart includes the algorithm's initial proximal step; it is not an
exact continuation from the middle of the saved loop. No production tolerance
was changed by these diagnostics. The certificate does not promise convergence
at tolerances below attainable floating-point accuracy, nor add a new inner
iteration cap when certification fails.
