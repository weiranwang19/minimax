# Latest four runs: successful SCSC solves, slow NCWC convergence

Audited October 23, 2026 using local W&B binary histories, configurations,
summaries, and complete output logs. `analyze.py` reproduces the numerical
results in `results.json`; it does not run or modify the solvers.

## What the logs establish

All four runs ended with `KeyboardInterrupt` after approximately 37,200 seconds
(10.33 hours). None exhausted its configured 100000-iteration budget. Every
completed NCWC iteration received a successful SCSC result, specifically an
`inner_certificate` return. There are no printed long-inner-loop progress
messages (printed every 1000 updates). Thus the earlier unbounded auxiliary
relative-test stall is not the behavior recorded in these runs.

| Run | Method / size | Completed SCSC solves across all stages | Current NCWC iterations | Last SCSC outer iterations / inner updates |
|---|---|---:|---:|---:|
| o3ce2psl | FOP / 100 | 1347 | 1035 | 927 / 255588 |
| iq2p8fl4 | FOP / 400 | 8088 | 5532 | 59 / 14635 |
| 22bvabs6 | SMO / 100 | 12273 | 4353 | 39 / 9589 |
| rgavkjk8 | GCMO / 100 | 3547 | 3547 | 77 / 19454 |

The active NCWC exit condition is `d <= epsilon/(4*L)`, where `d` is the
Euclidean change in the entire minimizing block. For FOP/SMO this includes
both x and y; GCMO additionally includes the minimizing multiplier block.
It is not an objective-change test.

| Run | Current stage epsilon | Current L | Last d | Required d at most | d / threshold | Last inner tau |
|---|---:|---:|---:|---:|---:|---:|
| FOP / 100 | .04 | 1616.2214 | 4.790710e-4 | 6.187271e-6 | 77.43 | 3.091787e-7 |
| FOP / 400 | .04 | 6359.3833 | 4.705838e-4 | 1.572480e-6 | 299.26 | 5.784526e-8 |
| SMO / 100 | .0439804651 | 607.9549 | 2.477428e-5 | 1.808541e-5 | 1.370 | 4.659410e-8 |
| GCMO / 100 | .01 | 32.1618 | 4.601538e-4 | 7.773196e-5 | 5.920 | 2.819284e-7 |

FOP is in stage index 3 (rho=25, mu=625), and SMO is in stage index 14.
W&B's last `fop/stage/*` and `smo/stage/*` summary values describe the previous
completed stage, because those metrics are emitted only after completion.
Using those summaries as the current constants would produce wrong thresholds.

## A visual objective plateau is not the stopping condition

Over the final 100 completed iterations of the active NCWC call:

| Run | Upper objective, first -> last | Other relevant final metrics |
|---|---|---|
| FOP / 100 | -140.030214 -> -140.109628 | feasibility .001910; lower gap -.004724 |
| FOP / 400 | -598.835387 -> -599.157442 | feasibility .002505; lower gap .067458 |
| SMO / 100 | -141.636051 -> -141.636125 | feasibility .001490; lower gap -.001038 |
| GCMO / 100 | -141.625763 -> -141.627111 | feasibility 1.22e-8; lower gap 2.94e-8; lower primal-dual gap 5.02e-6 |

The plotted objective is the upper-level objective, not the entire saddle
objective or its stationarity residual. In particular, FOP/400 still fails
the requested .01 lower-gap target. Small feasibility and lower-gap values
in GCMO do not establish stationarity of the upper/minimax problem.

SMO and GCMO have nearly constant step norms and small continuing descent.
Their observed per-iteration objective decreases are approximately 7.44e-7
and 1.362e-5. The proximal linear-face prediction `2*L*d^2` is 7.46e-7 and
1.362e-5 respectively. This close agreement is evidence consistent with slow
motion on a nearly linear face, rather than a floating-point objective floor.
It is not a reconstruction of the unlogged individual coordinates.

## Two different sources of strictness

1. **The fixed NCWC step threshold is conservative.** It reserves epsilon/2
   for inner error even when the measured/certified inner error is orders of
   magnitude smaller.
2. **Inner accuracy keeps tightening with NCWC iteration count.** The code
   prescribes `tau_j=tau_0/j`. At the end of these runs, tau is between
   4.66e-8 and 3.09e-7 while the current-stage target is .01--.044. This is
   the paper's schedule, not a newly found transcription error, but it makes
   every additional outer step expensive. These tolerances are also distinct
   from the earlier deliberately sub-roundoff 1e-14 diagnostics.

For these linear experiments, FOP has `L proportional to rho^3` and
`epsilon=1/rho`. Its step threshold therefore scales as `rho^-4`. The jump
from rho=5 to rho=25 makes L 125 times larger and the step threshold 625 times
smaller. The current SCSC declared smoothness/concavity ratios are approximately
4.85e6, 3.82e7, 1.66e6, and 8.64e6 for FOP/100, FOP/400, SMO, and GCMO.
Strong convexity/concavity guarantees neither good conditioning nor fast
convergence of the enclosing nonconvex proximal-point iteration.

FOP/100 is also costly within each SCSC solve: the last completed solve uses
255588 inner updates. The next incomplete call's extrapolated-point delta
reaches 1.185e7 while tau is 3.089e-7. The logs alone cannot identify the
coordinate contributions or establish a new update bug. Successful inner
certificates on preceding calls must not be confused with these intermediate
certificates at different extrapolated/projected points.

## A provably safe improvement already detectable for SMO

Write the original NCWC objective as H(u,v), and its SCSC subproblem as

`H_j(u,v) = H(u,v) + L||u-u_j||^2 - epsilon||v-v_0||^2/(4D)`.

Given a valid SCSC certificate at the returned candidate, its original
stationarity residuals have the bounds

`R_u <= tau_j + 2L*d`,

`R_v <= tau_j + epsilon/2`.

These follow by removing the two regularization gradients, retaining valid
normal/subgradient vectors at the same point, and applying the triangle
inequality. The existing step threshold substitutes the uniform worst-case
assumption `tau_j <= epsilon/2`. See the original
[Algorithm 2 and Theorem 2 proof, pp. 8 and 19](https://arxiv.org/pdf/2301.02060).

For SMO's final point:

- requested current-stage epsilon: .0439804651;
- upper bound on original minimizing residual: .0301233426;
- upper bound on original maximizing residual: .0219902791;
- even the joint-norm upper bound is only .0372959534.

Thus the current stage is already certified at its requested tolerance,
despite failing the stricter step surrogate. Along this recorded trajectory,
the blockwise bound first passes at NCWC iteration 1151, about 3.45 hours
before interruption. Even requiring the joint norm <= epsilon passes at
iteration 1244, about 3.37 hours before interruption. These are potential
savings for this stage, not an end-to-end runtime prediction; subsequent SMO
stages still need solving.

The same conservative bound does not pass for the other three runs. Failure
of an upper bound does not prove failure of the true residual. Direct
unregularized KKT evaluations could be tighter, but the histories do not
contain the iterate vectors needed to evaluate them retrospectively.

## Recommended next changes

1. Add an original-objective NCWC certificate after each successfully certified
   SCSC return. For these boxes, directly evaluate the minimum-norm KKT
   residuals of H using each block's correct bounds; do not include NCWC's
   regularization gradients. Keep the existing test as a sufficient fallback.
   An inexpensive first implementation can use the above certified bounds.
   This preserves the requested endpoint stationarity accuracy and the
   existing accepted-subproblem path; it does not use objective flatness as
   proof of convergence.
2. Log active epsilon, L, tau, d, d/threshold, original residuals by block,
   SCSC certificate/reason, and stage-start parameters. The current experiment
   logger drops `scsc_final_delta` and `scsc_termination_reason` even though
   NCWC supplies them. Add periodic iterate checkpoints for numerical replay.
3. Investigate adaptive subproblem accuracy and smaller, justified proximal
   regularization as separate algorithm changes. Do not simply restore an
   absolute `lhs` slack, truncate a loop as success, or replace the paper's
   decreasing error schedule with a constant without checking descent/error
   requirements. Decouple a valid smoothness bound from a proximal coefficient
   only with appropriate weak-curvature and acceptance checks.
4. For the structured linear experiments, benchmark a certified active-set or
   Newton solve for the quadratic/piecewise-quadratic SCSC subproblems. For
   GCMO, a justified tighter multiplier bound could also reduce its large
   max-domain diameter and improve conditioning. These are performance
   changes requiring comparison against the same residual certificate.
5. Separate NCWC, SCSC, and total wall-time budgets. Budget or stagnation exits
   should report incomplete convergence and save the current state. The
   existing 100000 setting applies at multiple nesting levels and provides
   no practical total-work bound.

No solver implementation or experiment settings were changed in this audit.
