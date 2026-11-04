**Current FOP run: slow valid progress and expensive inner solves — 2026-10-29**

Inspected local run `run-20261029_143612-5d4hhpdr`, named `fop-n100-m100-l5-idx1-seed0-lipauto`. This is the first `(100,100,5)` instance, not the subsequent `(400,400,20)` instance. The new deterministic NCWC certificate is enabled (`box_kkt`). No production code, settings, or running processes were changed during this investigation.

The fixed snapshot in [evidence.json](evidence.json) contains 1447 completed NCWC iterations in FOP stage 3 (the fourth stage), at runtime 14,330.9 seconds, or 3.98 hours. Subsequent log observations continued to show successful but expensive SCSC calls. This evidence establishes slow progress, not mathematical nontermination.

| Quantity at the snapshot | Value |
| --- | ---: |
| Stage penalty `rho`, `mu` | 25, 625 |
| NCWC target epsilon | 0.04 |
| Original-stage min KKT residual | 1.3445387574 |
| Original-stage max KKT residual | 0.0001322648 |
| Current SCSC target | 2.2114720e-7 |
| Returned SCSC residual | 2.2097160e-7 |
| Last SCSC inner updates | 490,503 |
| Total inner updates over all stages | 33,914,140 |
| Last minimizing-block displacement | 0.0004159513 |
| Feasibility norm | 0.0018962066 |
| Logged lower-level objective difference | -0.0048889848 |

The maximizing block passes the NCWC target; the minimizing block exceeds it by **33.61 times**. Every one of the 1447 completed SCSC calls in this stage reported success. NCWC is not mistakenly rejecting a point that its implemented certificate says is stationary. The current target is still 0.04, above the experiment's final target 0.01. A small constraint violation and a nearly flat upper-objective graph do not establish stage stationarity. The negative lower-level objective difference is evaluated at a slightly infeasible lower point; it is not a nonnegative optimality-gap certificate.

**Outer progress is limited by the regularization scale.** For this linear instance, `L_g^2 = 0.0258595421881` and the existing FOP bound is

\[
L=4\rho\mu L_g^2=4\rho^3L_g^2.
\]

At the active stage, `L = 1616.22138676`. The last displacement satisfies

\[
2L\|u_{j+1}-u_j\|=1.3445387608,
\]

almost exactly the directly logged original-objective residual 1.3445387574. This is consistent with the proximal regularizer balancing the original gradient and producing very small steps. A linear objective along a fixed constraint face can keep a nonzero residual while taking many small steps toward a boundary; that is a plausible explanation for the observed residual plateaus, but the logs do not contain the coordinates needed to identify the exact active face.

The upper objective decreases by about 0.000556 per NCWC iteration over the final 100 snapshot iterations. This looks flat on a plot with values near -140, yet is not zero progress and does not imply stationarity.

**Inner work becomes much more expensive.** At NCWC iterations 1000 and 1400, SCSC takes 10,822 and 12,150 inner updates. At iteration 1410, it takes 498,379. The increase is not monotone—some following calls are cheaper—but recent calls repeatedly cost around 500,000 updates, or approximately 200 seconds each in this run.

Between iterations 1400 and 1447, 47 further NCWC iterations consume **18,715,197 inner updates and 2.19 hours**, while reducing the min residual by only **0.00011908**. This is the dominant recent runtime cost.

The SCSC parameters explain why such work can be expensive despite the modest dimension:

| FOP stage | epsilon | L / declared sigma_x | declared sigma_y | smoothness / sigma_y | projected-certificate step |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.2 | 12.9298 | 0.005 | 7.76e3 | 3.32e-6 |
| 3, current | 0.04 | 1616.22 | 0.001 | 4.85e6 | 4.25e-11 |
| 4, next | 0.008 | 202027.67 | 0.0002 | 3.03e9 | 5.44e-16 |

The smoothness ratio uses the code's bound `3*L + sigma_y`, not a measured Hessian condition number. With `BASE_RHO=5`, L grows by 125 and sigma_y shrinks by 5 per stage, so this ratio grows approximately 625-fold. Strong convexity and concavity do not imply favorable conditioning.

The prescribed inner schedule is `tau_j = epsilon**2.5 / j` for one-based NCWC iteration j. Here it demands about 2.2e-7, approximately 181,000 times smaller than the NCWC target, while the original minimizing residual is still 1.34. The new certificate can avoid extra work once stage stationarity holds; it does not reduce this inner accuracy requirement or change the NCWC update itself.

**The old innermost relative-test stall is not what these logs show.** There are no long-inner-loop progress messages. The expensive calls repeatedly pass `lhs <= rhs` after roughly 200–280 inner updates and then perform many SCSC outer iterations. One recently completed call used 1759 outer iterations and eventually returned via `inner_certificate`.

Its displayed projected outer residual fell near 4e-5, later grew to millions, and then the feasible inner candidate passed approximately 2.2e-7. These are certificates at different points. The projected residual also includes projection quotients divided by a very small step; it is not the minimum-norm KKT residual used by the feasible inner certificate. The log alone cannot establish whether the sharp cost increase is caused by a boundary transition, unfavorable accelerated iterates, cancellation, or another implementation issue. In particular, large intermediate `delta` values do not establish failure of strong convexity/concavity. An expensive-call checkpoint is needed for a controlled diagnosis of that transition.

**Potential remedies, with their limits.**

- Capture an expensive SCSC call's input and regularization anchors, then compare inner strategies at the same independently verified NCWC accuracy. This is the next direct investigation of the sudden cost increase; another multi-hour run without snapshots would provide less information.
- Test adaptive inner accuracy while retaining the strict final NCWC certificate. This can reduce oversolving, but changes the prescribed inner-error schedule and needs a separate convergence/merit-bound argument. It must not be implemented as additive acceptance slack or a successful iteration-cap exit.
- Use a tighter proven smoothness bound for this particular linear squared-hinge objective. Its Hessian on each activity region is `2*rho*mu*(P.T*D_y*P - Q.T*D_z*Q)`, with `P=[A B 0]`, `Q=[A 0 B]`, and `||P||=||Q||=L_g`. Both terms are positive semidefinite before subtraction, so the absolute eigenvalues are bounded by `2*rho*mu*L_g^2`. Thus halving the current bound is justified for this linear objective. This gives a factor-of-two scale improvement, not a general cure, and must not be applied as a fixed override across different instances or nonlinear problems.
- Investigate preconditioning, a more suitable certified inner solver, or a less abrupt penalty continuation. Those change the algorithm's dynamics; a termination certificate by itself cannot fix the conditioning of the next stage.

The previous certificate work remains useful: it prevents unnecessary iteration at an already acceptable original-stage residual. This run has a substantial residual remaining, compounded by costly SCSC solves. The available evidence does not support loosening NCWC's 0.04 target or claiming that the run should already be considered converged.
