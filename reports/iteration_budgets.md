**Independent CBLO iteration budgets**

Edit each method's block at the top of `exp_cblo.py`. FOP, SMO, and GCMO each have independent `_NCWC_MAX_ITERS`, `_SCSC_MAX_OUTER_ITERS`, `_SCSC_MAX_INNER_ITERS`, and `_AGD_MAX_ITERS` settings. FOP additionally has `FOP_MAX_STAGES`; SMO has `SMO_MAX_STAGES`. GCMO performs one NCWC call.

The preserved defaults are 100000 NCWC iterations per stage, 100000 SCSC outer iterations per NCWC iteration, and no innermost or AGD cap (`None`). FOP permits 200 stages; SMO follows its tolerance schedule unless its stage cap is set. These limits apply to their individual loops, not cumulative work or elapsed time.

For example, a finite diagnostic configuration for FOP can be set as:

```python
FOP_MAX_STAGES = 20
FOP_NCWC_MAX_ITERS = 1000
FOP_SCSC_MAX_OUTER_ITERS = 2000
FOP_SCSC_MAX_INNER_ITERS = 10000
FOP_AGD_MAX_ITERS = 1000
```

These are illustrative caps, not convergence or runtime guarantees. Budgets do not change the stationarity certificates, tolerance schedules, or updates performed before a cap binds. Capped AGD remains an approximate starting point. An uncertified SCSC or NCWC result stops its method; SMO preserves its current multiplier and tolerance after such a failed stage. Method-stage exhaustion also reports an incomplete solve. The experiment finalizes and marks that instance failed, then its existing grid driver continues to the next instance.

SCSC checks its feasible-point certificate and relative completion condition before enforcing an inner cap, including when the cap is zero. Success at the exact limit remains valid. An inner-cap failure returns the current feasible candidate and its residual. Work counters include the attempted outer iteration and all completed inner updates.

Results expose `budget_exhausted` as `None`, `stages`, `ncwc`, `scsc_outer`, or `scsc_inner`. NCWC also exposes `last_scsc_stats`. The `termination_reason` provides the corresponding algorithm outcome; SCSC uses `inner_max_iter` for an inner cap and `max_iter` for an outer cap. NCWC reports `inner_solver_failed` when its child cannot certify an endpoint.

All configured budgets are recorded in W&B, with the old configuration keys retained as aliases. Exhaustion, counters, and available residuals are recorded in progress, stage, and instance diagnostics.

Existing wrapper calls remain valid. Their existing `max_iter` or `subproblem_max_iter` limits NCWC; new trailing `scsc_max_iter` and `scsc_max_inner_iter` arguments override the nested limits. Omitting the SCSC outer override retains the historical NCWC-limit fallback. SMO's new trailing `max_outer_iters` caps stages. Direct NCWC callers use `subproblem_max_iter` and the new `subproblem_max_inner_iter`; direct SCSC callers use `max_iter` and the new `max_inner_iter`. The new innermost cap is deterministic-only; stochastic NCWC keeps its existing behavior.

Validation covers unequal limits through all methods, actual nested solvers with small caps, exact-limit success, partial candidates and counters, multiplier preservation, failed-instance handling, and fake W&B finalization exactly once. Three successful analytic NCWC examples retain identical outputs and work counts with finite nonbinding caps.
