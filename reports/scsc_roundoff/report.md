# Strict inner stopping test: FOP, SMO, GCMO

Investigated October 23, 2026. No production solver or experiment settings were
changed. The source currently contains `if lhs <= rhs or t > 10000:`.

The earlier `1e-8` slack produced an inaccurate-inner-solve plateau. Removing
it removes that coarse error floor, but does not make the exact relative test
robust at floating-point precision. The latest completed uncapped runs show
this second failure. Their selected methods use analytic Lipschitz bounds,
so the previously identified `lip_scale=0.1` override is not the explanation.

## Evidence from the uncapped runs

These are the most recent completed uncapped stall logs for each available
method/size. New October 23 runs were still in progress during inspection;
their partial traces do not establish convergence or failure.

| Method / n=m | Run suffix | Frozen SCSC k | Last logged t | lhs | rhs |
|---|---|---:|---:|---:|---:|
| FOP / 400 | p1pmf888 | 87 | 113,939,000 | 2.131e-25 | 1.226e-25 |
| SMO / 400 | 1q5ongty | 83 | 26,544,000 | 5.758e-25 | 3.892e-25 |
| SMO / 100 | 4vkl6s2g | 70 | 110,407,000 | 3.138e-25 | 2.878e-26 |
| GCMO / 400 | rdsorhjn | 293 | 81,414,000 | 7.267e-27 | 6.496e-27 |
| GCMO / 100 | eqvrq3m8 | 906 | 28,528,000 | 4.609e-26 | 4.209e-26 |

The last repeated lines stop changing at printed precision. These numbers are
not near float64 underflow: the problem is cancellation and updates below
coordinate resolution, not inability to represent a squared norm of 1e-25.
Raw run identifiers/configurations are in `log_evidence.json`.

## gamma and the reference iterates

The implementation agrees with [Algorithm 5, page 25](https://arxiv.org/pdf/2301.01716v2#page=25):

```
gamma_x = gamma_y = gamma = 8 / sigma_x
x_m1 = -z_g / sigma_x
y_m1 = y_g
```

There is no aliasing update of `x_m1` or `y_m1` in the inner loop. They are
fixed for one outer index k, but recomputed for the next k. Also, rhs is not
literally constant in t: it contains the current inner iterate.

Write d=(x_t-x_m1, y_t-y_m1) and r=(a_x+b_x, a_y+b_y). Then

```
lhs = gamma * ||r||²
rhs = ||d||² / gamma
lhs <= rhs  <=>  ||r|| <= (sigma_x/8) * ||d||.
```

This is a relative residual requirement. For a fixed k away from the saddle,
rhs can approach a positive constant as t grows. That constant need not be
of order one, and is not bounded away from zero across k. At the exact saddle
the reference and inner solution coincide, so both sides are zero. Close to
the saddle the required relative accuracy can be below what float64 can
deliver. In these logs the recovered ||d|| ranges from 2.7e-14 to 1.4e-13.

Changing gamma alone makes the inequality easier or harder, but also changes
the proximal steps and invalidates the prescribed parameter relationships.
Freezing the reference at an old initial point changes the relative-error
condition. Neither is a principled cure for this numerical limit.

## Where the floating-point floor enters

The implementation forms `g_x - sigma_x*x` and then adds a term of comparable
size. In the y block it subtracts and re-adds `sigma_y*y`. It computes proximal
normals as `(before_prox - after_prox)/(zeta*gamma)` and then forms `a+b`.
At active bounds, a and b can be individually large while their sum is tiny.
Finally, rhs subtracts nearly identical coordinates. Once updates are below
an ulp, additional iterations cannot reliably reduce the computed residual.

Useful algebraic simplifications, in exact arithmetic, are

```
a_x = grad_x(h_bar) - (sigma_x/2)*(x-x_m1)
a_y = -grad_y(h_bar) + (sigma_x/8)*(y-y_g)
x_next = (x_current + x_f)/2
```

The centered x formula assumes x_m1=-z_g/sigma_x exactly; with rounded stored
values the two forms differ by roundoff. These rewrites reduce cancellation,
but do not remove finite coordinate resolution or all gradient/prox-normal
cancellation.

To test that distinction, isolated FOP and SMO subproblems were run to a
deliberately stringent tau=1e-14, with the final `+1e-8` allowance disabled in
memory. They use the same 400-dimensional generated instance and the relevant
stage coefficients, but are NOT replays of the long continuation trajectories.
Their stalled inner states were frozen and continued with NumPy arithmetic:

| Frozen problem | float64, 10,000 more updates | Extended precision | Simplified a, float64 |
|---|---|---|---|
| FOP | inequality still fails | passes after 6 updates | still fails |
| SMO | inequality still fails | passes after 5 updates | still fails |

The extended type on this machine has epsilon=1.0842e-19 versus 2.2204e-16
for float64. It is not arbitrary precision. Both NumPy comparisons use the
same stored coefficients, anchors, and initial state; NumPy float64 is not
bitwise identical to Torch. Results are in `*_precision.json`.

The direct feasible-point box KKT norms at the frozen Torch states are
1.1205e-13 (FOP) and 4.4610e-13 (SMO). These do NOT pass the deliberately
stringent 1e-14 stress-test target, and must not be called successful solves.
They do show the scale of the attainable accuracy and why higher precision
can be necessary when a requested target is actually that small.

A full GCMO continuation replay was bounded to 300 seconds and did not reach
the recorded stall. A separate isolated GCMO attempt reached its outer cap
without reproducing the roundoff stall. The GCMO diagnosis here rests on the
recorded frozen residuals and the identities below; no complete GCMO replay
or GCMO precision-comparison result is claimed.

## The current inner candidate can already solve the original SCSC task

This is the most useful distinction. b_x and b_y are proximal normal vectors
at the feasible inner point. Let d_x=x_t-x_m1 and d_y=y_t-y_g. Directly from
the operator definitions,

```
grad_x(h_bar) + b_x = r_x + (sigma_x/2)*d_x
grad_y(h_bar) - b_y = -r_y + (sigma_x/8)*d_y.
```

Thus these vectors are a stationarity certificate for the ORIGINAL regularized
SCSC objective at the CURRENT inner iterate, without first satisfying line 9
or extrapolating to the next outer iterate. In exact arithmetic,

```
stationarity_norm <= sqrt(lhs/gamma) + (sigma_x/2)*sqrt(gamma*rhs).
```

Applying this identity to the rounded logged values gives these estimates:

| Method / n=m | Implied stationarity upper bound | Requested tau |
|---|---:|---:|
| FOP / 400 | 4.70e-12 | 6.71e-6 |
| SMO / 400 | 1.48e-11 | 6.83e-7 |
| SMO / 100 | 7.73e-12 | 7.02e-7 |
| GCMO / 400 | 1.15e-12 | 7.14e-5 |
| GCMO / 100 | 2.08e-12 | 1.36e-6 |

These are inferred bounds, not independently measured KKT norms of the exact
historical states: the logs round values and floating-point operations incur
error. They strongly indicate that the inner candidates already meet their
requested SCSC accuracy by many orders of magnitude. A direct certificate
check should confirm that before returning success. `log_analysis.json`
contains the constants and calculations.

The existing final check is reached only AFTER the inner relative test passes,
and it examines a different, extrapolated point. In the isolated probes, the
stable output certificate at the next outer point is 9.61e-12 for FOP and
1.19e-10 for SMO, larger than the inner-point KKT norm. The two points should
not be conflated when diagnosing the logs.

## Recommended remedies

1. **Check genuine SCSC stationarity at the current feasible inner point.**
   Keep the strict relative criterion for ordinary progress. Periodically,
   or when progress stalls, evaluate the original objective's certificate at
   `(x_t,y_t)`. Return that point only when its certificate meets tau. This is
   an early success condition for the required stationarity goal, not an
   assertion that the relative test passed. The original line-25 projected
   certificate can also be evaluated from this point using stable arithmetic.

   For the present box problems, a direct KKT residual avoids tiny-step
   quotients. For each minimizing coordinate with gradient g, use g in the
   interior, min(g,0) at the lower bound, and max(g,0) at the upper bound.
   Apply the same rule to -g for maximizing coordinates. Take the joint norm.
   Use exact active-bound comparisons on the projected candidate; declaring
   nearby interior points active by a loose tolerance can falsely certify.

   FOP/SMO use [-1,1]. GCMO additionally has multiplier blocks in [0,200], so
   it needs per-block bounds. It must not use one common [-1,1] declaration.

2. **Reduce cancellation, but retain failure handling.** Use the simplified
   operators/outer updates and stable box proximal quotients/normals. GCMO
   currently lacks the stable certificate path because its blocks have
   different bounds. Generalizing to per-block bounds would address that.
   Its 400-dimensional certificate step is about 1.525e-10: a coordinate
   subtraction error near 2.22e-16 becomes about 1.46e-6 after division.
   Tiny extrapolation infeasibility is also amplified by that inverse step,
   even when the quotient is evaluated algebraically. A feasible inner-point
   certificate avoids that extrapolation issue.

3. **Use higher precision when the real certificate does not pass.** The
   precision experiment shows that additional significant digits can restore
   the relative test. Re-evaluating only the squared norm at higher precision
   is insufficient; the operator, proximal normal, and iterate updates need
   the extra precision. A roundoff allowance must depend on dtype and the
   magnitudes being canceled, and must not replace the final certificate.

4. **Treat iteration caps as explicit diagnostic failures.** A cap is useful
   to prevent infinite loops, but should report `inner_accuracy_not_met` or
   `numerical_stagnation` unless a true certificate passes. NCWC currently
   stops on its step-size test without requiring `solver_stats['terminated']`;
   that also needs guarding if failed inner solves can propagate outward.

Neither a universal `+1e-8` nor `rhs*(1+small_relative_tolerance)` resolves the
underlying problem. The former is enormous compared with these squared
residuals and recreates the earlier inexactness floor. The latter cannot
handle lhs/rhs ratios from 1.1 to 10.9 once rhs is at the numerical floor.

## Reproduction

Use the experiment Python environment. The scripts disable W&B and instrument
the solver in memory; they do not edit solver sources.

```
python reports/scsc_roundoff/analyze_logs.py
python reports/scsc_roundoff/replay.py fop isolated
python reports/scsc_roundoff/replay.py smo isolated
python reports/scsc_roundoff/precision_probe.py fop
python reports/scsc_roundoff/precision_probe.py smo
python reports/scsc_roundoff/certificate_probe.py fop
python reports/scsc_roundoff/certificate_probe.py smo
```

The isolated replay disables the final additive tolerance intentionally to
expose the numerical floor. The full replay option is separately bounded by
a 300-second alarm. Snapshots contain only locally generated problem tensors
and solver state. Production fixes and end-to-end convergence validation are
outside this investigation.
