**Implementation validation — 2026-10-28**

The deterministic NCWC certificate is active by default in the shared solver used by FOP, SMO, and GCMO. Declared boxes use direct stationarity of the original stage objective; generic proximal operators use residual upper bounds. Both require a successful SCSC certificate with finite `final_delta <= epsilon_k`. The old step size remains diagnostic. Stochastic NCWC retains its existing stopping rule.

The generic maximizing bound uses the actual displacement from the fixed NCWC anchor, `epsilon_k + epsilon/(2*D_y) * ||v-v_0||`. This is the refinement specified in the implementation plan; the earlier comparison experiments used the diameter bound `epsilon_k + epsilon/2`.

Results and progress callbacks now expose `final_residual_min`, `final_residual_max`, `residual_type`, and `termination_reason`. Progress distinguishes NCWC's `epsilon` from SCSC's `epsilon_k`. W&B history records these fields under `ncwc/`, plus `ncwc/terminated`, `ncwc/scsc_final_delta`, and `ncwc/scsc_termination_reason`. The existing callback `terminated` field retains its inner-solver meaning; `ncwc_terminated` remains the NCWC status.

All **68 solver tests passed**, including new analytic NCWC tests, original-objective certification through all three method wrappers with both gradient backends, and logger tests with a fake W&B run. The focused 14-test NCWC suite also passed after adding support for constant scalar objectives in the autograd helper.

The production implementation reproduces the saved direct-certificate experiments:

| Replay | NCWC iterations | SCSC inner updates | Verified endpoints |
| --- | ---: | ---: | ---: |
| Scalar linear | 1 | 52 | 1 |
| Scalar boundary | 1 | 1 | 1 |
| Scalar quadratic | 6 | 1,222 | 1 |
| SMO `(100,100,5)` through stage 6 | 36 | 7,336 | 7 |
| SMO `(100,100,5)` through stage 8 | 169 | 234,710 | 9 |
| SMO `(4,4,2)` through stage 14 | 248 | 158,703 | 15 |

All replays completed within their 180-second budgets. Work counts match the saved experiments exactly. SMO step norms, residuals, and final parameters match within `1e-12`; each of the 34 accepted endpoints was independently checked using autograd on the original scalar objective and passed the strict stage tolerance. These are bounded comparisons, not full runs to the original tight final target.

The [replay script](replay_implementation.py) calls production NCWC directly. It leaves the original report artifacts untouched and saves new results under [implementation_replays](implementation_replays/). It reuses problem setup from `experiment.py` without using that file's original source-replacement instrumentation. No online W&B runs were created.

From the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/scratch/yshen48/conda-envs/minimax/bin/python -m unittest discover -s tests -v

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/scratch/yshen48/conda-envs/minimax/bin/python \
  reports/ncwc_certificate/replay_implementation.py smo_stage8 --seconds 180
```

The other replay names are `analytic`, `smo_stage6`, and `smo_small_stage14`. Re-running a replay replaces only its implementation-replay result file. Method schedules, SCSC accuracy, multiplier updates, and the user's existing comment edit were preserved.
