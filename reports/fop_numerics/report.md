# FOP numerical changes: bounded benchmark

This report measures isolated regularized SCSC solves, not complete bilevel experiments. The baseline uses the original autograd and projection-subtraction path through the updated solver.

All measured trials terminated and passed the common accuracy checks. Combined median runtime was 1.807 s versus 4.101 s for baseline (2.27x). Residual-only median runtime changed by -0.29%; this small sample does not establish a speed improvement from the residual reformulation.

## Implemented changes

- `Minimax_SCSC` now annotates the input coefficients, gradient definitions, and update blocks against [Algorithm 5, page 25](https://arxiv.org/pdf/2301.01716#page=25). Existing deviations such as finite caps, the beta multiplier, and the strict stopping comparison are documented.
- Its optional box residual evaluates the line-25 projection quotients with algebraically equivalent clipped gradients. The step size, projected iterates, norm, and tolerance are unchanged.
- Optional explicit gradient/difference callbacks are passed through FOP and regularized by NCWC. They are enabled only for the linear CBLO FOP path; generic callers retain autograd and the generic prox residual.
- `FOP_STABLE_BOX_RESIDUAL` and `FOP_EXPLICIT_GRADIENTS` are enabled by default and recorded in run configuration. AGD, SMO, GCMO, and the existing experiment settings were not changed.

## Setup

- One `(400, 400, 20)` instance, index `1`, seed `0`, generated once and cloned for every trial.
- CPU float64; one Torch thread; initial `x=0`, `y=z=Y_HAT`; regularization anchors equal these initial values.
- `rho=5`, `mu=25`, `L_h=12.5`, `sigma_y=0.0025`, `tau=1e-6`; 100 SCSC outer iterations maximum.
- One untimed warm-up plus three measured solves per variant, each with a 30-second timeout.
- Instance generation and solver construction are outside solve timings. Accuracy checks run afterward; diagnostic LPs, logging, and W&B are disabled.
- Explicit-gradient variants also use direct new-minus-old gradient differences.
- The inner while loop is unchanged; the benchmark timeout bounds it externally.

Environment: `{"cpu": "Intel(R) Xeon(R) Gold 6438Y+", "dtype": "float64", "platform": "Linux-6.8.0-147-generic-x86_64-with-glibc2.39", "python": "3.10.20", "python_executable": "/data/scratch/yshen48/conda-envs/minimax/bin/python", "threads": 1, "torch": "2.11.0+cu130"}`

## Solve runtime

| Variant | Median seconds | Range seconds | Baseline / median | Statuses | SCSC outer counts | Inner counts |
|---|---:|---|---:|---|---|---|
| baseline | 4.1009 | 4.0768–4.1075 | 1.00x | terminated, terminated, terminated | 24, 24, 24 | 5033, 5033, 5033 |
| residual_only | 4.0888 | 4.0770–4.0899 | 1.00x | terminated, terminated, terminated | 24, 24, 24 | 5033, 5033, 5033 |
| gradients_only | 1.8050 | 1.8011–1.8053 | 2.27x | terminated, terminated, terminated | 24, 24, 24 | 5033, 5033, 5033 |
| combined | 1.8068 | 1.8037–1.8089 | 2.27x | terminated, terminated, terminated | 24, 24, 24 | 5033, 5033, 5033 |

## Accuracy and individual evaluation costs

Accuracy columns show the maximum across measured trials. Solution differences are infinity norms against the first completed baseline trial.

| Variant | Reported final residual | Common stable residual | Direct box KKT norm | Solution difference | Gradient µs | Residual µs |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 6.728e-07 | 6.728e-07 | 6.728e-07 | 0.000e+00 | 271.99 | 321.18 |
| residual_only | 6.728e-07 | 6.728e-07 | 6.728e-07 | 0.000e+00 | 271.23 | 331.90 |
| gradients_only | 6.728e-07 | 6.728e-07 | 6.728e-07 | 6.318e-15 | 56.52 | 101.77 |
| combined | 6.728e-07 | 6.728e-07 | 6.728e-07 | 6.318e-15 | 56.55 | 109.77 |

The common stable residual applies one certificate projection from each returned point using the same explicit gradient/difference oracle and the same step size. The direct box KKT norm is evaluated at the returned point itself. Neither metric depends on the variant’s termination flag.

Evaluation costs are medians of three batches of 200 calls after ten warm-up calls. Residual cost includes the second gradient or gradient difference and the norm, but excludes the first gradient and projection. These timings separate evaluation overhead from iteration-count changes.

## Higher-penalty cancellation checks

For a constant unit gradient at the interior coordinate `0.5`, the exact residual is `1`.

| rho | dtype | zeta_hat | Original residual | Stable residual |
|---:|---|---:|---:|---:|
| 25 | torch.float32 | 2.276e-11 | 0 | 1 |
| 25 | torch.float64 | 2.276e-11 | 1.00000108 | 1 |
| 125 | torch.float32 | 2.913e-16 | 0 | 1 |
| 125 | torch.float64 | 2.913e-16 | 0.952912066 | 1 |

## Interpretation and limits

- The stable residual addresses cancellation; it is not inherently a speed optimization. Different stopping decisions can change both runtime and iteration counts.
- Explicit gradients reduce evaluation work; the four variants isolate their effect and their interaction with the residual change.
- A short isolated solve from the initial point does not represent the full continuation schedule. No end-to-end speedup is established here.
- Penalty schedules, Lipschitz overrides, tolerances, AGD, and handling of failed inner solves were not tuned. The configured curvature override and later-stage conditioning remain separate concerns.
- Timeouts and caps are incomplete solves, not evidence of a speedup. A premature baseline stop must be assessed using the common accuracy metrics.

## Reproduction

From the repository directory:

```bash
/data/scratch/yshen48/conda-envs/minimax/bin/python benchmark_fop_numerics.py --output-dir reports/fop_numerics
```

Raw repetitions, warm-ups, environment information, and the generated-data checksum are in `results.json`.

## Numerical validation

The targeted tests cover 80-digit scalar references, both projection signs, non-unit bounds, interior/boundary/extrapolated points, tiny steps, explicit gradients and differences near constraint activation, callback purity and shape validation, FOP integration, and final parameter assignment.

Run the new and existing tests separately from benchmarking:

```bash
/data/scratch/yshen48/conda-envs/minimax/bin/python -m unittest discover -s tests -p 'test_*.py'
```
