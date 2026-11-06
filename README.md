# Penalty-Based First-Order Methods for Bilevel Optimization with Minimax and Constrained Lower-Level Problems
NeurIPS 2026 Poster

ArXiv: https://arxiv.org/abs/2605.08006

OpenReview: https://openreview.net/forum?id=K4vQ9YKOxc

## Driver files
* `exp_cblo.py`: Constrained bilevel linear optimization experiments (FOP, SMO, Ours (GCMO, generalized constrained minimax optimization)). `exp_cblo_mb.py` is used to check multiplier box constraint for problem instances. The authors of [FOP](https://arxiv.org/abs/2301.01716) and [SMO](https://arxiv.org/abs/2511.07398) did not release the code so we implemented the algorithm with their pseudocode in Python (originally done in MatLab).
* `exp_dro.py`: Hyperparameter tuning for Group DRO. Evaluate saved checkpoints with `exp_dro_eval.py`.

## Acknowledgements

We used the following repository for DRO experiments. Relevant datasets can be found there.

- [Group DRO](https://github.com/kohpangwei/group_DRO)
