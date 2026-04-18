import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from dro import CelebADROProblem, build_celeba_bundle, build_resnet50_backbone, init_classifier
from dro.problem import identity_prox, positive_eta_prox, simplex_prox
from minimax import optimize_NCWC
from utils import project_onto_simplex


torch.set_default_dtype(torch.float32)

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "celeba"
WANDB_PROJECT = "minimax"

# Implementation note for the bilevel variable mapping used below:
#   x_1 = ResNet50 backbone parameters
#   y_1 = classifier weights in R^{d x C}
#   y_2 = train-group simplex weights
#   x_2 = validation-group simplex weights
#   z_1 = classifier copy in the reformulation
#   z_2 = train-group simplex copy in the reformulation
#   eta = learnable scalar DRO regularizer coefficient


class _NoOpRun:
    def __init__(self):
        self.summary = {}

    def log(self, *args, **kwargs):
        del args, kwargs

    def finish(self, *args, **kwargs):
        del args, kwargs


def _resolve_wandb(mode):
    if mode == "disabled":
        return None
    import wandb

    return wandb


def init_run(args):
    wandb = _resolve_wandb(args.wandb_mode)
    if wandb is None:
        return _NoOpRun()
    run = wandb.init(
        project=WANDB_PROJECT,
        mode=args.wandb_mode,
        config=vars(args),
        reinit=True,
        name=f"dro-celeba-seed{args.seed}",
        tags=["dro", "celeba", "stochastic-ncwc"],
    )
    if hasattr(run, "define_metric"):
        step_metric = "ncwc/cumulative_inner_iters"
        run.define_metric(step_metric)
        for metric in (
            "ncwc/objective",
            "ncwc/final_diff",
            "ncwc/completed_outer_iters",
            "ncwc/inner_terminated",
            "dro/h",
            "dro/train_loss",
            "dro/val_loss",
        ):
            run.define_metric(metric, step_metric=step_metric)
    return run


def ensure_finite_metrics(metrics, context, solver_lip_h):
    for key, value in metrics.items():
        if not math.isfinite(float(value)):
            raise RuntimeError(
                f"Non-finite metric detected during {context}: {key}={value}. "
                f"Increase --solver_lip_h (current {solver_lip_h}) or reduce the solver iteration budget."
            )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Stochastic NCWC CelebA DRO experiment")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--target_name", default="Blond_Hair")
    parser.add_argument("--confounder_name", default="Male")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_fraction", type=float, default=1.0) # keeps that fraction of the official CelebA train split before building the balanced train loader.
    parser.add_argument("--val_fraction", type=float, default=1.0)
    parser.add_argument("--test_fraction", type=float, default=1.0)
    parser.add_argument("--augment_data", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_from_scratch", action="store_true", default=False)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--eta_init", type=float, default=10.0)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--solver_lip_h", type=float, default=10000.0)
    parser.add_argument("--solver_d_y", type=float, default=100.0)
    parser.add_argument("--solver_epsilon", type=float, default=1e-4)
    parser.add_argument("--solver_epsilon_0", type=float, default=1.0)
    parser.add_argument("--solver_lr", type=float, default=1.0)
    parser.add_argument("--solver_max_outer_iters", type=int, default=1000)
    parser.add_argument("--solver_inner_max_iters", type=int, default=1000)
    parser.add_argument("--monitor_batches", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--wandb_mode", default="online", choices=("online", "offline", "disabled"))
    return parser.parse_args()


def make_progress_logger(run):
    def callback(payload):
        step = payload["call_cumulative_scsc_inner_iters"]
        metrics = payload.get("metrics") or {}
        objective = payload.get("objective")
        config = getattr(run, "config", {})
        solver_lip_h = config.get("solver_lip_h", float("nan")) if hasattr(config, "get") else float("nan")
        if metrics:
            ensure_finite_metrics(metrics, "NCWC progress", solver_lip_h)
        if objective is not None and not math.isfinite(float(objective)):
            raise RuntimeError(
                f"Non-finite objective detected during NCWC progress: objective={objective}. "
                f"Increase --solver_lip_h (current {solver_lip_h}) or reduce the solver iteration budget."
            )
        log_payload = {
            "ncwc/cumulative_inner_iters": step,
            "ncwc/objective": objective,
            "ncwc/final_diff": payload.get("final_diff"),
            "ncwc/completed_outer_iters": payload.get("completed_outer_iters"),
            "ncwc/inner_terminated": float(bool(payload.get("inner_terminated"))),
        }
        log_payload.update(metrics)
        run.log(log_payload)
        completed_outer_iters = payload.get("completed_outer_iters")
        if completed_outer_iters is None:
            return
        print(
            f"outer={completed_outer_iters} "
            f"h={metrics.get('dro/h', float('nan')):.4f} "
            f"train={metrics.get('dro/train_loss', float('nan')):.4f} "
            f"val={metrics.get('dro/val_loss', float('nan')):.4f} "
            f"eta={metrics.get('dro/eta', float('nan')):.4f} "
            f"diff={payload.get('final_diff')}",
            flush=True,
        )

    return callback


def main():
    args = parse_args()
    if args.batch_size < 4:
        raise ValueError("batch_size must be at least 4 so each batch can contain all 4 groups")

    set_seed(args.seed)
    device = torch.device(args.device)
    run = init_run(args)
    start_time = time.perf_counter()

    data_bundle = build_celeba_bundle(
        root_dir=args.dataset_root,
        target_name=args.target_name,
        confounder_name=args.confounder_name,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=args.augment_data,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        monitor_batches=args.monitor_batches,
        device=device,
    )

    backbone, feature_dim = build_resnet50_backbone(pretrained=not args.train_from_scratch)
    backbone = backbone.to(device)
    classifier = init_classifier(feature_dim, 2, device=device)  # y_1
    eta = torch.nn.Parameter(torch.tensor([args.eta_init], device=device))
    train_simplex = torch.nn.Parameter(
        torch.full((data_bundle["num_groups"],), 1.0 / data_bundle["num_groups"], device=device)
    )  # y_2
    val_simplex = torch.nn.Parameter(
        torch.full((data_bundle["num_groups"],), 1.0 / data_bundle["num_groups"], device=device)
    )  # x_2
    classifier_copy = torch.nn.Parameter(classifier.detach().clone())  # z_1
    train_simplex_copy = torch.nn.Parameter(train_simplex.detach().clone())  # z_2

    # min block = (x_1, eta, y_1, y_2)
    min_block = list(backbone.parameters()) + [eta, classifier, train_simplex]
    # max block = (x_2, z_1, z_2)
    max_block = [val_simplex, classifier_copy, train_simplex_copy]

    problem = CelebADROProblem(
        backbone=backbone,
        num_groups=data_bundle["num_groups"],
        train_provider=data_bundle["train_provider"],
        val_provider=data_bundle["val_provider"],
        train_monitor_batches=data_bundle["train_monitor_batches"],
        val_monitor_batches=data_bundle["val_monitor_batches"],
        rho=args.rho,
        device=device,
    )

    min_prox = [identity_prox] * len(list(backbone.parameters())) + [
        positive_eta_prox(args.eta_min),
        identity_prox,
        simplex_prox(project_onto_simplex),
    ] # [prox for x_1] + [prox for eta] + [prox for y_1] + [prox for y_2]
    max_prox = [
        simplex_prox(project_onto_simplex),
        identity_prox,
        simplex_prox(project_onto_simplex),
    ] # [prox for x_2] + [prox for z_1] + [prox for z_2]

    initial_metrics = problem.monitor_metrics(min_block, max_block)
    ensure_finite_metrics(initial_metrics, "initial evaluation", args.solver_lip_h)
    print(
        f"initial h={initial_metrics['dro/h']:.4f} "
        f"train={initial_metrics['dro/train_loss']:.4f} "
        f"val={initial_metrics['dro/val_loss']:.4f}",
        flush=True,
    )

    solver_stats = optimize_NCWC(
        min_block,
        max_block,
        problem.h_func(min_block, max_block),
        lip_h=args.solver_lip_h,
        D_y=args.solver_d_y,
        prox_x=min_prox,
        prox_y=max_prox,
        epsilon=args.solver_epsilon,
        epsilon_0=args.solver_epsilon_0,
        sub_routine="stochastic",
        lr=args.solver_lr,
        max_iter=args.solver_max_outer_iters,
        subproblem_max_iter=args.solver_inner_max_iters,
        verbose=True,
        log_every=args.log_every,
        objective_func=problem.monitor_objective,
        metrics_func=problem.monitor_metrics,
        progress_callback=make_progress_logger(run),
    )

    final_metrics = problem.monitor_metrics(min_block, max_block)
    ensure_finite_metrics(final_metrics, "final evaluation", args.solver_lip_h)
    test_metrics = problem.evaluate_loader(classifier, data_bundle["test_loader"], prefix="test")
    ensure_finite_metrics(test_metrics, "test evaluation", args.solver_lip_h)
    elapsed = time.perf_counter() - start_time
    loss_decreased = (
        final_metrics["dro/h"] <= initial_metrics["dro/h"]
        and final_metrics["dro/train_loss"] <= initial_metrics["dro/train_loss"]
        and final_metrics["dro/val_loss"] <= initial_metrics["dro/val_loss"]
    )
    summary = {
        "instance/elapsed_sec": elapsed,
        "instance/feature_dim": feature_dim,
        "instance/solver_num_outer_iters": solver_stats["num_outer_iters"],
        "instance/solver_num_inner_iters": solver_stats["num_inner_iters"],
        "instance/solver_final_diff": solver_stats["final_diff"],
        "instance/loss_decreased": float(loss_decreased),
        "instance/initial_h": initial_metrics["dro/h"],
        "instance/final_h": final_metrics["dro/h"],
        "instance/initial_train_loss": initial_metrics["dro/train_loss"],
        "instance/final_train_loss": final_metrics["dro/train_loss"],
        "instance/initial_val_loss": initial_metrics["dro/val_loss"],
        "instance/final_val_loss": final_metrics["dro/val_loss"],
        "instance/test_accuracy": test_metrics["test/accuracy"],
        "instance/test_worst_group_accuracy": test_metrics["test/worst_group_accuracy"],
        "instance/test_loss": test_metrics["test/loss"],
    }
    summary.update({f"instance/{key.replace('/', '_')}": value for key, value in test_metrics.items()})
    if hasattr(run, "summary"):
        for key, value in summary.items():
            run.summary[key] = value
    run.log(
        {
            "ncwc/cumulative_inner_iters": solver_stats["num_inner_iters"],
            **test_metrics,
        }
    )
    run.finish()

    print(
        f"final h={final_metrics['dro/h']:.4f} "
        f"train={final_metrics['dro/train_loss']:.4f} "
        f"val={final_metrics['dro/val_loss']:.4f} "
        f"test_acc={test_metrics['test/accuracy']:.4f} "
        f"test_worst={test_metrics['test/worst_group_accuracy']:.4f} "
        f"eta={final_metrics['dro/eta']:.4f} "
        f"loss_decreased={int(loss_decreased)} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
