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
        run.define_metric("dro-simplex/*", step_metric=step_metric)
        for metric in (
            "ncwc/objective",
            "ncwc/final_diff",
            "ncwc/completed_outer_iters",
            "ncwc/inner_terminated",
            "dro/h",
            "dro/train_loss",
            "dro/val_loss",
            "dro/train_accuracy",
            "dro/val_accuracy",
            "dro/train_worst_group_accuracy",
            "dro/val_worst_group_accuracy",
            "dro/train_min_minus_train_max",
            "dro-simplex/reg_loss",
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


def resolve_best_checkpoint_path(args):
    return args.best_checkpoint_path


def _module_state_dict_to_cpu(module):
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in module.state_dict().items()
    }


def _serialize_args(args):
    serialized = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


class _BestCheckpointSaver:
    def __init__(
        self,
        checkpoint_path,
        args,
        backbone,
        classifier,
        eta,
        train_simplex,
        val_simplex,
        classifier_copy,
        train_simplex_copy,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.args = args
        self.backbone = backbone
        self.classifier = classifier
        self.eta = eta
        self.train_simplex = train_simplex
        self.val_simplex = val_simplex
        self.classifier_copy = classifier_copy
        self.train_simplex_copy = train_simplex_copy
        self.best_val_accuracy = None
        self.best_completed_outer_iters = None
        self.best_cumulative_inner_iters = None

    def maybe_save(self, payload):
        metrics = payload.get("metrics") or {}
        if "dro/val_accuracy" not in metrics:
            return False

        val_accuracy = float(metrics["dro/val_accuracy"])
        if self.best_val_accuracy is not None and val_accuracy <= self.best_val_accuracy:
            return False

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "args": _serialize_args(self.args),
                "best_metric_name": "dro/val_accuracy",
                "best_metric_value": val_accuracy,
                "completed_outer_iters": payload.get("completed_outer_iters"),
                "call_cumulative_scsc_inner_iters": payload.get("call_cumulative_scsc_inner_iters"),
                "final_diff": payload.get("final_diff"),
                "objective": payload.get("objective"),
                "metrics": dict(metrics),
                "backbone_state_dict": _module_state_dict_to_cpu(self.backbone),
                "classifier": self.classifier.detach().cpu().clone(),
                "eta": self.eta.detach().cpu().clone(),
                "train_simplex": self.train_simplex.detach().cpu().clone(),
                "val_simplex": self.val_simplex.detach().cpu().clone(),
                "classifier_copy": self.classifier_copy.detach().cpu().clone(),
                "train_simplex_copy": self.train_simplex_copy.detach().cpu().clone(),
            },
            self.checkpoint_path,
        )
        self.best_val_accuracy = val_accuracy
        self.best_completed_outer_iters = payload.get("completed_outer_iters")
        self.best_cumulative_inner_iters = payload.get("call_cumulative_scsc_inner_iters")
        return True


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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_fraction", type=float, default=1.0) # keeps that fraction of the official CelebA train split before building the balanced train loader.
    parser.add_argument("--val_fraction", type=float, default=1.0)
    parser.add_argument("--test_fraction", type=float, default=1.0)
    parser.add_argument("--augment_data", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_from_scratch", action="store_true", default=False)
    parser.add_argument("--rho", type=float, default=100.0)
    parser.add_argument("--eta_init", type=float, default=10.0)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    # L2 coefficient used for R(x_1), R(y_1), and R(z_1) in the DRO objective.
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--solver_lip_h", type=float, default=5000.0)
    parser.add_argument("--solver_lip_tau", type=float, default=None)
    parser.add_argument("--solver_theta", type=float, default=0.1)
    parser.add_argument("--solver_d_y", type=float, default=100.0)
    parser.add_argument("--solver_epsilon", type=float, default=1.0)
    parser.add_argument("--solver_epsilon_0", type=float, default=10.0)
    parser.add_argument("--solver_lr", type=float, default=1.0)
    parser.add_argument("--solver_max_outer_iters", type=int, default=100000)
    parser.add_argument("--solver_inner_max_iters", type=int, default=1)
    parser.add_argument("--monitor_batches", type=int, default=1)
    parser.add_argument("--test_eval_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--best_checkpoint_path", type=Path, default=None)
    parser.add_argument("--wandb_mode", default="online", choices=("online", "offline", "disabled"))
    return parser.parse_args()


def make_progress_logger(run, problem, classifier, test_loader, test_eval_every, solver_lip_h, checkpoint_saver):
    def callback(payload):
        step = payload["call_cumulative_scsc_inner_iters"]
        metrics = payload.get("metrics") or {}
        objective = payload.get("objective")
        if metrics:
            ensure_finite_metrics(metrics, "NCWC progress", solver_lip_h)
            if checkpoint_saver is not None and checkpoint_saver.maybe_save(payload):
                print(
                    f"saved best checkpoint outer={payload.get('completed_outer_iters')} "
                    f"val_acc={checkpoint_saver.best_val_accuracy:.4f} "
                    f"path={checkpoint_saver.checkpoint_path}",
                    flush=True,
                )
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
        completed_outer_iters = payload.get("completed_outer_iters")
        test_metrics = None
        if (
            test_eval_every > 0
            and completed_outer_iters is not None
            and completed_outer_iters > 0
            and completed_outer_iters % test_eval_every == 0
        ):
            test_metrics = problem.evaluate_loader(classifier, test_loader, prefix="test")
            ensure_finite_metrics(
                test_metrics,
                f"test evaluation at outer={completed_outer_iters}",
                solver_lip_h,
            )
            log_payload.update(test_metrics)

        run.log(log_payload)
        if completed_outer_iters is None:
            return
        print(
            f"outer={completed_outer_iters} "
            f"h={metrics.get('dro/h', float('nan')):.4f} "
            f"train={metrics.get('dro/train_loss', float('nan')):.4f} "
            f"train_acc={metrics.get('dro/train_accuracy', float('nan')):.4f} "
            f"val={metrics.get('dro/val_loss', float('nan')):.4f} "
            f"val_acc={metrics.get('dro/val_accuracy', float('nan')):.4f} "
            f"train_gap={metrics.get('dro/train_min_minus_train_max', float('nan')):.4f} "
            f"eta={metrics.get('dro/eta', float('nan')):.4f} "
            f"diff={payload.get('final_diff')}",
            flush=True,
        )
        if test_metrics is not None:
            print(
                f"test@outer={completed_outer_iters} "
                f"acc={test_metrics.get('test/accuracy', float('nan')):.4f} "
                f"worst={test_metrics.get('test/worst_group_accuracy', float('nan')):.4f} "
                f"loss={test_metrics.get('test/loss', float('nan')):.4f}",
                flush=True,
            )

    return callback


def main():
    args = parse_args()
    if args.batch_size < 4:
        raise ValueError("batch_size must be at least 4 so each batch can contain all 4 groups")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

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
    backbone.train()
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
        weight_decay=args.weight_decay,
        device=device,
    )
    checkpoint_path = resolve_best_checkpoint_path(args)
    checkpoint_saver = None
    if checkpoint_path is not None:
        checkpoint_saver = _BestCheckpointSaver(
            checkpoint_path=checkpoint_path,
            args=args,
            backbone=backbone,
            classifier=classifier,
            eta=eta,
            train_simplex=train_simplex,
            val_simplex=val_simplex,
            classifier_copy=classifier_copy,
            train_simplex_copy=train_simplex_copy,
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
        lip_tau=args.solver_lip_tau,
        theta=args.solver_theta,
        objective_func=problem.monitor_objective,
        metrics_func=problem.monitor_metrics,
        progress_callback=make_progress_logger(
            run,
            problem,
            classifier,
            data_bundle["test_loader"],
            args.test_eval_every,
            args.solver_lip_h,
            checkpoint_saver,
        ),
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
    if checkpoint_saver is not None:
        summary["instance/best_val_accuracy"] = checkpoint_saver.best_val_accuracy
        summary["instance/best_checkpoint_outer_iters"] = checkpoint_saver.best_completed_outer_iters
        summary["instance/best_checkpoint_cumulative_inner_iters"] = checkpoint_saver.best_cumulative_inner_iters
    if checkpoint_saver is not None and checkpoint_saver.best_val_accuracy is not None:
        summary["instance/best_checkpoint_path"] = str(checkpoint_saver.checkpoint_path)
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

    final_message = (
        f"final h={final_metrics['dro/h']:.4f} "
        f"train={final_metrics['dro/train_loss']:.4f} "
        f"val={final_metrics['dro/val_loss']:.4f} "
        f"test_acc={test_metrics['test/accuracy']:.4f} "
        f"test_worst={test_metrics['test/worst_group_accuracy']:.4f} "
        f"eta={final_metrics['dro/eta']:.4f} "
    )
    if checkpoint_saver is not None:
        final_message += (
            f"best_val_acc={checkpoint_saver.best_val_accuracy:.4f} "
            f"best_ckpt={checkpoint_saver.checkpoint_path} "
        )
    final_message += f"loss_decreased={int(loss_decreased)} elapsed={elapsed:.1f}s"
    print(final_message, flush=True)


if __name__ == "__main__":
    main()
