"""Checkpoint and JSON summary helpers for SVM experiments."""

import json
import math
from pathlib import Path

import numpy as np
import torch


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return value
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def artifact_stem(dataset_name, split_idx, method, seed):
    return f"{dataset_name}_split{split_idx}_{method}_seed{seed}"


def save_artifacts(
    model_root,
    result_root,
    config,
    problem,
    method,
    c_vec,
    y_vec,
    diagnostics,
    solver_result,
    elapsed,
    extra_tensors=None,
):
    model_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem(problem.dataset_name, problem.split_idx, method, problem.seed)
    model_path = model_root / f"{stem}.pt"
    result_path = result_root / f"{stem}.json"
    w_vec, b_val, xi_vec = problem.unpack_lower(y_vec)
    extra_tensors = extra_tensors or {}

    checkpoint = {
        "config": config,
        "dataset_name": problem.dataset_name,
        "split_idx": int(problem.split_idx),
        "train_indices": problem.train_indices.detach().cpu().clone(),
        "val_indices": problem.val_indices.detach().cpu().clone(),
        "metrics": dict(diagnostics),
        "elapsed_sec": float(elapsed),
        "c": c_vec.detach().cpu().clone(),
        "w": w_vec.detach().cpu().clone(),
        "b": b_val.detach().cpu().clone(),
        "xi": xi_vec.detach().cpu().clone(),
        "solver_result": solver_result,
    }
    for key, value in extra_tensors.items():
        if torch.is_tensor(value):
            checkpoint[key] = value.detach().cpu().clone()
        elif isinstance(value, (list, tuple)):
            checkpoint[key] = [v.detach().cpu().clone() if torch.is_tensor(v) else v for v in value]
        else:
            checkpoint[key] = value
    torch.save(checkpoint, model_path)

    summary = {
        "dataset_name": problem.dataset_name,
        "split_idx": int(problem.split_idx),
        "method": method,
        "config": config,
        "metrics": dict(diagnostics),
        "elapsed_sec": float(elapsed),
        "model_path": model_path,
        "result_path": result_path,
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, indent=2, sort_keys=True)
    return model_path, result_path
