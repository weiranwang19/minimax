"""Local LIBSVM data loading and deterministic train/validation splits."""

import random

import numpy as np
import torch
from sklearn.datasets import load_svmlight_file


DEFAULT_DATASETS = (
    # "breast-cancer_scale",
    # "heart_scale",
    # "ionosphere_scale",
    "german.numer_scale",
    # "australian_scale",
)


def stable_seed(base_seed, *parts):
    seed = int(base_seed) % (2**63 - 1)
    for part in parts:
        for ch in str(part):
            seed = (seed * 1000003 + ord(ch) + 0x9E3779B97F4A7C15) % (2**63 - 1)
    return seed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_dataset_path(data_root, dataset_name):
    return data_root / dataset_name


def processed_dataset_path(processed_data_root, dataset_name, seed, split_idx, validation_fraction):
    val_tag = int(round(validation_fraction * 10000))
    return processed_data_root / f"{dataset_name}_seed{seed}_split{split_idx}_val{val_tag}.pt"


def ensure_raw_dataset(data_root, dataset_name):
    raw_path = raw_dataset_path(data_root, dataset_name)
    if raw_path.exists():
        return raw_path
    raise FileNotFoundError(
        f"Missing raw LIBSVM data file {raw_path}. "
        "This experiment only reads local files; place the dataset under svm/data/ first."
    )


def map_binary_labels(labels):
    labels = np.asarray(labels, dtype=np.float64)
    unique = np.unique(labels)
    if unique.size != 2:
        raise ValueError(f"Expected exactly two labels, got {unique.tolist()}")
    mapped = np.where(labels == unique[0], -1.0, 1.0)
    return mapped.astype(np.float64)


def load_libsvm_dense(path):
    features, labels = load_svmlight_file(path)
    x_np = features.toarray().astype(np.float64, copy=False)
    y_np = map_binary_labels(labels)
    return x_np, y_np


def make_split_indices(num_samples, dataset_name, split_idx, seed, validation_fraction):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"VALIDATION_FRACTION must be in (0, 1), got {validation_fraction}")
    num_val = max(1, int(round(num_samples * validation_fraction)))
    if num_val >= num_samples:
        raise ValueError(f"Validation split leaves no training samples for {num_samples} examples")

    generator = torch.Generator()
    generator.manual_seed(stable_seed(seed, dataset_name, split_idx, "split"))
    permutation = torch.randperm(num_samples, generator=generator)
    val_idx = permutation[:num_val]
    train_idx = permutation[num_val:]
    return train_idx, val_idx


def prepare_dataset_split(
    dataset_name,
    split_idx,
    data_root,
    processed_data_root,
    seed,
    validation_fraction,
    save_processed_data=True,
):
    processed_path = processed_dataset_path(
        processed_data_root,
        dataset_name,
        seed,
        split_idx,
        validation_fraction,
    )
    if save_processed_data and processed_path.exists():
        return torch.load(processed_path, map_location="cpu", weights_only=False)

    raw_path = ensure_raw_dataset(data_root, dataset_name)
    x_np, y_np = load_libsvm_dense(raw_path)
    train_idx, val_idx = make_split_indices(x_np.shape[0], dataset_name, split_idx, seed, validation_fraction)

    x_tensor = torch.as_tensor(x_np, dtype=torch.get_default_dtype())
    y_tensor = torch.as_tensor(y_np, dtype=torch.get_default_dtype())
    split = {
        "dataset_name": dataset_name,
        "split_idx": int(split_idx),
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "x_train": x_tensor[train_idx].clone(),
        "y_train": y_tensor[train_idx].clone(),
        "x_val": x_tensor[val_idx].clone(),
        "y_val": y_tensor[val_idx].clone(),
        "train_indices": train_idx.clone(),
        "val_indices": val_idx.clone(),
    }
    if save_processed_data:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(split, processed_path)
    return split
