import math
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CELEBA_SPLITS = {"train": 0, "val": 1, "test": 2}


def build_celeba_transform(train, augment):
    target_resolution = (224, 224)
    if train and augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(1.0, 1.3333333333333333),
                    interpolation=Image.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CelebASplitDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        target_name,
        confounder_name,
        augment=False,
        fraction=1.0,
        seed=0,
        eval_mode=False,
    ):
        if split not in CELEBA_SPLITS:
            raise ValueError(f"Unknown split: {split}")
        if not 0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")

        root_dir = Path(root_dir)
        data_dir = root_dir / "data"
        attrs_df = pd.read_csv(data_dir / "list_attr_celeba.csv")
        split_df = pd.read_csv(data_dir / "list_eval_partition.csv")
        merged = attrs_df.merge(split_df, on="image_id")
        merged = merged[merged["partition"] == CELEBA_SPLITS[split]].reset_index(drop=True)
        if fraction < 1.0:
            merged = merged.sample(frac=fraction, random_state=seed).reset_index(drop=True)

        attr_cols = [col for col in merged.columns if col not in ("image_id", "partition")]
        merged.loc[:, attr_cols] = merged.loc[:, attr_cols].replace(-1, 0)
        if target_name not in merged.columns:
            raise ValueError(f"Unknown target attribute: {target_name}")
        if confounder_name not in merged.columns:
            raise ValueError(f"Unknown confounder attribute: {confounder_name}")

        self.root_dir = root_dir
        self.data_dir = data_dir / "img_align_celeba"
        self.split = split
        self.target_name = target_name
        self.confounder_name = confounder_name
        self.filenames = merged["image_id"].tolist()
        self.labels = torch.as_tensor(merged[target_name].to_numpy(), dtype=torch.long)
        self.confounders = torch.as_tensor(merged[confounder_name].to_numpy(), dtype=torch.long)
        self.groups = (self.labels * 2 + self.confounders).to(dtype=torch.long)
        self.num_groups = 4
        self.transform = build_celeba_transform(train=split == "train" and not eval_mode, augment=augment)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.data_dir / self.filenames[index]).convert("RGB")
        image = self.transform(image)
        return image, self.labels[index], self.groups[index]

    def group_counts(self):
        counts = torch.bincount(self.groups, minlength=self.num_groups)
        return counts.to(dtype=torch.long)


class GroupBalancedBatchSampler(Sampler):
    def __init__(self, group_array, batch_size, num_groups=4, num_batches=None, seed=0):
        if batch_size < num_groups:
            raise ValueError(f"batch_size must be at least {num_groups}, got {batch_size}")
        self.group_array = torch.as_tensor(group_array, dtype=torch.long)
        self.batch_size = batch_size
        self.num_groups = num_groups
        self.num_batches = num_batches or max(1, math.ceil(len(self.group_array) / batch_size))
        self.seed = seed
        self._epoch = 0
        self.group_indices = []
        for group_idx in range(num_groups):
            indices = torch.nonzero(self.group_array == group_idx, as_tuple=False).view(-1).tolist()
            if not indices:
                raise ValueError(f"group {group_idx} has no samples")
            self.group_indices.append(indices)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        base_order = list(range(self.num_groups))
        for _ in range(self.num_batches):
            rng.shuffle(base_order)
            counts = [1] * self.num_groups
            for extra_idx in range(self.batch_size - self.num_groups):
                counts[base_order[extra_idx % self.num_groups]] += 1
            batch = []
            for group_idx, group_count in enumerate(counts):
                pool = self.group_indices[group_idx]
                for _ in range(group_count):
                    batch.append(pool[rng.randrange(len(pool))])
            rng.shuffle(batch)
            yield batch


class CyclingBatchProvider:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.iterator = iter(loader)

    def next_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return tuple(t.to(self.device, non_blocking=True) for t in batch)


def capture_monitor_batches(loader, num_batches):
    iterator = iter(loader)
    batches = []
    for _ in range(num_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batches.append(tuple(t.clone() for t in batch))
    return batches


def _make_loader(dataset, batch_size, num_workers, seed, num_batches=None):
    sampler = GroupBalancedBatchSampler(
        dataset.groups,
        batch_size=batch_size,
        num_groups=dataset.num_groups,
        num_batches=num_batches,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def _make_eval_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def build_celeba_bundle(
    root_dir,
    target_name,
    confounder_name,
    batch_size,
    eval_batch_size,
    num_workers,
    seed,
    augment,
    train_fraction=1.0,
    val_fraction=1.0,
    test_fraction=1.0,
    monitor_batches=1,
    device=None,
):
    train_dataset = CelebASplitDataset(
        root_dir,
        split="train",
        target_name=target_name,
        confounder_name=confounder_name,
        augment=augment,
        fraction=train_fraction,
        seed=seed,
        eval_mode=False,
    )
    train_eval_dataset = CelebASplitDataset(
        root_dir,
        split="train",
        target_name=target_name,
        confounder_name=confounder_name,
        augment=False,
        fraction=train_fraction,
        seed=seed,
        eval_mode=True,
    )
    val_dataset = CelebASplitDataset(
        root_dir,
        split="val",
        target_name=target_name,
        confounder_name=confounder_name,
        augment=False,
        fraction=val_fraction,
        seed=seed,
        eval_mode=True,
    )
    test_dataset = CelebASplitDataset(
        root_dir,
        split="test",
        target_name=target_name,
        confounder_name=confounder_name,
        augment=False,
        fraction=test_fraction,
        seed=seed,
        eval_mode=True,
    )

    train_loader = _make_loader(train_dataset, batch_size, num_workers, seed)
    val_loader = _make_loader(val_dataset, batch_size, num_workers, seed + 1)
    train_monitor_loader = _make_loader(
        train_eval_dataset,
        batch_size,
        num_workers,
        seed + 2,
        num_batches=monitor_batches,
    )
    val_monitor_loader = _make_loader(
        val_dataset,
        batch_size,
        num_workers,
        seed + 3,
        num_batches=monitor_batches,
    )
    test_loader = _make_eval_loader(test_dataset, eval_batch_size, num_workers)

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_provider": CyclingBatchProvider(train_loader, device=device),
        "val_provider": CyclingBatchProvider(val_loader, device=device),
        "train_monitor_batches": capture_monitor_batches(train_monitor_loader, monitor_batches),
        "val_monitor_batches": capture_monitor_batches(val_monitor_loader, monitor_batches),
        "test_loader": test_loader,
        "num_groups": train_dataset.num_groups,
        "train_group_counts": train_dataset.group_counts(),
        "val_group_counts": val_dataset.group_counts(),
        "test_group_counts": test_dataset.group_counts(),
    }
