#!/usr/bin/env python3
"""Dataset controller for CLIP source training.

Only two dataset modes are supported:
- m58
- visda
"""

from pathlib import Path
from typing import List, Optional
import random

import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_classnames(data_dir: Path, classname_file: Optional[str]) -> List[str]:
    """Load class names from file or infer from subfolder names."""
    if classname_file:
        with open(classname_file, "r", encoding="utf-8-sig") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        if names:
            return names

    # Fallback: infer classes from folder names.
    return sorted([p.name for p in data_dir.iterdir() if p.is_dir()])


class M58Dataset(Dataset):
    """M58 dataset with class-balanced split per class."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        classname_file: Optional[str] = None,
        classnames: Optional[List[str]] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.split_seed = split_seed
        self.image_extensions = {".jpg", ".jpeg", ".png"}
        self.samples = []

        if classnames is not None:
            self.classnames = list(classnames)
        else:
            self.classnames = _load_classnames(self.data_dir, classname_file)

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classnames)}
        self._load_data()

    def _split_samples(self, class_samples: List):
        rng = random.Random(self.split_seed)
        shuffled = list(class_samples)
        rng.shuffle(shuffled)

        total = len(shuffled)
        train_size = max(1, int(self.train_ratio * total))
        val_size = max(1, int(self.val_ratio * total))

        if train_size >= total:
            train_size = total
            val_size = 0
        elif train_size + val_size > total:
            val_size = max(0, total - train_size)

        if self.split == "train":
            return shuffled[:train_size]
        if self.split == "val":
            return shuffled[train_size:train_size + val_size]
        if self.split == "test":
            return shuffled[train_size + val_size:]
        if self.split == "all":
            return shuffled

        raise ValueError(f"Unknown split: {self.split}")

    def _load_data(self):
        """Load M58 data by split for each class."""
        for class_name in self.classnames:
            class_dir = self.data_dir / class_name
            if not class_dir.is_dir():
                continue

            class_idx = self.class_to_idx[class_name]
            class_samples = []

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.image_extensions:
                    class_samples.append((str(img_path), class_idx, class_name))

            if not class_samples:
                continue

            selected = self._split_samples(class_samples)
            self.samples.extend(selected)

        print(f"載入 {self.split} 集: {len(self.samples)} 張圖片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, classname = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "img": image,
            "label": torch.tensor(label, dtype=torch.long),
            "impath": img_path,
            "classname": classname,
        }


class VISDADataset(Dataset):
    """VISDA dataset with class-balanced split from a provided folder."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        classname_file: Optional[str] = None,
        classnames: Optional[List[str]] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.split_seed = split_seed
        self.image_extensions = {".jpg", ".jpeg", ".png"}
        self.samples = []

        if classnames is not None:
            self.classnames = list(classnames)
        else:
            self.classnames = _load_classnames(self.data_dir, classname_file)

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classnames)}
        self._load_data()

    def _split_samples(self, class_samples: List):
        rng = random.Random(self.split_seed)
        shuffled = list(class_samples)
        rng.shuffle(shuffled)

        total = len(shuffled)
        train_size = max(1, int(self.train_ratio * total))
        val_size = max(1, int(self.val_ratio * total))

        if train_size >= total:
            train_size = total
            val_size = 0
        elif train_size + val_size > total:
            val_size = max(0, total - train_size)

        if self.split == "train":
            return shuffled[:train_size]
        if self.split == "val":
            return shuffled[train_size:train_size + val_size]
        if self.split == "test":
            return shuffled[train_size + val_size:]
        if self.split == "all":
            return shuffled

        raise ValueError(f"Unknown split: {self.split}")

    def _load_data(self):
        """Load VISDA data by split for each class from the provided folder."""
        for class_name in self.classnames:
            class_dir = self.data_dir / class_name
            if not class_dir.is_dir():
                continue

            class_idx = self.class_to_idx[class_name]
            class_samples = []
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.image_extensions:
                    class_samples.append((str(img_path), class_idx, class_name))

            if not class_samples:
                continue

            selected = self._split_samples(class_samples)
            self.samples.extend(selected)

        print(f"載入 {self.split} 集: {len(self.samples)} 張圖片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, classname = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "img": image,
            "label": torch.tensor(label, dtype=torch.long),
            "impath": img_path,
            "classname": classname,
        }


def create_dataset(
    data_dir: str,
    split: str,
    classname_file: Optional[str] = None,
    transform=None,
    classnames: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 42,
    dataset_name: str = "m58",
):
    if dataset_name == "m58":
        return M58Dataset(
            data_dir=data_dir,
            split=split,
            transform=transform,
            classname_file=classname_file,
            classnames=classnames,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
        )

    if dataset_name == "visda":
        return VISDADataset(
            data_dir=data_dir,
            split=split,
            transform=transform,
            classname_file=classname_file,
            classnames=classnames,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
        )

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def resolve_dataset_dirs(
    data_dir: str,
    dataset_name: str = "m58",
):
    """Use the provided data_dir directly for both source/target paths."""
    _ = dataset_name
    return data_dir, data_dir


def create_train_val_test_datasets(
    data_dir: str,
    classname_file: Optional[str],
    train_transform=None,
    eval_transform=None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 42,
    dataset_name: str = "m58",
):
    source_dir, _ = resolve_dataset_dirs(
        data_dir=data_dir,
        dataset_name=dataset_name,
    )

    if dataset_name not in {"m58", "visda"}:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    split_map = {
        "train": "train",
        "val": "val",
        "test": "test",
    }

    def _build(split_name, transform, classnames=None):
        kwargs = {
            "data_dir": source_dir,
            "dataset_name": dataset_name,
            "classname_file": classname_file,
            "split": split_map[split_name],
            "transform": transform,
            "classnames": classnames,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "split_seed": split_seed,
        }
        return create_dataset(**kwargs)

    train_dataset = _build("train", train_transform)
    shared_classnames = train_dataset.classnames
    val_dataset = _build("val", eval_transform, classnames=shared_classnames)
    test_dataset = _build("test", eval_transform, classnames=shared_classnames)

    return train_dataset, val_dataset, test_dataset
