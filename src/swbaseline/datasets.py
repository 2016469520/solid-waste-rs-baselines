from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class NpyPatchDataset(Dataset):
    def __init__(
        self,
        x_path: str | Path,
        y_path: str | Path,
        scale: float = 2047.0,
        augment: bool = False,
        hflip_p: float = 0.5,
        vflip_p: float = 0.5,
        rotate90: bool = True,
        noise_std: float = 0.0,
    ) -> None:
        self.x = np.load(str(x_path), mmap_mode="r")
        self.y = np.load(str(y_path), mmap_mode="r").reshape(-1).astype(np.int64)
        if len(self.x) != len(self.y):
            raise ValueError(f"x/y length mismatch: {x_path} has {len(self.x)}, {y_path} has {len(self.y)}")
        self.scale = float(scale)
        self.augment = augment
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotate90 = rotate90
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = np.asarray(self.x[idx]).astype(np.float32)
        label = int(self.y[idx])
        image = self._augment(image) if self.augment else image
        image = np.ascontiguousarray(image / self.scale)
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image).float(), torch.tensor(label, dtype=torch.long)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.hflip_p:
            image = np.flip(image, axis=1)
        if np.random.rand() < self.vflip_p:
            image = np.flip(image, axis=0)
        if self.rotate90:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k, axes=(0, 1))
        if self.noise_std > 0:
            image = image + np.random.normal(0, self.noise_std * self.scale, size=image.shape).astype(np.float32)
        return image


class PatchFolderDataset(Dataset):
    def __init__(self, patch_dir: str | Path, scale: float = 2047.0) -> None:
        self.patch_dir = Path(patch_dir)
        info_path = self.patch_dir / "patches_spatial_info.json"
        if info_path.exists():
            with info_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
            self.filenames = [row["filename"] for row in info]
        else:
            self.filenames = sorted(p.name for p in self.patch_dir.glob("*.npy"))
        if not self.filenames:
            raise ValueError(f"No patch .npy files found in {self.patch_dir}")
        self.scale = float(scale)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        filename = self.filenames[idx]
        image = np.load(str(self.patch_dir / filename)).astype(np.float32)
        image = np.ascontiguousarray(image / self.scale)
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image).float(), filename


def make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    labels = labels.astype(np.int64)
    counts = np.bincount(labels, minlength=2)
    counts = np.maximum(counts, 1)
    class_weights = 1.0 / counts
    sample_weights = torch.from_numpy(class_weights[labels]).float()
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def dataset_summary(x_path: str | Path, y_path: str | Path) -> Dict[str, object]:
    x = np.load(str(x_path), mmap_mode="r")
    y = np.load(str(y_path), mmap_mode="r").reshape(-1)
    labels, counts = np.unique(y.astype(int), return_counts=True)
    return {
        "x_path": str(x_path),
        "y_path": str(y_path),
        "x_shape": tuple(int(v) for v in x.shape),
        "x_dtype": str(x.dtype),
        "y_shape": tuple(int(v) for v in y.shape),
        "y_dtype": str(y.dtype),
        "label_counts": {int(k): int(v) for k, v in zip(labels, counts)},
    }

