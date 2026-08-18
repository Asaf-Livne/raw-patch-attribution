"""A generic class-folder-per-generator dataset.

Every benchmark (DRAGON, OpenFake, GenImage, ...) uses the same layout: a JSON
label map `{class_name: relative_path}` pointing at per-class image folders
(a dict is also accepted directly, e.g. a known/unknown subset carved out of
a larger map for open-set experiments), either already split into
`<class>/{train,val,test}/` (`presplit=True`) or randomly split here via a
seeded per-class shuffle.

`mode="train"` returns one random `patch x patch` crop with augmentation
applied on-the-fly (fresh per call, never cached to disk): a full-image
corruption before cropping (so "resize" is a genuine downscale), then a
cheap post-crop flip. `mode="eval"` returns the full image tensor; patch
tiling happens at inference time via `datalib.patches.patchify`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from datalib.augment import freq_filter

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _list_class_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _split_indices(n: int, train_frac: float, val_frac: float, seed: int):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr = round(n * train_frac)
    n_va = round(n * val_frac)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


class PatchFolderDataset(Dataset):
    def __init__(
        self,
        label_map_path: str | Path | dict,
        data_root: str | Path,
        split: str = "train",
        mode: str = "train",
        patch: int = 256,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        split_seed: int = 0,
        augment: Optional[Callable] = None,
        pre_crop_augment: Optional[Callable] = None,
        max_per_class: Optional[int] = None,
        presplit: bool = False,
        input_filter: str = "none",
        filter_k: int = 5,
        deterministic: bool = False,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode must be train/eval, got {mode!r}")

        self.split = split
        self.mode = mode
        self.patch = patch
        self.augment = augment
        self.pre_crop_augment = pre_crop_augment
        self.deterministic = deterministic
        self.data_root = Path(data_root)
        self.input_filter = input_filter
        self.filter_k = int(filter_k)

        if isinstance(label_map_path, dict):
            label_map = label_map_path
        else:
            with open(label_map_path) as f:
                label_map = json.load(f)
        self.class_names: List[str] = sorted(label_map.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples: List[Tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.data_root / label_map[cls]
            if presplit:
                split_dir = cls_dir / split
                if not split_dir.exists():
                    raise FileNotFoundError(f"presplit dir missing: {split_dir}")
                files = _list_class_files(split_dir)
                if max_per_class is not None:
                    files = files[:max_per_class]
                self.samples += [(f, self.class_to_idx[cls]) for f in files]
            else:
                files = _list_class_files(cls_dir)
                if max_per_class is not None:
                    files = files[:max_per_class]
                tr, va, te = _split_indices(len(files), train_frac, val_frac,
                                            split_seed + self.class_to_idx[cls])
                chosen = {"train": tr, "val": va, "test": te}[split]
                self.samples += [(files[i], self.class_to_idx[cls]) for i in chosen]

        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._mean = mean.view(-1, 1, 1).float()
        self._std = std.view(-1, 1, 1).float()

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def class_counts(self) -> List[int]:
        counts = [0] * self.num_classes
        for _, lbl in self.samples:
            counts[lbl] += 1
        return counts

    def __len__(self) -> int:
        return len(self.samples)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        from torchvision.transforms.functional import pil_to_tensor
        t = pil_to_tensor(img.convert("RGB")).float() / 255.0
        if self.input_filter != "none":
            t = freq_filter(t, self.input_filter, self.filter_k)
        if self._mean is not None:
            t = (t - self._mean) / self._std
        return t

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.mode == "eval":
            return self._to_tensor(img), label, {"path": str(path)}

        seed = idx if self.deterministic else None
        if self.pre_crop_augment is not None:
            img = self.pre_crop_augment(img, seed=seed)

        w, h = img.size
        if w < self.patch or h < self.patch:
            scale = self.patch / min(w, h)
            img = img.resize((max(self.patch, round(w * scale)),
                              max(self.patch, round(h * scale))), Image.BICUBIC)
            w, h = img.size

        rng = random.Random(idx) if self.deterministic else random
        x = rng.randint(0, w - self.patch)
        y = rng.randint(0, h - self.patch)
        crop = img.crop((x, y, x + self.patch, y + self.patch))
        if self.augment is not None:
            crop = self.augment(crop, seed=seed)
        return self._to_tensor(crop), label, {"path": str(path)}
