"""Per-channel normalization stats over a training dataset.

Streams the dataset once (unnormalized), accumulates per-channel sum and
sum-of-squares, returns `(mean, std)` as 1D `[C]` tensors. Cached to disk so
repeat runs against the same data/config skip the pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_channel_stats(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """`dataset` must be unnormalized (raw [0, 1] tensors)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)

    n_pixels = 0
    csum = csqsum = None
    for i, batch in enumerate(tqdm(loader, desc="channel stats")):
        if max_batches is not None and i >= max_batches:
            break
        x = batch[0]
        B, C, H, W = x.shape
        if csum is None:
            csum = torch.zeros(C, dtype=torch.float64)
            csqsum = torch.zeros(C, dtype=torch.float64)
        csum += x.sum(dim=(0, 2, 3)).double()
        csqsum += (x.double() ** 2).sum(dim=(0, 2, 3))
        n_pixels += B * H * W

    if n_pixels == 0:
        raise RuntimeError("compute_channel_stats: empty dataset")

    mean = (csum / n_pixels).float()
    var = (csqsum / n_pixels - mean.double() ** 2).clamp(min=0.0)
    std = var.sqrt().float() + 1e-8
    return mean, std


def load_or_compute_stats(
    cache_path: Path,
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_path = Path(cache_path)
    if cache_path.exists():
        d = torch.load(cache_path, map_location="cpu", weights_only=True)
        return d["mean"], d["std"]
    mean, std = compute_channel_stats(dataset, batch_size, num_workers, max_batches)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std}, cache_path)
    return mean, std
