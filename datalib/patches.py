"""Deterministic patch tiling for inference.

When a side isn't divisible by `patch`, generate overlapping patches such that
every pixel is covered at least once. Step is `floor((dim - patch) / (k - 1))`
where `k = ceil(dim / patch)`.

Per-patch weight `w_i = 1 / overlap_count_for_patch_pixels` is consumed at
aggregation so each pixel contributes uniformly to the image-level prediction.
"""

from __future__ import annotations

from math import ceil

import torch


def _starts(dim: int, patch: int) -> list[int]:
    if dim < patch:
        raise ValueError(f"image dim {dim} smaller than patch {patch}")
    if dim == patch:
        return [0]
    k = ceil(dim / patch)
    if k == 1:
        return [0]
    step = (dim - patch) // (k - 1)
    return [i * step for i in range(k - 1)] + [dim - patch]


def patchify(image: torch.Tensor, patch: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile an image into overlapping patches with per-patch weights.

    Args:
        image: float tensor `C x H x W` in [0, 1] or normalized.
        patch: patch side length in pixels.

    Returns:
        patches: `N x C x patch x patch`
        weights: `N` -- each = 1 / mean overlap count of the patch's pixels
    """
    if image.dim() != 3:
        raise ValueError(f"expected C x H x W, got shape {tuple(image.shape)}")
    _, H, W = image.shape

    ys = _starts(H, patch)
    xs = _starts(W, patch)

    cover = torch.zeros((H, W), dtype=torch.float32, device=image.device)
    for y in ys:
        for x in xs:
            cover[y:y + patch, x:x + patch] += 1.0

    patches, weights = [], []
    for y in ys:
        for x in xs:
            patches.append(image[:, y:y + patch, x:x + patch])
            weights.append(1.0 / cover[y:y + patch, x:x + patch].mean().item())

    return torch.stack(patches, dim=0), torch.tensor(weights, dtype=torch.float32)


def aggregate(logits: torch.Tensor, weights: torch.Tensor, mode: str) -> torch.Tensor:
    """Combine per-patch logits into one image-level score (Eq. 3 of the paper).

    Args:
        logits: `N x C` per-patch class logits.
        weights: `N` per-patch weights from `patchify` (uniform pixel coverage).
        mode: one of `AGGREGATIONS`. `logit_avg` is the paper's protocol
            everywhere; `prob_avg` averages in probability space instead.

    Returns:
        `C` logit-like score whose argmax is the image-level prediction.
    """
    w = weights / weights.sum()
    if mode == "logit_avg":
        return (logits * w.unsqueeze(1)).sum(0)
    if mode == "prob_avg":
        probs = logits.softmax(-1)
        return (probs * w.unsqueeze(1)).sum(0).log()
    raise ValueError(f"unknown aggregation mode: {mode!r}")


AGGREGATIONS = {"logit_avg", "prob_avg"}
