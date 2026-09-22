"""Deterministic patch tiling for inference.

Patches sit on the `patch`-pixel grid. When a side isn't divisible by `patch`,
one extra patch is placed flush with the far edge, overlapping its neighbour, so
every pixel is covered at least once while every other patch keeps the grid
alignment the classifier was trained on.

Per-patch weight `w_i = 1 / overlap_count_for_patch_pixels` is consumed at
aggregation so each pixel contributes uniformly to the image-level prediction.
"""

from __future__ import annotations

import torch


def _starts(dim: int, patch: int) -> list[int]:
    if dim < patch:
        raise ValueError(f"image dim {dim} smaller than patch {patch}")
    starts = [i * patch for i in range(dim // patch)]
    if dim % patch:
        starts.append(dim - patch)
    return starts


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

    dev = image.device
    ar = torch.arange(patch, device=dev)
    rows = torch.tensor(_starts(H, patch), device=dev)[:, None] + ar   # n_y x patch
    cols = torch.tensor(_starts(W, patch), device=dev)[:, None] + ar   # n_x x patch

    # One gather materialises every tile in row-major order: C x n_y x n_x x P x P.
    patches = image[:, rows[:, None, :, None], cols[None, :, None, :]]
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, image.shape[0], patch, patch)

    # Overlap count factorises into a row count times a column count, so each
    # patch's mean coverage is the product of its mean row and column counts.
    cover_y = torch.zeros(H, device=dev).index_add_(0, rows.reshape(-1), torch.ones(rows.numel(), device=dev))
    cover_x = torch.zeros(W, device=dev).index_add_(0, cols.reshape(-1), torch.ones(cols.numel(), device=dev))
    weights = 1.0 / (cover_y[rows].mean(1)[:, None] * cover_x[cols].mean(1)[None, :])

    return patches, weights.reshape(-1)


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
