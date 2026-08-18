"""Evaluate a trained checkpoint with multi-patch aggregation (closed-set
attribution). Also the entry point for the robustness experiment (run on a
config whose dataset points at corrupted/held-out data) and the "what does
the CNN see" frequency-filter analysis (`dataset.input_filter` in the
eval config: none/lowpass/highpass/fftmag).

Usage:
    python eval.py --checkpoint runs/dragon/best.pt --config configs/dragon.yaml \
                   --split test --aggregation logit_avg --num_patches "1 4 16"

Aggregation rules (datalib.patches.AGGREGATIONS): logit_avg (the paper's
protocol) or prob_avg.

`--num_patches` accepts a list (e.g. "1 4 16"); eligibility is decided per
image (the largest requested n with ceil(sqrt(n))*patch <= image size), so a
mixed-resolution dataset scores each image at the right budget. A "best"
accumulator additionally reports the max-eligible-n number per image.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from math import ceil, sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import load_config
from datalib.dataset import PatchFolderDataset
from datalib.patches import AGGREGATIONS, aggregate, patchify
from models.cnn import build_model


def build_eval_dataset(cfg: dict, split: str) -> PatchFolderDataset:
    ds_cfg = cfg["dataset"]
    return PatchFolderDataset(
        label_map_path=ds_cfg["label_map"],
        data_root=ds_cfg["data_root"],
        split=split,
        mode="eval",
        patch=ds_cfg.get("patch", 256),
        train_frac=ds_cfg.get("train_frac", 0.8),
        val_frac=ds_cfg.get("val_frac", 0.1),
        split_seed=ds_cfg.get("split_seed", 0),
        max_per_class=ds_cfg.get("max_per_class"),
        presplit=ds_cfg.get("presplit", False),
        input_filter=ds_cfg.get("input_filter", "none"),
        filter_k=ds_cfg.get("filter_k", 5),
    )


def _center_crop(image: torch.Tensor, patch: int) -> torch.Tensor:
    _, H, W = image.shape
    y, x = max(0, (H - patch) // 2), max(0, (W - patch) // 2)
    return image[:, y:y + patch, x:x + patch]


def _resize_up_to_patch(image: torch.Tensor, patch: int) -> torch.Tensor:
    _, H, W = image.shape
    if H >= patch and W >= patch:
        return image
    scale = patch / min(H, W)
    new_h, new_w = max(patch, round(H * scale)), max(patch, round(W * scale))
    return F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bicubic",
                         align_corners=False).squeeze(0).clamp(0.0, 1.0)


def _select_central_patches(patches, weights, n_y, n_x, k):
    cy, cx = (n_y - 1) / 2.0, (n_x - 1) / 2.0
    coords = [(i, j) for i in range(n_y) for j in range(n_x)]
    dist = sorted((abs(i - cy) + abs(j - cx), idx) for idx, (i, j) in enumerate(coords))
    keep = sorted(idx for _, idx in dist[:k])
    return patches[keep], weights[keep]


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = build_model(ckpt["config"], num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean = ckpt["channel_mean"].view(-1, 1, 1).to(device).float()
    std = ckpt["channel_std"].view(-1, 1, 1).to(device).float()
    return model, mean, std, class_names


def _new_acc(num_classes: int) -> dict:
    return {
        "n_correct": 0, "n_top5": 0, "n_total": 0,
        "confusion": np.zeros((num_classes, num_classes), dtype=np.int64),
        "per_class_correct": np.zeros(num_classes, dtype=np.int64),
        "per_class_total": np.zeros(num_classes, dtype=np.int64),
        "probs": [], "labels": [], "secs": [],
    }


def _finalize(acc: dict, class_names: list[str]) -> dict | None:
    if acc["n_total"] == 0:
        return None
    num_classes = len(class_names)
    per_class = acc["per_class_correct"] / np.maximum(1, acc["per_class_total"])
    probs_arr = np.stack(acc["probs"])
    labels_arr = np.asarray(acc["labels"])
    macro_auc = None
    if num_classes >= 2 and len(np.unique(labels_arr)) >= 2:
        from sklearn.metrics import roc_auc_score
        y_onehot = np.eye(num_classes)[labels_arr]
        try:
            macro_auc = float(roc_auc_score(y_onehot, probs_arr, average="macro", multi_class="ovr"))
        except ValueError:
            pass
    times_arr = np.asarray(acc["secs"], dtype=np.float64)
    return {
        "top1": acc["n_correct"] / acc["n_total"],
        "top5": acc["n_top5"] / acc["n_total"],
        "macro_auc": macro_auc,
        "per_class_accuracy": {class_names[i]: float(per_class[i]) for i in range(num_classes)},
        "n_samples": int(acc["n_total"]),
        "mean_latency_s": float(times_arr.mean()) if len(times_arr) else 0.0,
        "confusion": acc["confusion"],
    }


@torch.no_grad()
def run_eval(model, class_names, mean, std, dataset, aggregation: str,
            requested_ns: list[int], device: torch.device) -> dict:
    """Evaluate `dataset` with multi-patch aggregation. Returns a dict keyed by
    `n{N}` (one accumulator per requested patch budget that was eligible for at
    least one image) plus `best_per_image` (each image's largest eligible N).
    """
    num_classes = len(class_names)
    patch = dataset.patch

    def eligible_for(image_size: int) -> list[int]:
        return [n for n in requested_ns if n == 1 or ceil(sqrt(n)) * patch <= image_size]

    state = {n: _new_acc(num_classes) for n in requested_ns}
    best = _new_acc(num_classes)
    is_cuda = device.type == "cuda"

    for i in range(len(dataset)):
        image, label, _ = dataset[i]
        image = image.to(device)
        if min(image.shape[1], image.shape[2]) < patch:
            image = _resize_up_to_patch(image, patch)
        _, H, W = image.shape
        img_size = min(H, W)
        ns_here = eligible_for(img_size)
        if not ns_here:
            continue
        max_n = max(ns_here)

        if max_n == 1:
            base_tiles, base_w = _center_crop(image, patch).unsqueeze(0), torch.ones(1, device=device)
            n_y = n_x = 1
        else:
            tiles, w = patchify(image.cpu(), patch=patch)
            base_tiles, base_w = tiles.to(device), w.to(device)
            n_y, n_x = max(1, ceil(H / patch)), max(1, ceil(W / patch))

        for n in ns_here:
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if n == 1:
                patches, weights = (base_tiles, base_w) if max_n == 1 else \
                    _select_central_patches(base_tiles, base_w, n_y, n_x, 1)
            elif base_tiles.shape[0] > n:
                patches, weights = _select_central_patches(base_tiles, base_w, n_y, n_x, n)
            else:
                patches, weights = base_tiles, base_w

            logits = model((patches - mean) / std)
            score = aggregate(logits.float().cpu(), weights.float().cpu(), aggregation)
            probs = score.softmax(-1).numpy()
            pred = int(score.argmax().item())
            top5 = score.topk(min(5, num_classes)).indices.tolist()

            for acc in ([state[n]] if n != max_n else [state[n], best]):
                acc["n_total"] += 1
                acc["n_correct"] += int(pred == label)
                acc["n_top5"] += int(label in top5)
                acc["confusion"][label, pred] += 1
                acc["per_class_total"][label] += 1
                acc["per_class_correct"][label] += int(pred == label)
                acc["probs"].append(probs)
                acc["labels"].append(int(label))
                if is_cuda:
                    torch.cuda.synchronize()
                acc["secs"].append(time.perf_counter() - t0)

    results = {}
    for n in sorted(requested_ns):
        r = _finalize(state[n], class_names)
        if r is not None:
            results[f"n{n}"] = r
    best_r = _finalize(best, class_names)
    if best_r is not None:
        results["best_per_image"] = best_r
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--aggregation", default="logit_avg", choices=sorted(AGGREGATIONS))
    p.add_argument("--num_patches", type=str, default="4",
                   help='single int or space/comma-separated list, e.g. "1 4 16"')
    p.add_argument("--output", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, mean, std, class_names = load_checkpoint(args.checkpoint, device)
    dataset = build_eval_dataset(load_config(args.config), args.split)

    requested_ns = sorted({int(x) for x in args.num_patches.replace(",", " ").split() if x})
    results = run_eval(model, class_names, mean, std, dataset, args.aggregation, requested_ns, device)

    out = Path(args.output) if args.output else Path(args.checkpoint).parent / f"eval_{args.split}_{args.aggregation}"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"split": args.split, "aggregation": args.aggregation, "results": {}}
    for key, r in results.items():
        np.save(out / f"confusion_{key}.npy", r.pop("confusion"))
        summary["results"][key] = r
        auc_s = "None" if r["macro_auc"] is None else f"{r['macro_auc']:.4f}"
        print(f"[{key}] top1={r['top1']:.4f} top5={r['top5']:.4f} auc={auc_s} n={r['n_samples']}", flush=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
