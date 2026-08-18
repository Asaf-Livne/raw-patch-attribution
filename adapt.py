"""Adapt a frozen backbone to new classes by fitting only a fresh linear
head, on CACHED penultimate features. The backbone runs once per image (to
build the feature cache), not once per epoch -- fitting the head is then a
few seconds of linear-layer training per epoch, matching the paper's
reported ~7-minute wall clock for a 27-way head. One script covers all three
published adaptation regimes -- they differ only in what target data feeds
the cache, not in the mechanism:

    full adaptation   target = the source benchmark's full class set (e.g.
                      a 17-known OpenFake backbone -> all 27); pass
                      --transplant so classes shared with the source
                      checkpoint keep their trained head row (by name) and
                      only the new classes' rows are freshly initialized.
    cross-dataset     target = a DIFFERENT benchmark's full, disjoint class
                      set. --transplant is a no-op (no shared names); omit
                      --shots to use all of the target's training data.
    few-shot          target = a (possibly disjoint) benchmark, using only
                      `--shots` labeled training images per class.

The backbone is frozen entirely (a true linear probe): its parameters keep
`requires_grad=False`, and it never sees a backward pass. Channel
normalization is reused from the source checkpoint, never recomputed, since
the frozen filters are calibrated to it. Each training/val image contributes
one cached feature vector per patch (all patches, deterministic tiling, no
augmentation -- the linear head trains on this fixed cache for every epoch).

Usage:
    python adapt.py --checkpoint runs/openset_seed42/best.pt \
        --config configs/openfake_27class.yaml --output runs/adapt_full27 --transplant
    python adapt.py --checkpoint runs/dragon/best.pt \
        --config configs/openfake_27class.yaml --output runs/cross_d2o
    python adapt.py --checkpoint runs/openfake/best.pt \
        --config configs/genimage_9class.yaml --output runs/fewshot_10shot --shots 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import load_config
from datalib.augment import build_augment
from datalib.dataset import PatchFolderDataset
from datalib.patches import patchify
from eval import _resize_up_to_patch, build_eval_dataset, run_eval
from models.cnn import build_model
from train import set_seed


def transplant_and_freeze(model, source_ckpt: dict, target_names: list[str],
                          do_transplant: bool) -> list[str]:
    """Copy the frozen backbone into `model`; optionally copy head rows for
    classes present in both the source checkpoint and the target by name.
    Freezes every backbone parameter. Returns the transplanted class names."""
    source_state = source_ckpt["model"]
    backbone_state = {k: v for k, v in source_state.items() if not k.startswith("head.")}
    model.load_state_dict(backbone_state, strict=False)

    transplanted = []
    if do_transplant:
        source_names = source_ckpt["class_names"]
        source_idx = {c: i for i, c in enumerate(source_names)}
        with torch.no_grad():
            for name in target_names:
                if name in source_idx:
                    i = source_idx[name]
                    j = target_names.index(name)
                    model.head.weight[j].copy_(source_state["head.weight"][i])
                    model.head.bias[j].copy_(source_state["head.bias"][i])
                    transplanted.append(name)

    for p in model.backbone.parameters():
        p.requires_grad = False
    return transplanted


def subsample_shots(dataset: PatchFolderDataset, shots: int, seed: int) -> None:
    """Keep only `shots` random training images per class (in place)."""
    by_class: dict[int, list] = {}
    for sample in dataset.samples:
        by_class.setdefault(sample[1], []).append(sample)
    rng = random.Random(seed)
    kept = []
    for label, samples in by_class.items():
        rng.shuffle(samples)
        kept.extend(samples[:shots])
    dataset.samples = kept


class _PatchifyWrapper(torch.utils.data.Dataset):
    """Wraps an eval-mode PatchFolderDataset so patchify (CPU-bound: resize +
    tiling) runs inside DataLoader workers, in parallel, instead of serially
    in the main process. Yields all deterministic tiles of one image per
    call -- the right amount of natural diversity when training data is
    plentiful (full/cross-dataset adaptation)."""

    def __init__(self, dataset: PatchFolderDataset):
        self.dataset = dataset
        self.patch = dataset.patch

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label, _ = self.dataset[idx]
        if min(image.shape[1], image.shape[2]) < self.patch:
            image = _resize_up_to_patch(image, self.patch)
        tiles, _ = patchify(image, patch=self.patch)
        return tiles, label


class _RepeatedAugmentedWrapper(torch.utils.data.Dataset):
    """Wraps a mode="train" PatchFolderDataset (on-the-fly random crop +
    flip, fresh per call) and samples each image `views` times, expanding a
    tiny few-shot image pool into a larger augmented feature bank -- without
    this, caching a single deterministic view per shot leaves too few
    training vectors and the head overfits badly (see README)."""

    def __init__(self, dataset: PatchFolderDataset, views: int):
        self.dataset = dataset
        self.views = views

    def __len__(self) -> int:
        return len(self.dataset) * self.views

    def __getitem__(self, idx: int):
        image, label, _ = self.dataset[idx // self.views]
        return image.unsqueeze(0), label


@torch.no_grad()
def _cache_features(loader, model, mean, std, device, batch_patches: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Push whatever `loader` yields (tiles-per-item, label) through the
    frozen backbone, batching tiles across items so the GPU sees large
    calls instead of one tiny call per image."""
    model.eval()
    feats, labels = [], []
    buf_tiles, buf_labels, buf_n = [], [], 0

    def flush():
        nonlocal buf_n
        if buf_n == 0:
            return
        batch = torch.cat(buf_tiles, dim=0).to(device)
        f = model.features((batch - mean) / std)
        feats.append(f.cpu())
        labels.extend(buf_labels)
        buf_tiles.clear()
        buf_labels.clear()
        buf_n = 0

    for tiles, label in loader:
        buf_tiles.append(tiles)
        buf_labels.extend([label] * tiles.shape[0])
        buf_n += tiles.shape[0]
        if buf_n >= batch_patches:
            flush()
    flush()
    return torch.cat(feats, dim=0), torch.tensor(labels, dtype=torch.long)


def extract_patch_features(model, dataset, mean, std, device,
                           batch_patches: int = 512, num_workers: int = 8
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen backbone once over every deterministic tile of every
    image in `dataset` (an eval-mode PatchFolderDataset). Each tile becomes
    one cached (feature, label) row -- this is the one-time cost; every
    epoch of head-fitting afterward only touches this cache."""
    loader = DataLoader(_PatchifyWrapper(dataset), batch_size=1, num_workers=num_workers,
                        collate_fn=lambda batch: batch[0])
    return _cache_features(loader, model, mean, std, device, batch_patches)


def extract_augmented_features(model, dataset, mean, std, device, views: int,
                               batch_patches: int = 512, num_workers: int = 8
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Like `extract_patch_features`, but samples each image `views` times
    through on-the-fly random-crop + flip augmentation (a mode="train"
    dataset) instead of deterministic tiling -- for expanding a tiny
    few-shot pool into a larger feature bank. See `--train_views`."""
    loader = DataLoader(_RepeatedAugmentedWrapper(dataset, views), batch_size=1,
                        num_workers=num_workers, collate_fn=lambda batch: batch[0])
    return _cache_features(loader, model, mean, std, device, batch_patches)


def fit_head(model, train_feats, train_labels, val_feats, val_labels, device,
            epochs: int, lr: float, weight_decay: float, label_smoothing: float,
            patience: int, batch_size: int, curves_path: Path) -> dict:
    """Train only `model.head` on cached features. `model.dropout` still
    applies its usual stochastic regularization each training step."""
    class_counts = torch.bincount(train_labels, minlength=model.head.out_features).float()
    cw = (class_counts.sum() / (len(class_counts) * class_counts.clamp(min=1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_labels), batch_size=batch_size, shuffle=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * max(1, len(train_loader)), eta_min=1e-6)

    with open(curves_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    best_val_loss, best_val_acc, best_epoch, no_improve, best_state = float("inf"), -1.0, -1, 0, None
    for epoch in range(epochs):
        model.head.train()
        total_loss, n_seen, n_correct = 0.0, 0, 0
        for f, y in train_loader:
            f, y = f.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.head(model.dropout(f))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * f.size(0)
            n_correct += (logits.argmax(1) == y).sum().item()
            n_seen += f.size(0)
        train_loss, train_acc = total_loss / max(1, n_seen), n_correct / max(1, n_seen)

        model.head.eval()
        val_loss_sum, val_correct, val_seen = 0.0, 0, 0
        with torch.no_grad():
            for f, y in val_loader:
                f, y = f.to(device), y.to(device)
                logits = model.head(f)  # no dropout at eval
                val_loss_sum += criterion(logits, y).item() * f.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_seen += f.size(0)
        val_loss, val_acc = val_loss_sum / max(1, val_seen), val_correct / max(1, val_seen)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch+1:03d}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:.2e}", flush=True)
        with open(curves_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{lr_now:.6e}\n")

        new_best = val_loss < best_val_loss or (val_loss == best_val_loss and val_acc > best_val_acc)
        if new_best:
            best_val_loss, best_val_acc, best_epoch, no_improve = val_loss, val_acc, epoch + 1, 0
            best_state = {k: v.clone() for k, v in model.head.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[early-stop] no val improvement for {patience} epochs", flush=True)
                break

    model.head.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_acc": best_val_acc}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--transplant", action="store_true",
                   help="copy source head rows for classes shared by name with the target")
    p.add_argument("--shots", type=int, default=None,
                   help="limit training data to this many images/class (few-shot)")
    p.add_argument("--train_views", type=int, default=1,
                   help="augmented views (random crop + flip) cached per training image, "
                        "instead of one deterministic tiling -- set > 1 for extreme few-shot "
                        "(e.g. --shots 10 --train_views 20), where a handful of raw images "
                        "is too few feature vectors to fit a head on without overfitting")
    p.add_argument("--val_split", type=str, default="val",
                   help="split used for head-selection during fitting")
    p.add_argument("--test_split", type=str, default="test",
                   help="split used for the final reported score; some benchmarks "
                        "(e.g. GenImage-9 under LIDA's protocol) only have train/val, "
                        "no held-out test -- pass --test_split val to score on val "
                        "instead (matches the paper's few-shot protocol: no separate "
                        "val to early-stop on, the query set doubles as both)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--batch_patches", type=int, default=512,
                   help="patches per backbone forward call during feature caching")
    p.add_argument("--num_workers", type=int, default=8,
                   help="parallel workers for the one-time feature-caching pass")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    source_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mean_cpu, std_cpu = source_ckpt["channel_mean"].cpu(), source_ckpt["channel_std"].cpu()
    mean, std = mean_cpu.view(-1, 1, 1).to(device).float(), std_cpu.view(-1, 1, 1).to(device).float()
    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    common = dict(
        label_map_path=ds_cfg["label_map"], data_root=ds_cfg["data_root"],
        patch=ds_cfg.get("patch", 256), train_frac=ds_cfg.get("train_frac", 0.8),
        val_frac=ds_cfg.get("val_frac", 0.1), split_seed=ds_cfg.get("split_seed", 0),
        max_per_class=ds_cfg.get("max_per_class"), presplit=ds_cfg.get("presplit", False),
    )

    train_mode = "train" if args.train_views > 1 else "eval"
    train_aug = build_augment({"flip": True}) if args.train_views > 1 else None
    train_ds = PatchFolderDataset(split="train", mode=train_mode,
                                  augment=train_aug.post_crop if train_aug else None, **common)
    val_ds = PatchFolderDataset(split=args.val_split, mode="eval", **common)
    if args.shots is not None:
        subsample_shots(train_ds, args.shots, args.seed)
    target_names = train_ds.class_names
    print(f"[dataset] train_images={len(train_ds)} val_images={len(val_ds)} "
          f"classes={len(target_names)} shots={args.shots} train_views={args.train_views}",
          flush=True)

    model = build_model(source_ckpt["config"], num_classes=len(target_names)).to(device)
    transplanted = transplant_and_freeze(model, source_ckpt, target_names, args.transplant)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable={n_train} / total={n_total} ({100*n_train/n_total:.2f}%) "
          f"transplanted={len(transplanted)}/{len(target_names)}", flush=True)

    t_cache = time.time()
    if args.train_views > 1:
        train_feats, train_labels = extract_augmented_features(
            model, train_ds, mean, std, device, args.train_views,
            args.batch_patches, args.num_workers)
    else:
        train_feats, train_labels = extract_patch_features(model, train_ds, mean, std, device,
                                                           args.batch_patches, args.num_workers)
    val_feats, val_labels = extract_patch_features(model, val_ds, mean, std, device,
                                                   args.batch_patches, args.num_workers)
    cache_sec = time.time() - t_cache
    print(f"[cache] train_patches={train_feats.shape[0]} val_patches={val_feats.shape[0]} "
          f"({cache_sec:.0f}s)", flush=True)

    t_start = time.time()
    train_stats = fit_head(model, train_feats, train_labels, val_feats, val_labels, device,
                           args.epochs, args.lr, args.weight_decay, args.label_smoothing,
                           args.patience, args.batch_size, out / "curves.csv")
    wall_sec = time.time() - t_start

    torch.save({"model": model.state_dict(), "config": source_ckpt["config"],
               "channel_mean": mean_cpu, "channel_std": std_cpu, "class_names": target_names,
               **train_stats}, out / "best.pt")

    t_test = time.time()
    model.eval()
    test_dataset = build_eval_dataset(cfg, args.test_split)
    # "best_per_image" uses each image's largest ELIGIBLE patch budget --
    # requesting only n4 would silently drop every class whose images are
    # smaller than 512px (2 * patch), which happens on mixed-resolution
    # benchmarks like GenImage-9 (128px-1024px across classes).
    test_results = run_eval(model, target_names, mean, std, test_dataset,
                            "logit_avg", [1, 4, 16], device)
    test_top1 = test_results["best_per_image"]["top1"] if "best_per_image" in test_results else None
    test_eval_sec = time.time() - t_test

    per_class = test_results.get("best_per_image", {}).get("per_class_accuracy", {})
    transplanted_set = set(transplanted)
    new_classes = [c for c in target_names if c not in transplanted_set]
    result = {
        **train_stats, "cache_sec": cache_sec, "head_fit_sec": wall_sec,
        "test_eval_sec": test_eval_sec,
        "n_transplanted": len(transplanted), "n_new": len(new_classes),
        "test_top1_overall": test_top1,
        "test_top1_transplanted": (float(np.mean([per_class[c] for c in transplanted]))
                                   if transplanted else None),
        "test_top1_new": float(np.mean([per_class[c] for c in new_classes])) if new_classes else None,
        "per_class_accuracy": per_class,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(f"\n[result] overall={test_top1} transplanted={result['test_top1_transplanted']} "
          f"new={result['test_top1_new']} head_fit={wall_sec:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
