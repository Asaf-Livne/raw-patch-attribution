"""Train a PatchCNN classifier on a configured dataset.

Usage:
    python train.py --config configs/dragon.yaml --output runs/dragon_seed42 --seed 42

On-the-fly augmentation (JPEG/blur/resize corruption, for the robustness
variant) is a config option under `augment_train` -- clean vs. robustness
training is the same code path with different config. A run directory
contains: best.pt, last.pt, config.yaml (snapshot), curves.csv, curves.png,
channel_stats.pt. Reruns with the same --output auto-resume from last.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import load_config
from datalib.augment import build_augment
from datalib.dataset import PatchFolderDataset
from datalib.stats import load_or_compute_stats
from models.cnn import build_model


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def _dataset_common(ds_cfg: dict) -> dict:
    return dict(
        label_map_path=ds_cfg["label_map"],
        data_root=ds_cfg["data_root"],
        patch=ds_cfg.get("patch", 256),
        train_frac=ds_cfg.get("train_frac", 0.8),
        val_frac=ds_cfg.get("val_frac", 0.1),
        split_seed=ds_cfg.get("split_seed", 0),
        max_per_class=ds_cfg.get("max_per_class"),
        presplit=ds_cfg.get("presplit", False),
        input_filter=ds_cfg.get("input_filter", "none"),
        filter_k=ds_cfg.get("filter_k", 5),
    )


def build_datasets(cfg: dict):
    """Build (train_ds, val_ds, test_ds) per the config.

    If `augment_val.match_train` is set, the val set draws the SAME full-image
    corruption as train but deterministically (seeded per sample index), so
    val_loss measures the corrupted distribution and best.pt is selected for
    robustness rather than clean accuracy.
    """
    ds_cfg = cfg["dataset"]
    common = _dataset_common(ds_cfg)

    aug = build_augment(cfg.get("augment_train"))
    val_match = bool((cfg.get("augment_val") or {}).get("match_train", False))

    train_ds = PatchFolderDataset(
        split="train", mode="train",
        augment=aug.post_crop if aug else None,
        pre_crop_augment=aug.pre_crop if aug else None,
        **common,
    )
    if val_match:
        val_ds = PatchFolderDataset(
            split="val", mode="train", augment=None,
            pre_crop_augment=aug.pre_crop if aug else None,
            deterministic=True, **common,
        )
    else:
        val_ds = PatchFolderDataset(split="val", mode="train", augment=None, **common)
    test_ds = PatchFolderDataset(split="test", mode="train", augment=None, **common)
    return train_ds, val_ds, test_ds


def class_weights(counts: list[int]) -> torch.Tensor:
    counts_t = torch.tensor(counts, dtype=torch.float32)
    return counts_t.sum() / (len(counts) * counts_t.clamp(min=1))


def evaluate(model, loader, device, criterion) -> tuple[float, float]:
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            n_correct += (logits.argmax(1) == y).sum().item()
            n_total += x.size(0)
    return total_loss / max(1, n_total), n_correct / max(1, n_total)


def _plot_curves(csv_path: Path, png_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("acc"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no_resume", action="store_true",
                   help="ignore any existing <output>/last.pt; always start from scratch")
    args = p.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    cfg = load_config(args.config)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")

    last_ckpt_path = out / "last.pt"
    resume_state = None
    if last_ckpt_path.exists() and not args.no_resume:
        resume_state = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        print(f"[auto-resume] {last_ckpt_path} epoch={resume_state['epoch']} "
              f"best_val_acc={resume_state.get('best_val_acc', float('nan')):.4f}", flush=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}", flush=True)

    train_ds, val_ds, test_ds = build_datasets(cfg)
    print(f"[dataset] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"classes={train_ds.num_classes}", flush=True)

    tr_cfg = cfg["train"]
    if resume_state is not None:
        mean, std = resume_state["channel_mean"], resume_state["channel_std"]
    else:
        unnormalized_train = PatchFolderDataset(split="train", mode="train", augment=None,
                                                **_dataset_common(cfg["dataset"]))
        mean, std = load_or_compute_stats(
            out / "channel_stats.pt", unnormalized_train,
            batch_size=tr_cfg.get("stats_batch_size", 32),
            num_workers=tr_cfg.get("num_workers", 4),
            max_batches=tr_cfg.get("stats_max_batches"),
        )
    print(f"[stats] mean={mean.tolist()} std={std.tolist()}", flush=True)
    train_ds.set_normalization(mean, std)
    val_ds.set_normalization(mean, std)
    test_ds.set_normalization(mean, std)

    model = build_model(cfg, num_classes=train_ds.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M", flush=True)

    bs = tr_cfg.get("batch_size", 16)
    nw = tr_cfg.get("num_workers", 8)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                            pin_memory=(device.type == "cuda"))

    cw = class_weights(train_ds.class_counts()).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=tr_cfg.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr_cfg.get("lr", 1e-3),
                                  weight_decay=tr_cfg.get("weight_decay", 1e-3))

    epochs = tr_cfg.get("epochs", 75)
    total_steps = epochs * max(1, len(train_loader))
    eta_min = tr_cfg.get("lr_min", 1e-6)
    is_plateau = tr_cfg.get("lr_scheduler", "cosine") == "plateau"
    if is_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=tr_cfg.get("plateau_factor", 0.5),
            patience=tr_cfg.get("plateau_patience", 15), min_lr=eta_min)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

    patience = tr_cfg.get("early_stopping_patience", 15)
    best_val_acc, best_val_loss, best_epoch, epochs_no_improve, start_epoch = -1.0, float("inf"), -1, 0, 0

    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        best_val_acc = resume_state["best_val_acc"]
        best_val_loss = resume_state.get("best_val_loss", float("inf"))
        best_epoch = resume_state["best_epoch"]
        epochs_no_improve = resume_state["epochs_no_improve"]
        start_epoch = resume_state["epoch"]
        torch.set_rng_state(resume_state["torch_rng_state"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
        np.random.set_state(resume_state["numpy_rng_state"])
        random.setstate(resume_state["python_rng_state"])
        print(f"[auto-resume] resuming at epoch {start_epoch + 1}/{epochs}", flush=True)
        if start_epoch >= epochs:
            print(json.dumps({"best_val_acc": best_val_acc, "best_epoch": best_epoch}), flush=True)
            return 0

    curves_path = out / "curves.csv"
    if not curves_path.exists() or resume_state is None:
        with open(curves_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        total_loss, n_seen, n_correct_tr = 0.0, 0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if not is_plateau:
                scheduler.step()
            total_loss += loss.item() * x.size(0)
            n_correct_tr += (logits.argmax(1) == y).sum().item()
            n_seen += x.size(0)

        train_loss, train_acc = total_loss / max(1, n_seen), n_correct_tr / max(1, n_seen)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        if is_plateau:
            scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch+1:03d}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:.2e} ({time.time()-t0:.1f}s)",
              flush=True)

        new_best = val_loss < best_val_loss or (val_loss == best_val_loss and val_acc > best_val_acc)
        if new_best:
            best_val_acc, best_val_loss, best_epoch, epochs_no_improve = val_acc, val_loss, epoch + 1, 0
        else:
            epochs_no_improve += 1

        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "val_acc": val_acc,
            "best_val_acc": best_val_acc, "best_val_loss": best_val_loss, "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
            "config": cfg, "channel_mean": mean, "channel_std": std,
            "class_names": train_ds.class_names,
        }
        tmp = out / "last.pt.tmp"
        torch.save(state, tmp)
        tmp.replace(out / "last.pt")

        with open(curves_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}",
                                    f"{val_loss:.6f}", f"{val_acc:.6f}", f"{lr_now:.6e}"])

        if new_best:
            best_tmp = out / "best.pt.tmp"
            shutil.copy(out / "last.pt", best_tmp)
            best_tmp.replace(out / "best.pt")

        if epochs_no_improve >= patience:
            print(f"[early-stop] no val_loss improvement for {patience} epochs; "
                  f"best val_loss={best_val_loss:.4f} @ ep{best_epoch}", flush=True)
            break

    try:
        _plot_curves(curves_path, out / "curves.png")
    except Exception as e:
        print(f"[plot] skipped: {e}", flush=True)

    print(json.dumps({"best_val_acc": best_val_acc, "best_epoch": best_epoch}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
