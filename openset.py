"""Open-set attribution: does a classifier's own confidence flag images from
generators it never trained on?

The checkpoint's `class_names` (whatever it was trained on) are the "known"
classes. `dataset.label_map` in the config should be the FULL candidate label
map (every generator, known and unseen); everything in it that ISN'T one of
the checkpoint's classes is scored as "unknown". Use
scripts/split_known_unknown.py beforehand to carve out a known-only label map
to train on in the first place.

Score = max-softmax-probability (MSP) under all-patches logit-averaging (the
paper's protocol); reported metrics are threshold-free (AUROC, AU-OSCR) plus
per-generator rejection recall at the operating point that keeps 95% of known
images (5th percentile of the known-image score distribution).

Usage:
    python openset.py --checkpoint runs/openset_seed42/best.pt \
                      --config configs/openset.yaml --output runs/openset_seed42/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from config import load_config
from datalib.dataset import PatchFolderDataset
from datalib.openset_metrics import au_oscr, auroc
from datalib.patches import aggregate, patchify
from eval import _resize_up_to_patch, load_checkpoint


def split_known_unknown(full_label_map: dict, known_names: list[str]) -> tuple[dict, dict]:
    known = {c: full_label_map[c] for c in known_names}
    unknown = {c: p for c, p in full_label_map.items() if c not in known_names}
    return known, unknown


@torch.no_grad()
def score_dataset(model, dataset, mean, std, device) -> tuple[np.ndarray, np.ndarray]:
    """Per-image (max-softmax-probability, predicted-class-index)."""
    scores, preds = [], []
    for i in range(len(dataset)):
        image, _, _ = dataset[i]
        image = image.to(device)
        patch = dataset.patch
        if min(image.shape[1], image.shape[2]) < patch:
            image = _resize_up_to_patch(image, patch)
        tiles, weights = patchify(image.cpu(), patch=patch)
        tiles, weights = tiles.to(device), weights.to(device)
        logits = model((tiles - mean) / std)
        score = aggregate(logits.float().cpu(), weights.float().cpu(), "logit_avg")
        probs = score.softmax(-1).numpy()
        scores.append(float(probs.max()))
        preds.append(int(probs.argmax()))
    return np.asarray(scores), np.asarray(preds)


def per_generator_rejection(unknown_scores, unknown_labels, class_names, tau) -> dict:
    out = {}
    for idx, name in enumerate(class_names):
        sel = unknown_labels == idx
        out[name] = float((unknown_scores[sel] < tau).mean()) if sel.sum() else None
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, mean, std, known_names = load_checkpoint(args.checkpoint, device)
    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]

    with open(ds_cfg["label_map"]) as f:
        full_label_map = json.load(f)
    known_map, unknown_map = split_known_unknown(full_label_map, known_names)
    if not unknown_map:
        raise SystemExit("no classes in the label map fall outside the checkpoint's "
                         "known classes -- nothing to treat as unseen")

    common = dict(data_root=ds_cfg["data_root"], patch=ds_cfg.get("patch", 256),
                 presplit=ds_cfg.get("presplit", False), mode="eval", split="test")
    known_ds = PatchFolderDataset(label_map_path=known_map, **common)
    unknown_ds = PatchFolderDataset(label_map_path=unknown_map, **common)

    known_scores, known_preds = score_dataset(model, known_ds, mean, std, device)
    unknown_scores, unknown_preds = score_dataset(model, unknown_ds, mean, std, device)
    known_labels = np.asarray([lbl for _, lbl in known_ds.samples])
    unknown_labels = np.asarray([lbl for _, lbl in unknown_ds.samples])

    is_known = np.r_[np.ones(len(known_scores), int), np.zeros(len(unknown_scores), int)]
    scores = np.r_[known_scores, unknown_scores]
    correct = np.r_[(known_preds == known_labels).astype(int), np.zeros(len(unknown_scores), int)]

    closed_set_acc = float((known_preds == known_labels).mean())
    tau95 = float(np.percentile(known_scores, 5))  # 95% known-TPR operating point

    result = {
        "closed_set_acc_known": closed_set_acc,
        "AUROC": auroc(scores, is_known),
        "AU_OSCR": au_oscr(scores, correct, is_known),
        "tau_known95pct": tau95,
        "known_false_reject_at_tau95": float((known_scores < tau95).mean()),
        "per_generator_reject_at_tau95": per_generator_rejection(
            unknown_scores, unknown_labels, unknown_ds.class_names, tau95),
        "n_known": len(known_scores), "n_unknown": len(unknown_scores),
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"closed_set_acc={closed_set_acc:.4f}  AUROC={result['AUROC']:.4f}  "
          f"AU-OSCR={result['AU_OSCR']:.4f}  (n_known={result['n_known']} "
          f"n_unknown={result['n_unknown']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
