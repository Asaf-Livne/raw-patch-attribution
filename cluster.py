"""Recover model lineage and discover unseen generators from a trained
classifier's own penultimate features -- no lineage/discovery labels, no
retraining.

Two subcommands:
    lineage    Per-image features (4 central patches, mean-pooled) -> PCA(64)
               (fit on all images, since a handful of generators is too few
               samples for a 64-component PCA on their own) -> per-generator
               mean -> correlation distance -> average-linkage hierarchical
               clustering -> cophenetic correlation + dendrogram.
    discovery  Same per-image features, but for generators the checkpoint was
               NEVER trained on (its own class_names mark "known"; everything
               else in the label map is "unseen", same convention as
               openset.py) -> UMAP(min_dist=0.0) -> HDBSCAN (auto cluster
               count) -> ARI/NMI/purity against the true (unseen) generator
               labels.

Usage:
    python cluster.py lineage --checkpoint runs/openfake/best.pt \
        --config configs/openfake.yaml --output runs/openfake/lineage
    python cluster.py discovery --checkpoint runs/openset_seed42/best.pt \
        --config configs/openset.yaml --output runs/openset_seed42/discovery
"""

from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path

import numpy as np
import torch

from config import load_config
from datalib.dataset import PatchFolderDataset
from datalib.patches import patchify
from eval import _resize_up_to_patch, _select_central_patches, load_checkpoint
from openset import split_known_unknown


@torch.no_grad()
def extract_features(model, dataset, mean, std, device, num_patches: int = 4):
    """Per-image penultimate-layer feature, mean-pooled over the `num_patches`
    most-central patches. Returns (features [N, D], labels [N])."""
    feats, labels = [], []
    for i in range(len(dataset)):
        image, label, _ = dataset[i]
        image = image.to(device)
        patch = dataset.patch
        if min(image.shape[1], image.shape[2]) < patch:
            image = _resize_up_to_patch(image, patch)
        _, H, W = image.shape
        tiles, weights = patchify(image.cpu(), patch=patch)
        if tiles.shape[0] > num_patches:
            n_y, n_x = max(1, ceil(H / patch)), max(1, ceil(W / patch))
            tiles, weights = _select_central_patches(tiles, weights, n_y, n_x, num_patches)
        tiles, weights = tiles.to(device), weights.to(device)
        f = model.features((tiles - mean) / std)
        w = (weights / weights.sum()).unsqueeze(1)
        feats.append((f * w).sum(0).cpu().numpy())
        labels.append(int(label))
    return np.stack(feats), np.asarray(labels)


def run_lineage(features: np.ndarray, labels: np.ndarray, class_names: list[str],
                pca_components: int = 64) -> dict:
    from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
    from scipy.spatial.distance import pdist
    from sklearn.decomposition import PCA

    n_components = min(pca_components, *features.shape)
    reduced = PCA(n_components=n_components, random_state=0).fit_transform(features)
    centroids = np.stack([reduced[labels == i].mean(0) for i in range(len(class_names))])

    dist = pdist(centroids, metric="correlation")
    Z = linkage(dist, method="average")
    cophenetic_corr, _ = cophenet(Z, dist)
    return {"linkage": Z, "cophenetic_correlation": float(cophenetic_corr),
           "class_names": class_names, "dendrogram_order": dendrogram(Z, no_plot=True)["ivl"]}


def run_discovery(features: np.ndarray, labels: np.ndarray,
                  min_cluster_size: int = 50, min_samples: int = 10) -> tuple[np.ndarray, dict]:
    import hdbscan
    import umap
    from sklearn.metrics import (adjusted_rand_score, confusion_matrix,
                                 normalized_mutual_info_score)

    embedding = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.0,
                          metric="euclidean", random_state=42).fit_transform(features)
    cluster_labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                     min_samples=min_samples,
                                     metric="euclidean").fit_predict(embedding)

    mask = cluster_labels >= 0
    n_kept = int(mask.sum())
    if n_kept < 2:
        return cluster_labels, {"ARI": float("nan"), "NMI": float("nan"), "purity": float("nan"),
                                "n_clusters_found": 0, "n_noise": int((~mask).sum())}

    cm = confusion_matrix(labels[mask], cluster_labels[mask])
    # "purity": the fraction of (non-noise) images whose true generator's own
    # dominant cluster contains them -- ported to match the paper's reported
    # number, not the more common per-cluster-majority definition.
    purity = float(cm.max(axis=1).sum() / n_kept)
    metrics = {
        "ARI": float(adjusted_rand_score(labels[mask], cluster_labels[mask])),
        "NMI": float(normalized_mutual_info_score(labels[mask], cluster_labels[mask])),
        "purity": purity,
        "n_clusters_found": int(len(set(cluster_labels[mask]))),
        "n_noise": int((~mask).sum()),
    }
    return cluster_labels, metrics


def _build_dataset(label_map, data_root: str, patch: int, presplit: bool) -> PatchFolderDataset:
    return PatchFolderDataset(label_map_path=label_map, data_root=data_root, split="test",
                              mode="eval", patch=patch, presplit=presplit)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["lineage", "discovery"])
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--num_patches", type=int, default=4)
    p.add_argument("--pca_components", type=int, default=64)
    p.add_argument("--min_cluster_size", type=int, default=50)
    p.add_argument("--min_samples", type=int, default=10)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, mean, std, known_names = load_checkpoint(args.checkpoint, device)
    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    patch, presplit = ds_cfg.get("patch", 256), ds_cfg.get("presplit", False)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.command == "lineage":
        dataset = _build_dataset(ds_cfg["label_map"], ds_cfg["data_root"], patch, presplit)
        features, labels = extract_features(model, dataset, mean, std, device, args.num_patches)
        result = run_lineage(features, labels, dataset.class_names, args.pca_components)
        np.save(out / "linkage.npy", result.pop("linkage"))
        with open(out / "summary.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"cophenetic_correlation={result['cophenetic_correlation']:.4f} "
              f"(n_generators={len(dataset.class_names)})", flush=True)
        return 0

    with open(ds_cfg["label_map"]) as f:
        full_label_map = json.load(f)
    _, unknown_map = split_known_unknown(full_label_map, known_names)
    if not unknown_map:
        raise SystemExit("no classes in the label map fall outside the checkpoint's "
                         "known classes -- nothing unseen to cluster")
    dataset = _build_dataset(unknown_map, ds_cfg["data_root"], patch, presplit)
    features, labels = extract_features(model, dataset, mean, std, device, args.num_patches)
    cluster_labels, metrics = run_discovery(features, labels, args.min_cluster_size, args.min_samples)
    np.save(out / "cluster_labels.npy", cluster_labels)
    metrics["class_names"] = dataset.class_names
    with open(out / "summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"ARI={metrics['ARI']:.4f} NMI={metrics['NMI']:.4f} purity={metrics['purity']:.4f} "
          f"n_clusters_found={metrics['n_clusters_found']} (true={len(dataset.class_names)}) "
          f"n_noise={metrics['n_noise']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
