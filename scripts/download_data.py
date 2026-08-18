"""Download DRAGON and OpenFake into the class-folder layout PatchFolderDataset
expects. Images only; nothing here is Python-package data.

Usage:
    python scripts/download_data.py dragon --output-dir datasets --config Regular
    python scripts/download_data.py openfake --output-dir datasets/openfake --samples-per-class 1000

DRAGON already ships pre-split (train/val/test, per HuggingFace metadata) --
write it presplit-ready (each `<output-dir>/dragon_<config>/<model>/{train,val,test}/`).
OpenFake ships as one flat pool per model -- write it flat and let
PatchFolderDataset's `presplit=False` (train_frac/val_frac/split_seed) do the
750/250/400-per-class split at load time, matching the paper's protocol.

GenImage (used only for the few-shot experiment, as a 9-generator subset
under LIDA's protocol) isn't included here -- its provenance is a specific
third-party benchmark curation, not a plain HuggingFace download; prepare it
per LIDA's (Wang et al.) released protocol instead.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def sanitize_model_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def download_dragon(args) -> None:
    from datasets import load_dataset

    output_dir = Path(args.output_dir) / f"dragon_{args.config.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DRAGON (config: {args.config})...")
    dataset = load_dataset("lesc-unifi/dragon", args.config)

    model_meta = defaultdict(lambda: {"train": [], "test": []})
    for split_name in dataset.keys():
        hf_split = "train" if split_name == "train" else "test"
        for idx, sample in enumerate(dataset[split_name]):
            json_data = sample.get("json", {})
            model = sample.get("model.txt") or json_data.get("model", "unknown")
            model_meta[model][hf_split].append({
                "prompt": json_data.get("prompt", ""), "_hf_split": split_name, "_hf_idx": idx,
            })

    total = sum(len(m["train"]) + len(m["test"]) for m in model_meta.values())
    print(f"Found {total} samples across {len(model_meta)} models")

    for model_name, data in model_meta.items():
        model_dir = output_dir / sanitize_model_name(model_name)

        train_prompts = sorted(set(m["prompt"] for m in data["train"]))
        random.Random(args.seed).shuffle(train_prompts)
        n_val = max(1, round(len(train_prompts) * args.val_ratio))
        val_prompts = set(train_prompts[:n_val])

        splits = {"train": [], "val": [], "test": data["test"]}
        for m in data["train"]:
            splits["val" if m["prompt"] in val_prompts else "train"].append(m)

        for split_name, metas in splits.items():
            if not metas:
                continue
            images_dir = model_dir / split_name / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            for i, meta in enumerate(metas):
                sample = dataset[meta["_hf_split"]][meta["_hf_idx"]]
                image = sample.get("png")
                if image is not None and hasattr(image, "save"):
                    image.save(images_dir / f"sample_{i}.png")
            print(f"  {model_name}/{split_name}: {len(metas)}")

    print(f"Done -> {output_dir}")


def download_openfake(args) -> None:
    from datasets import Image as HfImage, load_dataset
    from PIL import Image as PILImage

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading OpenFake (cached after first download)...")
    ds = load_dataset("ComplexDataLab/OpenFake", split=args.split)
    ds = ds.cast_column("image", HfImage(decode=False))

    all_models = sorted(set(ds["model"]))
    print(f"Found {len(all_models)} models")
    targets = set(args.models) if args.models else set(all_models)

    for model_name in all_models:
        if model_name not in targets:
            continue
        dir_name = sanitize_model_name(model_name).replace(".", "-")
        model_dir = output_path / dir_name
        model_dir.mkdir(parents=True, exist_ok=True)

        have = len(list(model_dir.glob("*.png")))
        if have >= args.samples_per_class:
            print(f"  skip {dir_name}: {have}/{args.samples_per_class}")
            continue

        subset = ds.filter(lambda x: x["model"] == model_name)
        saved = have
        for sample in subset:
            if saved - have >= args.samples_per_class - have:
                break
            image = PILImage.open(io.BytesIO(sample["image"]["bytes"]))
            image.save(model_dir / f"{saved:04d}.png")
            saved += 1
        print(f"  {dir_name}: {saved}/{args.samples_per_class}")

    print(f"Done -> {output_path} (flat per class; PatchFolderDataset(presplit=False) "
          f"does the seeded 750/250/400 train/val/test split at load time)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("dragon", help="Download DRAGON (presplit train/val/test)")
    d.add_argument("--output-dir", default="datasets")
    d.add_argument("--config", default="Regular",
                   choices=["ExtraSmall", "Small", "Regular", "Large", "ExtraLarge"])
    d.add_argument("--val-ratio", type=float, default=0.25,
                  help="fraction of HF-train prompts held out as val (default matches "
                       "the paper's 750/250 train/val split)")
    d.add_argument("--seed", type=int, default=42)
    d.set_defaults(func=download_dragon)

    o = sub.add_parser("openfake", help="Download OpenFake (flat per class)")
    o.add_argument("--output-dir", default="datasets/openfake")
    o.add_argument("--samples-per-class", type=int, default=1400,
                  help="750+250+400 per the paper's per-class split")
    o.add_argument("--split", default="train", choices=["train", "test"])
    o.add_argument("--models", nargs="+", default=None)
    o.set_defaults(func=download_openfake)

    args = p.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
