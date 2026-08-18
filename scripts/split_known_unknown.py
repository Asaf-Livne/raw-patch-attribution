"""Carve a random known-class subset out of a full label map, for training an
open-set classifier. The held-out remainder is scored as "unknown" by
openset.py directly from the checkpoint's class_names -- no separate
unknown-map file is needed.

Usage:
    python scripts/split_known_unknown.py --label-map configs/label_maps/openfake_27class.json \
        --n-known 17 --seed 42 --output configs/label_maps/openfake_17known_seed42.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label-map", required=True, type=str)
    p.add_argument("--n-known", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--output", required=True, type=str)
    args = p.parse_args()

    with open(args.label_map) as f:
        full_map = json.load(f)
    names = sorted(full_map.keys())
    if args.n_known > len(names):
        raise SystemExit(f"--n-known ({args.n_known}) exceeds the label map's "
                         f"{len(names)} classes")

    rng = random.Random(args.seed)
    rng.shuffle(names)
    known = sorted(names[:args.n_known])
    known_map = {c: full_map[c] for c in known}

    with open(args.output, "w") as f:
        json.dump(known_map, f, indent=2, sort_keys=True)
    print(f"wrote {len(known_map)}/{len(full_map)} known classes to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
