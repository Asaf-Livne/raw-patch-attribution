# Pretrained checkpoints

The two headline classifiers are distributed as
[release assets](https://github.com/Asaf-Livne/raw-patch-attribution/releases/tag/v1.0) rather
than tracked in git — they are ~71 MB each.

| File | Benchmark | Paper | Validated | Aggregation |
|:--|:--|:--|:--|:--|
| `dragon_25class.pt` | DRAGON, 25 generators | 98.0 % | 98.3 % | all patches |
| `openfake_27class.pt` | OpenFake, 27 generators | 92.9 % | 92.7 % | 4-patch `logit_avg` |

## Download

```bash
BASE=https://github.com/Asaf-Livne/raw-patch-attribution/releases/download/v1.0
curl -L -o checkpoints/dragon_25class.pt   $BASE/dragon_25class.pt
curl -L -o checkpoints/openfake_27class.pt $BASE/openfake_27class.pt
shasum -a 256 -c checkpoints/SHA256SUMS
```

## Use

Through `eval.py`:

```bash
python eval.py --checkpoint checkpoints/dragon_25class.pt \
    --config configs/dragon_25class.yaml --split test --num_patches "1 4 16"
```

Or directly:

```python
import torch
from models.cnn import build_model

ckpt = torch.load("checkpoints/dragon_25class.pt", map_location="cpu", weights_only=False)
model = build_model(ckpt["config"], num_classes=len(ckpt["class_names"]))
model.load_state_dict(ckpt["model"])
model.eval()
```

Each checkpoint also carries `class_names` (the label order) and
`channel_mean` / `channel_std` — patches must be normalized with these before
the forward pass.
