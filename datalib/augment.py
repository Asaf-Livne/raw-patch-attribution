"""On-the-fly stochastic augmentation, applied fresh per sample per epoch
(never pre-generated to disk). Matches the paper's robustness recipe: JPEG,
Gaussian blur, and resize are each applied independently with their own
probability over a continuous parameter range, to the full image *before* it
is cropped to a patch -- so "resize" is a genuine downscale, not an
upscale-then-recrop. Random flips are cheap, always on, applied after
cropping. Picklable so DataLoader workers can use it under `spawn`.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image, ImageFilter


@dataclass
class AugmentConfig:
    flip: bool = True
    p_jpeg: float = 0.0
    p_blur: float = 0.0
    p_resize: float = 0.0
    jpeg_quality: Tuple[int, int] = (60, 90)
    blur_sigma: Tuple[float, float] = (0.1, 4.0)
    resize_short_edge: Tuple[int, int] = (384, 768)


class Augment:
    """Picklable (PIL.Image, rng) -> PIL.Image augmentation pipeline."""

    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg

    def pre_crop(self, img: Image.Image, seed: Optional[int] = None) -> Image.Image:
        """Full-image corruption applied before the patch crop."""
        c = self.cfg
        rng = random.Random(seed) if seed is not None else random
        if rng.random() < c.p_resize:
            img = self._resize(img, rng.uniform(*c.resize_short_edge), rng)
        if rng.random() < c.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*c.blur_sigma)))
        if rng.random() < c.p_jpeg:
            img = self._jpeg(img, rng.randint(*c.jpeg_quality))
        return img

    def post_crop(self, img: Image.Image, seed: Optional[int] = None) -> Image.Image:
        """Cheap, always-on flips applied to the already-cropped patch."""
        if not self.cfg.flip:
            return img
        rng = random.Random(seed) if seed is not None else random
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    @staticmethod
    def _resize(img: Image.Image, short_edge: float, rng) -> Image.Image:
        w, h = img.size
        if min(w, h) <= 0:
            return img
        scale = short_edge / min(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        return img.resize((new_w, new_h), Image.BICUBIC)

    @staticmethod
    def _jpeg(img: Image.Image, quality: int) -> Image.Image:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB").copy()


def build_augment(cfg: Optional[dict]) -> Optional[Augment]:
    """Build an `Augment` from a config dict, or `None` if nothing is configured.

    Config schema (all optional; robustness training sets p_jpeg/p_blur/p_resize):
        flip: bool (default True)
        p_jpeg, p_blur, p_resize: float in [0, 1] (default 0.0 -- clean training)
        jpeg_quality: [min, max] (default [60, 90])
        blur_sigma: [min, max] (default [0.1, 4.0])
        resize_short_edge: [min, max] (default [384, 768])
    """
    if cfg is None:
        return None
    return Augment(AugmentConfig(
        flip=cfg.get("flip", True),
        p_jpeg=cfg.get("p_jpeg", 0.0),
        p_blur=cfg.get("p_blur", 0.0),
        p_resize=cfg.get("p_resize", 0.0),
        jpeg_quality=tuple(cfg.get("jpeg_quality", (60, 90))),
        blur_sigma=tuple(cfg.get("blur_sigma", (0.1, 4.0))),
        resize_short_edge=tuple(cfg.get("resize_short_edge", (384, 768))),
    ))


def freq_filter(t, mode: str, k: int = 5):
    """Split a CHW [0,1] image tensor into low/high spatial frequency via box
    blur, or return its log-magnitude 2D FFT spectrum. Used by `eval.py`'s
    `--input-filter` option for the "what does the CNN see" analysis.
    """
    import torch
    import torch.nn.functional as F

    if mode == "none":
        return t
    if mode == "fftmag":
        import torch.fft as tfft
        spec = tfft.fftshift(tfft.fft2(t), dim=(-2, -1))
        mag = torch.log1p(spec.abs())
        mn = mag.amin(dim=(-2, -1), keepdim=True)
        mx = mag.amax(dim=(-2, -1), keepdim=True)
        return (mag - mn) / (mx - mn + 1e-8)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    x = t.unsqueeze(0)
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    low = F.avg_pool2d(xp, kernel_size=k, stride=1).squeeze(0)
    if mode == "lowpass":
        return low
    if mode == "highpass":
        return t - low
    raise ValueError(f"unknown frequency filter mode: {mode!r}")
