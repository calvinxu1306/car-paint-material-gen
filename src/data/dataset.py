"""
dataset.py — PyTorch Dataset for the rendered car-paint samples.

Reads the folder produced by generate_dataset_v2.py and hands the network
(photo -> ground-truth) pairs as tensors.

WHAT EACH SAMPLE LOOKS LIKE
    photo    float tensor (3, H, W)   the flash-lit photo  = the INPUT
    maps     float tensor (8, H, W)   per-pixel answer key
    scalars  float tensor (5,)        per-sample answer key, normalised 0..1
    params   float tensor (6,)        legacy raw values (kept for eval scripts)
    index    int                      which sample this is

The 8 map channels, in order:
    0,1,2  base color  R G B
    3      roughness
    4      metallic
    5,6,7  normal      x y z   (stored 0..1, where 0.5,0.5,1.0 = flat)

WHY SCALARS ARE SEPARATE FROM MAPS
Clear coat and flakes are LAYER properties, not per-pixel ones: a sample has
one coat roughness and one flake size, not a different value at every pixel.
Predicting them as maps would waste capacity learning to output a constant.
They are the quantities no learning-based SVBRDF paper in the reading list
predicts at all, so they are the point of the project - hence their own head.

Scalars are normalised to 0..1 using SCALAR_RANGES, which MUST match the
sampling ranges in generate_dataset_v2.py. A mismatch is checked for at load
time and warned about rather than silently corrupting the targets.

ONE SUBTLETY WORTH KNOWING
basecolor_*.png is sRGB-encoded (the convention for colour textures), while
roughness/metallic/normal are stored raw. So base colour is decoded back to
linear here and the others are used as-is.

QUICK TEST:
    python src/data/dataset.py --root data/blender_gen/dataset_v2
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MAP_CHANNELS = ["basecolor_r", "basecolor_g", "basecolor_b",
                "roughness", "metallic", "normal_x", "normal_y", "normal_z"]

# Per-sample layer properties: the clear coat and the flakes.
# KEEP IN SYNC with sample_paint_params() in generate_dataset_v2.py.
SCALAR_RANGES = {
    "coat_weight": (0.8, 1.0),
    "coat_roughness": (0.01, 0.1),
    "flake_scale": (75.0, 150.0),
    "flake_strength": (0.12, 0.40),
    "peel_strength": (0.02, 0.09),
}
SCALAR_KEYS = list(SCALAR_RANGES)

# Legacy vector kept so older eval scripts keep working.
PARAM_KEYS = ["metallic", "roughness", "coat_weight", "coat_roughness",
              "flake_scale", "flake_strength"]


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Undo sRGB encoding. Inverse of the linear_to_srgb used in the Blender check."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load_png(path: str) -> np.ndarray:
    """Load a PNG as float32 RGB in 0..1, shape (H, W, 3)."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def normalise_scalars(p: dict) -> np.ndarray:
    """Map the raw layer parameters into 0..1 so one loss weight suits them all.

    Without this, flake_scale (~100) would dominate coat_roughness (~0.05) by
    three orders of magnitude and the network would ignore the coat entirely.
    """
    out = np.empty(len(SCALAR_KEYS), dtype=np.float32)
    for i, k in enumerate(SCALAR_KEYS):
        lo, hi = SCALAR_RANGES[k]
        out[i] = (float(p[k]) - lo) / (hi - lo)
    return out


def denormalise_scalars(v) -> dict:
    """Back to real units, for reporting errors people can interpret."""
    arr = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    return {k: float(arr[i] * (SCALAR_RANGES[k][1] - SCALAR_RANGES[k][0])
                     + SCALAR_RANGES[k][0])
            for i, k in enumerate(SCALAR_KEYS)}


class CarPaintDataset(Dataset):
    """Samples rendered by the Blender generator.

    Args:
        root: the dataset folder.
        split: "train", "val", or "all".
        val_fraction: share of samples held out for validation.
        limit: optionally use only the first N samples (handy while debugging).
        require_scalars: fail loudly on a v1 dataset that has no layer params.
    """

    def __init__(self, root: str, split: str = "train",
                 val_fraction: float = 0.1, limit: int | None = None,
                 require_scalars: bool = True):
        self.root = root
        self.require_scalars = require_scalars
        self.indices = self._discover(root)
        if limit is not None:
            self.indices = self.indices[:limit]
        if not self.indices:
            raise RuntimeError(
                f"No samples found in {root}. Has the generator run yet?")

        self._check_scalar_ranges()

        # Deterministic split: the LAST slice is validation. Because seed ==
        # index, the same samples are always held out, on any machine.
        n_val = max(1, int(round(len(self.indices) * val_fraction)))
        if split == "train":
            self.indices = self.indices[:-n_val]
        elif split == "val":
            self.indices = self.indices[-n_val:]
        elif split != "all":
            raise ValueError(f"split must be train/val/all, got {split!r}")

    @staticmethod
    def _discover(root: str) -> list[int]:
        """Find complete samples. Trusts the files on disk rather than the
        manifest, so a half-written last sample can't break training."""
        found = []
        for p in sorted(glob.glob(os.path.join(root, "params_*.json"))):
            tag = os.path.basename(p)[len("params_"):-len(".json")]
            if all(os.path.exists(os.path.join(root, f"{n}_{tag}.png"))
                   for n in ("photo", "basecolor", "roughness", "metallic", "normal")):
                found.append(int(tag))
        return found

    def _params(self, index: int) -> dict:
        with open(os.path.join(self.root, f"params_{index:06d}.json")) as f:
            return json.load(f)

    def _check_scalar_ranges(self) -> None:
        """Warn if the data was generated with different ranges than
        SCALAR_RANGES says. Silent mismatch here would corrupt every target."""
        if not self.require_scalars:
            return
        sample = self._params(self.indices[0])
        missing = [k for k in SCALAR_KEYS if k not in sample]
        if missing:
            raise RuntimeError(
                f"{self.root} has no {missing} in its params - that's a v1 "
                f"dataset. Regenerate with generate_dataset_v2.py, or pass "
                f"require_scalars=False to train on maps only.")

        probe = self.indices[::max(1, len(self.indices) // 50)]
        lo_hi = {k: [float("inf"), float("-inf")] for k in SCALAR_KEYS}
        for i in probe:
            p = self._params(i)
            for k in SCALAR_KEYS:
                lo_hi[k][0] = min(lo_hi[k][0], float(p[k]))
                lo_hi[k][1] = max(lo_hi[k][1], float(p[k]))
        for k, (lo, hi) in lo_hi.items():
            exp_lo, exp_hi = SCALAR_RANGES[k]
            slack = 0.05 * (exp_hi - exp_lo)
            if lo < exp_lo - slack or hi > exp_hi + slack:
                print(f"WARNING: {k} in the data spans [{lo:.3g}, {hi:.3g}] but "
                      f"SCALAR_RANGES says [{exp_lo:.3g}, {exp_hi:.3g}]. "
                      f"Update SCALAR_RANGES in dataset.py to match the generator.")

    def __len__(self) -> int:
        return len(self.indices)

    def _path(self, name: str, index: int) -> str:
        return os.path.join(self.root, f"{name}_{index:06d}.png")

    def __getitem__(self, i: int) -> dict:
        index = self.indices[i]

        photo = load_png(self._path("photo", index))
        basecolor = srgb_to_linear(load_png(self._path("basecolor", index)))
        roughness = load_png(self._path("roughness", index))[..., :1]
        metallic = load_png(self._path("metallic", index))[..., :1]
        normal = load_png(self._path("normal", index))
        maps = np.concatenate([basecolor, roughness, metallic, normal], axis=-1)

        p = self._params(index)
        params = np.array([p[k] for k in PARAM_KEYS], dtype=np.float32)
        scalars = (normalise_scalars(p) if self.require_scalars
                   else np.zeros(len(SCALAR_KEYS), dtype=np.float32))

        to_chw = lambda a: torch.from_numpy(np.ascontiguousarray(
            a.transpose(2, 0, 1)).astype(np.float32))

        return {
            "photo": to_chw(photo),          # (3, H, W)
            "maps": to_chw(maps),            # (8, H, W)
            "scalars": torch.from_numpy(scalars),   # (5,) normalised 0..1
            "params": torch.from_numpy(params),
            "index": index,
        }


def make_loaders(root: str, batch_size: int = 8, val_fraction: float = 0.1,
                 num_workers: int = 0, limit: int | None = None):
    """Convenience: returns (train_loader, val_loader)."""
    from torch.utils.data import DataLoader
    common = dict(root=root, val_fraction=val_fraction, limit=limit)
    train = CarPaintDataset(split="train", **common)
    val = CarPaintDataset(split="val", **common)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, drop_last=True),
        DataLoader(val, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )


# --- Self-test --------------------------------------------------------------
def self_test(root: str) -> bool:
    """Load a few samples and check the decoded data matches the params JSON."""
    ds = CarPaintDataset(root, split="all")
    print(f"found {len(ds)} complete samples in {root}")

    s = ds[0]
    print(f"photo   {tuple(s['photo'].shape)}  "
          f"range [{s['photo'].min():.3f}, {s['photo'].max():.3f}]")
    print(f"maps    {tuple(s['maps'].shape)}  "
          f"range [{s['maps'].min():.3f}, {s['maps'].max():.3f}]")
    print(f"scalars {tuple(s['scalars'].shape)} "
          f"range [{s['scalars'].min():.3f}, {s['scalars'].max():.3f}]")

    ok = True
    print("\nchecking decoded maps against params JSON:")
    for i in range(min(3, len(ds))):
        s = ds[i]
        with open(os.path.join(root, f"params_{s['index']:06d}.json")) as f:
            p = json.load(f)
        m = s["maps"]
        for label, expected, actual in [
            ("base_color R", p["base_color"][0], m[0].mean().item()),
            ("roughness", p["roughness"], m[3].mean().item()),
            ("metallic", p["metallic"], m[4].mean().item()),
        ]:
            # Wider tolerance than v1: these channels vary spatially now, so the
            # image MEAN only approximates the parameter it was built from.
            good = abs(expected - actual) < 0.05
            ok &= good
            if not good:
                print(f"  sample {s['index']:06d}  {label:<14} "
                      f"expected {expected:.3f}  got {actual:.3f}  MISMATCH")
        # Scalars should round-trip exactly.
        back = denormalise_scalars(s["scalars"])
        for k in SCALAR_KEYS:
            if abs(back[k] - float(p[k])) > 1e-3 * max(1.0, abs(float(p[k]))):
                ok = False
                print(f"  sample {s['index']:06d}  scalar {k} round-trip failed: "
                      f"{p[k]} -> {back[k]}")
        print(f"  sample {s['index']:06d}  {'OK' if ok else 'see above'}")

    nz = ds[0]["maps"][7].mean().item()
    print(f"\nnormal z mean {nz:.3f} (expect near 1.0 for a mostly flat sample)")
    print(f"normal x std {ds[0]['maps'][5].std().item():.4f} "
          f"(near 0 means the flakes carry no signal)")

    train, val = make_loaders(root, batch_size=min(4, len(ds)))
    print(f"\nloaders: {len(train.dataset)} train / {len(val.dataset)} val")
    batch = next(iter(train))
    print(f"one batch: photo {tuple(batch['photo'].shape)}, "
          f"maps {tuple(batch['maps'].shape)}, "
          f"scalars {tuple(batch['scalars'].shape)}")

    print("\nRESULT:", "dataset looks correct." if ok else
          "MISMATCH - do not train on this.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/blender_gen/dataset_v2")
    a = ap.parse_args()
    self_test(a.root)