"""
dataset.py — PyTorch Dataset for the rendered car-paint samples.

Reads the folder produced by generate_dataset.py and hands the network
(photo -> ground-truth maps) pairs as tensors.

WHAT EACH SAMPLE LOOKS LIKE
    photo   float tensor (3, H, W)   the flash-lit photo  = the INPUT
    maps    float tensor (8, H, W)   the answer key       = the TARGET
    params  float tensor (7,)        the raw paint numbers (handy for eval)
    index   int                      which sample this is

The 8 target channels, in order:
    0,1,2  base color  R G B
    3      roughness
    4      metallic
    5,6,7  normal      x y z   (stored 0..1, where 0.5,0.5,1.0 = flat)

ONE SUBTLETY WORTH KNOWING
basecolor_*.png is sRGB-encoded (the normal convention for colour textures),
while roughness/metallic/normal are stored raw. So base color is decoded back
to linear here, and the others are used as-is. Get this wrong and every base
colour target is quietly too bright - which is exactly what the self-check in
Blender was guarding against, now enforced on the Python side too.

The photo is left display-encoded (AgX tone curve baked in). That's what a real
camera gives you, so it's the honest input. Deschaintre et al. work in log space
instead; worth revisiting in Week 3 if training struggles.

QUICK TEST (from the repo root, with the dataset rendered):
    python src/data/dataset.py --root data/blender_gen/dataset_output
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

# The order params are packed into the params vector.
PARAM_KEYS = ["metallic", "roughness", "coat_weight", "coat_roughness",
              "flake_scale", "flake_strength"]

MAP_CHANNELS = ["basecolor_r", "basecolor_g", "basecolor_b",
                "roughness", "metallic", "normal_x", "normal_y", "normal_z"]


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Undo sRGB encoding. Inverse of the linear_to_srgb used in the Blender check."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load_png(path: str) -> np.ndarray:
    """Load a PNG as float32 RGB in 0..1, shape (H, W, 3)."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return arr


class CarPaintDataset(Dataset):
    """Samples rendered by generate_dataset.py.

    Args:
        root: the dataset_output folder.
        split: "train", "val", or "all".
        val_fraction: share of samples held out for validation.
        limit: optionally use only the first N samples (handy while debugging).
    """

    def __init__(self, root: str, split: str = "train",
                 val_fraction: float = 0.1, limit: int | None = None):
        self.root = root
        self.indices = self._discover(root)
        if limit is not None:
            self.indices = self.indices[:limit]
        if not self.indices:
            raise RuntimeError(
                f"No samples found in {root}. Has generate_dataset.py run yet?")

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

    def __len__(self) -> int:
        return len(self.indices)

    def _path(self, name: str, index: int) -> str:
        return os.path.join(self.root, f"{name}_{index:06d}.png")

    def __getitem__(self, i: int) -> dict:
        index = self.indices[i]

        photo = load_png(self._path("photo", index))

        # sRGB-encoded -> back to linear, to match the params.
        basecolor = srgb_to_linear(load_png(self._path("basecolor", index)))
        # Stored raw; take one channel since they're greyscale.
        roughness = load_png(self._path("roughness", index))[..., :1]
        metallic = load_png(self._path("metallic", index))[..., :1]
        normal = load_png(self._path("normal", index))

        maps = np.concatenate([basecolor, roughness, metallic, normal], axis=-1)

        with open(os.path.join(self.root, f"params_{index:06d}.json")) as f:
            p = json.load(f)
        params = np.array([p[k] for k in PARAM_KEYS], dtype=np.float32)

        to_chw = lambda a: torch.from_numpy(np.ascontiguousarray(
            a.transpose(2, 0, 1)).astype(np.float32))

        return {
            "photo": to_chw(photo),     # (3, H, W)
            "maps": to_chw(maps),       # (8, H, W)
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
    """Load a few samples and check the decoded maps match the params JSON.

    Same spirit as the Blender self-check: don't trust the data, verify it.
    Catches sRGB mistakes, channel mix-ups and misaligned files.
    """
    ds = CarPaintDataset(root, split="all")
    print(f"found {len(ds)} complete samples in {root}")

    s = ds[0]
    print(f"photo  {tuple(s['photo'].shape)}  "
          f"range [{s['photo'].min():.3f}, {s['photo'].max():.3f}]")
    print(f"maps   {tuple(s['maps'].shape)}  "
          f"range [{s['maps'].min():.3f}, {s['maps'].max():.3f}]")

    ok = True
    print("\nchecking decoded maps against params JSON:")
    for i in range(min(3, len(ds))):
        s = ds[i]
        with open(os.path.join(root, f"params_{s['index']:06d}.json")) as f:
            p = json.load(f)
        m = s["maps"]
        checks = [
            ("base_color R", p["base_color"][0], m[0].mean().item()),
            ("base_color G", p["base_color"][1], m[1].mean().item()),
            ("roughness", p["roughness"], m[3].mean().item()),
            ("metallic", p["metallic"], m[4].mean().item()),
        ]
        for label, expected, actual in checks:
            good = abs(expected - actual) < 0.02
            ok &= good
            if not good:
                print(f"  sample {s['index']:06d}  {label:<14} "
                      f"expected {expected:.3f}  got {actual:.3f}  MISMATCH")
        print(f"  sample {s['index']:06d}  "
              f"{'all OK' if ok else 'see mismatches above'}")

    # Flat surfaces should sit near (0.5, 0.5, 1.0) before the flakes tilt them.
    nz = ds[0]["maps"][7].mean().item()
    print(f"\nnormal z mean {nz:.3f} (expect near 1.0 for a flat sample)")

    train, val = make_loaders(root, batch_size=min(4, len(ds)))
    print(f"loaders: {len(train.dataset)} train / {len(val.dataset)} val")
    batch = next(iter(train))
    print(f"one batch: photo {tuple(batch['photo'].shape)}, "
          f"maps {tuple(batch['maps'].shape)}")

    print("\nRESULT:", "dataset looks correct." if ok else
          "MISMATCH - decoding is wrong, do not train on this.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/blender_gen/dataset_output")
    a = ap.parse_args()
    self_test(a.root)