"""
dataset_multi.py — several photos of the same paint, one answer key.

Reads v5 folders (a flash photo plus side-lit photos per sample, from
generate_dataset_v5.py). Also reads v3/v4 folders, which only have the flash
photo, as one photo per sample - so one loader serves every dataset and the
v2 model can be compared with v1 on v1's own data.

WHAT EACH SAMPLE LOOKS LIKE
    photos       float (K, 3, H, W)  photos[0] is always the flash photo (when
                                     the selection includes it)
    maps         float (8, H, W)     same 8 channels as dataset.py
    scalars      float (S,)          layer parameters, normalised 0..1
    scalar_mask  float (S,)          1 = the parameter means something for this
                                     paint, 0 = ignore it in the loss/metrics
    pigment      int                 index into .pigments, -1 if not recorded
    matte        float               1 = matte finish, 0 = gloss
    index        int

WHY A MASK
Every paint has the same parameter list, but some entries are meaningless for
some paints: solid paint has no flakes, so its flake SIZE is noise; a paint with
no thin film has no film IOR. Training the network to predict those would teach
it to guess random numbers. The mask switches them off:
    flake_scale  ignored when flake_strength == 0   (solid paint)
    film_ior     ignored when film_thickness == 0   (no film)
Everything else is always meaningful - "no film" is a real answer
(film_thickness = 0), and so is "no tint" (coat_tint = white).

The scalar list, its ranges, the photo names and the pigment names all come
from the folder's meta.json, so the loader can't silently disagree with the
generator.

QUICK TEST:
    python src/data/dataset_multi.py --root data/blender_gen/dataset_v5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (MAP_CHANNELS, SCALAR_KEYS as V1_SCALAR_KEYS,  # noqa: E402
                     SCALAR_RANGES as V1_SCALAR_RANGES, load_png, srgb_to_linear)

MAP_FILES = ["basecolor", "roughness", "metallic", "normal"]

# Which scalar is ignored when, as (key, the key that switches it off).
MASK_RULES = [("flake_scale", "flake_strength"), ("film_ior", "film_thickness")]


def read_meta(root: str) -> dict:
    path = os.path.join(root, "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def describe(root: str) -> dict:
    """What a dataset folder contains: scalar keys + ranges, photos, pigments."""
    meta = read_meta(root)
    keys = meta.get("scalar_keys", V1_SCALAR_KEYS)
    ranges_src = meta.get("ranges", {})
    if meta:
        missing = [k for k in keys if k not in ranges_src]
        if missing:
            raise RuntimeError(f"{root}/meta.json has no range for {missing}")
        ranges = {k: tuple(ranges_src[k]) for k in keys}
    else:
        ranges = {k: tuple(V1_SCALAR_RANGES[k]) for k in keys}
    return {
        "version": meta.get("version"),
        "scalar_keys": list(keys),
        "scalar_ranges": ranges,
        "photos": meta.get("photos", ["photo"]),
        "pigments": list(meta.get("pigments", {})),
    }


def normalise(p: dict, keys, ranges) -> np.ndarray:
    return np.array([(float(p[k]) - ranges[k][0]) / (ranges[k][1] - ranges[k][0])
                     for k in keys], dtype=np.float32)


def denormalise(v, keys, ranges) -> dict:
    arr = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    return {k: float(arr[i] * (ranges[k][1] - ranges[k][0]) + ranges[k][0])
            for i, k in enumerate(keys)}


def scalar_mask(p: dict, keys) -> np.ndarray:
    m = np.ones(len(keys), dtype=np.float32)
    for key, switch in MASK_RULES:
        if key in keys and switch in p and float(p[switch]) == 0.0:
            m[keys.index(key)] = 0.0
    return m


class MultiLightDataset(Dataset):
    """Args:
        root: dataset folder.
        split: "train", "val" (last val_fraction of samples) or "all".
        photos: which photo names to load (default: all the folder has).
        limit: use only the first N samples.
    """

    def __init__(self, root: str, split: str = "train", val_fraction: float = 0.1,
                 photos: list[str] | None = None, limit: int | None = None):
        self.root = root
        info = describe(root)
        self.version = info["version"]
        self.scalar_keys = info["scalar_keys"]
        self.scalar_ranges = info["scalar_ranges"]
        self.pigments = info["pigments"]
        names = list(photos or info["photos"])
        unknown = [n for n in names if n not in info["photos"]]
        if unknown:
            raise ValueError(f"{root} has no photos called {unknown}; "
                             f"it has {info['photos']}")
        # The flash photo always comes first: training ("--photos flash") and
        # evaluation ("flash" set) rely on photos[0] being the flash photo.
        if "photo" in names:
            names.remove("photo")
            names.insert(0, "photo")
        self.photo_names = names

        self.indices = self._discover()
        if limit is not None:
            self.indices = self.indices[:limit]
        if not self.indices:
            raise RuntimeError(f"No complete samples in {root}. Has the generator run?")
        self._check_ranges()

        n_val = max(1, int(round(len(self.indices) * val_fraction)))
        if split == "train":
            self.indices = self.indices[:-n_val]
        elif split == "val":
            self.indices = self.indices[-n_val:]
        elif split != "all":
            raise ValueError(f"split must be train/val/all, got {split!r}")

    def _discover(self) -> list[int]:
        """Complete samples only: a half-written last sample is skipped."""
        found = []
        for name in sorted(os.listdir(self.root)):
            if not (name.startswith("params_") and name.endswith(".json")):
                continue
            tag = name[len("params_"):-len(".json")]
            needed = self.photo_names + MAP_FILES
            if all(os.path.exists(os.path.join(self.root, f"{n}_{tag}.png"))
                   for n in needed):
                found.append(int(tag))
        return found

    def params(self, index: int) -> dict:
        with open(os.path.join(self.root, f"params_{index:06d}.json")) as f:
            return json.load(f)

    def _check_ranges(self) -> None:
        probe = self.indices[::max(1, len(self.indices) // 50)]
        for k in self.scalar_keys:
            vals = [float(self.params(i)[k]) for i in probe]
            lo, hi = self.scalar_ranges[k]
            slack = 0.05 * (hi - lo)
            if min(vals) < lo - slack or max(vals) > hi + slack:
                print(f"WARNING: {k} in the data spans [{min(vals):.3g}, "
                      f"{max(vals):.3g}] but meta.json says [{lo:.3g}, {hi:.3g}]")

    def __len__(self) -> int:
        return len(self.indices)

    def _png(self, name: str, index: int) -> np.ndarray:
        return load_png(os.path.join(self.root, f"{name}_{index:06d}.png"))

    def __getitem__(self, i: int) -> dict:
        index = self.indices[i]
        photos = np.stack([self._png(n, index) for n in self.photo_names])
        basecolor = srgb_to_linear(self._png("basecolor", index))
        rough = self._png("roughness", index)[..., :1]
        metal = self._png("metallic", index)[..., :1]
        normal = self._png("normal", index)
        maps = np.concatenate([basecolor, rough, metal, normal], axis=-1)

        p = self.params(index)
        pigment = (self.pigments.index(p["pigment"])
                   if p.get("pigment") in self.pigments else -1)
        return {
            "photos": torch.from_numpy(np.ascontiguousarray(
                photos.transpose(0, 3, 1, 2))),                  # (K, 3, H, W)
            "maps": torch.from_numpy(np.ascontiguousarray(
                maps.transpose(2, 0, 1)).astype(np.float32)),     # (8, H, W)
            "scalars": torch.from_numpy(
                normalise(p, self.scalar_keys, self.scalar_ranges)),
            "scalar_mask": torch.from_numpy(scalar_mask(p, self.scalar_keys)),
            "pigment": pigment,
            "matte": torch.tensor(float(p.get("finish") == "matte"),
                                  dtype=torch.float32),
            "index": index,
        }


# --- Self-test --------------------------------------------------------------
def self_test(root: str) -> bool:
    ds = MultiLightDataset(root, split="all")
    print(f"found {len(ds)} complete samples in {root} (dataset v{ds.version})")
    print(f"photos per sample: {len(ds.photo_names)} {ds.photo_names}")
    print(f"scalars: {len(ds.scalar_keys)} {ds.scalar_keys}")
    print(f"pigments: {ds.pigments or '(not recorded - v3/v4 data)'}")

    ok = True
    for i in range(min(4, len(ds))):
        s = ds[i]
        p = ds.params(s["index"])
        m = s["maps"]
        checks = [("base_color R", p["base_color"][0], m[0].mean().item()),
                  ("roughness", p["roughness"], m[3].mean().item()),
                  ("metallic", p["metallic"], m[4].mean().item())]
        for label, expected, actual in checks:
            if abs(expected - actual) > 0.05:
                ok = False
                print(f"  {s['index']:06d} {label}: expected {expected:.3f} "
                      f"got {actual:.3f}  MISMATCH")
        back = denormalise(s["scalars"], ds.scalar_keys, ds.scalar_ranges)
        for k in ds.scalar_keys:
            if abs(back[k] - float(p[k])) > 1e-3 * max(1.0, abs(float(p[k]))):
                ok = False
                print(f"  {s['index']:06d} scalar {k} round-trip failed")
        masked = [k for k, v in zip(ds.scalar_keys, s["scalar_mask"]) if v == 0]
        # Masks must follow the rules exactly.
        for key, switch in MASK_RULES:
            if key in ds.scalar_keys and switch in p:
                should = float(p[switch]) == 0.0
                if should != (key in masked):
                    ok = False
                    print(f"  {s['index']:06d} mask for {key} is wrong")
        ph = s["photos"]
        spread = (ph - ph[:1]).abs().mean(dim=(1, 2, 3)).tolist()
        print(f"  {s['index']:06d} {p.get('pigment', '?'):<10} "
              f"{p.get('finish', ''):<5} photos {tuple(ph.shape)} "
              f"masked={masked or '-'} "
              f"diff-from-flash per photo={[round(d, 3) for d in spread]}")
        if len(spread) > 1 and min(spread[1:]) < 1e-3:
            ok = False
            print("  a side photo is identical to the flash photo - lights not moving?")

    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(ds, batch_size=min(2, len(ds)))))
    print(f"\none batch: photos {tuple(batch['photos'].shape)}, maps "
          f"{tuple(batch['maps'].shape)}, scalars {tuple(batch['scalars'].shape)}, "
          f"pigment {batch['pigment'].tolist()}")
    print("\nRESULT:", "dataset looks correct." if ok else "MISMATCH - do not train on this.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/blender_gen/dataset_v5")
    self_test(ap.parse_args().root)
