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

HDR PHOTOS (hdr=True; off by default)
The PNG photos are 8-bit, through Blender's AgX tone curve, which squeezes or
clips the clear coat's highlight. A folder rendered with
generate_dataset_v5.py --hdr also has every photo as <name>_XXXXXX.exr:
linear radiance x, highlights above 1 kept. hdr=True reads those instead
and feeds them in log space, as Kaltheuner et al. 2021 do:
    I = (log(x + 0.01) - log(0.01)) / (log(1.01) - log(0.01))   per channel
x = 0 -> 0 and x = 1 -> 1, like the PNGs' range, but a highlight 10x brighter
than 1 is ~1.5 and 100x is ~2.0 - compressed, not clipped. The maps and
parameters don't change. Needs the OpenEXR package (pip install OpenEXR),
imported only when hdr=True.

QUICK TEST (also checks the EXR photos when the folder has them):
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

# Log encoding of HDR photos (Kaltheuner et al. 2021): 0 -> 0, 1 -> 1.
HDR_EPS = 0.01
HDR_HALF_MAX = 65504.0   # largest half float: where an overflowed (inf) pixel goes


def hdr_log_encode(x: np.ndarray) -> np.ndarray:
    """Linear radiance -> (log(x + 0.01) - log 0.01) / (log 1.01 - log 0.01).
    Negative values (a denoiser can undershoot a little) count as 0, and a
    non-finite pixel is replaced, so one bad pixel can't make the loss NaN."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0,
                      posinf=HDR_HALF_MAX, neginf=0.0)
    x = np.maximum(x, 0.0)
    # log(x + e) - log(e) = log1p(x / e): the same formula, but exactly 0 at
    # x = 0 in float32 (the difference of two logs leaves a ~1e-8 residue).
    return (np.log1p(x / HDR_EPS) / np.log1p(1.0 / HDR_EPS)).astype(np.float32)


def _openexr():
    """The OpenEXR module, imported only when HDR photos are used."""
    try:
        import OpenEXR
    except ImportError as e:
        raise ImportError("Reading HDR photos (.exr) needs the OpenEXR package: "
                          "pip install OpenEXR") from e
    return OpenEXR


def load_exr(path: str) -> np.ndarray:
    """Load an RGB OpenEXR as float32 linear values, shape (H, W, 3)."""
    with _openexr().File(path, separate_channels=True) as f:
        channels = {name: ch.pixels for name, ch in f.channels().items()}

    def channel(c):
        if c in channels:
            return channels[c]
        # e.g. "Combined.R", if a file was ever saved with layer names
        hits = [n for n in channels if n.endswith("." + c)]
        if len(hits) != 1:
            raise RuntimeError(f"{path}: no single {c} channel among {sorted(channels)}")
        return channels[hits[0]]
    return np.stack([channel(c) for c in "RGB"], axis=-1).astype(np.float32)


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
        "hdr": bool(meta.get("hdr_photos")),   # written by generate_dataset_v5.py --hdr
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
        hdr: read the photos from the .exr files, log-encoded (see HDR PHOTOS
             above). The folder must have been rendered with --hdr.
    """

    def __init__(self, root: str, split: str = "train", val_fraction: float = 0.1,
                 photos: list[str] | None = None, limit: int | None = None,
                 hdr: bool = False):
        self.root = root
        self.hdr = hdr
        info = describe(root)
        if hdr:
            if not info["hdr"]:
                raise RuntimeError(
                    f"HDR photos were asked for (hdr=True: train_multi.py --hdr-input, "
                    f"or eval_multi.py on a run trained with it), but {root} has none: "
                    f"its meta.json has no 'hdr_photos' entry. Render a new folder "
                    f"with generate_dataset_v5.py --hdr.")
            _openexr()   # fail now, with the install hint, not in the first batch
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
        # The files __getitem__ reads: the photos as .exr with hdr, else .png.
        photo_ext = ".exr" if self.hdr else ".png"
        for name in sorted(os.listdir(self.root)):
            if not (name.startswith("params_") and name.endswith(".json")):
                continue
            tag = name[len("params_"):-len(".json")]
            needed = ([f"{n}_{tag}{photo_ext}" for n in self.photo_names]
                      + [f"{n}_{tag}.png" for n in MAP_FILES])
            if all(os.path.exists(os.path.join(self.root, f)) for f in needed):
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

    def _photo(self, name: str, index: int) -> np.ndarray:
        if self.hdr:
            return hdr_log_encode(
                load_exr(os.path.join(self.root, f"{name}_{index:06d}.exr")))
        return self._png(name, index)

    def __getitem__(self, i: int) -> dict:
        index = self.indices[i]
        photos = np.stack([self._photo(n, index) for n in self.photo_names])
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
    if describe(root)["hdr"]:
        ok = hdr_check(root) and ok
    print("\nRESULT:", "dataset looks correct." if ok else "MISMATCH - do not train on this.")
    return ok


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def hdr_check(root: str, n_samples: int = 4) -> bool:
    """For a folder rendered with --hdr: how bright the EXR photos get, and
    whether each EXR is the same picture as its PNG (same render, so their
    brightness patterns must match pixel for pixel). Fails if a gloss sample
    was checked and nothing goes above 1: a gloss coat's flash hotspot, a
    reflection of the 500 W flash, should be well above 1 in linear, so
    the EXRs were probably written through the view transform (AgX), which
    is the untested Blender assumption."""
    print("\nHDR photos (.exr, read with hdr=True):")
    try:
        ds = MultiLightDataset(root, split="all", hdr=True)
    except ImportError as e:
        print(f"  NOT CHECKED: {e}")
        return True
    except RuntimeError as e:          # e.g. meta.json says HDR but no .exr files
        print(f"  {e}  MISMATCH")
        return False
    n_png = len(MultiLightDataset(root, split="all"))
    if len(ds) != n_png:
        print(f"  WARNING: {n_png} samples are complete as PNGs but {len(ds)} with "
              f"EXRs - some .exr files are missing")
    ok = True
    peak = 0.0
    checked = list(ds.indices[:n_samples])
    finish = {i: ds.params(i).get("finish") for i in checked}
    if "gloss" not in finish.values():
        # The peak test below needs a gloss sample: add the first one, if any.
        gloss = next((i for i in ds.indices[n_samples:]
                      if ds.params(i).get("finish") == "gloss"), None)
        if gloss is not None:
            checked.append(gloss)
            finish[gloss] = "gloss"
    for index in checked:
        lin = np.stack([load_exr(os.path.join(root, f"{n}_{index:06d}.exr"))
                        for n in ds.photo_names])                  # (K, H, W, 3)
        png = np.stack([ds._png(n, index) for n in ds.photo_names])
        if lin.shape != png.shape:
            print(f"  {index:06d} EXR photos {lin.shape} but PNGs {png.shape}  MISMATCH")
            ok = False
            continue
        if not np.isfinite(lin).all():
            print(f"  {index:06d} has NaN/inf pixels (hdr_log_encode replaces them)")
        enc = hdr_log_encode(lin)
        # Same render: the log-encoded EXR and the tone-mapped PNG both rise
        # with brightness, so they must correlate strongly. The same with the
        # PNG shifted 2 pixels is shown for comparison: lower if they line up.
        y_enc, y_png = enc.mean(-1), png.mean(-1)
        r = _corr(y_enc, y_png)
        r_shift = _corr(y_enc, np.roll(y_png, 2, axis=-1))
        peak = max(peak, float(np.nanmax(lin)))
        print(f"  {index:06d} {finish[index] or '':<5} linear max {np.nanmax(lin):7.2f}, "
              f"above 1: {100 * (lin.max(-1) > 1).mean():5.2f}% of pixels | encoded "
              f"{enc.min():.3f}..{enc.max():.3f} | EXR vs PNG r = {r:.3f} "
              f"(shifted 2 px: {r_shift:.3f})")
        if r < 0.5:
            ok = False
            print(f"  {index:06d} EXR and PNG don't look like the same picture  MISMATCH")
    if peak <= 1.0:
        n_gloss = sum(finish[i] == "gloss" for i in checked)
        if n_gloss:
            ok = False
            print(f"  no pixel above 1, yet {n_gloss} of these samples are gloss, whose "
                  "flash hotspot should be well above 1. Were the EXRs written through "
                  "the view transform? See save_last_render_exr in "
                  "generate_dataset_v5.py  MISMATCH")
        else:
            print("  WARNING: no pixel above 1, but the folder has no gloss sample to "
                  "test this on (a gloss paint's flash hotspot should go above 1)")
    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(ds, batch_size=min(2, len(ds)))))
    print(f"  one HDR batch: photos {tuple(batch['photos'].shape)}, "
          f"{batch['photos'].min().item():.3f}..{batch['photos'].max().item():.3f}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/blender_gen/dataset_v5")
    self_test(ap.parse_args().root)
