"""
export_demo.py — turn model predictions into assets for the web viewer.

Runs the trained model on a handful of VALIDATION samples (never seen in
training) and writes, for each one:

    photo.png           the flash photo the model was given
    pred_basecolor.png  predicted base colour  (sRGB-encoded, like any texture)
    pred_orm.png        predicted roughness in G, metallic in B  (R = 1)
    pred_normal.png     predicted flake normal (xyz stored as v*0.5+0.5)
    pred_rough.png      greyscale roughness, for display in the page only
    gt_*.png            the same four from the ground truth, for comparison

plus demo/assets/samples/manifest.json with the layer parameters (clear coat,
flakes) in real units, predicted and true.

WHY THE ROUGHNESS/METALLIC PACKING
three.js (following glTF) reads roughness from a texture's GREEN channel and
metalness from its BLUE channel. Packing both into one "ORM" image is the
standard layout, so the viewer can hand the same texture to both slots.

UNITS CARRY STRAIGHT ACROSS
Blender and three.js both square roughness into the microfacet alpha, so the
predicted roughness and clear-coat roughness go into three.js unchanged.

RUN (from the repo root, after training):
    python src/export_demo.py --root data/blender_gen/dataset_v3 \
        --ckpt runs/v3_joint/best.pt --n 12
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("data", "models"):
    p = os.path.join(HERE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset import CarPaintDataset, SCALAR_KEYS, denormalise_scalars  # noqa: E402
from model import CarPaintNet                                          # noqa: E402


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save_rgb(arr_hwc: np.ndarray, path: str) -> None:
    Image.fromarray((np.clip(arr_hwc, 0, 1) * 255).round().astype(np.uint8)).save(path)


def maps_to_textures(maps_chw: np.ndarray) -> dict:
    """8-channel map stack (linear, 0..1) -> the three web textures."""
    m = maps_chw.transpose(1, 2, 0)                    # (H, W, 8)
    basecolor = linear_to_srgb(m[..., 0:3])
    orm = np.stack([np.ones_like(m[..., 3]), m[..., 3], m[..., 4]], axis=-1)
    # Re-normalise the normal: the network's sigmoid outputs are not exactly
    # unit length, and an unnormalised normal map shades too dark.
    n = m[..., 5:8] * 2.0 - 1.0
    n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(min=1e-6)
    normal = n * 0.5 + 0.5
    # Greyscale roughness purely for display: the packed ORM texture shows up
    # as meaningless magenta if you put it on screen.
    rough = np.repeat(m[..., 3:4], 3, axis=-1)
    return {"basecolor": basecolor, "orm": orm, "normal": normal, "rough": rough}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/blender_gen/dataset_v3")
    ap.add_argument("--ckpt", default="runs/v3_joint/best.pt")
    ap.add_argument("--out", default="demo/assets/samples")
    ap.add_argument("--n", type=int, default=12, help="how many samples to export")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    ds = CarPaintDataset(args.root, split="val")
    ranges = ds.scalar_ranges

    model = CarPaintNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Spread the picks across the validation set rather than taking the first N.
    n = min(args.n, len(ds))
    picks = np.linspace(0, len(ds) - 1, n).round().astype(int)

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)      # stale samples from an older model mislead
    os.makedirs(args.out)

    samples = []
    with torch.no_grad():
        for k, i in enumerate(picks):
            s = ds[int(i)]
            out = model(s["photo"][None].to(device))
            pred_maps = out["maps"][0].cpu().numpy()
            gt_maps = s["maps"].numpy()

            tag = f"{s['index']:06d}"
            folder = os.path.join(args.out, tag)
            os.makedirs(folder)

            save_rgb(s["photo"].numpy().transpose(1, 2, 0),
                     os.path.join(folder, "photo.png"))
            for prefix, maps in (("pred", pred_maps), ("gt", gt_maps)):
                for name, img in maps_to_textures(maps).items():
                    save_rgb(img, os.path.join(folder, f"{prefix}_{name}.png"))

            with open(os.path.join(args.root, f"params_{tag}.json")) as f:
                p = json.load(f)
            pred_scalars = (denormalise_scalars(out["scalars"][0], ranges)
                            if "scalars" in out else {})
            gt_scalars = {k2: float(p[k2]) for k2 in SCALAR_KEYS}

            samples.append({
                "id": tag,
                "folder": tag,
                "pred": pred_scalars,
                "gt": gt_scalars,
                "base": {  # mean map values: table + fallback for models without UVs
                    "pred_roughness": float(pred_maps[3].mean()),
                    "gt_roughness": float(gt_maps[3].mean()),
                    "pred_metallic": float(pred_maps[4].mean()),
                    "gt_metallic": float(gt_maps[4].mean()),
                    "pred_basecolor": pred_maps[0:3].mean(axis=(1, 2)).tolist(),
                    "gt_basecolor": gt_maps[0:3].mean(axis=(1, 2)).tolist(),
                },
            })
            print(f"[{k + 1}/{n}] exported sample {tag}")

    manifest = {
        "checkpoint": os.path.relpath(args.ckpt).replace("\\", "/"),
        "checkpoint_epoch": ckpt.get("epoch"),
        "dataset": os.path.relpath(args.root).replace("\\", "/"),
        "split": "validation (never seen in training)",
        "scalar_ranges": {k2: list(v) for k2, v in ranges.items()},
        "samples": samples,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {n} samples + manifest.json to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
