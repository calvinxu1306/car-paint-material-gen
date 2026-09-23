"""
eval_baseline.py — is the model actually learning, or predicting averages?

A low loss means nothing on its own. A model that ignores the input and always
predicts the training-set average can score well when targets are flat. So this
compares, on the validation set:

    1. MEAN PREDICTOR   ignores the photo, always predicts the training-set
                        average. The "did nothing" baseline.
    2. THE MODEL        the trained network.
    3. SKILL            how much of the mean predictor's error the model
                        removes. 0% = learned nothing, 100% = perfect.

This is an L1 analogue of R-squared. It is a development sanity check, not a
headline metric - papers report PSNR/SSIM on maps, a re-rendered comparison,
and a table against prior methods.

Layer parameters are also reported in REAL units, which is what's actually
interpretable: "coat_roughness is off by 0.012 on a 0.01-0.1 range" says
something; a normalised number doesn't.

RUN:
    python src/eval_baseline.py --root data/blender_gen/dataset_v2 \
        --ckpt runs/v3/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("data", "models"):
    p = os.path.join(HERE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset import (CarPaintDataset, MAP_CHANNELS, SCALAR_KEYS,  # noqa: E402
                     SCALAR_RANGES)
from model import CarPaintNet                      # noqa: E402


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def training_means(loader):
    """Average of each target across the training set."""
    m_tot = torch.zeros(len(MAP_CHANNELS))
    s_tot = torch.zeros(len(SCALAR_KEYS))
    n = 0
    for batch in loader:
        b = batch["maps"].shape[0]
        m_tot += batch["maps"].mean(dim=(0, 2, 3)) * b
        s_tot += batch["scalars"].mean(dim=0) * b
        n += b
    return m_tot / max(n, 1), s_tot / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, map_means, scalar_means):
    n_m, n_s = len(MAP_CHANNELS), len(SCALAR_KEYS)
    model_m, mean_m = torch.zeros(n_m), torch.zeros(n_m)
    model_s, mean_s = torch.zeros(n_s), torch.zeros(n_s)
    n = 0

    for batch in loader:
        photo = batch["photo"].to(device)
        t_maps, t_scal = batch["maps"], batch["scalars"]
        b = t_maps.shape[0]

        pred = model(photo)
        model_m += (pred["maps"].cpu() - t_maps).abs().mean(dim=(0, 2, 3)) * b
        mean_m += (map_means[None, :, None, None].expand_as(t_maps)
                   - t_maps).abs().mean(dim=(0, 2, 3)) * b

        if "scalars" in pred:
            model_s += (pred["scalars"].cpu() - t_scal).abs().mean(dim=0) * b
            mean_s += (scalar_means[None, :].expand_as(t_scal)
                       - t_scal).abs().mean(dim=0) * b
        n += b

    return (model_m / n).numpy(), (mean_m / n).numpy(), \
           (model_s / n).numpy(), (mean_s / n).numpy()


def table(names, mean_err, model_err, unit_scale=None, unit_note=""):
    print(f"{'channel':<16}{'mean pred':>11}{'model':>11}{'skill':>9}"
          + (f"{unit_note:>18}" if unit_scale is not None else ""))
    print("-" * (47 + (16 if unit_scale is not None else 0)))
    for i, name in enumerate(names):
        skill = 100.0 * (1 - model_err[i] / mean_err[i]) if mean_err[i] > 1e-9 else 0.0
        verdict = "" if skill > 50 else ("  <- weak" if skill > 10
                                         else "  <- NOT LEARNED")
        line = (f"{name:<16}{mean_err[i]:>11.4f}{model_err[i]:>11.4f}"
                f"{skill:>8.1f}%")
        if unit_scale is not None:
            line += f"{model_err[i] * unit_scale[i]:>18.4f}"
        print(line + verdict)
    overall = 100.0 * (1 - model_err.mean() / mean_err.mean())
    print("-" * (47 + (16 if unit_scale is not None else 0)))
    print(f"{'OVERALL':<16}{mean_err.mean():>11.4f}{model_err.mean():>11.4f}"
          f"{overall:>8.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/blender_gen/dataset_v2")
    ap.add_argument("--ckpt", default="runs/v3/best.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from torch.utils.data import DataLoader
    device = pick_device(args.device)

    train_ds = CarPaintDataset(args.root, split="train")
    val_ds = CarPaintDataset(args.root, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = CarPaintNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch', '?')}) on {device}")
    print(f"train {len(train_ds)} / val {len(val_ds)} samples\n")

    m_means, s_means = training_means(train_loader)
    model_m, mean_m, model_s, mean_s = evaluate(model, val_loader, device,
                                                m_means, s_means)

    print("PER-PIXEL MAPS")
    table(MAP_CHANNELS, mean_m, model_m)

    print("\nLAYER PARAMETERS (clear coat + flakes)")
    spans = [SCALAR_RANGES[k][1] - SCALAR_RANGES[k][0] for k in SCALAR_KEYS]
    table(SCALAR_KEYS, mean_s, model_s, unit_scale=spans,
          unit_note="err (real units)")

    print("\nSkill is how much of the do-nothing baseline's error the model")
    print("removes. A tiny loss with low skill means the targets were easy.")


if __name__ == "__main__":
    main()