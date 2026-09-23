"""
train.py — train the model: per-pixel maps + per-sample layer parameters.

LOSSES (each can be weighted, or switched off with 0)
    map      L1 on the 8 per-pixel maps                     --map-weight
    scalar   L1 on the 5 layer parameters (coat, flakes)    --scalar-weight
    render   Deschaintre-style rendering-aware loss         --render-weight

Render defaults to 0 (off) so results stay comparable with earlier runs. Turn
it on with --render-weight 1 once you want to compare with and without it; that
comparison is an ablation worth having in the write-up.

FIRST THING TO RUN — the overfit test:
    python src/train.py --root data/blender_gen/dataset_v2 --out runs/overfit \
        --overfit 4 --epochs 200
The loss should collapse towards ~0. If it can't overfit 4 samples, something
is broken and training on 2000 would hide it behind noise.

THEN the real run:
    python src/train.py --root data/blender_gen/dataset_v2 --out runs/v3 \
        --epochs 50 --batch-size 8

Give every run its OWN --out directory. Mixing runs in one folder is how
previews and logs get misattributed to the wrong model.

Outputs, under --out:
    best.pt          weights with the lowest validation loss
    last.pt          weights from the final epoch
    log.csv          per-epoch losses, for plotting later
    preview_XXX.png  predicted vs. ground-truth maps, saved periodically

WHY L1 AND NOT L2
L1 blurs less on image tasks - L2 hedges towards the average when it's unsure,
which would wash out the flake detail.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Make the sibling folders importable without any packaging ceremony.
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("data", "models"):
    p = os.path.join(HERE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset import (CarPaintDataset, MAP_CHANNELS, SCALAR_KEYS,  # noqa: E402
                     SCALAR_RANGES)
from model import CarPaintNet, count_params        # noqa: E402
from render import RenderLoss                      # noqa: E402


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Criterion(nn.Module):
    """map L1 + scalar L1 + optional rendering loss, reported separately."""

    def __init__(self, map_weight=1.0, scalar_weight=1.0, render_weight=0.0):
        super().__init__()
        self.map_weight = map_weight
        self.scalar_weight = scalar_weight
        self.render_weight = render_weight
        self.l1 = nn.L1Loss()
        self.render_loss = RenderLoss() if render_weight > 0 else None

    def forward(self, pred: dict, target_maps, target_scalars):
        parts = {}
        parts["map"] = self.l1(pred["maps"], target_maps)
        parts["scalar"] = (self.l1(pred["scalars"], target_scalars)
                           if "scalars" in pred
                           else pred["maps"].new_zeros(()))
        parts["render"] = (self.render_loss(pred["maps"], target_maps)
                           if self.render_loss is not None
                           else pred["maps"].new_zeros(()))
        total = (self.map_weight * parts["map"]
                 + self.scalar_weight * parts["scalar"]
                 + self.render_weight * parts["render"])
        return total, parts


@torch.no_grad()
def save_preview(pred_maps, target_maps, photo, path):
    """PNG: top row = ground truth, bottom row = prediction.
    Columns: photo | basecolor | roughness | metallic | normal."""
    def to_img(t):
        a = t.detach().float().clamp(0, 1).cpu().numpy()
        if a.shape[0] == 1:
            a = np.repeat(a, 3, axis=0)
        return (a.transpose(1, 2, 0) * 255).round().astype(np.uint8)

    def row(maps, ph):
        return np.concatenate([to_img(ph), to_img(maps[0:3]), to_img(maps[3:4]),
                               to_img(maps[4:5]), to_img(maps[5:8])], axis=1)

    grid = np.concatenate([row(target_maps[0], photo[0]),
                           row(pred_maps[0], photo[0])], axis=0)
    Image.fromarray(grid).save(path)


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass over a loader. With an optimizer it trains, without it evaluates."""
    training = optimizer is not None
    model.train(training)
    criterion.train(training)

    total, n_batches = 0.0, 0
    parts_total = {"map": 0.0, "scalar": 0.0, "render": 0.0}
    chan_totals = np.zeros(len(MAP_CHANNELS))
    scalar_totals = np.zeros(len(SCALAR_KEYS))
    last = None

    for batch in loader:
        photo = batch["photo"].to(device, non_blocking=True)
        t_maps = batch["maps"].to(device, non_blocking=True)
        t_scalars = batch["scalars"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            pred = model(photo)
            loss, parts = criterion(pred, t_maps, t_scalars)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total += loss.item()
        for k in parts_total:
            parts_total[k] += parts[k].item()
        chan_totals += (pred["maps"] - t_maps).abs().mean(
            dim=(0, 2, 3)).detach().cpu().numpy()
        if "scalars" in pred:
            scalar_totals += (pred["scalars"] - t_scalars).abs().mean(
                dim=0).detach().cpu().numpy()
        n_batches += 1
        last = (pred["maps"], t_maps, photo)

    n = max(n_batches, 1)
    return (total / n, {k: v / n for k, v in parts_total.items()},
            chan_totals / n, scalar_totals / n, last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/blender_gen/dataset_v2")
    ap.add_argument("--out", default="runs/v3")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overfit", type=int, default=0,
                    help="sanity check: train on N samples, no validation")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--preview-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-weight", type=float, default=1.0)
    ap.add_argument("--scalar-weight", type=float, default=1.0)
    ap.add_argument("--render-weight", type=float, default=0.0,
                    help="rendering-aware loss; 0 = off (default)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)

    from torch.utils.data import DataLoader

    if args.overfit:
        ds = CarPaintDataset(args.root, split="all", limit=args.overfit)
        bs = min(args.batch_size, len(ds))
        train_loader = DataLoader(ds, batch_size=bs, shuffle=True,
                                  num_workers=args.workers)
        val_loader = DataLoader(ds, batch_size=bs, num_workers=args.workers)
        print(f"OVERFIT MODE: {len(ds)} samples, expecting the loss to approach 0")
    else:
        common = dict(root=args.root, val_fraction=0.1, limit=args.limit)
        train_ds = CarPaintDataset(split="train", **common)
        val_ds = CarPaintDataset(split="val", **common)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                num_workers=args.workers)
        print(f"train {len(train_ds)} / val {len(val_ds)} samples")

    model = CarPaintNet().to(device)
    criterion = Criterion(args.map_weight, args.scalar_weight,
                          args.render_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"device: {device} | parameters: {count_params(model):,}")
    print(f"loss: {args.map_weight} x map + {args.scalar_weight} x scalar"
          + (f" + {args.render_weight} x render" if args.render_weight else ""))

    log_path = os.path.join(args.out, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_map", "val_scalar",
             "val_render", "seconds"]
            + [f"val_{c}" for c in MAP_CHANNELS]
            + [f"val_{k}" for k in SCALAR_KEYS])

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, _, _, _, _ = run_epoch(model, train_loader, device,
                                           criterion, optimizer)
        val_loss, vp, vchan, vscal, last = run_epoch(model, val_loader, device,
                                                     criterion)
        dt = time.time() - t0

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                 f"{vp['map']:.6f}", f"{vp['scalar']:.6f}", f"{vp['render']:.6f}",
                 f"{dt:.1f}"]
                + [f"{c:.6f}" for c in vchan] + [f"{s:.6f}" for s in vscal])

        flag = ""
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss}, os.path.join(args.out, "best.pt"))
            flag = "  <- best"
        torch.save({"model": model.state_dict(), "epoch": epoch},
                   os.path.join(args.out, "last.pt"))

        print(f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.4f}  "
              f"val {val_loss:.4f}  (map {vp['map']:.4f} scalar {vp['scalar']:.4f})"
              f"  {dt:.1f}s{flag}", flush=True)

        if epoch % args.preview_every == 0 or epoch == args.epochs:
            save_preview(*last, os.path.join(args.out, f"preview_{epoch:03d}.png"))

    print(f"\nbest validation loss: {best:.4f}")
    print("\nlayer parameters, mean error in REAL units:")
    for i, k in enumerate(SCALAR_KEYS):
        lo, hi = SCALAR_RANGES[k]
        print(f"  {k:<16} {vscal[i] * (hi - lo):.4f}   (range {lo} to {hi})")
    worst = np.argsort(vchan)[::-1][:3]
    print("\nworst map channels: " + ", ".join(
        f"{MAP_CHANNELS[i]} {vchan[i]:.4f}" for i in worst))
    print(f"outputs in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()