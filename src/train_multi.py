"""
train_multi.py — train the v2 model: several photos of one paint -> its material.

What's new compared with train.py (v1):
    * Each sample has K photos (flash + side lights). Every batch the network
      sees a RANDOM number of them, in random order, so one trained model works
      with 1 photo or with all of them. That is what lets eval_multi.py ask
      "how much does each extra photo buy?" with the same weights.
    * 10 layer parameters instead of 5 (coat tint and thin film are new), and
      a mask so meaningless ones (flake size of a solid paint, film IOR when
      there's no film) don't count. See dataset_multi.py.
    * A pigment-type head (solid / metallic / pearl / colorshift / candy),
      trained with cross-entropy. It's mostly for reporting, and it gives the
      network an easy, related target early in training.

LOSSES (each weighted; 0 switches it off)
    map      L1 on the 8 per-pixel maps                  --map-weight
    scalar   masked L1 on the layer parameters           --scalar-weight
    pigment  cross-entropy on the pigment type           --pigment-weight

WHICH PHOTOS THE NETWORK SEES (--photos)
    random  (default) each batch: k ~ uniform{1..--max-photos}, chosen at
            random from all the sample's photos. --max-photos defaults to
            every photo the dataset has, so validation and evaluation never
            show the network more photos than it trained with (max-pooled
            features grow with the photo count, so "more than in training"
            would be an untested condition).
    flash   only the flash photo - v1's input, with v2's network: the
            single-flash baseline for the v2 comparison
    flash+random
            the flash photo every time, plus k-1 side photos chosen at
            random (k ~ uniform{1..--max-photos}, the same spread as random).
            A real capture always includes the flash photo; random leaves it
            out of ~42% of samples. Run 12 in ablations.md compares the two.
    all     the first --max-photos photos (flash first), every time

MODEL OPTIONS (off by default; see MultiLightPaintNet in model.py)
    --coords       pixel x/y as two extra input channels per photo
    --flash-slot   the flash photo gets its own slot next to the max/mean
                   pooling; needs --photos flash+random, all or flash

HDR INPUT (--hdr-input, off by default)
    Reads the photos from the .exr files of a folder rendered with
    generate_dataset_v5.py --hdr, in log space (HDR PHOTOS in dataset_multi.py):
    the coat highlight keeps its height instead of being squeezed by the PNGs'
    tone curve. Needs pip install OpenEXR. Recorded in config.json, so
    eval_multi.py reads the photos the same way.
    python src/train_multi.py --root data/blender_gen/dataset_v5_hdr --out runs/v5_hdr --photos flash+random --hdr-input
    Compare with the same command without --hdr-input (same folder, PNG photos).

FIRST: the overfit test (should drive the loss towards 0)
    python src/train_multi.py --root data/blender_gen/dataset_v5 \
        --out runs/v5_overfit --overfit 4 --epochs 200 --schedule constant
(constant: the default cosine schedule shrinks the learning rate too fast to
judge an overfit test)
THEN the main run and its single-flash baseline (identical except --photos):
    python src/train_multi.py --root data/blender_gen/dataset_v5 \
        --out runs/v5_multi --epochs 150 --schedule cosine
    python src/train_multi.py --root data/blender_gen/dataset_v5 \
        --out runs/v5_flash --epochs 150 --schedule cosine --photos flash

MEMORY: a step pushes batch-size x k photos through the network - with the
defaults (4 samples, up to 6 photos) up to 24 images, about 3x v1's batch of
8. If the GPU runs out of memory: --batch-size 2, or a lower --max-photos
(validation and eval_multi.py then stop at that many). --amp (16-bit maths) is
not recommended: Run 12's first attempt with it turned to NaN in epoch 2,
and on an RTX 4070 it was only ~10% faster.

If the validation loss becomes NaN or infinite, training stops with a message
instead of running out its remaining epochs on a broken model.

Outputs under --out: config.json, best.pt, last.pt, log.csv, preview_XXX.png.
best.pt is chosen on the validation loss with ALL photos.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("data", "models"):
    p = os.path.join(HERE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset_multi import MultiLightDataset, MAP_CHANNELS  # noqa: E402
from model import MultiLightPaintNet, count_params           # noqa: E402


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_photos(photos: torch.Tensor, policy: str, max_photos: int,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """photos (B, K, 3, H, W) -> (B, k, 3, H, W) according to the policy.
    The same k for the whole batch; WHICH photos differs per sample."""
    b, k_all = photos.shape[:2]
    if policy == "flash" or k_all == 1:
        return photos[:, :1]
    if policy == "all":
        return photos[:, :max_photos]
    k = int(torch.randint(1, min(max_photos, k_all) + 1, (1,), generator=generator))
    if policy == "flash+random":
        # photos[:, 0] is the flash photo (dataset_multi.py puts it first).
        sides = torch.argsort(torch.rand(b, k_all - 1, generator=generator),
                              dim=1)[:, :k - 1] + 1
        order = torch.cat([torch.zeros(b, 1, dtype=torch.long), sides], dim=1)
    else:
        order = torch.argsort(torch.rand(b, k_all, generator=generator), dim=1)[:, :k]
    idx = order.to(photos.device)[:, :, None, None, None].expand(
        -1, -1, *photos.shape[2:])
    return photos.gather(1, idx)


class Criterion(nn.Module):
    def __init__(self, map_weight=1.0, scalar_weight=1.0, pigment_weight=0.2):
        super().__init__()
        self.w = {"map": map_weight, "scalar": scalar_weight, "pigment": pigment_weight}

    def forward(self, pred, batch):
        parts = {"map": F.l1_loss(pred["maps"], batch["maps"])}
        err = (pred["scalars"] - batch["scalars"]).abs() * batch["scalar_mask"]
        parts["scalar"] = err.sum() / batch["scalar_mask"].sum().clamp(min=1)
        known = batch["pigment"] >= 0
        parts["pigment"] = (F.cross_entropy(pred["pigment"][known], batch["pigment"][known])
                            if known.any() else pred["maps"].new_zeros(()))
        total = sum(self.w[k] * v for k, v in parts.items())
        return total, parts


@torch.no_grad()
def save_preview(photos, pred_maps, target_maps, path):
    """Row 1: the photos the network saw. Row 2: truth. Row 3: prediction.
    Map columns: basecolor | roughness | metallic | normal."""
    def img(t):
        a = t.detach().float().clamp(0, 1).cpu().numpy()
        if a.shape[0] == 1:
            a = np.repeat(a, 3, axis=0)
        return (a.transpose(1, 2, 0) * 255).round().astype(np.uint8)

    def maps_row(m):
        return np.concatenate([img(m[0:3]), img(m[3:4]), img(m[4:5]), img(m[5:8])], 1)

    truth, pred = maps_row(target_maps[0]), maps_row(pred_maps[0])
    w = truth.shape[1]
    ph = np.concatenate([img(p) for p in photos[0]], 1)
    if ph.shape[1] < w:
        ph = np.pad(ph, ((0, 0), (0, w - ph.shape[1]), (0, 0)), constant_values=255)
    Image.fromarray(np.concatenate([ph[:, :w], truth, pred], 0)).save(path)


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def run_epoch(model, loader, device, criterion, policy, max_photos,
              optimizer=None, amp=False, gen=None):
    training = optimizer is not None
    model.train(training)
    n_s = None
    totals = {"loss": 0.0, "map": 0.0, "scalar": 0.0, "pigment": 0.0}
    chan = np.zeros(len(MAP_CHANNELS))
    scal_err, scal_cnt = None, None
    correct = known = 0
    n_batches = 0
    last = None
    def autocast():
        return (torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp and device.type == "cuda" else contextlib.nullcontext())
    scaler = getattr(run_epoch, "scaler", None)

    for batch in loader:
        batch = to_device(batch, device)
        photos = choose_photos(batch["photos"], policy, max_photos, gen)
        with torch.set_grad_enabled(training), autocast():
            pred = model(photos)
            pred = {k: v.float() for k, v in pred.items()}
            loss, parts = criterion(pred, batch)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        totals["loss"] += loss.item()
        for k in parts:
            totals[k] += parts[k].item()
        chan += (pred["maps"] - batch["maps"]).abs().mean(dim=(0, 2, 3)).detach().cpu().numpy()
        e = ((pred["scalars"] - batch["scalars"]).abs() * batch["scalar_mask"]).detach()
        if scal_err is None:
            n_s = e.shape[1]
            scal_err, scal_cnt = np.zeros(n_s), np.zeros(n_s)
        scal_err += e.sum(0).cpu().numpy()
        scal_cnt += batch["scalar_mask"].sum(0).cpu().numpy()
        k_mask = batch["pigment"] >= 0
        correct += (pred["pigment"].argmax(1)[k_mask] == batch["pigment"][k_mask]).sum().item()
        known += k_mask.sum().item()
        n_batches += 1
        last = (photos, pred["maps"], batch["maps"])

    n = max(n_batches, 1)
    out = {k: v / n for k, v in totals.items()}
    out["chan"] = chan / n
    # nan where no sample in this split uses the parameter (e.g. film IOR when
    # no paint has a film) - "no data" must not print as "zero error".
    out["scalar_err"] = np.where(scal_cnt > 0, scal_err / np.maximum(scal_cnt, 1), np.nan)
    out["pigment_acc"] = correct / known if known else float("nan")
    return out, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/blender_gen/dataset_v5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--schedule", choices=["constant", "cosine"], default="cosine")
    ap.add_argument("--photos", choices=["random", "flash", "flash+random", "all"],
                    default="random")
    ap.add_argument("--coords", action="store_true",
                    help="pixel coordinates as two extra input channels")
    ap.add_argument("--flash-slot", action="store_true",
                    help="give the flash photo its own slot in the pooling")
    ap.add_argument("--hdr-input", action="store_true",
                    help="photos from the folder's .exr files (rendered with "
                         "--hdr), log-encoded; needs pip install OpenEXR")
    ap.add_argument("--max-photos", type=int, default=None,
                    help="most photos per sample (default: all the dataset has)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision (CUDA only); diverged to NaN on this model once")
    ap.add_argument("--init-from", default=None,
                    help="v1 checkpoint (e.g. runs/v4_long/best.pt) to start the trunk from")
    ap.add_argument("--preview-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-weight", type=float, default=1.0)
    ap.add_argument("--scalar-weight", type=float, default=1.0)
    ap.add_argument("--pigment-weight", type=float, default=0.2)
    args = ap.parse_args()

    if args.flash_slot and args.photos == "random":
        sys.exit("--flash-slot needs the flash photo first in every sample; "
                 "--photos random can leave it out. Use --photos flash+random.")
    if args.out is None:
        args.out = os.path.join("runs", time.strftime("run_%Y%m%d_%H%M%S"))
    if os.path.exists(os.path.join(args.out, "log.csv")) and not args.overwrite:
        sys.exit(f"{args.out} already holds a run (log.csv exists). Pick a new "
                 f"--out, or pass --overwrite if you really mean to replace it.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    print(f"run folder: {args.out} | device: {device}")

    from torch.utils.data import DataLoader
    if args.overfit:
        train_ds = MultiLightDataset(args.root, split="all", limit=args.overfit,
                                     hdr=args.hdr_input)
        val_ds = train_ds
        bs = min(args.batch_size, len(train_ds))
        print(f"OVERFIT MODE: {len(train_ds)} samples, expecting the loss to approach 0")
    else:
        train_ds = MultiLightDataset(args.root, split="train", limit=args.limit,
                                     hdr=args.hdr_input)
        val_ds = MultiLightDataset(args.root, split="val", limit=args.limit,
                                   hdr=args.hdr_input)
        bs = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=args.workers,
                              drop_last=len(train_ds) > bs)
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=args.workers)
    keys = train_ds.scalar_keys
    n_photos = len(train_ds.photo_names)
    args.max_photos = min(args.max_photos or n_photos, n_photos)
    print(f"train {len(train_ds)} / val {len(val_ds)} samples, "
          f"{len(train_ds.photo_names)} photos each, {len(keys)} layer parameters"
          + (" | HDR photos (.exr, log-encoded)" if args.hdr_input else ""))

    meta_path = os.path.join(args.root, "meta.json")
    model_cfg = dict(n_scalars=len(keys), n_pigments=max(len(train_ds.pigments), 1),
                     prenorm_global=True, coords=args.coords,
                     flash_slot=args.flash_slot)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "model": "MultiLightPaintNet",
                   "model_cfg": model_cfg, "scalar_keys": keys,
                   "scalar_ranges": train_ds.scalar_ranges,
                   "pigments": train_ds.pigments, "photos": train_ds.photo_names,
                   "val_indices": list(val_ds.indices),
                   "dataset_meta": (json.load(open(meta_path))
                                    if os.path.exists(meta_path) else None)},
                  f, indent=2)

    model = MultiLightPaintNet(**model_cfg).to(device)
    if args.init_from:
        n = model.load_trunk_from(args.init_from, map_location=device)
        print(f"trunk initialised from {args.init_from} ({n} tensors)")
    criterion = Criterion(args.map_weight, args.scalar_weight, args.pigment_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
        if args.schedule == "cosine" else None)
    run_epoch.scaler = (torch.amp.GradScaler("cuda")
                        if args.amp and device.type == "cuda" else None)

    spread = {"random": f" (1-{args.max_photos})",
              "flash+random": f" (flash + 0-{args.max_photos - 1} side)"}
    print(f"parameters: {count_params(model):,} | lr {args.lr} ({args.schedule}) | "
          f"photos: {args.photos}{spread.get(args.photos, '')}")

    log_path = os.path.join(args.out, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss_all", "val_loss_flash", "val_map",
             "val_scalar", "val_pigment", "val_pigment_acc", "val_pigment_acc_flash",
             "seconds", "lr"]
            + [f"val_{c}" for c in MAP_CHANNELS] + [f"val_{k}" for k in keys]
            + [f"valflash_{k}" for k in keys])

    best, best_epoch = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr, _ = run_epoch(model, train_loader, device, criterion, args.photos,
                          args.max_photos, optimizer, args.amp, gen)
        # Validate twice: with every photo, and with the flash photo alone.
        val_policy = "flash" if args.photos == "flash" else "all"
        va, last = run_epoch(model, val_loader, device, criterion, val_policy,
                             args.max_photos, amp=args.amp)
        vf, _ = run_epoch(model, val_loader, device, criterion, "flash",
                          args.max_photos, amp=args.amp)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{tr['loss']:.6f}", f"{va['loss']:.6f}", f"{vf['loss']:.6f}",
                 f"{va['map']:.6f}", f"{va['scalar']:.6f}", f"{va['pigment']:.6f}",
                 f"{va['pigment_acc']:.4f}", f"{vf['pigment_acc']:.4f}",
                 f"{dt:.1f}", f"{lr_now:.2e}"]
                + [f"{c:.6f}" for c in va["chan"]]
                + [f"{s:.6f}" for s in va["scalar_err"]]
                + [f"{s:.6f}" for s in vf["scalar_err"]])

        # A NaN/inf validation loss means the weights themselves are broken (a
        # single bad TRAINING batch under --amp is normal: GradScaler skips
        # that step). Nothing recovers from it, so stop now, not 150 epochs on.
        if not (np.isfinite(va["loss"]) and np.isfinite(vf["loss"])):
            print(f"epoch {epoch:3d}/{args.epochs}  train {tr['loss']:.4f}  "
                  f"val all {va['loss']:.4f} flash {vf['loss']:.4f}", flush=True)
            sys.exit(f"\nSTOPPED: the validation loss became NaN/inf in epoch {epoch}; "
                     f"training cannot recover from that. "
                     + (f"best.pt (epoch {best_epoch}) is from before it. "
                        if best_epoch else "No usable checkpoint was saved. ")
                     + ("This run used --amp, which has diverged on this model "
                        "before: rerun without it (add --overwrite to reuse --out)."
                        if args.amp else "Try a lower --lr, then report it."))

        flag = ""
        if va["loss"] < best:
            best, best_epoch = va["loss"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": va["loss"]}, os.path.join(args.out, "best.pt"))
            flag = "  <- best"
        torch.save({"model": model.state_dict(), "epoch": epoch},
                   os.path.join(args.out, "last.pt"))
        print(f"epoch {epoch:3d}/{args.epochs}  train {tr['loss']:.4f}  "
              f"val all {va['loss']:.4f} flash {vf['loss']:.4f}  "
              f"(map {va['map']:.4f} scalar {va['scalar']:.4f} "
              f"pigment acc {va['pigment_acc']:.2f})  {dt:.1f}s{flag}", flush=True)

        if epoch % args.preview_every == 0 or epoch == args.epochs:
            save_preview(*last, os.path.join(args.out, f"preview_{epoch:03d}.png"))

    print(f"\nbest validation loss (all photos): {best:.4f}")
    print("layer parameters, mean error in real units at the FINAL epoch "
          "(all photos | flash only):")
    for i, k in enumerate(keys):
        lo, hi = train_ds.scalar_ranges[k]
        a, f = va["scalar_err"][i], vf["scalar_err"][i]
        show = lambda e: f"{'n/a':>9}" if np.isnan(e) else f"{e * (hi - lo):9.4f}"
        print(f"  {k:<16} {show(a)} | {show(f)}   (range {lo} to {hi})")
    print("run eval_multi.py on best.pt for skill scores per photo set and paint type.")
    print(f"outputs in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
