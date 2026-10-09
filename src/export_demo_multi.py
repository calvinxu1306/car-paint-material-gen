"""
export_demo_multi.py — turn a v2 run's predictions into assets for demo/v2.html.

The v2 counterpart of export_demo.py. It runs a MultiLightPaintNet trained by
train_multi.py on a handful of samples it never trained on and writes, for
each one, into --out/<sample id>/:

    photo.png, side1.png ...  every photo the dataset has for the sample, flash
                              first, for display (the network may have been
                              shown fewer, see below)
    pred_basecolor.png        predicted maps as web textures, written by the
    pred_orm.png              same functions as v1's export_demo.py: sRGB base
    pred_normal.png           colour, roughness (G) + metallic (B) packed in
    pred_rough.png            one ORM image, flake normals, greyscale roughness
    gt_*.png                  the same four from the ground truth

plus --out/manifest.json with, per sample: the paint type (true, predicted,
and the network's probability for every type), the finish (gloss/matte), all
layer parameters in real units (predicted and true), which of them mean
something for this paint, and the mean map values (as v1's "base" block).

WHICH SAMPLES
By default the run's own validation split: exactly the samples train_multi.py
validated on (config.json's val_indices), the same ones eval_multi.py scores
without --test-root. They were never trained on, but did pick best.pt. With
--test-root, a held-out test folder instead. The picks are spread over the
paint types as evenly as the data allows (12 samples of 5 types -> 3, 3, 2, 2,
2) and over each type's samples, so a run always exports the same samples.

The page tells viewers that none of the samples was used for training only
when the script could check it ("held_out" in the manifest):
  - with --test-root, it warns (as eval_multi.py does) when test indices also
    exist in the training folder (same index = same seed = same paint), e.g.
    when --test-root is the training folder itself;
  - an --overfit run validates on its own training samples, so those are
    exported, labelled as training samples.

WHICH PHOTOS THE NETWORK SEES
The ones the run trained to see: the flash photo alone for a --photos flash
run; otherwise the flash photo first, then side photos up to the run's
--max-photos (eval_multi.py's "all N" set). The page shows every photo and
marks the ones the network was not given.

MEANINGLESS PARAMETERS
Some parameters mean nothing for some paints (flake size of a solid paint,
film IOR without a film; see dataset_multi.py). "mask" in the manifest flags
them (false = meaningless), their true value is written as null, and the page
shows "n/a". The predicted value is kept: it is what the network said, and the
page renders it (e.g. a film the network imagined on a paint that has none).

RUN (from the repo root, after training):
    python src/export_demo_multi.py --run runs/v5_multi
    python src/export_demo_multi.py --run runs/v5_multi --test-root data/blender_gen/dataset_v5_test
then view it (the page needs a web server):
    cd demo
    python -m http.server 8000
and open http://localhost:8000/v2.html
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "data"), os.path.join(HERE, "models")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset_multi import MultiLightDataset, denormalise              # noqa: E402
from export_demo import maps_to_textures, pick_device, save_rgb      # noqa: E402
from model import MultiLightPaintNet                                  # noqa: E402

EXPORTER = "export_demo_multi.py"   # recorded in the manifest; see check_out


def rel(path: str) -> str:
    """Path relative to the working directory, with forward slashes."""
    try:
        path = os.path.relpath(path)
    except ValueError:                  # another drive on Windows
        path = os.path.abspath(path)
    return path.replace("\\", "/")


def check_out(out: str) -> None:
    """--out gets deleted and rewritten, so make sure this script wrote it. A
    mistyped --out (say "demo", or v1's demo/assets/samples, which has a
    manifest.json of its own) must not wipe the site."""
    if os.path.exists(out) and not os.path.isdir(out):
        sys.exit(f"--out {out} is a file, not a folder.")
    if not (os.path.isdir(out) and os.listdir(out)):
        return
    path = os.path.join(out, "manifest.json")
    if not os.path.exists(path):
        sys.exit(f"{out} is not empty and has no manifest.json, so it wasn't "
                 f"written by {EXPORTER}. Refusing to delete it.")
    try:
        with open(path) as f:
            writer = json.load(f).get("exporter")
    except (OSError, ValueError, AttributeError):
        writer = None
    if writer != EXPORTER:
        sys.exit(f"{path} was not written by {EXPORTER} (v1's export_demo.py "
                 f"writes demo/assets/samples). Refusing to delete {out}.")


def indices_in(root: str) -> set[int] | None:
    """The sample indices in a data folder, or None if it can't be read."""
    try:
        return set(MultiLightDataset(root, split="all").indices)
    except (OSError, RuntimeError, ValueError) as err:
        print(f"NOTE: can't read the training data {root} ({err}), so can't "
              f"check that the test set is held out (pass --root to check).")
        return None


def samples_to_export(cfg: dict, root: str, test_root: str | None):
    """The dataset to export from, its folder, a description for the page, and
    the indices in it that the run may have learned from (None: can't tell).
    Without --test-root: the validation split, as eval_multi.py builds it."""
    if test_root:
        ds = MultiLightDataset(test_root, split="all")
        folder, where = test_root, "held-out test set"
        # As eval_multi.py: the same index in the training folder is the same
        # seed, so the same paint. Catches a --test-root that IS the training
        # folder, too.
        trained = indices_in(root)
        learned = None if trained is None else trained & set(ds.indices)
        if learned:
            print(f"WARNING: {len(learned)} test indices also exist in {root} "
                  f"(same seed = same paint). The test set is not fully held out.")
            where = "test set (overlaps the training data)"
        elif learned is None:
            where = "test set (not checked against the training data)"
    else:
        overfit = cfg["args"].get("overfit")
        ds = MultiLightDataset(root, split="val")
        # The samples train_multi.py actually validated on, even if the folder
        # has grown since (the split is "last 10%", which would move).
        saved = cfg.get("val_indices")
        if saved is not None and saved != ds.indices:
            have = set(MultiLightDataset(root, split="all").indices)
            ds.indices = [i for i in saved if i in have]
            if not overfit:     # those differ from the split by design, below
                print(f"NOTE: {root} changed since training; exporting from the "
                      f"{len(ds.indices)} samples the run actually validated on.")
        if overfit:
            # train_multi.py --overfit validates on the samples it trains on.
            print(f"WARNING: this is an --overfit run, which validates on its "
                  f"{len(ds.indices)} training samples; exporting those, labelled "
                  f"as training samples. Use --test-root for unseen samples.")
            folder, where = root, "training samples (an --overfit run validates on them)"
            learned = set(ds.indices)
        else:
            folder, where = root, "validation split (never trained on; it did pick best.pt)"
            learned = set()
    if not ds.indices:
        sys.exit(f"no samples to export in {folder}")

    # Must match the run, or its outputs would be misread (as in eval_multi.py).
    ranges = {k: list(v) for k, v in ds.scalar_ranges.items()}
    for name, have, want in (("scalar_keys", ds.scalar_keys, cfg["scalar_keys"]),
                             ("scalar_ranges", ranges, cfg.get("scalar_ranges", ranges)),
                             ("pigments", ds.pigments, cfg.get("pigments", ds.pigments)),
                             ("photos", ds.photo_names, cfg.get("photos", ds.photo_names))):
        if have != want:
            sys.exit(f"{folder} has {name} {have}, but the run was trained on "
                     f"{want}; the export would misread the model's outputs.")
    if ds.photo_names[0] != "photo":
        sys.exit(f"no flash photo ('photo') among {ds.photo_names} in {folder}")
    return ds, folder, where, learned


def pick(ds: MultiLightDataset, pigments: list[str], n: int) -> list[int]:
    """Positions in ds to export: spread over the paint types as evenly as the
    data allows (types in the run's order, then any unlabelled samples), and
    over each type's samples. Deterministic: no randomness involved."""
    groups: dict = {}
    for pos, index in enumerate(ds.indices):
        name = ds.params(index).get("pigment")
        groups.setdefault(name if name in pigments else None, []).append(pos)
    order = [p for p in pigments if p in groups] + ([None] if None in groups else [])
    # Deal the n slots out one per type in turn, skipping types that run out.
    take = dict.fromkeys(order, 0)
    left = min(n, len(ds))
    while left:
        for key in order:
            if left and take[key] < len(groups[key]):
                take[key] += 1
                left -= 1
    picks = []
    for key in order:
        g = groups[key]
        if take[key]:
            picks += [g[int(j)] for j in np.linspace(0, len(g) - 1, take[key]).round()]
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder from train_multi.py")
    ap.add_argument("--ckpt", default="best.pt", help="checkpoint inside --run")
    ap.add_argument("--root", default=None, help="training data (default: from config)")
    ap.add_argument("--test-root", default=None,
                    help="held-out test folder (default: the run's validation split)")
    ap.add_argument("--n", type=int, default=12, help="how many samples to export")
    ap.add_argument("--out", default="demo/assets/samples_v2")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.n < 1:
        sys.exit("--n must be at least 1")

    # Before anything slow: a refusal should not wait for the model to load.
    check_out(args.out)

    with open(os.path.join(args.run, "config.json")) as f:
        cfg = json.load(f)
    root = args.root or cfg["args"]["root"]
    ds, folder, where, learned = samples_to_export(cfg, root, args.test_root)
    keys, pigments = cfg["scalar_keys"], cfg.get("pigments") or []
    ranges = {k: tuple(v) for k, v in cfg["scalar_ranges"].items()}

    policy = cfg["args"].get("photos", "random")
    # config.json records the dataset's photo count as max_photos even for a
    # flash-only run, which never saw more than the one photo.
    max_photos = (1 if policy == "flash"
                  else cfg["args"].get("max_photos") or len(ds.photo_names))
    seen = ds.photo_names[:max_photos]     # flash first (dataset_multi.py)

    device = pick_device(args.device)
    # Runs from before prenorm_global existed were trained without it; coords
    # and flash_slot default to off (as in eval_multi.py).
    model = MultiLightPaintNet(**{"prenorm_global": False, **cfg["model_cfg"]}).to(device)
    ckpt_path = os.path.join(args.run, args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')}), trained with "
          f"photos={policy}; showing it {seen}")

    picks = pick(ds, pigments, args.n)
    # Whether the page may say that none of these samples was used for
    # training: null when that couldn't be checked.
    picked = {ds.indices[pos] for pos in picks}
    held_out = None if learned is None else not (learned & picked)

    # Stale samples from an older model mislead, so the folder is replaced
    # (checked above: only one this script wrote).
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)
    # A placeholder first: if the export stops halfway, the next run may still
    # replace the folder, and the page shows "no samples" instead of breaking.
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"exporter": EXPORTER, "incomplete": True, "samples": []}, f)

    samples = []
    with torch.no_grad():
        for k, pos in enumerate(picks):
            s = ds[pos]
            photos = s["photos"]                               # (K, 3, H, W)
            out = model(photos[None, :len(seen)].to(device))
            if not all(torch.isfinite(v).all() for v in out.values()):
                sys.exit(f"the model output NaN/inf for sample {s['index']}: "
                         f"{ckpt_path} is broken. Export a checkpoint from before it.")
            pred_maps = out["maps"][0].float().cpu().numpy()
            gt_maps = s["maps"].numpy()

            tag = f"{s['index']:06d}"
            folder_out = os.path.join(args.out, tag)
            os.makedirs(folder_out)
            for name, img in zip(ds.photo_names, photos):
                save_rgb(img.numpy().transpose(1, 2, 0),
                         os.path.join(folder_out, f"{name}.png"))
            for prefix, maps in (("pred", pred_maps), ("gt", gt_maps)):
                for name, img in maps_to_textures(maps).items():
                    save_rgb(img, os.path.join(folder_out, f"{prefix}_{name}.png"))

            p = ds.params(s["index"])
            mask = {key: bool(m) for key, m in zip(keys, s["scalar_mask"].tolist())}
            if pigments and "pigment" in out:
                probs = torch.softmax(out["pigment"][0].float(), dim=0).cpu().tolist()
                pigment_probs = dict(zip(pigments, probs))
                pigment_pred = pigments[int(np.argmax(probs))]
            else:                                  # v3/v4 data: no types
                pigment_probs, pigment_pred = {}, None

            samples.append({
                "id": tag,
                "folder": tag,
                "photos": [{"name": name, "file": f"{name}.png", "seen": name in seen}
                           for name in ds.photo_names],
                "pigment_gt": pigments[s["pigment"]] if s["pigment"] >= 0 else None,
                "pigment_pred": pigment_pred,
                "pigment_probs": pigment_probs,
                "finish": p.get("finish"),
                "pred": denormalise(out["scalars"][0], keys, ranges),
                "gt": {key: float(p[key]) if mask[key] else None for key in keys},
                "mask": mask,   # false = meaningless for this paint ("n/a")
                "base": {  # mean map values, as in v1's manifest
                    "pred_roughness": float(pred_maps[3].mean()),
                    "gt_roughness": float(gt_maps[3].mean()),
                    "pred_metallic": float(pred_maps[4].mean()),
                    "gt_metallic": float(gt_maps[4].mean()),
                    "pred_basecolor": pred_maps[0:3].mean(axis=(1, 2)).tolist(),
                    "gt_basecolor": gt_maps[0:3].mean(axis=(1, 2)).tolist(),
                },
            })
            print(f"[{k + 1}/{len(picks)}] exported sample {tag} "
                  f"({samples[-1]['pigment_gt'] or '?'}, {p.get('finish', '?')})")

    manifest = {
        "exporter": EXPORTER,
        "run": rel(args.run),
        "checkpoint": rel(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "dataset": rel(folder),
        "split": where,
        "held_out": held_out,
        "photo_policy": policy,
        "photos": seen,                    # what the network was shown, in order
        "scalar_keys": keys,
        "scalar_ranges": {key: list(v) for key, v in ranges.items()},
        "pigments": pigments,
        "samples": samples,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {len(samples)} samples + manifest.json to {os.path.abspath(args.out)}")
    print("view: cd demo; python -m http.server 8000; open http://localhost:8000/v2.html")


if __name__ == "__main__":
    main()
