"""
eval_multi.py — what does each extra photo buy, for each kind of paint?

The v2 question. One trained MultiLightPaintNet is scored on the same samples
several times, each time shown a different set of photos:

    flash            the phone's flash only (what v1 had)
    side1            one side-lit photo only
    flash+side1      flash and one side light
    flash+3 sides    flash and three side lights
    all N            every photo, up to the number the run trained with
                     (--max-photos); never more, since the network has never
                     seen more and max-pooled features grow with the count

For every set it reports SKILL (as eval_baseline.py: how much of the
do-nothing baseline's error the model removes; 0% = learned nothing) for the
maps, for each layer parameter, and the pigment-type accuracy.

WHAT PHYSICS PREDICTS (write these down before looking at the numbers)
    film_thickness  partial with the flash alone (the "face" colour hints at it),
                    clearly better once a side light is added
    film_ior        ~0 with the flash alone: it only shows in how the colour
                    CHANGES with angle, and the flash shows one angle
    coat_weight     better with side lights (the coat's own reflection moves
                    away from the base's)
    pigment         pearl/colorshift vs metallic confusions drop with side lights

SCORES "SKILL" HOW
    Every layer parameter is scored only on samples where it means something
    (the mask in dataset_multi.py). The baseline is the training-set average
    over those same samples.
    The per-pigment table compares against a STRICTER baseline: the average for
    that pigment type. So it measures what the network gets right beyond
    knowing the paint type. "n/a" = the parameter is constant for that type
    (e.g. film thickness of a solid paint is always 0).

RUN
    python src/eval_multi.py --run runs/v5_multi
    python src/eval_multi.py --run runs/v5_multi --test-root data/blender_gen/dataset_v5_test
Without --test-root it scores the validation split, which also chose best.pt
(slightly optimistic). Use the held-out test set for numbers you report.
--json results.json saves everything for the write-up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("data", "models"):
    p = os.path.join(HERE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset_multi import MultiLightDataset, MAP_CHANNELS, read_meta  # noqa: E402
from model import MultiLightPaintNet                         # noqa: E402

SHORT = {"coat_weight": "coat wt", "coat_roughness": "coat rgh",
         "coat_tint_r": "tint R", "coat_tint_g": "tint G", "coat_tint_b": "tint B",
         "flake_scale": "flk size", "flake_strength": "flk str",
         "peel_strength": "peel", "film_thickness": "film d", "film_ior": "film n"}


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def photo_sets(names: list[str], max_photos: int) -> dict[str, list[int]]:
    """Named photo subsets, as indices into the dataset's photo list. No set is
    larger than max_photos: the network never trained on more photos than
    that, and max-pooled features keep growing with the photo count."""
    if "photo" not in names:
        raise SystemExit(f"no flash photo ('photo') among {names}")
    flash = names.index("photo")
    sides = [i for i, n in enumerate(names) if n != "photo"]
    sets = {"flash": [flash]}
    if sides:
        sets["side1"] = [sides[0]]
    if sides and max_photos >= 2:
        sets["flash+side1"] = [flash, sides[0]]
    if len(sides) >= 3 and max_photos >= 4:
        sets["flash+3 sides"] = [flash] + sides[:3]
    n_all = min(max_photos, len(names))
    if n_all > 4:
        sets[f"all {n_all}"] = [flash] + sides[:n_all - 1]
    return sets


@torch.no_grad()
def training_means(ds, n_pig):
    """Masked means of every target over the training split: overall and per
    pigment type. Read from params files and maps directly."""
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=16)
    n_s = len(ds.scalar_keys)
    m_sum, m_n = torch.zeros(len(MAP_CHANNELS)), 0
    s_sum, s_cnt = torch.zeros(n_s), torch.zeros(n_s)
    ps_sum, ps_cnt = torch.zeros(n_pig, n_s), torch.zeros(n_pig, n_s)
    for b in loader:
        m_sum += b["maps"].mean(dim=(2, 3)).sum(0)
        m_n += b["maps"].shape[0]
        mask = b["scalar_mask"]
        s_sum += (b["scalars"] * mask).sum(0)
        s_cnt += mask.sum(0)
        for t in range(n_pig):
            sel = b["pigment"] == t
            if sel.any():
                ps_sum[t] += (b["scalars"][sel] * mask[sel]).sum(0)
                ps_cnt[t] += mask[sel].sum(0)
    return (m_sum / max(m_n, 1), s_sum / s_cnt.clamp(min=1),
            ps_sum / ps_cnt.clamp(min=1))


@torch.no_grad()
def score_all(model, ds, sets, device, map_mean, s_mean, ps_mean, n_pig, bs=4):
    """Score every photo set in one pass over the data (each sample's PNGs
    are read once, then the model runs once per set)."""
    from torch.utils.data import DataLoader
    n_s = len(ds.scalar_keys)

    def fresh():
        a = {k: torch.zeros(len(MAP_CHANNELS)) for k in ("map_model", "map_base")}
        for k in ("s_model", "s_base", "s_cnt"):
            a[k] = torch.zeros(n_s)
        for k in ("ps_model", "ps_base", "ps_cnt"):
            a[k] = torch.zeros(n_pig, n_s)
        a["conf"] = torch.zeros(n_pig, n_pig, dtype=torch.long)
        return a

    accs = {name: fresh() for name in sets}
    n = 0
    for b in DataLoader(ds, batch_size=bs):
        t_maps, t_s, mask, pig = b["maps"], b["scalars"], b["scalar_mask"], b["pigment"]
        base_maps = (map_mean[None, :, None, None] - t_maps).abs().mean(dim=(2, 3)).sum(0)
        base_s = ((s_mean[None] - t_s).abs() * mask).sum(0)
        for name, idx in sets.items():
            acc = accs[name]
            pred = {k: v.float().cpu() for k, v in model(b["photos"][:, idx].to(device)).items()}
            acc["map_model"] += (pred["maps"] - t_maps).abs().mean(dim=(2, 3)).sum(0)
            acc["map_base"] += base_maps
            e_model = (pred["scalars"] - t_s).abs() * mask
            acc["s_model"] += e_model.sum(0)
            acc["s_base"] += base_s
            acc["s_cnt"] += mask.sum(0)
            for t in range(n_pig):
                sel = pig == t
                if sel.any():
                    acc["ps_model"][t] += e_model[sel].sum(0)
                    acc["ps_base"][t] += ((ps_mean[t][None] - t_s[sel]).abs()
                                          * mask[sel]).sum(0)
                    acc["ps_cnt"][t] += mask[sel].sum(0)
            if "pigment" in pred:
                known = pig >= 0
                for tr, pr in zip(pig[known], pred["pigment"].argmax(1)[known]):
                    acc["conf"][tr, pr] += 1
        n += t_maps.shape[0]
    for acc in accs.values():
        acc["map_model"] /= n
        acc["map_base"] /= n
    return accs


def skill(model_err, base_err):
    return 100.0 * (1.0 - model_err / base_err) if base_err > 1e-6 else float("nan")


def fmt(x, w=8):
    return f"{'n/a':>{w}}" if x != x else f"{x:>{w - 1}.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder from train_multi.py")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--root", default=None, help="training data (default: from config)")
    ap.add_argument("--test-root", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--json", default=None, help="save all results here")
    args = ap.parse_args()

    with open(os.path.join(args.run, "config.json")) as f:
        cfg = json.load(f)
    root = args.root or cfg["args"]["root"]
    device = pick_device(args.device)

    train_ds = MultiLightDataset(root, split="train")
    if args.test_root:
        eval_ds = MultiLightDataset(args.test_root, split="all")
        # Must match exactly, or the model's outputs would be misread.
        for attr in ("scalar_keys", "scalar_ranges", "pigments", "photo_names"):
            if getattr(eval_ds, attr) != getattr(train_ds, attr):
                raise SystemExit(f"{args.test_root} and {root} differ in {attr}; "
                                 f"scoring would misread the model's outputs.")
        # Should match (same generator settings), or the test set is from a
        # different distribution - allowed, but say so.
        m_train, m_test = read_meta(root), read_meta(args.test_root)
        differ = [k for k in sorted(set(m_train) | set(m_test))
                  if k != "generator" and m_train.get(k) != m_test.get(k)]
        if differ:
            print(f"NOTE: test and training folders were generated with different "
                  f"settings: {differ}. Fine if intended (e.g. a harder test set).")
        overlap = set(eval_ds.indices) & set(MultiLightDataset(root, split="all").indices)
        if overlap:
            print(f"WARNING: {len(overlap)} test indices also exist in {root} "
                  f"(same seed = same paint). The test set is not fully held out.")
        where = f"held-out test set {args.test_root}"
    else:
        eval_ds = MultiLightDataset(root, split="val")
        # Score exactly the samples train_multi.py validated on, even if the
        # folder has grown since (the split is "last 10%", which would move).
        saved = cfg.get("val_indices")
        if saved is not None and saved != eval_ds.indices:
            have = set(MultiLightDataset(root, split="all").indices)
            eval_ds.indices = [i for i in saved if i in have]
            print(f"NOTE: {root} changed since training; scoring the "
                  f"{len(eval_ds.indices)} samples the run actually validated on.")
        where = f"validation split of {root} (also used to pick best.pt)"

    keys, pigs = train_ds.scalar_keys, train_ds.pigments
    if keys != cfg["scalar_keys"]:
        raise SystemExit(f"the run was trained on {cfg['scalar_keys']}, "
                         f"the data has {keys}")
    if pigs != cfg.get("pigments", pigs):
        raise SystemExit(f"the run's pigment order {cfg['pigments']} differs from "
                         f"the data's {pigs}; the type head would be misread")
    max_photos = cfg["args"].get("max_photos") or len(eval_ds.photo_names)
    n_pig = max(len(pigs), 1)
    # Runs from before prenorm_global existed were trained without it.
    model = MultiLightPaintNet(**{"prenorm_global": False, **cfg["model_cfg"]}).to(device)
    ckpt = torch.load(os.path.join(args.run, args.ckpt), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.run}/{args.ckpt} (epoch {ckpt.get('epoch', '?')}), "
          f"trained with photos={cfg['args']['photos']}, up to {max_photos} per sample")
    print(f"scoring {len(eval_ds)} samples: {where}")
    print(f"baselines from {len(train_ds)} training samples\n")

    # Baselines only need targets, so load just one photo per training sample.
    map_mean, s_mean, ps_mean = training_means(
        MultiLightDataset(root, split="train", photos=["photo"]), n_pig)
    sets = photo_sets(eval_ds.photo_names, max_photos)
    results = {}
    accs = score_all(model, eval_ds, sets, device, map_mean, s_mean, ps_mean,
                     n_pig, args.batch_size)
    for name, idx in sets.items():
        acc, conf = accs[name], accs[name]["conf"]
        cnt = acc["s_cnt"].clamp(min=1)
        r = {
            "photos": [eval_ds.photo_names[i] for i in idx],
            "map_skill": {c: skill(acc["map_model"][i].item(), acc["map_base"][i].item())
                          for i, c in enumerate(MAP_CHANNELS)},
            "map_skill_overall": skill(acc["map_model"].mean().item(),
                                       acc["map_base"].mean().item()),
            "scalar_skill": {k: skill((acc["s_model"][i] / cnt[i]).item(),
                                      (acc["s_base"][i] / cnt[i]).item())
                             for i, k in enumerate(keys)},
            "scalar_err_real": {k: ((acc["s_model"][i] / cnt[i]).item()
                                    * (train_ds.scalar_ranges[k][1] - train_ds.scalar_ranges[k][0])
                                    if acc["s_cnt"][i] > 0 else float("nan"))
                                for i, k in enumerate(keys)},
            "pigment_acc": ((conf.diag().sum() / conf.sum()).item()
                            if conf.sum() > 0 else float("nan")),
            "confusion": conf.tolist(),
            "per_pigment_skill": {
                pigs[t] if pigs else "?": {
                    k: skill((acc["ps_model"][t, i] / acc["ps_cnt"][t, i].clamp(min=1)).item(),
                             (acc["ps_base"][t, i] / acc["ps_cnt"][t, i].clamp(min=1)).item())
                    if acc["ps_cnt"][t, i] > 0 else float("nan")
                    for i, k in enumerate(keys)}
                for t in range(n_pig)},
        }
        results[name] = r

    # 1. Headline: skill per photo set.
    print("SKILL BY WHICH PHOTOS THE NETWORK SEES")
    head = f"{'photos':<15}{'maps':>8}" + "".join(f"{SHORT.get(k, k)[:8]:>9}" for k in keys) \
        + f"{'pigment':>9}"
    print(head)
    print("-" * len(head))
    for name, r in results.items():
        print(f"{name:<15}{fmt(r['map_skill_overall'])}"
              + "".join(" " + fmt(r["scalar_skill"][k]) for k in keys)
              + (f"{'n/a':>9}" if r["pigment_acc"] != r["pigment_acc"]
                 else f"{100 * r['pigment_acc']:>8.0f}%"))
    conf0 = np.array(results["flash"]["confusion"])
    if conf0.sum():
        print(f"(pigment = type accuracy, not skill. Always guessing the commonest "
              f"type would score {100 * conf0.sum(1).max() / conf0.sum():.0f}%)")

    # 2. Real-unit errors, flash vs. most photos.
    best_set = list(results)[-1]
    print(f"\nLAYER PARAMETERS, mean error in real units: flash | {best_set}")
    for k in keys:
        lo, hi = train_ds.scalar_ranges[k]
        show = lambda e: f"{'n/a':>10}" if e != e else f"{e:>10.4f}"
        print(f"  {k:<16}{show(results['flash']['scalar_err_real'][k])} |"
              f"{show(results[best_set]['scalar_err_real'][k])}   (range {lo:g} to {hi:g})")

    # 3. Per-pigment skill beyond knowing the type.
    if pigs:
        for name in ("flash", best_set):
            print(f"\nPER PIGMENT ({name}): skill vs that pigment's own average")
            print(f"{'pigment':<12}" + "".join(f"{SHORT.get(k, k)[:8]:>9}" for k in keys))
            for t, pg in enumerate(pigs):
                row = results[name]["per_pigment_skill"][pg]
                print(f"{pg:<12}" + "".join(" " + fmt(row[k]) for k in keys))

        # 4. Confusions.
        for name in ("flash", best_set):
            print(f"\nPIGMENT CONFUSION ({name}): rows = true, columns = predicted")
            print(f"{'':<12}" + "".join(f"{p[:9]:>10}" for p in pigs))
            for t, pg in enumerate(pigs):
                print(f"{pg:<12}" + "".join(f"{v:>10d}" for v in results[name]["confusion"][t]))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"run": args.run, "ckpt": args.ckpt, "epoch": ckpt.get("epoch"),
                       "scored": where, "n": len(eval_ds), "results": results}, f, indent=2)
        print(f"\nsaved {args.json}")


if __name__ == "__main__":
    main()
