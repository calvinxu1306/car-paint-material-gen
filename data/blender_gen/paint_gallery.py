"""
paint_gallery.py — one of each paint type, under the flash and under side lights.

Two jobs:
1. A visual check of the v5 recipes before rendering thousands of samples:
   every row should look like the paint it claims to be.
2. The figure that motivates v2. The last two rows are a colour-shift paint and
   a plain metallic paint whose colour was TUNED so the two look the same under
   the flash. From the side they don't. A single flash photo cannot tell them
   apart; a side-lit photo can.

RUN (from data/blender_gen):
    blender -b -P paint_gallery.py -- --out gallery
Writes gallery/tiles/*.png, gallery/gallery.png (rows = paints, columns =
lights) and gallery/gallery.json (what each row and column is).
Takes a few minutes on a GPU. --res 128 --samples 16 for a quick look.
"""

import json
import math
import os
import random
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_dataset_v5 as g  # noqa: E402


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gallery")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


def side_light(elevation_deg, azimuth_deg=0.0, dist=3.2):
    """A side light whose centre irradiance matches the flash (as in v5)."""
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    return {"x": dist * math.cos(el) * math.cos(az),
            "y": dist * math.cos(el) * math.sin(az),
            "z": dist * math.sin(el),
            "energy": g.FLASH_ENERGY * (dist / g.CAMERA_HEIGHT) ** 2 / math.sin(el)}


LIGHTS = [
    ("flash", {"x": 0.0, "y": 0.0, "z": g.CAMERA_HEIGHT, "energy": g.FLASH_ENERGY}),
    ("side 60 deg", side_light(60)),
    ("side 40 deg", side_light(40)),
    ("side 25 deg", side_light(25)),
]

# Shared settings so rows differ only in what makes each paint type.
COMMON = dict(roughness=0.3, coat_weight=1.0, coat_roughness=0.12,
              flake_scale=110.0, flake_strength=0.25, peel_strength=0.04,
              coat_tint_r=1.0, coat_tint_g=1.0, coat_tint_b=1.0,
              film_thickness=0.0, film_ior=g.NO_FILM_IOR,
              color_var_amount=0.1, scratch_amount=0.05, dust_amount=0.04)

PAINTS = [
    ("solid (red, gloss)", dict(pigment="solid", metallic=0.0, flake_strength=0.0,
                                base_color=(0.55, 0.04, 0.04, 1.0))),
    ("metallic (blue)", dict(pigment="metallic", metallic=0.9,
                             base_color=(0.15, 0.3, 0.6, 1.0))),
    ("pearl (film 250 nm, n 2.3)", dict(pigment="pearl", metallic=0.7,
                                        base_color=(0.25, 0.25, 0.3, 1.0),
                                        film_thickness=250.0, film_ior=2.3)),
    ("colour-shift (film 400 nm, n 2.0)", dict(pigment="colorshift", metallic=0.95,
                                               base_color=(0.6, 0.6, 0.62, 1.0),
                                               film_thickness=400.0, film_ior=2.0)),
    ("candy (red coat over silver)", dict(pigment="candy", metallic=0.9,
                                          base_color=(0.8, 0.8, 0.8, 1.0),
                                          coat_tint_r=0.8, coat_tint_g=0.08,
                                          coat_tint_b=0.1)),
    ("matte metallic (rough coat)", dict(pigment="metallic", finish="matte",
                                         metallic=0.9, coat_roughness=0.55,
                                         base_color=(0.3, 0.3, 0.32, 1.0))),
]

# The "same under the flash" pair: a colour-shift paint, and a plain metallic
# paint whose base colour is solved for below.
METAMER_TARGET = ("colour-shift (film 300 nm, n 2.0)",
                  dict(pigment="colorshift", metallic=0.95,
                       base_color=(0.6, 0.6, 0.62, 1.0),
                       film_thickness=300.0, film_ior=2.0))
METAMER_MATCH_LABEL = "plain metallic, colour matched under the flash"


def make_paint(overrides):
    p = g.sample_paint_params(random.Random(7))   # supplies the noise seeds
    p.update(COMMON)
    p.update(overrides)
    return p


def read_png(path):
    """PNG -> (H, W, 3) array of the stored 0..1 values, top row first."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]
    bpy.data.images.remove(img)
    return arr


def mean_linear(path):
    a = read_png(path)
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return lin.reshape(-1, 3).mean(axis=0)


def render_row(plane, light_obj, p, tile_dir, row):
    g.apply_material(plane, g.build_paint_material(p, name=f"Gallery{row}"))
    paths = []
    for col, (_, light) in enumerate(LIGHTS):
        g.place_light(light_obj, light)
        path = os.path.join(tile_dir, f"row{row}_col{col}.png")
        g.render_to(path)
        paths.append(path)
    return paths


def match_under_flash(plane, light_obj, target_path, p, tile_dir, iters=8):
    """Tune a plain metallic paint's base colour until its FLASH photo has the
    same mean colour as the target's flash photo. Rendering is monotonic in
    base colour, so a proportional update converges. The AgX tone curve
    compresses changes (roughly a square root here), so the ratio is applied
    with an exponent of ~2 to converge in a few steps instead of many."""
    target = mean_linear(target_path)
    g.place_light(light_obj, LIGHTS[0][1])
    probe = os.path.join(tile_dir, "_probe.png")
    for it in range(iters):
        g.apply_material(plane, g.build_paint_material(p, name="GalleryProbe"))
        g.render_to(probe)
        got = mean_linear(probe)
        ratio = target / np.maximum(got, 1e-4)
        err = float(np.abs(got - target).max())
        print(f"[match] iter {it}: flash mean {got.round(4)} target "
              f"{target.round(4)} max diff {err:.4f}")
        if err < 0.003:
            break
        c = np.clip(np.array(p["base_color"][:3]) * ratio ** 1.8, 0.0, 1.0)
        p["base_color"] = (*c.tolist(), 1.0)
    os.remove(probe)
    return p, err


def assemble(rows_of_paths, out_path, gap=4):
    tiles = [[read_png(p) for p in row] for row in rows_of_paths]
    h, w, _ = tiles[0][0].shape
    n_r, n_c = len(tiles), len(tiles[0])
    sheet = np.ones((n_r * (h + gap) - gap, n_c * (w + gap) - gap, 3), np.float32)
    for r, row in enumerate(tiles):
        for c, t in enumerate(row):
            sheet[r * (h + gap): r * (h + gap) + h, c * (w + gap): c * (w + gap) + w] = t
    H, W, _ = sheet.shape
    img = bpy.data.images.new("gallery", W, H, alpha=False)
    rgba = np.concatenate([sheet[::-1], np.ones((H, W, 1), np.float32)], axis=2)
    img.pixels[:] = rgba.ravel()
    img.filepath_raw = out_path
    img.file_format = "PNG"
    img.save()


def main():
    args = parse_args()
    out = os.path.abspath(args.out)
    tile_dir = os.path.join(out, "tiles")
    os.makedirs(tile_dir, exist_ok=True)

    g.clear_scene()
    plane, light_obj = g.setup_scene()
    g.setup_render(args.res, args.cpu)
    scene = bpy.context.scene
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    g.set_view_transform("AgX")

    labels, rows = [], []
    for row, (label, over) in enumerate(PAINTS):
        print(f"[gallery] {label}")
        rows.append(render_row(plane, light_obj, make_paint(over), tile_dir, row))
        labels.append(label)

    row = len(rows)
    label, over = METAMER_TARGET
    print(f"[gallery] {label}")
    target_paths = render_row(plane, light_obj, make_paint(over), tile_dir, row)
    rows.append(target_paths)
    labels.append(label)

    match = make_paint(dict(pigment="metallic", metallic=0.95,
                            base_color=(0.5, 0.5, 0.5, 1.0)))
    match, err = match_under_flash(plane, light_obj, target_paths[0], match, tile_dir)
    print(f"[gallery] {METAMER_MATCH_LABEL}")
    rows.append(render_row(plane, light_obj, match, tile_dir, row + 1))
    labels.append(METAMER_MATCH_LABEL)

    # How different are the pair under each light? Mean colour, linear RGB.
    diffs = []
    for col, (name, _) in enumerate(LIGHTS):
        a = mean_linear(rows[-2][col])
        b = mean_linear(rows[-1][col])
        diffs.append({"light": name, "colourshift_mean": a.round(4).tolist(),
                      "matched_metallic_mean": b.round(4).tolist(),
                      "max_channel_diff": round(float(np.abs(a - b).max()), 4)})
        print(f"[pair] {name:<12} colour-shift {a.round(3)}  metallic {b.round(3)}"
              f"  max diff {np.abs(a - b).max():.3f}")

    sheet = os.path.join(out, "gallery.png")
    assemble(rows, sheet)
    with open(os.path.join(out, "gallery.json"), "w") as f:
        json.dump({"rows": labels, "columns": [n for n, _ in LIGHTS],
                   "tiles": rows, "pair_mean_colours": diffs,
                   "matched_metallic_base_color": list(match["base_color"]),
                   "resolution": args.res, "samples": args.samples}, f, indent=2)
    print(f"\n[gallery] wrote {sheet}")
    print("rows:    " + " | ".join(labels))
    print("columns: " + " | ".join(n for n, _ in LIGHTS))


if __name__ == "__main__":
    main()
