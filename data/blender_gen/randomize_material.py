"""
randomize_material.py — sample a random-but-plausible car paint.

Reuses build_car_paint_material() from build_material.py and feeds it
randomly sampled parameters. The RANGES are anchored to the real measured
car paints in Gunther et al. 2005 (Table 1): a palette spanning near-black
through silver/gray to saturated colors, a metallic-dominant base, and a
very smooth clearcoat.

Honest caveat: Gunther fit a Cook-Torrance model, which is NOT the same
shading model as Blender's Principled BSDF. So we use their measurements to
set plausible *ranges*, not to copy exact numbers. Anchoring the distribution
to real paints still beats guessing from nothing.

Run it inside Blender the same way as build_material.py. Change the seed (or
remove it) to get a different paint each time.
"""

import bpy
import colorsys
import random
import sys
import os

# --- Find build_material.py so we can import it ---
# Easiest setup: keep BOTH .py files in the same folder (data/blender_gen/)
# and OPEN this script from disk (don't paste it), so __file__ is defined.
try:
    here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback if the script was pasted: use the folder of the open .blend file.
    here = bpy.path.abspath("//")
if here and here not in sys.path:
    sys.path.append(here)

from build_material import build_car_paint_material, apply_material_to_active


def sample_paint_params(rng):
    """Return a dict of randomly sampled, plausible car-paint parameters."""
    # Sample color in HSV so we get realistic paints, not garish random RGB.
    hue = rng.uniform(0.0, 1.0)
    sat = rng.uniform(0.0, 0.9)    # ~0 -> silver/gray/white; high -> candy colors
    val = rng.uniform(0.02, 0.9)   # low -> near-black; high -> bright
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)

    return {
        "base_color": (r, g, b, 1.0),
        "metallic": rng.uniform(0.7, 1.0),         # car paint is metallic-dominant
        "roughness": rng.uniform(0.2, 0.5),        # base coat, moderately polished
        "coat_weight": rng.uniform(0.8, 1.0),      # clearcoat almost always present
        "coat_roughness": rng.uniform(0.01, 0.1),  # clearcoat is very smooth
        "flake_scale": rng.uniform(120.0, 300.0),  # higher = denser, finer flakes
        "flake_strength": rng.uniform(0.05, 0.25), # how strongly flakes tilt light
    }


def build_random_paint(seed=None):
    """Sample parameters and build the material. Returns (material, params)."""
    rng = random.Random(seed)
    params = sample_paint_params(rng)
    mat = build_car_paint_material(name="CarPaint", **params)
    return mat, params


if __name__ == "__main__":
    # A fixed seed gives the SAME paint every run (good for debugging).
    # Change the number for a different paint; use seed=None for fully random.
    mat, params = build_random_paint(seed=0)
    apply_material_to_active(mat)

    print("Built random paint with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")