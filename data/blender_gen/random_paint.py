"""
random_paint.py — self-contained: build a random car-paint material.

This is build_material.py + randomize_material.py merged into ONE file, so
there are no cross-file imports to trip over inside Blender. Open it in the
Scripting workspace, select your sphere, and hit Run. Change the seed (or set
it to None) for a different paint.

Three layers, same as your hand-built version:
  - shade   : base color + metallic
  - gloss   : clearcoat (Coat Weight / Coat Roughness)
  - glitter : flakes (Voronoi -> Bump -> Normal)

Written for Blender 4.x / 5.x (clearcoat inputs are "Coat Weight" /
"Coat Roughness").
"""

import bpy
import colorsys
import random


# ---------------------------------------------------------------------------
# Material builder
# ---------------------------------------------------------------------------
def build_car_paint_material(
    name="CarPaint",
    base_color=(0.5, 0.02, 0.02, 1.0),
    metallic=0.9,
    roughness=0.4,
    coat_weight=1.0,
    coat_roughness=0.03,
    flake_scale=200.0,
    flake_strength=0.1,
):
    """Build (or rebuild) the car-paint material and return it."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()  # so re-running doesn't stack duplicate nodes

    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (0, 0)
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)

    # Shade: base coat
    principled.inputs["Base Color"].default_value = base_color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness

    # Gloss: clearcoat
    principled.inputs["Coat Weight"].default_value = coat_weight
    principled.inputs["Coat Roughness"].default_value = coat_roughness

    # Glitter: flakes
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.location = (-600, -200)
    voronoi.inputs["Scale"].default_value = flake_scale

    bump = nodes.new("ShaderNodeBump")
    bump.location = (-300, -200)
    bump.inputs["Strength"].default_value = flake_strength

    links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    return mat


def apply_material_to_active(mat):
    """Put the material on whatever object is currently active/selected."""
    obj = bpy.context.active_object
    if obj is None:
        raise RuntimeError(
            "No active object. Click your sphere in the viewport first, "
            "then run again."
        )
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Randomization (ranges anchored to Gunther et al. 2005 real paints)
# ---------------------------------------------------------------------------
def sample_paint_params(rng):
    """Return a dict of randomly sampled, plausible car-paint parameters."""
    hue = rng.uniform(0.0, 1.0)
    sat = rng.uniform(0.0, 0.9)    # ~0 -> silver/gray/white; high -> candy colors
    val = rng.uniform(0.02, 0.9)   # low -> near-black; high -> bright
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)

    return {
        "base_color": (r, g, b, 1.0),
        "metallic": rng.uniform(0.7, 1.0),
        "roughness": rng.uniform(0.2, 0.5),
        "coat_weight": rng.uniform(0.8, 1.0),
        "coat_roughness": rng.uniform(0.01, 0.1),
        "flake_scale": rng.uniform(120.0, 300.0),
        "flake_strength": rng.uniform(0.05, 0.25),
    }


def build_random_paint(seed=None):
    """Sample parameters and build the material. Returns (material, params)."""
    rng = random.Random(seed)
    params = sample_paint_params(rng)
    mat = build_car_paint_material(name="CarPaint", **params)
    return mat, params


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mat, params = build_random_paint(seed=0)  # change the number for a new paint
    apply_material_to_active(mat)

    print("Built random paint with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")