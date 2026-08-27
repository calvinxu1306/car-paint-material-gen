"""
build_material.py — recreate the car-paint shader in Python.

This rebuilds, from scratch, the same three-layer material you made by hand:
  - shade   : base color + metallic (the underlying pigment)
  - gloss   : the clearcoat (Coat Weight / Coat Roughness)
  - glitter : flakes, via Voronoi texture -> Bump -> Normal

Run it inside Blender (Scripting workspace, Run button). With your sphere
selected, it wipes whatever material is there and rebuilds this node graph,
so you can eyeball it against your hand-built reference_paint.blend.

Written for Blender 4.x / 5.x, where the clearcoat inputs are named
"Coat Weight" / "Coat Roughness". (In old 3.x they were "Clearcoat" — not
your case, since you're on the latest LTS.)
"""

import bpy


def build_car_paint_material(
    name="CarPaint",
    base_color=(0.5, 0.02, 0.02, 1.0),  # deep red, as RGBA (0-1 each)
    metallic=0.9,
    roughness=0.4,
    coat_weight=1.0,
    coat_roughness=0.03,
    flake_scale=200.0,
    flake_strength=0.1,
):
    """Build (or rebuild) the car-paint material and return it."""
    # Reuse a material of this name if it exists, else make a new one.
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear existing nodes so re-running doesn't stack duplicates.
    nodes.clear()

    # --- Core: the Principled shader and the Material Output ---
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (0, 0)
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)

    # --- Shade: base coat ---
    principled.inputs["Base Color"].default_value = base_color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness

    # --- Gloss: clearcoat ---
    principled.inputs["Coat Weight"].default_value = coat_weight
    principled.inputs["Coat Roughness"].default_value = coat_roughness

    # --- Glitter: flakes (Voronoi -> Bump -> Normal) ---
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.location = (-600, -200)
    voronoi.inputs["Scale"].default_value = flake_scale

    bump = nodes.new("ShaderNodeBump")
    bump.location = (-300, -200)
    bump.inputs["Strength"].default_value = flake_strength

    # --- Wire everything together ---
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


if __name__ == "__main__":
    mat = build_car_paint_material()
    apply_material_to_active(mat)
    print("Car paint material built and applied.")

    # If a socket name ever errors on a future Blender version, uncomment
    # this to print the real input names so you can fix the string:
    # for i in mat.node_tree.nodes["Principled BSDF"].inputs:
    #     print(i.name)