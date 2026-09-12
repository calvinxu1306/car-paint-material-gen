"""
render_one.py — set up the flash-capture scene and render ONE photo.

This builds the data-generation scene from scratch:
  - a flat plane (the paint sample), viewed straight-on
  - a camera looking straight down at it
  - a POINT light collocated with the camera (mimics an on-camera flash)
  - a dark world, so only the flash lights the paint (like a black-felt studio)

then applies a random car paint and renders a single 256x256 image to disk.
This is just the PHOTO for now; exporting the ground-truth maps comes next.

IMPORTANT: this clears all objects to build a clean scene. Run it in a FRESH
Blender file, not your reference_paint.blend. Suggested workflow:
  File > New > General, then File > Save As data/blender_gen/generate_dataset.blend,
  then open and run this script.

Self-contained (no imports) so it just runs. Written for Blender 4.x / 5.x.
"""

import bpy
import colorsys
import random
import os


# --- Config -----------------------------------------------------------------
RESOLUTION = 256
SAMPLES = 64  # render quality; low is fine for a flat sample

# Where images go. Uses the folder of the saved .blend; falls back to home.
if bpy.data.filepath:
    OUTPUT_DIR = bpy.path.abspath("//dataset_output")
else:
    OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "car_paint_dataset")


# --- Material (same three-layer paint as before) ----------------------------
def build_car_paint_material(name="CarPaint", base_color=(0.5, 0.02, 0.02, 1.0),
                             metallic=0.9, roughness=0.4, coat_weight=1.0,
                             coat_roughness=0.03, flake_scale=200.0,
                             flake_strength=0.1):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()

    principled = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)

    principled.inputs["Base Color"].default_value = base_color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Coat Weight"].default_value = coat_weight
    principled.inputs["Coat Roughness"].default_value = coat_roughness

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


def sample_paint_params(rng):
    hue, sat, val = rng.uniform(0, 1), rng.uniform(0, 0.9), rng.uniform(0.02, 0.9)
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


# --- Scene setup ------------------------------------------------------------
def clear_scene():
    """Delete all objects so we build a clean, reproducible scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def setup_scene():
    """Plane + top-down camera + collocated flash + dark world."""
    # The paint sample
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
    plane = bpy.context.active_object

    # Camera looking straight down (-Z) from above
    cam_data = bpy.data.cameras.new("Cam")
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0, 0, 3)
    cam_obj.rotation_euler = (0, 0, 0)  # zero rotation = looks straight down
    bpy.context.scene.camera = cam_obj

    # Collocated "flash": a point light at the camera's position
    light_data = bpy.data.lights.new("Flash", type="POINT")
    light_data.energy = 500.0  # may need tuning; see notes
    light_obj = bpy.data.objects.new("Flash", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (0, 0, 3)

    # Dark world so only the flash lights the paint
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0, 0, 0, 1)
        bg.inputs["Strength"].default_value = 0.0

    return plane


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.image_settings.file_format = "PNG"


def apply_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clear_scene()
    plane = setup_scene()
    setup_render()

    rng = random.Random(0)  # fixed seed -> same paint every run
    params = sample_paint_params(rng)
    mat = build_car_paint_material(**params)
    apply_material(plane, mat)

    out_path = os.path.join(OUTPUT_DIR, "photo_0000.png")
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

    print("Rendered photo to:", out_path)
    print("Paint parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")