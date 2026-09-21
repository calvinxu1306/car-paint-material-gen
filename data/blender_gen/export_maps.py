"""
export_maps.py — render a photo PLUS its ground-truth maps (one sample).

For each paint this saves:
    photo_0000.png       <- the flash-lit photo (the network's INPUT)
    basecolor_0000.png   <- ground-truth maps (what the network learns to PREDICT)
    roughness_0000.png
    metallic_0000.png
    normal_0000.png
    params_0000.json     <- the exact numbers used, for reproducibility

HOW THE MAPS ARE MADE
Roughness isn't something you can photograph; it's an input to a lighting
calculation. So to record it, we temporarily swap the material for an EMISSION
shader that glows with the value directly, with the flash switched off. Each
pixel then holds the raw value instead of a lit appearance.

THREE GOTCHAS THIS HANDLES
1. View transform: the photo uses AgX (a photographic tone curve). Data maps
   (roughness, metallic, normal) use "Raw" so the numbers are stored exactly,
   with no tone curve and no sRGB encoding. Base color uses "Standard"
   (sRGB-encoded), which is the normal convention for color textures.
2. Normals run -1..1 but images store 0..1, so they're remapped with
   (v * 0.5 + 0.5). That's why normal maps look lavender-blue.
3. Denoising is turned OFF for maps, because it would blur the flake normals.

At the end the script READS BACK the saved roughness and base color files and
prints expected vs. stored values. If they match, the maps can be trusted.

Run it in a FRESH Blender file (it clears the scene), same as render_one.py.
Self-contained. Blender 4.x / 5.x.
"""

import bpy
import colorsys
import random
import json
import os


# --- Config -----------------------------------------------------------------
RESOLUTION = 256
PHOTO_SAMPLES = 64  # render quality for the photo
MAP_SAMPLES = 16    # a few samples average out the flake normals within a pixel
PLANE_SIZE = 2.5    # slightly larger than the camera's view -> no black border

if bpy.data.filepath:
    OUTPUT_DIR = bpy.path.abspath("//dataset_output")
else:
    OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "car_paint_dataset")


# --- Paint parameters (same ranges as random_paint.py) ----------------------
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


# --- Materials --------------------------------------------------------------
def fresh_material(name):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat, mat.node_tree.nodes, mat.node_tree.links


def add_flakes(nodes, links, flake_scale, flake_strength):
    """Voronoi -> Bump. Returns the Bump node (its 'Normal' output is the
    flake-perturbed normal). Shared by the photo material and the normal map,
    so both see exactly the same flakes."""
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.location = (-800, -200)
    voronoi.inputs["Scale"].default_value = flake_scale
    bump = nodes.new("ShaderNodeBump")
    bump.location = (-600, -200)
    bump.inputs["Strength"].default_value = flake_strength
    links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
    return bump


def build_car_paint_material(base_color, metallic, roughness, coat_weight,
                             coat_roughness, flake_scale, flake_strength):
    """The real, lit paint material (used for the PHOTO)."""
    mat, nodes, links = fresh_material("CarPaint")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)

    principled.inputs["Base Color"].default_value = base_color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Coat Weight"].default_value = coat_weight
    principled.inputs["Coat Roughness"].default_value = coat_roughness

    bump = add_flakes(nodes, links, flake_scale, flake_strength)
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return mat


def build_constant_map_material(name, rgba):
    """Emission shader glowing with one constant value everywhere
    (base color / roughness / metallic are uniform across our sample)."""
    mat, nodes, links = fresh_material(name)
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)
    emission.inputs["Color"].default_value = rgba
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def build_normal_map_material(flake_scale, flake_strength):
    """Emission shader glowing with the flake-perturbed normal, remapped 0..1.
    Chain: Voronoi -> Bump -> (* 0.5) -> (+ 0.5) -> Emission."""
    mat, nodes, links = fresh_material("M_Normal")
    bump = add_flakes(nodes, links, flake_scale, flake_strength)

    mult = nodes.new("ShaderNodeVectorMath")
    mult.location = (-400, 0)
    mult.operation = "MULTIPLY"
    mult.inputs[1].default_value = (0.5, 0.5, 0.5)

    add = nodes.new("ShaderNodeVectorMath")
    add.location = (-200, 0)
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)

    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)

    links.new(bump.outputs["Normal"], mult.inputs[0])
    links.new(mult.outputs["Vector"], add.inputs[0])
    links.new(add.outputs["Vector"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


# --- Scene ------------------------------------------------------------------
def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def setup_scene():
    """Plane + top-down camera + collocated flash + dark world."""
    bpy.ops.mesh.primitive_plane_add(size=PLANE_SIZE, location=(0, 0, 0))
    plane = bpy.context.active_object

    cam_data = bpy.data.cameras.new("Cam")
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0, 0, 3)
    cam_obj.rotation_euler = (0, 0, 0)  # looks straight down
    bpy.context.scene.camera = cam_obj

    light_data = bpy.data.lights.new("Flash", type="POINT")
    light_data.energy = 500.0
    light_obj = bpy.data.objects.new("Flash", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (0, 0, 3)  # same spot as the camera = collocated flash

    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0, 0, 0, 1)
        bg.inputs["Strength"].default_value = 0.0

    return plane, light_obj


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def set_view_transform(name, fallback="Standard"):
    try:
        bpy.context.scene.view_settings.view_transform = name
    except TypeError:
        print(f"  (view transform '{name}' not available, using '{fallback}')")
        bpy.context.scene.view_settings.view_transform = fallback


def set_photo_mode():
    scene = bpy.context.scene
    scene.cycles.samples = PHOTO_SAMPLES
    scene.cycles.use_denoising = True
    set_view_transform("AgX")


def set_map_mode(data_map):
    """data_map=True  -> Raw: store the number exactly (roughness/metallic/normal).
       data_map=False -> Standard: sRGB-encoded, the convention for base color."""
    scene = bpy.context.scene
    scene.cycles.samples = MAP_SAMPLES
    scene.cycles.use_denoising = False
    set_view_transform("Raw" if data_map else "Standard")


def apply_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


# --- One complete sample ----------------------------------------------------
def render_sample(index, seed, plane, light_obj):
    p = sample_paint_params(random.Random(seed))
    tag = f"{index:04d}"
    out = lambda name: os.path.join(OUTPUT_DIR, f"{name}_{tag}.png")

    # 1) The photo: real material, flash on, photographic tone curve.
    set_photo_mode()
    light_obj.hide_render = False
    apply_material(plane, build_car_paint_material(**p))
    render_to(out("photo"))

    # 2) The maps: emission shaders, flash off.
    light_obj.hide_render = True

    set_map_mode(data_map=False)
    apply_material(plane, build_constant_map_material("M_BaseColor", p["base_color"]))
    render_to(out("basecolor"))

    set_map_mode(data_map=True)
    rv, mv = p["roughness"], p["metallic"]
    apply_material(plane, build_constant_map_material("M_Rough", (rv, rv, rv, 1.0)))
    render_to(out("roughness"))

    apply_material(plane, build_constant_map_material("M_Metal", (mv, mv, mv, 1.0)))
    render_to(out("metallic"))

    apply_material(plane, build_normal_map_material(p["flake_scale"], p["flake_strength"]))
    render_to(out("normal"))

    # 3) The exact parameters, so any sample can be regenerated or inspected.
    with open(os.path.join(OUTPUT_DIR, f"params_{tag}.json"), "w") as f:
        json.dump({"seed": seed,
                   "base_color_note": "linear RGB; basecolor PNG stores it sRGB-encoded",
                   **p}, f, indent=2)

    light_obj.hide_render = False
    return p


# --- Self-check: read the saved files back and compare ----------------------
def linear_to_srgb(c):
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def center_pixel(path):
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    i = ((h // 2) * w + (w // 2)) * 4
    px = img.pixels[i:i + 3]  # stored values, 0..1
    bpy.data.images.remove(img)
    return px


def verify(index, p):
    tag = f"{index:04d}"
    ok = True
    print("\nSELF-CHECK (expected vs. stored in file):")

    stored = center_pixel(os.path.join(OUTPUT_DIR, f"roughness_{tag}.png"))[0]
    good = abs(stored - p["roughness"]) < 0.01
    ok &= good
    print(f"  roughness : expected {p['roughness']:.3f}, stored {stored:.3f}  "
          f"{'OK' if good else 'MISMATCH'}")

    exp = linear_to_srgb(p["base_color"][0])
    stored = center_pixel(os.path.join(OUTPUT_DIR, f"basecolor_{tag}.png"))[0]
    good = abs(stored - exp) < 0.01
    ok &= good
    print(f"  basecolor R (sRGB): expected {exp:.3f}, stored {stored:.3f}  "
          f"{'OK' if good else 'MISMATCH'}")

    print("  -> maps look trustworthy." if ok else
          "  -> MISMATCH: color management is altering the values. Paste this output.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clear_scene()
    plane, light_obj = setup_scene()
    setup_render()

    params = render_sample(index=0, seed=0, plane=plane, light_obj=light_obj)
    print("Wrote one complete sample to:", OUTPUT_DIR)
    verify(0, params)