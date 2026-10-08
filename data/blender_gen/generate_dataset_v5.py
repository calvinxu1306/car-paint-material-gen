"""
generate_dataset_v5.py — every kind of car paint, photographed under several lights.

WHY V5
v1-v4 photographed each paint once, with the light at (or near) the camera.
That works for solid and metallic paint, but it cannot work for pearlescent
or colour-shift paint. Their colour comes from thin-film interference, and
that colour depends on the angle between the light and the camera. With a
phone flash ~1 cm from the lens, that angle is ~0 for every pixel, so a flash
photo only ever shows one colour (the "face" colour) and never the colour shift.
Every paint instrument agrees: ASTM E2539 says interference pigments need
"measurement at multiple angles of illumination and detection".

So v5 changes two things:

1. PAINT TYPES. One sampler covers five pigment types and two finishes:
       solid       no flakes, dielectric base
       metallic    aluminium flakes in a metallic base (as in v3/v4)
       pearl       flakes plus a high-index thin film on a part-metallic
                   base (TiO2-on-mica pigments)
       colorshift  flakes plus a thicker thin film on a metallic base
                   (multi-layer "chameleon" pigments, modelled as one film)
       candy       a bright silver metallic base under a COLOURED clear coat
     x gloss or matte (matte = a rough clear coat)
   All of this is one Principled BSDF: Coat Tint for candy, Thin Film for
   pearl/colour-shift (Blender 4.2 added thin film; 5.0 extended it to metals).
   "Off" is just a value (film thickness 0, flake strength 0, white tint), so
   every sample has the same parameter list.

2. LIGHTS. The camera stays put and the light moves - the capture you can do
   with a phone on a stand and a second phone's torch:
       photo_XXXXXX.png    light at the camera (the phone's own flash)
       side1..sideN        light off to the side, random direction,
                           20-65 degrees above the surface
   The camera never moves, so every photo lines up pixel-for-pixel with the
   others AND with the ground-truth maps (which don't depend on the light).
   The side lights' brightness is set so the centre of the sample gets about
   the same light as the flash photo, +/-30% (a phone's auto-exposure does
   roughly this too).

The ground-truth maps and their encodings are unchanged from v3/v4, and the
flash photo keeps v4's name (photo_XXXXXX.png), so the v1 loader and model can
train on v5's flash photos alone - that's the single-flash baseline.

RUN (from data/blender_gen):
    blender -b -P generate_dataset_v5.py -- --count 4 --out dataset_v5
then look at the photos, and for the full set:
    blender -b -P generate_dataset_v5.py -- --count 4000 --out dataset_v5
    blender -b -P generate_dataset_v5.py -- --start 100000 --count 400 --out dataset_v5_test
The test set starts at 100000 so its random seeds can never overlap the
training set's, even if the training set grows later.

Quick smoke test with fewer render samples (noisier, ~4x faster):
    blender -b -P generate_dataset_v5.py -- --count 4 --out v5_smoke --samples 16
"""

import bpy
import colorsys
import json
import math
import os
import random
import sys
import time
import argparse


PHOTO_SAMPLES = 64
MAP_SAMPLES = 32
PLANE_SIZE = 2.5
DATASET_VERSION = 5
CAMERA_HEIGHT = 3.0
FLASH_ENERGY = 500.0          # same as v3/v4's flash

N_SIDE_LIGHTS = 5             # photos per sample = 1 flash + this many
SIDE_ELEVATION_DEG = (20.0, 65.0)   # angle of the light above the surface
SIDE_DISTANCE = (2.5, 4.0)          # distance from the sample centre
SIDE_BRIGHTNESS_JITTER = (0.7, 1.3)

MAP_NAMES = ["basecolor", "roughness", "metallic", "normal", "coat_normal"]

# How often each pigment type is drawn, and how often the finish is matte.
PIGMENTS = {"solid": 0.15, "metallic": 0.30, "pearl": 0.20,
            "colorshift": 0.15, "candy": 0.20}
MATTE_PROB = 0.2

# The LAYER parameters the network predicts as one number per sample. These
# are the normalisation ranges (written to meta.json); the per-type recipes
# below draw from sub-ranges of them.
SCALAR_RANGES = {
    "coat_weight": (0.5, 1.0),
    # gloss 0.08-0.22 brackets Guenther et al. 2005; matte 0.35-0.70
    "coat_roughness": (0.08, 0.70),
    # coat tint (candy): white = no tint
    "coat_tint_r": (0.0, 1.0),
    "coat_tint_g": (0.0, 1.0),
    "coat_tint_b": (0.0, 1.0),
    # flakes: strength 0 = no flakes (solid). Scale sets cell size (~2-4 px).
    "flake_scale": (75.0, 150.0),
    "flake_strength": (0.0, 0.40),
    # orange peel on the coat
    "peel_strength": (0.02, 0.09),
    # thin film: thickness in nanometres, 0 = no film
    "film_thickness": (0.0, 700.0),
    "film_ior": (1.3, 2.6),
}

# Everything else that is sampled (not predicted as a scalar).
OTHER_RANGES = {
    "roughness": (0.2, 0.5),          # base roughness, also a per-pixel map
    "peel_scale": (4.0, 14.0),
    "color_var_amount": (0.05, 0.35),
    "color_var_scale": (2.0, 8.0),
    "scratch_amount": (0.04, 0.18),
    "scratch_scale": (40.0, 140.0),
    "dust_amount": (0.03, 0.15),
    "dust_scale": (2.0, 9.0),
    "noise_w": (0.0, 1000.0),
}

# Per-pigment recipes. Each entry is a range to draw from; a single number is
# fixed. These are written to meta.json so the data documents itself.
RECIPES = {
    "solid": dict(metallic=0.0, flake_strength=0.0, film_thickness=0.0,
                  base_hsv=((0, 1), (0, 0.9), (0.02, 0.9))),
    "metallic": dict(metallic=(0.7, 1.0), flake_strength=(0.12, 0.40),
                     film_thickness=0.0,
                     base_hsv=((0, 1), (0, 0.9), (0.02, 0.9))),
    # TiO2-coated mica: high-index film, physical thickness ~100-400 nm.
    # Tuned by eye (see docs/v2_plan.md, "recipe check"): on a mostly
    # dielectric, pale base the film only tints the weak specular and the
    # paint just looks grey, so pearl uses a part-metallic, mid-dark base.
    "pearl": dict(metallic=(0.4, 0.8), flake_strength=(0.12, 0.40),
                  film_thickness=(100.0, 400.0), film_ior=(1.8, 2.6),
                  base_hsv=((0, 1), (0, 0.5), (0.15, 0.7))),
    # "Chameleon" pigments are multi-layer stacks; one film over metal is
    # the closest single-layer stand-in. A film below IOR ~1.5 on metal looks
    # almost colourless in Blender, so the range starts at 1.5.
    "colorshift": dict(metallic=(0.85, 1.0), flake_strength=(0.12, 0.40),
                       film_thickness=(250.0, 700.0), film_ior=(1.5, 2.4),
                       base_hsv=((0, 1), (0, 0.3), (0.3, 0.9))),
    # bright silver base; the colour lives in the coat
    "candy": dict(metallic=(0.8, 1.0), flake_strength=(0.12, 0.40),
                  film_thickness=0.0,
                  base_hsv=((0, 1), (0, 0.15), (0.6, 0.95)),
                  tint_hsv=((0, 1), (0.5, 1.0), (0.35, 0.9))),
}
GLOSS_COAT_ROUGHNESS = (0.08, 0.22)
MATTE_COAT_ROUGHNESS = (0.35, 0.70)
NO_FILM_IOR = 1.33   # Blender's default; ignored when thickness is 0


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", type=str, default="dataset_v5")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--samples", type=int, default=PHOTO_SAMPLES,
                    help="render samples per photo (lower = faster, noisier)")
    ap.add_argument("--side-lights", type=int, default=N_SIDE_LIGHTS)
    return ap.parse_args(argv)


# --- Parameters -------------------------------------------------------------
def draw(rng, spec):
    """A (lo, hi) range -> uniform draw; a plain number -> that number."""
    if isinstance(spec, (tuple, list)):
        return rng.uniform(spec[0], spec[1])
    return float(spec)


def draw_hsv(rng, hsv):
    h, s, v = (draw(rng, c) for c in hsv)
    return colorsys.hsv_to_rgb(h, s, v)


def sample_paint_params(rng):
    """One random paint. The draw order is fixed, so seed -> paint is stable."""
    names = list(PIGMENTS)
    pigment = rng.choices(names, weights=[PIGMENTS[n] for n in names])[0]
    finish = "matte" if rng.random() < MATTE_PROB else "gloss"
    r = RECIPES[pigment]

    p = {"pigment": pigment, "finish": finish}
    p["base_color"] = (*draw_hsv(rng, r["base_hsv"]), 1.0)
    p["metallic"] = draw(rng, r["metallic"])
    p["coat_weight"] = draw(rng, SCALAR_RANGES["coat_weight"])
    p["coat_roughness"] = draw(rng, MATTE_COAT_ROUGHNESS if finish == "matte"
                               else GLOSS_COAT_ROUGHNESS)
    tint = draw_hsv(rng, r["tint_hsv"]) if "tint_hsv" in r else (1.0, 1.0, 1.0)
    p["coat_tint_r"], p["coat_tint_g"], p["coat_tint_b"] = tint
    # Flake size is drawn even for solid paint (where it is unused and masked
    # out of the loss), so every params file has the same keys.
    p["flake_scale"] = draw(rng, SCALAR_RANGES["flake_scale"])
    p["flake_strength"] = draw(rng, r["flake_strength"])
    p["peel_strength"] = draw(rng, SCALAR_RANGES["peel_strength"])
    p["film_thickness"] = draw(rng, r["film_thickness"])
    p["film_ior"] = draw(rng, r["film_ior"]) if "film_ior" in r else NO_FILM_IOR
    for key, rng_range in OTHER_RANGES.items():
        p[key] = draw(rng, rng_range)
    return p


def sample_lights(rng, n_side):
    """Light 0 is the flash (at the camera). The rest are side lights."""
    lights = [{"name": "photo", "kind": "flash",
               "x": 0.0, "y": 0.0, "z": CAMERA_HEIGHT, "energy": FLASH_ENERGY}]
    for k in range(1, n_side + 1):
        az = rng.uniform(0.0, 2.0 * math.pi)
        elev = math.radians(rng.uniform(*SIDE_ELEVATION_DEG))
        dist = rng.uniform(*SIDE_DISTANCE)
        x = dist * math.cos(elev) * math.cos(az)
        y = dist * math.cos(elev) * math.sin(az)
        z = dist * math.sin(elev)
        # Irradiance at the centre is energy * cos(incidence) / dist^2. Match
        # the flash's (FLASH_ENERGY / CAMERA_HEIGHT^2), then jitter.
        energy = (FLASH_ENERGY * (dist / CAMERA_HEIGHT) ** 2 / math.sin(elev)
                  * rng.uniform(*SIDE_BRIGHTNESS_JITTER))
        lights.append({"name": f"side{k}", "kind": "side",
                       "x": x, "y": y, "z": z, "energy": energy,
                       "azimuth_deg": math.degrees(az),
                       "elevation_deg": math.degrees(elev), "distance": dist})
    return lights


# --- Node helpers -----------------------------------------------------------
def fresh_material(name):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat, mat.node_tree.nodes, mat.node_tree.links


def noise_node(nodes, scale, w, detail=2.0, location=(0, 0)):
    n = nodes.new("ShaderNodeTexNoise")
    n.noise_dimensions = "4D"
    n.location = location
    n.inputs["Scale"].default_value = scale
    n.inputs["W"].default_value = w
    n.inputs["Detail"].default_value = detail
    return n


def math_node(nodes, op, value=None, clamp=False, location=(0, 0)):
    n = nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    n.location = location
    if value is not None:
        n.inputs[1].default_value = value
    return n


def vec_node(nodes, op, location=(0, 0)):
    n = nodes.new("ShaderNodeVectorMath")
    n.operation = op
    n.location = location
    return n


def centre_vector(nodes, links, src_socket, location=(0, 0)):
    """Map a 0..1 vector to -1..1:  v * 2 - 1."""
    n = vec_node(nodes, "MULTIPLY_ADD", location)
    n.inputs[1].default_value = (2.0, 2.0, 2.0)
    n.inputs[2].default_value = (-1.0, -1.0, -1.0)
    links.new(src_socket, n.inputs[0])
    return n.outputs["Vector"]


def tilt_to_normal(nodes, links, tilt_socket, location=(0, 0)):
    """Keep a tilt vector's x and y, force z = 1, normalise -> unit normal."""
    x0, y0 = location
    sep = nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (x0, y0)
    links.new(tilt_socket, sep.inputs[0])
    comb = nodes.new("ShaderNodeCombineXYZ")
    comb.location = (x0 + 140, y0)
    comb.inputs["Z"].default_value = 1.0
    links.new(sep.outputs["X"], comb.inputs["X"])
    links.new(sep.outputs["Y"], comb.inputs["Y"])
    norm = vec_node(nodes, "NORMALIZE", (x0 + 280, y0))
    links.new(comb.outputs["Vector"], norm.inputs[0])
    return norm.outputs["Vector"]


# --- Channel definitions (shared by the photo and the map export) -----------
def basecolor_socket(nodes, links, p):
    """Base colour with large-scale brightness variation, clamped to <= 1:
    colour = min(base_rgb * (1 + amount * (noise - 0.5)), 1)."""
    n = noise_node(nodes, p["color_var_scale"], p["noise_w"] + 11.0,
                   location=(-1000, 400))
    centred = math_node(nodes, "SUBTRACT", 0.5, location=(-800, 400))
    scaled = math_node(nodes, "MULTIPLY", p["color_var_amount"], location=(-640, 400))
    factor = math_node(nodes, "ADD", 1.0, location=(-480, 400))
    links.new(n.outputs["Fac"], centred.inputs[0])
    links.new(centred.outputs[0], scaled.inputs[0])
    links.new(scaled.outputs[0], factor.inputs[0])

    tint = vec_node(nodes, "SCALE", location=(-320, 400))
    tint.inputs[0].default_value = p["base_color"][:3]
    links.new(factor.outputs[0], tint.inputs["Scale"])

    cap = vec_node(nodes, "MINIMUM", location=(-160, 400))
    cap.inputs[1].default_value = (1.0, 1.0, 1.0)
    links.new(tint.outputs["Vector"], cap.inputs[0])
    return cap.outputs["Vector"]


def roughness_socket(nodes, links, p):
    """Base roughness + fine scratches + broader dust patches, clamped to 0..1."""
    scratch = noise_node(nodes, p["scratch_scale"], p["noise_w"] + 23.0,
                         detail=4.0, location=(-1000, 100))
    s_c = math_node(nodes, "SUBTRACT", 0.5, location=(-800, 160))
    s_a = math_node(nodes, "MULTIPLY", p["scratch_amount"], location=(-640, 160))
    links.new(scratch.outputs["Fac"], s_c.inputs[0])
    links.new(s_c.outputs[0], s_a.inputs[0])

    dust = noise_node(nodes, p["dust_scale"], p["noise_w"] + 37.0,
                      location=(-1000, -100))
    d_c = math_node(nodes, "SUBTRACT", 0.5, location=(-800, -60))
    d_a = math_node(nodes, "MULTIPLY", p["dust_amount"], location=(-640, -60))
    links.new(dust.outputs["Fac"], d_c.inputs[0])
    links.new(d_c.outputs[0], d_a.inputs[0])

    base = math_node(nodes, "ADD", p["roughness"], location=(-480, 100))
    links.new(s_a.outputs[0], base.inputs[0])
    total = math_node(nodes, "ADD", clamp=True, location=(-320, 100))
    links.new(base.outputs[0], total.inputs[0])
    links.new(d_a.outputs[0], total.inputs[1])
    return total.outputs[0]


def flake_normal_socket(nodes, links, p):
    """BASE-layer normal: a random tilt per Voronoi cell (the flakes).

        n = normalise( flake_strength * (2*cellColour.xy - 1),  z = 1 )

    flake_strength 0 (solid paint) gives a flat normal."""
    flakes = nodes.new("ShaderNodeTexVoronoi")
    flakes.voronoi_dimensions = "4D"
    flakes.location = (-1100, -400)
    flakes.inputs["Scale"].default_value = p["flake_scale"]
    flakes.inputs["W"].default_value = p["noise_w"] + 71.0
    centred = centre_vector(nodes, links, flakes.outputs["Color"], (-900, -400))
    scaled = vec_node(nodes, "SCALE", (-720, -400))
    scaled.inputs["Scale"].default_value = p["flake_strength"]
    links.new(centred, scaled.inputs[0])
    return tilt_to_normal(nodes, links, scaled.outputs["Vector"], (-560, -400))


def peel_normal_socket(nodes, links, p):
    """COAT-layer normal: a slow low-frequency ripple (orange peel)."""
    peel = noise_node(nodes, p["peel_scale"], p["noise_w"] + 53.0,
                      location=(-1100, -700))
    centred = centre_vector(nodes, links, peel.outputs["Color"], (-900, -700))
    scaled = vec_node(nodes, "SCALE", (-720, -700))
    scaled.inputs["Scale"].default_value = p["peel_strength"]
    links.new(centred, scaled.inputs[0])
    return tilt_to_normal(nodes, links, scaled.outputs["Vector"], (-560, -700))


# --- Materials --------------------------------------------------------------
def build_paint_material(p, name="CarPaintV5"):
    mat, nodes, links = fresh_material(name)
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (0, 0)
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    links.new(basecolor_socket(nodes, links, p), principled.inputs["Base Color"])
    links.new(roughness_socket(nodes, links, p), principled.inputs["Roughness"])
    # Flakes go to the base layer, peel to the coat. Coat Normal MUST be
    # connected, otherwise Cycles copies the base (flake) normal onto the coat.
    links.new(flake_normal_socket(nodes, links, p), principled.inputs["Normal"])
    links.new(peel_normal_socket(nodes, links, p), principled.inputs["Coat Normal"])
    principled.inputs["Metallic"].default_value = p["metallic"]
    principled.inputs["Coat Weight"].default_value = p["coat_weight"]
    principled.inputs["Coat Roughness"].default_value = p["coat_roughness"]
    principled.inputs["Coat Tint"].default_value = (
        p["coat_tint_r"], p["coat_tint_g"], p["coat_tint_b"], 1.0)
    # Thin film sits on the BASE layer (under the coat) and follows the flake
    # normals, like the coating on a pearl pigment's platelets.
    principled.inputs["Thin Film Thickness"].default_value = p["film_thickness"]
    principled.inputs["Thin Film IOR"].default_value = p["film_ior"]

    links.new(principled.outputs["BSDF"], out.inputs["Surface"])
    return mat


def build_emission_material(name, socket_fn, p, remap_normal=False):
    """Emission material glowing with whatever socket_fn builds - the same
    builder the photo uses, so the map cannot disagree with the render."""
    mat, nodes, links = fresh_material(name)
    emission = nodes.new("ShaderNodeEmission")
    emission.location = (0, 0)
    emission.inputs["Strength"].default_value = 1.0
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    src = socket_fn(nodes, links, p)
    if remap_normal:
        # Normals run -1..1 but images store 0..1.
        remap = vec_node(nodes, "MULTIPLY_ADD", (-200, 0))
        remap.inputs[1].default_value = (0.5, 0.5, 0.5)
        remap.inputs[2].default_value = (0.5, 0.5, 0.5)
        links.new(src, remap.inputs[0])
        src = remap.outputs["Vector"]

    links.new(src, emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def build_constant_material(name, rgba):
    mat, nodes, links = fresh_material(name)
    emission = nodes.new("ShaderNodeEmission")
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    emission.inputs["Color"].default_value = rgba
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


# --- Scene ------------------------------------------------------------------
def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def setup_scene():
    bpy.ops.mesh.primitive_plane_add(size=PLANE_SIZE, location=(0, 0, 0))
    plane = bpy.context.active_object

    cam_data = bpy.data.cameras.new("Cam")
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0, 0, CAMERA_HEIGHT)
    cam_obj.rotation_euler = (0, 0, 0)
    bpy.context.scene.camera = cam_obj

    light_data = bpy.data.lights.new("Light", type="POINT")
    light_data.energy = FLASH_ENERGY
    light_obj = bpy.data.objects.new("Light", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (0, 0, CAMERA_HEIGHT)

    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0, 0, 0, 1)
        bg.inputs["Strength"].default_value = 0.0
    return plane, light_obj


def setup_render(res, use_cpu):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    if use_cpu:
        scene.cycles.device = "CPU"
        print("[setup] rendering on CPU (forced)")
        return
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if gpus:
                for d in prefs.devices:
                    d.use = (d.type != "CPU")
                scene.cycles.device = "GPU"
                print(f"[setup] rendering on GPU via {backend}")
                return
    except Exception as e:
        print(f"[setup] GPU setup failed ({e})")
    scene.cycles.device = "CPU"
    print("[setup] no GPU found, rendering on CPU")


def set_view_transform(name, fallback="Standard"):
    try:
        bpy.context.scene.view_settings.view_transform = name
    except TypeError:
        bpy.context.scene.view_settings.view_transform = fallback


def apply_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def place_light(light_obj, light):
    light_obj.location = (light["x"], light["y"], light["z"])
    light_obj.data.energy = light["energy"]


def render_photos(plane, light_obj, p, lights, path_for, samples):
    """One photo per light, same camera, same paint."""
    scene = bpy.context.scene
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.dither_intensity = PHOTO_DITHER
    set_view_transform("AgX")
    light_obj.hide_render = False
    apply_material(plane, build_paint_material(p))
    for light in lights:
        place_light(light_obj, light)
        render_to(path_for(light["name"]))


def render_maps(plane, light_obj, p, path_for):
    """Ground-truth maps (emission shaders, no light needed)."""
    scene = bpy.context.scene
    light_obj.hide_render = True
    scene.cycles.samples = MAP_SAMPLES
    scene.cycles.use_denoising = False
    # Blender dithers 8-bit output by default (+-~1 code value of noise per
    # pixel). Fine for a photo, wrong for ground truth: it is the same size as
    # the flake-normal errors being measured. Maps store exact values.
    scene.render.dither_intensity = MAP_DITHER

    set_view_transform("Standard")  # colour -> sRGB-encoded
    apply_material(plane, build_emission_material("M_BaseColor", basecolor_socket, p))
    render_to(path_for("basecolor"))

    set_view_transform("Raw")       # data -> stored exactly
    apply_material(plane, build_emission_material("M_Rough", roughness_socket, p))
    render_to(path_for("roughness"))
    mv = p["metallic"]
    apply_material(plane, build_constant_material("M_Metal", (mv, mv, mv, 1.0)))
    render_to(path_for("metallic"))
    apply_material(plane, build_emission_material("M_Normal", flake_normal_socket, p,
                                                  remap_normal=True))
    render_to(path_for("normal"))
    apply_material(plane, build_emission_material("M_CoatNormal", peel_normal_socket,
                                                  p, remap_normal=True))
    render_to(path_for("coat_normal"))
    light_obj.hide_render = False


# --- One sample -------------------------------------------------------------
def photo_names(n_side):
    return ["photo"] + [f"side{k}" for k in range(1, n_side + 1)]


def render_sample(index, out_dir, plane, light_obj, n_side, samples):
    rng = random.Random(index)
    p = sample_paint_params(rng)
    lights = sample_lights(rng, n_side)   # drawn after the paint
    tag = f"{index:06d}"
    path_for = lambda name: os.path.join(out_dir, f"{name}_{tag}.png")

    render_photos(plane, light_obj, p, lights, path_for, samples)
    render_maps(plane, light_obj, p, path_for)

    with open(os.path.join(out_dir, f"params_{tag}.json"), "w") as f:
        json.dump({"index": index, "seed": index, "version": DATASET_VERSION,
                   **p, "lights": lights}, f, indent=2)
    return p


def already_done(index, out_dir, n_side):
    tag = f"{index:06d}"
    names = photo_names(n_side) + MAP_NAMES
    return (all(os.path.exists(os.path.join(out_dir, f"{n}_{tag}.png"))
                for n in names)
            and os.path.exists(os.path.join(out_dir, f"params_{tag}.json")))


PHOTO_DITHER = 1.0   # Blender's default; hides 8-bit banding like camera noise
MAP_DITHER = 0.0     # ground truth stores exact values (see render_maps)


def write_meta(out_dir, n_side, samples, res):
    """Record exactly how this dataset was generated. The loaders read the
    ranges and photo names from here, so the code can't desync from the data."""
    meta = {
        "version": DATASET_VERSION,
        "generator": os.path.basename(__file__),
        "ranges": {k: list(v) for k, v in {**SCALAR_RANGES, **OTHER_RANGES}.items()},
        "scalar_keys": list(SCALAR_RANGES),
        "pigments": PIGMENTS,
        "matte_prob": MATTE_PROB,
        "recipes": RECIPES,
        "coat_roughness": {"gloss": list(GLOSS_COAT_ROUGHNESS),
                           "matte": list(MATTE_COAT_ROUGHNESS)},
        "base_color": "HSV per pigment recipe (base_hsv)",
        "normal_png": "base-layer flake normal, xyz stored as v*0.5+0.5",
        "coat_normal_png": "coat-layer orange-peel normal, same encoding",
        "resolution": res,
        "photo_samples": samples,
        "dither": {"photos": PHOTO_DITHER, "maps": MAP_DITHER},
        "photos": photo_names(n_side),
        "lighting": {
            "camera": [0, 0, CAMERA_HEIGHT],
            "flash": "point light at the camera, energy %g" % FLASH_ENERGY,
            "side_elevation_deg": list(SIDE_ELEVATION_DEG),
            "side_distance": list(SIDE_DISTANCE),
            "side_azimuth_deg": [0, 360],
            "side_brightness": "centre irradiance matched to the flash, x U%s"
                               % (SIDE_BRIGHTNESS_JITTER,),
        },
    }
    path = os.path.join(out_dir, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        new = json.loads(json.dumps(meta))   # tuples -> lists, as on disk
        # EVERY setting must match (ranges, recipes, lights, render samples...),
        # or a resumed run would quietly mix two kinds of data in one folder.
        differ = [k for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
        if differ:
            raise RuntimeError(
                f"{path} was written with different settings: {differ}. Mixing "
                f"two distributions in one folder would corrupt the dataset - "
                f"use a new --out folder, or the same flags as before.")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m" if h else (f"{m}m {s}s" if m else f"{s}s")


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    clear_scene()
    plane, light_obj = setup_scene()
    setup_render(args.res, args.cpu)
    write_meta(out_dir, args.side_lights, args.samples, args.res)

    indices = list(range(args.start, args.start + args.count))
    todo = [i for i in indices if not already_done(i, out_dir, args.side_lights)]
    print(f"[batch] output: {out_dir}")
    print(f"[batch] {len(todo)} to render, {len(indices) - len(todo)} already done "
          f"| {1 + args.side_lights} photos per sample at {args.samples} samples")

    t0 = time.time()
    counts = {}
    for n, i in enumerate(todo, start=1):
        p = render_sample(i, out_dir, plane, light_obj, args.side_lights, args.samples)
        counts[p["pigment"]] = counts.get(p["pigment"], 0) + 1
        el = time.time() - t0
        per = el / n
        print(f"[batch] {n}/{len(todo)} {p['pigment']:<10} {p['finish']:<5} | "
              f"{per:.1f}s each | remaining ~{fmt(per * (len(todo) - n))}", flush=True)

    if todo:
        per = (time.time() - t0) / len(todo)
        print(f"\n[batch] done in {fmt(time.time() - t0)} ({per:.1f}s each)")
        print(f"[batch] 4000 samples would take ~{fmt(per * 4000)}")
        print(f"[batch] pigment counts this run: {counts}")
    print("\nCHECK before a full run (open a few samples):")
    print("  photo_*      flash at the camera: hotspot dead centre")
    print("  side1..N_*   same paint lit from the side; pearl/colorshift samples")
    print("               should change colour between side photos")
    print("  basecolor_*  for candy paint this is the SILVER base; the colour is")
    print("               in coat_tint_* in the params file")


if __name__ == "__main__":
    main()
