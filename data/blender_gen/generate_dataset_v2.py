"""
generate_dataset_v2.py — richer car-paint dataset with real spatial variation.

WHY V2
In v1 every map was flat except the normals, and those were nearly flat too: a
mean predictor scored 0.0022 L1 on them, i.e. there was nothing to learn.

WHY THE FLAKE NORMALS ARE BUILT DIFFERENTLY HERE
The first attempt fed a Voronoi texture into a Bump node. Bump derives the
tilt from the GRADIENT of the height field, so the tilt depends on how sharply
the height changes across space - which made the flakes both sub-pixel (at high
scale) and nearly flat (at low scale). Dead end.

Instead the flake normal is now built analytically: every Voronoi cell gets its
own random tilt, taken from the texture's per-cell random Colour output.

    n = normalise( flake_strength * (2*cellColour.xy - 1)
                 + peel_strength  * (2*noiseColour.xy - 1),  z = 1 )

That is discrete randomly-oriented facets - which is what metallic flakes
physically ARE, and close to the procedural flake-normal texture used by
Gunther et al. 2005. Measured: this puts 11-33 levels (out of 255) of variation
into the normal map, against 0.6 in v1.

CAVEAT worth stating in the write-up: the tilt vector is applied as if texture
space and world space agree. That holds here because the sample is a flat plane
facing +Z. It would need a proper tangent frame on curved geometry.

WHAT ELSE CHANGED
  base colour   large-scale brightness variation instead of one flat colour
  roughness     fine scratch/swirl texture plus broader dust patches
  orange peel   slow ripple in the clear coat, folded into the normal above
  metallic      unchanged (constant) - one thing at a time

GROUND TRUTH STAYS HONEST
Each channel's node sub-graph is built by ONE function used for both the lit
photo and the emission pass that exports the map, so the two cannot drift.

RUN (delete any previous dataset_v2 first - those samples used the old normals):
    blender -b -P generate_dataset_v2.py -- --count 2 --out dataset_v2
Check normal_000000.png has visible speckle, THEN:
    blender -b -P generate_dataset_v2.py -- --count 2000 --out dataset_v2

File names and 6-digit tags match v1, so src/data/dataset.py reads this folder
unchanged.
"""

import bpy
import colorsys
import random
import json
import os
import sys
import time
import argparse


PHOTO_SAMPLES = 64
MAP_SAMPLES = 32     # there is real high-frequency detail now
PLANE_SIZE = 2.5


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", type=str, default="dataset_v2")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


# --- Parameters -------------------------------------------------------------
def sample_paint_params(rng):
    hue, sat, val = rng.uniform(0, 1), rng.uniform(0, 0.9), rng.uniform(0.02, 0.9)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return {
        # base coat
        "base_color": (r, g, b, 1.0),
        "metallic": rng.uniform(0.7, 1.0),
        "roughness": rng.uniform(0.2, 0.5),
        # clear coat
        "coat_weight": rng.uniform(0.8, 1.0),
        "coat_roughness": rng.uniform(0.01, 0.1),
        # flakes: scale sets cell size (cells are ~5-20 px), strength sets tilt
        "flake_scale": rng.uniform(75.0, 150.0),
        "flake_strength": rng.uniform(0.12, 0.40),
        # orange peel: slow wobble in the coat
        "peel_scale": rng.uniform(4.0, 14.0),
        "peel_strength": rng.uniform(0.02, 0.09),
        # spatial variation in the base colour
        "color_var_amount": rng.uniform(0.05, 0.35),
        "color_var_scale": rng.uniform(2.0, 8.0),
        # surface wear
        "scratch_amount": rng.uniform(0.04, 0.18),
        "scratch_scale": rng.uniform(40.0, 140.0),
        "dust_amount": rng.uniform(0.03, 0.15),
        "dust_scale": rng.uniform(2.0, 9.0),
        # offset into 4D noise so no two samples share a pattern
        "noise_w": rng.uniform(0.0, 1000.0),
    }


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


# --- Channel definitions (shared by the photo and the map export) -----------
def basecolor_socket(nodes, links, p):
    """Base colour with large-scale brightness variation:
    colour = base_rgb * (1 + amount * (noise - 0.5))."""
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
    return tint.outputs["Vector"]


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


def normal_socket(nodes, links, p):
    """Per-cell random flake tilts plus a slow orange-peel wobble.

        n = normalise( flake_strength * (2*cellColour.xy - 1)
                     + peel_strength  * (2*noiseColour.xy - 1),  z = 1 )

    No Bump node: the tilt is set directly, so its size is exactly what the
    strength parameters say rather than a side effect of a height gradient.
    """
    # Flakes: Voronoi's Colour output is a different random RGB per cell.
    flakes = nodes.new("ShaderNodeTexVoronoi")
    flakes.voronoi_dimensions = "4D"
    flakes.location = (-1100, -400)
    flakes.inputs["Scale"].default_value = p["flake_scale"]
    flakes.inputs["W"].default_value = p["noise_w"] + 71.0
    f_centred = centre_vector(nodes, links, flakes.outputs["Color"], (-900, -400))
    f_scaled = vec_node(nodes, "SCALE", (-720, -400))
    f_scaled.inputs["Scale"].default_value = p["flake_strength"]
    links.new(f_centred, f_scaled.inputs[0])

    # Orange peel: smooth low-frequency random direction.
    peel = noise_node(nodes, p["peel_scale"], p["noise_w"] + 53.0,
                      location=(-1100, -650))
    p_centred = centre_vector(nodes, links, peel.outputs["Color"], (-900, -650))
    p_scaled = vec_node(nodes, "SCALE", (-720, -650))
    p_scaled.inputs["Scale"].default_value = p["peel_strength"]
    links.new(p_centred, p_scaled.inputs[0])

    tilt = vec_node(nodes, "ADD", (-560, -500))
    links.new(f_scaled.outputs["Vector"], tilt.inputs[0])
    links.new(p_scaled.outputs["Vector"], tilt.inputs[1])

    # Keep the tilt's x and y, force z = 1, then normalise.
    sep = nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-400, -500)
    links.new(tilt.outputs["Vector"], sep.inputs[0])
    comb = nodes.new("ShaderNodeCombineXYZ")
    comb.location = (-260, -500)
    comb.inputs["Z"].default_value = 1.0
    links.new(sep.outputs["X"], comb.inputs["X"])
    links.new(sep.outputs["Y"], comb.inputs["Y"])

    norm = vec_node(nodes, "NORMALIZE", (-120, -500))
    links.new(comb.outputs["Vector"], norm.inputs[0])
    return norm.outputs["Vector"]


# --- Materials --------------------------------------------------------------
def build_paint_material(p):
    mat, nodes, links = fresh_material("CarPaintV2")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (0, 0)
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    links.new(basecolor_socket(nodes, links, p), principled.inputs["Base Color"])
    links.new(roughness_socket(nodes, links, p), principled.inputs["Roughness"])
    links.new(normal_socket(nodes, links, p), principled.inputs["Normal"])
    principled.inputs["Metallic"].default_value = p["metallic"]
    principled.inputs["Coat Weight"].default_value = p["coat_weight"]
    principled.inputs["Coat Roughness"].default_value = p["coat_roughness"]

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
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def setup_scene():
    bpy.ops.mesh.primitive_plane_add(size=PLANE_SIZE, location=(0, 0, 0))
    plane = bpy.context.active_object

    cam_data = bpy.data.cameras.new("Cam")
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0, 0, 3)
    cam_obj.rotation_euler = (0, 0, 0)
    bpy.context.scene.camera = cam_obj

    light_data = bpy.data.lights.new("Flash", type="POINT")
    light_data.energy = 500.0
    light_obj = bpy.data.objects.new("Flash", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (0, 0, 3)

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


# --- One sample -------------------------------------------------------------
def render_sample(index, out_dir, plane, light_obj):
    p = sample_paint_params(random.Random(index))
    tag = f"{index:06d}"
    out = lambda name: os.path.join(out_dir, f"{name}_{tag}.png")
    scene = bpy.context.scene

    scene.cycles.samples = PHOTO_SAMPLES
    scene.cycles.use_denoising = True
    set_view_transform("AgX")
    light_obj.hide_render = False
    apply_material(plane, build_paint_material(p))
    render_to(out("photo"))

    light_obj.hide_render = True
    scene.cycles.samples = MAP_SAMPLES
    scene.cycles.use_denoising = False

    set_view_transform("Standard")  # colour -> sRGB-encoded
    apply_material(plane, build_emission_material("M_BaseColor", basecolor_socket, p))
    render_to(out("basecolor"))

    set_view_transform("Raw")       # data -> stored exactly
    apply_material(plane, build_emission_material("M_Rough", roughness_socket, p))
    render_to(out("roughness"))
    mv = p["metallic"]
    apply_material(plane, build_constant_material("M_Metal", (mv, mv, mv, 1.0)))
    render_to(out("metallic"))
    apply_material(plane, build_emission_material("M_Normal", normal_socket, p,
                                                  remap_normal=True))
    render_to(out("normal"))

    light_obj.hide_render = False
    with open(os.path.join(out_dir, f"params_{tag}.json"), "w") as f:
        json.dump({"index": index, "seed": index, "version": 2, **p}, f, indent=2)
    return p


def already_done(index, out_dir):
    tag = f"{index:06d}"
    names = ["photo", "basecolor", "roughness", "metallic", "normal"]
    return (all(os.path.exists(os.path.join(out_dir, f"{n}_{tag}.png")) for n in names)
            and os.path.exists(os.path.join(out_dir, f"params_{tag}.json")))


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

    indices = list(range(args.start, args.start + args.count))
    todo = [i for i in indices if not already_done(i, out_dir)]
    print(f"[batch] output: {out_dir}")
    print(f"[batch] {len(todo)} to render, {len(indices) - len(todo)} already done")

    t0 = time.time()
    for n, i in enumerate(todo, start=1):
        render_sample(i, out_dir, plane, light_obj)
        el = time.time() - t0
        per = el / n
        print(f"[batch] {n}/{len(todo)} | {per:.1f}s each | "
              f"remaining ~{fmt(per * (len(todo) - n))}", flush=True)

    if todo:
        per = (time.time() - t0) / len(todo)
        print(f"\n[batch] done in {fmt(time.time() - t0)} ({per:.1f}s each)")
        print(f"[batch] 2000 samples would take ~{fmt(per * 2000)}")


if __name__ == "__main__":
    main()