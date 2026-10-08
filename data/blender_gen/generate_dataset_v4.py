"""
generate_dataset_v4.py — v3 paint, with the flash moved around the frame.

WHY V4
v2 and v3 always put the light exactly at the camera, so the specular hotspot
sat dead centre in every training photo. The network could learn a shortcut
("the bright blob is always in the middle"), and when that shortcut is
imperfect it bakes a blob into the centre of the predicted base colour - which
tiling then repeats across the demo sphere.

What v4 does and does not model: a phone's flash is ~1 cm from its lens, so in
a real photo the hotspot leaves the centre mainly when the phone is TILTED.
v4 keeps the camera square to the sample and moves the light sideways instead
(Deschaintre's setup). That breaks the fixed-position shortcut, but it is not
the geometry of a tilted phone, which would also give an oblique view.

Deschaintre et al. 2018 render their training photos this way: "The light is a
small white emitting sphere positioned in a plane parallel to the material
sample, at a random offset from the camera center." v4 does the same: the light
stays at the camera's height but moves up to +/-1.6 units sideways, which puts
the hotspot anywhere within the central ~75% of the frame (for a flat surface
the hotspot lands halfway between the camera and the light).

WHAT DID NOT CHANGE
The paint. Sample i in v4 is the same paint as sample i in v3 (the light offset
is drawn AFTER all the paint parameters, so the random stream is unchanged).
The ground-truth maps don't depend on the light at all, so:

FAST PATH (recommended): reuse v3's maps, render only the photos (~4x faster)
    blender -b -P generate_dataset_v4.py -- --count 2 --out dataset_v4 --reuse-maps dataset_v3
Each sample's paint is checked against v3's params file before its maps are
copied; any mismatch stops the run rather than pairing a photo with the wrong
answer key. Then the full set:
    blender -b -P generate_dataset_v4.py -- --count 2000 --out dataset_v4 --reuse-maps dataset_v3

Without --reuse-maps everything is rendered from scratch, as in v3.
"""

import bpy
import colorsys
import random
import json
import os
import sys
import time
import argparse
import shutil


PHOTO_SAMPLES = 64
MAP_SAMPLES = 32
PLANE_SIZE = 2.5
DATASET_VERSION = 4

# Light position: same height as the camera (a plane parallel to the sample,
# as in Deschaintre et al.), random sideways offset up to +/-LIGHT_OFFSET.
LIGHT_HEIGHT = 3.0
LIGHT_OFFSET = 1.6
MAP_NAMES = ["basecolor", "roughness", "metallic", "normal", "coat_normal"]

# Every sampled parameter and its range, in one place. Written to meta.json so
# the loader normalises with exactly these numbers.
RANGES = {
    # base coat
    "metallic": (0.7, 1.0),
    "roughness": (0.2, 0.5),
    # clear coat - roughness brackets Guenther Table 1 (sqrt(m3) ~ 0.11-0.18)
    "coat_weight": (0.8, 1.0),
    "coat_roughness": (0.08, 0.22),
    # flakes (base layer): scale sets cell size (~2-4 px at 256), strength tilt
    "flake_scale": (75.0, 150.0),
    "flake_strength": (0.12, 0.40),
    # orange peel (coat layer): slow ripple of the coat surface
    "peel_scale": (4.0, 14.0),
    "peel_strength": (0.02, 0.09),
    # spatial variation in the base colour
    "color_var_amount": (0.05, 0.35),
    "color_var_scale": (2.0, 8.0),
    # surface wear
    "scratch_amount": (0.04, 0.18),
    "scratch_scale": (40.0, 140.0),
    "dust_amount": (0.03, 0.15),
    "dust_scale": (2.0, 9.0),
    "noise_w": (0.0, 1000.0),
}


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", type=str, default="dataset_v3")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--reuse-maps", type=str, default=None,
                    help="v3 folder to copy ground-truth maps from (photos only)")
    return ap.parse_args(argv)


# --- Parameters -------------------------------------------------------------
def sample_paint_params(rng):
    # Colour first, in HSV, so paints look like paints rather than random RGB.
    hue, sat, val = rng.uniform(0, 1), rng.uniform(0, 0.9), rng.uniform(0.02, 0.9)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    p = {"base_color": (r, g, b, 1.0)}
    for key, (lo, hi) in RANGES.items():
        p[key] = rng.uniform(lo, hi)
    return p


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
    """BASE-layer normal: a random tilt per Voronoi cell (the metallic flakes).

        n = normalise( flake_strength * (2*cellColour.xy - 1),  z = 1 )
    """
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
    """COAT-layer normal: a slow low-frequency ripple (orange peel).

        n = normalise( peel_strength * (2*noiseColour.xy - 1),  z = 1 )
    """
    peel = noise_node(nodes, p["peel_scale"], p["noise_w"] + 53.0,
                      location=(-1100, -700))
    centred = centre_vector(nodes, links, peel.outputs["Color"], (-900, -700))
    scaled = vec_node(nodes, "SCALE", (-720, -700))
    scaled.inputs["Scale"].default_value = p["peel_strength"]
    links.new(centred, scaled.inputs[0])
    return tilt_to_normal(nodes, links, scaled.outputs["Vector"], (-560, -700))


# --- Materials --------------------------------------------------------------
def build_paint_material(p):
    mat, nodes, links = fresh_material("CarPaintV3")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (0, 0)
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    links.new(basecolor_socket(nodes, links, p), principled.inputs["Base Color"])
    links.new(roughness_socket(nodes, links, p), principled.inputs["Roughness"])
    # The fix: flakes go to the base layer, peel to the coat. Coat Normal MUST be
    # connected, otherwise Cycles copies the base (flake) normal onto the coat.
    links.new(flake_normal_socket(nodes, links, p), principled.inputs["Normal"])
    links.new(peel_normal_socket(nodes, links, p), principled.inputs["Coat Normal"])
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
OUTPUT_NAMES = ["photo", "basecolor", "roughness", "metallic", "normal",
                "coat_normal"]


def same_paint(p, q):
    """True if two params dicts describe the same paint (ignores lighting)."""
    keys = list(RANGES) + ["base_color"]
    for k in keys:
        a, b = p.get(k), q.get(k)
        if isinstance(a, (list, tuple)):
            if len(a) != len(b) or any(abs(x - y) > 1e-9 for x, y in zip(a, b)):
                return False
        elif a is None or b is None or abs(a - b) > 1e-9:
            return False
    return True


def render_sample(index, out_dir, plane, light_obj, reuse_dir=None):
    rng = random.Random(index)
    p = sample_paint_params(rng)
    # Drawn AFTER the paint, so the paint is identical to v3's sample `index`.
    light = {"light_x": rng.uniform(-LIGHT_OFFSET, LIGHT_OFFSET),
             "light_y": rng.uniform(-LIGHT_OFFSET, LIGHT_OFFSET)}
    tag = f"{index:06d}"
    out = lambda name: os.path.join(out_dir, f"{name}_{tag}.png")
    scene = bpy.context.scene

    if reuse_dir:
        # Check the answer key really belongs to this paint BEFORE rendering.
        src_params = os.path.join(reuse_dir, f"params_{tag}.json")
        if not os.path.exists(src_params):
            raise RuntimeError(f"--reuse-maps: {src_params} not found")
        with open(src_params) as f:
            if not same_paint(p, json.load(f)):
                raise RuntimeError(
                    f"--reuse-maps: sample {tag} in {reuse_dir} is a different paint. "
                    f"Its maps would be the wrong answer key; stopping.")

    scene.cycles.samples = PHOTO_SAMPLES
    scene.cycles.use_denoising = True
    set_view_transform("AgX")
    light_obj.hide_render = False
    light_obj.location = (light["light_x"], light["light_y"], LIGHT_HEIGHT)
    apply_material(plane, build_paint_material(p))
    render_to(out("photo"))

    if reuse_dir:
        for name in MAP_NAMES:
            shutil.copyfile(os.path.join(reuse_dir, f"{name}_{tag}.png"), out(name))
        with open(os.path.join(out_dir, f"params_{tag}.json"), "w") as f:
            json.dump({"index": index, "seed": index, "version": DATASET_VERSION,
                       **p, **light, "maps_from": os.path.basename(reuse_dir)},
                      f, indent=2)
        return p

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
    apply_material(plane, build_emission_material("M_Normal", flake_normal_socket, p,
                                                  remap_normal=True))
    render_to(out("normal"))
    apply_material(plane, build_emission_material("M_CoatNormal", peel_normal_socket,
                                                  p, remap_normal=True))
    render_to(out("coat_normal"))

    light_obj.hide_render = False
    with open(os.path.join(out_dir, f"params_{tag}.json"), "w") as f:
        json.dump({"index": index, "seed": index, "version": DATASET_VERSION,
                   **p, **light}, f, indent=2)
    return p


def already_done(index, out_dir):
    tag = f"{index:06d}"
    return (all(os.path.exists(os.path.join(out_dir, f"{n}_{tag}.png"))
                for n in OUTPUT_NAMES)
            and os.path.exists(os.path.join(out_dir, f"params_{tag}.json")))


def write_meta(out_dir):
    """Record exactly how this dataset was generated. dataset.py reads the
    ranges from here, so changing RANGES above can't desync the loader."""
    meta = {
        "version": DATASET_VERSION,
        "generator": os.path.basename(__file__),
        "ranges": {k: list(v) for k, v in RANGES.items()},
        "base_color": "HSV: hue U(0,1), sat U(0,0.9), val U(0.02,0.9)",
        "normal_png": "base-layer flake normal, xyz stored as v*0.5+0.5",
        "coat_normal_png": "coat-layer orange-peel normal, same encoding",
        "resolution": bpy.context.scene.render.resolution_x,
        "lighting": {
            "camera": [0, 0, 3.0],
            "light_height": LIGHT_HEIGHT,
            "light_x": [-LIGHT_OFFSET, LIGHT_OFFSET],
            "light_y": [-LIGHT_OFFSET, LIGHT_OFFSET],
            "note": "light in a plane parallel to the sample at a random offset "
                    "(Deschaintre et al. 2018); hotspot lands halfway between "
                    "camera and light",
        },
    }
    path = os.path.join(out_dir, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old.get("ranges") != meta["ranges"]:
            raise RuntimeError(
                f"{path} was written with different RANGES. Mixing two "
                f"distributions in one folder would corrupt the dataset - "
                f"use a new --out folder.")
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

    reuse_dir = os.path.abspath(args.reuse_maps) if args.reuse_maps else None
    if reuse_dir:
        # Maps can only be reused from a dataset built with the same paint
        # ranges at the same resolution.
        with open(os.path.join(reuse_dir, "meta.json")) as f:
            src_meta = json.load(f)
        if src_meta.get("ranges") != {k: list(v) for k, v in RANGES.items()}:
            raise RuntimeError(f"{reuse_dir} was generated with different paint ranges")
        if src_meta.get("resolution") != args.res:
            raise RuntimeError(f"{reuse_dir} is {src_meta.get('resolution')} px, "
                               f"this run is {args.res} px")

    clear_scene()
    plane, light_obj = setup_scene()
    setup_render(args.res, args.cpu)
    write_meta(out_dir)

    indices = list(range(args.start, args.start + args.count))
    todo = [i for i in indices if not already_done(i, out_dir)]
    print(f"[batch] output: {out_dir}")
    print(f"[batch] {len(todo)} to render, {len(indices) - len(todo)} already done"
          + (f" | maps reused from {reuse_dir}" if reuse_dir else ""))

    t0 = time.time()
    for n, i in enumerate(todo, start=1):
        render_sample(i, out_dir, plane, light_obj, reuse_dir)
        el = time.time() - t0
        per = el / n
        print(f"[batch] {n}/{len(todo)} | {per:.1f}s each | "
              f"remaining ~{fmt(per * (len(todo) - n))}", flush=True)

    if todo:
        per = (time.time() - t0) / len(todo)
        print(f"\n[batch] done in {fmt(time.time() - t0)} ({per:.1f}s each)")
        print(f"[batch] 2000 samples would take ~{fmt(per * 2000)}")
    print("\nCHECK before a full run:")
    print("  photo_*       the hotspot is OFF-centre (different place per sample),")
    print("                at the position light_x/2, light_y/2 in the params file")
    print("  other files   identical to the same-numbered v3 files")


if __name__ == "__main__":
    main()
