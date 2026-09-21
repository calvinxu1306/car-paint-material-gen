"""
generate_dataset.py — render the full training set (batch version).

Same per-sample logic as export_maps.py, wrapped in a loop and made runnable
from the terminal so it can grind away without Blender's window open.

RUN IT (headless, from a terminal):

  Windows (adjust the version folder to match yours):
    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b -P generate_dataset.py -- --count 20

  Mac:
    /Applications/Blender.app/Contents/MacOS/Blender -b -P generate_dataset.py -- --count 20

The "--" matters: everything before it is for Blender, everything after is for
this script.

  --count N     how many samples to render (default 10)
  --start N     first sample index (default 0) - use this to add more later
  --out PATH    output folder (default ./dataset_output)
  --res N       resolution (default 256)
  --cpu         force CPU rendering

START SMALL. Run --count 20 first and read the reported timing before
launching thousands.

Already-finished samples are skipped, so if you interrupt it (Ctrl+C) you can
just run the same command again and it picks up where it stopped.
"""

import bpy
import colorsys
import random
import json
import os
import sys
import time
import argparse


# --- Command-line arguments -------------------------------------------------
def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", type=str, default="dataset_output")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


PHOTO_SAMPLES = 64
MAP_SAMPLES = 16
PLANE_SIZE = 2.5


# --- Paint parameters -------------------------------------------------------
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
    mat, nodes, links = fresh_material(name)
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)
    emission.inputs["Color"].default_value = rgba
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def build_normal_map_material(flake_scale, flake_strength):
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

    # Try to enable GPU rendering. Falls back to CPU if nothing is found.
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
                print(f"[setup] rendering on GPU via {backend}: "
                      f"{', '.join(d.name for d in gpus)}")
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
    """Seed == index, so sample 0042 is always the same paint, forever."""
    seed = index
    p = sample_paint_params(random.Random(seed))
    tag = f"{index:06d}"
    out = lambda name: os.path.join(out_dir, f"{name}_{tag}.png")

    scene = bpy.context.scene

    # Photo
    scene.cycles.samples = PHOTO_SAMPLES
    scene.cycles.use_denoising = True
    set_view_transform("AgX")
    light_obj.hide_render = False
    apply_material(plane, build_car_paint_material(**p))
    render_to(out("photo"))

    # Maps
    light_obj.hide_render = True
    scene.cycles.samples = MAP_SAMPLES
    scene.cycles.use_denoising = False

    set_view_transform("Standard")  # colour -> sRGB-encoded
    apply_material(plane, build_constant_map_material("M_BaseColor", p["base_color"]))
    render_to(out("basecolor"))

    set_view_transform("Raw")  # data -> stored exactly
    rv, mv = p["roughness"], p["metallic"]
    apply_material(plane, build_constant_map_material("M_Rough", (rv, rv, rv, 1.0)))
    render_to(out("roughness"))
    apply_material(plane, build_constant_map_material("M_Metal", (mv, mv, mv, 1.0)))
    render_to(out("metallic"))
    apply_material(plane, build_normal_map_material(p["flake_scale"], p["flake_strength"]))
    render_to(out("normal"))

    light_obj.hide_render = False

    with open(os.path.join(out_dir, f"params_{tag}.json"), "w") as f:
        json.dump({"index": index, "seed": seed, **p}, f, indent=2)
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


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    clear_scene()
    plane, light_obj = setup_scene()
    setup_render(args.res, args.cpu)

    indices = list(range(args.start, args.start + args.count))
    todo = [i for i in indices if not already_done(i, out_dir)]
    skipped = len(indices) - len(todo)

    print(f"[batch] output: {out_dir}")
    print(f"[batch] {len(todo)} to render"
          + (f", {skipped} already done (skipping)" if skipped else ""))

    t0 = time.time()
    manifest = os.path.join(out_dir, "manifest.jsonl")
    for n, i in enumerate(todo, start=1):
        p = render_sample(i, out_dir, plane, light_obj)
        with open(manifest, "a") as f:
            f.write(json.dumps({"index": i, **p}) + "\n")

        elapsed = time.time() - t0
        per = elapsed / n
        left = per * (len(todo) - n)
        print(f"[batch] {n}/{len(todo)} done | {per:.1f}s each | "
              f"elapsed {fmt(elapsed)} | remaining ~{fmt(left)}", flush=True)

    total = time.time() - t0
    if todo:
        per = total / len(todo)
        print(f"\n[batch] finished {len(todo)} samples in {fmt(total)} "
              f"({per:.1f}s each)")
        print(f"[batch] at that rate: 1000 samples ~= {fmt(per * 1000)}, "
              f"5000 ~= {fmt(per * 5000)}")
    else:
        print("[batch] nothing to do - everything was already rendered.")


if __name__ == "__main__":
    main()