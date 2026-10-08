# Car paint from photos

A neural network that looks at photos of a painted surface and recovers the
paint's **layers**: base colour, roughness and metalness, the statistics of its
metallic flakes, and the clear coat on top. The result is a material you can
put on any 3D object.

**Live demo:** https://calvinxu1306.github.io/car-paint-material-gen/
(validation samples the network never trained on, predicted vs. true,
rendered in three.js)

Most learned material-capture methods predict per-pixel maps (colour,
roughness, normals). Car paint is layered, and much of its look comes from
properties of whole layers: how big and how tilted the flakes are, and how
glossy and how tinted the clear coat is. This project predicts those
layer properties directly, as numbers a renderer understands.

## Status

| | |
|---|---|
| **v1 — metallic paint, one flash photo** | done. Results below, full log in [`docs/ablations.md`](docs/ablations.md) |
| **v2 — every paint type, a few photos** | in progress. Plan in [`docs/v2_plan.md`](docs/v2_plan.md) |

## v1 results (one flash photo, metallic paint)

Trained on 2,000 Blender-rendered paints. Scores are **skill**: how much of a
"predict the average" baseline's error the model removes (0% = learned
nothing, 100% = perfect). Validation split, Run 9 (`docs/ablations.md`).

| layer parameter | skill | typical error |
|---|---|---|
| flake strength (tilt) | 91% | 0.007 on a 0.12–0.40 range |
| flake size | 86% | 2.5 on a 75–150 range |
| clear-coat roughness | 68% | 0.011 on a 0.08–0.22 range |
| clear-coat weight, orange peel | ~0% | not recoverable from this capture |

What v1 showed:
- **Flake statistics survive physically correct paint.** Flakes sit under a
  smooth clear coat, and their statistics are still recovered.
- **Statistics, not individual flakes.** From one photo the network can tell
  *how* flaky the paint is, but not which way each flake points. That is a
  geometric ambiguity, not a training failure (Finding 11).
- **The capture limits what can be learned.** In a dark room the clear coat
  reflects only the flash, so its weight and ripple leave almost no trace.
  This motivated v2.

## v2: why one flash photo isn't enough

![Paint types under the flash and side lights](docs/figures/v2_paint_gallery.jpg)

Pearlescent and colour-shift paints get their colour from thin-film
interference, which depends on the angle between light and camera. A phone's
flash sits next to its lens, so that angle is ~0 and a flash photo shows only
one colour.

The bottom two rows above are a colour-shift paint and a plain metallic paint
tuned to look the same under the flash. They come apart once the light moves
to the side.

v2 keeps the camera still and adds a few side-lit photos (a second phone's
torch is enough). The network runs on each photo and pools what it sees. One
generator now covers solid, metallic, pearl, colour-shift and candy paint, in
gloss or matte. See [`docs/v2_plan.md`](docs/v2_plan.md) for the physics,
the experiments (with predictions written down before running them), and the
real-paint capture protocol.

## How it works

1. **Synthetic data** (`data/blender_gen/`). Blender's Principled BSDF builds
   each paint as layers:
   - a base coat, with procedural flakes as per-cell normal tilts;
   - for pearl and colour-shift, a thin film on the base;
   - a clear coat with orange peel and, for candy, a tint.

   Each paint is photographed under a flash (and, in v2, side lights). The
   ground-truth maps are rendered from the same shader nodes, so they cannot
   disagree with the photo.
2. **Network** (`src/models/model.py`). A U-Net with a global-feature track
   (Deschaintre et al. 2018) predicts the per-pixel maps. A separate head
   reads the global vector and predicts the layer parameters. v2 runs the
   same trunk on each photo and max+mean pools across photos (Deschaintre et
   al. 2019; Deep Sets), plus a pigment-type head.
3. **Evaluation** (`src/eval_baseline.py`, `src/eval_multi.py`). Skill per
   map channel and layer parameter on held-out test sets. In v2 it is also
   reported per photo set and per paint type.
4. **Demo** (`demo/`). three.js `MeshPhysicalMaterial` with clear coat,
   deployed by GitHub Actions.

## Repository layout

```
data/blender_gen/     Blender generators (v1-v5), paint gallery
src/data/             PyTorch datasets (single photo; multi-photo)
src/models/           CarPaintNet (v1), MultiLightPaintNet (v2), renderer
src/train.py          v1 training         src/train_multi.py   v2 training
src/eval_baseline.py  v1 evaluation       src/eval_multi.py    v2 evaluation
src/export_demo.py    writes demo assets from a checkpoint
demo/                 three.js viewer (GitHub Pages)
docs/                 plan, design, reading notes, ablations, v2 plan
```

## Running it

Requirements: Python 3.11, PyTorch, Pillow, NumPy; Blender 5.x for data
(thin film on metals needs 5.0+).

```
# v1: render, check, train, evaluate
blender -b -P data/blender_gen/generate_dataset_v3.py -- --count 2000 --out data/blender_gen/dataset_v3
python src/data/dataset.py --root data/blender_gen/dataset_v3
python src/train.py --root data/blender_gen/dataset_v3 --out runs/v3_joint
python src/eval_baseline.py --root data/blender_gen/dataset_v3 --ckpt runs/v3_joint/best.pt

# v2: see docs/v2_plan.md section 5 for the full sequence
```

## Documents

- [`docs/design.md`](docs/design.md): the original design and what changed
- [`docs/ablations.md`](docs/ablations.md): every run, findings and audits
- [`docs/v2_plan.md`](docs/v2_plan.md): v2 physics, experiments and protocol
- [`docs/reading_notes.md`](docs/reading_notes.md),
  [`docs/reading_notes_v2.md`](docs/reading_notes_v2.md): sources

## Credits

Car model in the demo: *FREE Concept Car 004* by Unity Fan (@unityfan777),
CC0, via [Sketchfab](https://sketchfab.com/3d-models/free-concept-car-004-public-domain-cc0-4cba124633eb494eadc3bb0c4660ad7e).
Key references: Deschaintre et al. 2018 and 2019; Günther et al. 2005;
Belcour & Barla 2017. The full list is in the reading notes.
