# Ablations and Results

Running log of every training run and what it showed. Numbers are **skill**:
how much of a mean-predictor baseline's L1 error the model removes, measured on
the held-out validation split (the last 10% of samples, fixed because seed ==
index). 0% = no better than predicting the dataset average; 100% = perfect.

Skill is a development sanity check (an L1 analogue of R²), not a headline
metric — the report should also carry PSNR/SSIM and a re-rendered comparison.

> **Read the audit section at the bottom first.** A 2026-09-27 audit found that
> the v2 dataset put the flake normals on the clear coat as well as the base
> layer, which is not physically how paint works. Runs 2 and 3 are kept as
> recorded, but their findings are annotated and must be re-measured on v3.

---

## Run 1 — v1 data, maps only

Dataset `dataset_output` (2000 samples), 50 epochs, L1 on 8 map channels.

| channel | mean pred | model | skill |
|---|---|---|---|
| basecolor_r | 0.2000 | 0.0144 | 92.8% |
| basecolor_g | 0.1925 | 0.0142 | 92.6% |
| basecolor_b | 0.1869 | 0.0184 | 90.2% |
| roughness | 0.0731 | 0.0047 | 93.6% |
| metallic | 0.0724 | 0.0101 | 86.0% |
| normal_x | 0.0022 | 0.0023 | −4.9% |
| normal_y | 0.0022 | 0.0023 | −6.4% |
| normal_z | 0.0006 | 0.0004 | 40.8% |
| **overall** | **0.0912** | **0.0083** | **90.8%** |

**Finding: the headline number is misleading.** The normal channels show a
mean-predictor error of 0.0022 — the naive baseline is already almost perfect,
so there was nothing to learn. The v1 materials were spatially uniform, the
flake cells were 1–2.5 px wide (Voronoi scale 120–300; cell ≈ 296/scale px),
and the tilts came from a Bump node with strength 0.05–0.25, so they were tiny
to begin with. 90.8% overall reflects easy targets, not a strong model.

*Audit note:* the normal-channel baselines (0.0022, 0.0006) are **below one
8-bit quantisation step** (1/255 ≈ 0.0039). The normal skill numbers in this
run (−4.9%, −6.4%, 40.8%) are measurement noise and mean nothing either way.

Validation loss plateaued (best 0.008348 at epoch 45, 0.008791 at epoch 50):
converged, not diverging.

---

## Run 2 — v2 data, maps only

Dataset `dataset_v2` (2000 samples), 50 epochs, same loss and model as Run 1.

v2 changes: flake normals set analytically per Voronoi cell rather than through
a Bump node, flake scale 75–150 (cells ≈2–4 px), plus orange peel, spatially
varying base colour, and roughness scratches/dust.

| channel | mean pred | model | skill |
|---|---|---|---|
| basecolor_r | 0.2000 | 0.0325 | 83.7% |
| basecolor_g | 0.1925 | 0.1065 | 44.7% |
| basecolor_b | 0.1869 | 0.0261 | 86.0% |
| roughness | 0.0732 | 0.0142 | 80.7% |
| metallic | 0.0724 | 0.0183 | 74.7% |
| normal_x | 0.0468 | 0.0367 | 21.7% |
| normal_y | 0.0468 | 0.0425 | 21.5% |
| normal_z | 0.0075 | 0.0066 | 11.2% |
| **overall** | **0.1033** | **0.0347** | **66.4%** |

**Finding: the lower overall number is the better result.** Mean-predictor error
on the normals rose from 0.0022 to 0.0468 — 20× more signal present — and the
model went from negative skill to ~21%. Predicted normals show flake speckle
rather than mush; the model captures flake *statistics*, not the exact per-cell
orientations, which is the right expectation for a single-image method.

`basecolor_g` at 44.7% was an outlier here (R and B at 84–86%). Hypothesised as
the AgX tone curve skewing green in bright regions, but **Run 3 did not
reproduce it** (all three channels landed together), so it was probably
run-to-run variance rather than a systematic effect. Unresolved — and a warning
that single-seed differences of tens of points in base colour are possible.

*Audit notes:* `normal_z`'s baseline (0.0075) is under two quantisation steps,
so its skill is unreliable. It is also redundant — z is determined by x and y
for a unit normal, which is why Deschaintre et al. predict only 2 normal
channels. Read normals from x and y. And in this dataset the normal map was
flakes + peel, applied to both the base layer and the clear coat (see audit).

---

## Run 3 — v2 data, maps + layer-parameter head

Dataset `dataset_v2`, 50 epochs, L1 on maps + L1 on 5 layer scalars, equal
weights. Model gains a second head reading the layer parameters off the global
feature vector rather than the per-pixel decoder.

### Per-pixel maps

| channel | mean pred | model | skill |
|---|---|---|---|
| basecolor_r | 0.2000 | 0.0873 | 56.3% |
| basecolor_g | 0.1925 | 0.0870 | 54.8% |
| basecolor_b | 0.1869 | 0.0850 | 54.5% |
| roughness | 0.0732 | 0.0147 | 79.9% |
| metallic | 0.0724 | 0.0207 | 71.4% |
| normal_x | 0.0468 | 0.0410 | 12.5% |
| normal_y | 0.0468 | 0.0425 | 9.1% |
| normal_z | 0.0075 | 0.0049 | 34.0% |
| **overall** | **0.1033** | **0.0479** | **53.6%** |

### Layer parameters (the contribution)

| parameter | mean pred | model | skill | error in real units |
|---|---|---|---|---|
| coat_weight | 0.2565 | 0.2577 | −0.5% | 0.0515 (range 0.8–1.0) |
| coat_roughness | 0.2492 | 0.0384 | **84.6%** | 0.0035 (range 0.01–0.1) |
| flake_scale | 0.2391 | 0.0346 | **85.5%** | 2.60 (range 75–150) |
| flake_strength | 0.2745 | 0.0251 | **90.9%** | 0.0070 (range 0.12–0.40) |
| peel_strength | 0.2732 | 0.2714 | 0.7% | 0.0190 (range 0.02–0.09) |
| **overall** | **0.2585** | **0.1254** | **51.5%** |

**Finding 1 (provisional) — three of five layer parameters were recoverable
on v2 data.** Flake size, flake strength and clearcoat roughness landed at
84–91% skill, with errors of a few percent of their range.

*Audit caveat — likely inflated.* In v2 the clear coat carried the flake
tilts (see audit), so the sharp coat glinted off every flake cell across the
whole image. That makes both flake statistics and coat sharpness unusually
visible, far more than with a smooth coat over flakes. On v2 data the coat and
flake layers are physically entangled, so this run cannot support a
"layers disentangled" claim. Re-measure on v3 before quoting these numbers.

*Positioning:* none of the learning-based SVBRDF papers in the reading list
(Deschaintre 2018, MaterialGAN 2020, Highlight-Aware 2021) predicts these
parameters, and the car-paint papers (Günther 2005, Ershov 2001, Kneiphof 2022)
obtain them by measurement or simulation. But predicting a procedural
generator's parameters from an image is **inverse procedural material
modelling**, an established line (Hu et al. 2019, 2022; MATch, Shi et al.
2020; Li et al. 2023). The honest claim is a car-paint-specific instance of
that, not a new problem.

**Finding 2 — the two failures: one explanation supported, one revised.**
- `coat_weight` (sampled 0.8–1.0): **supported.** The coat's highlight is a
  small blob that clips — 32–68 pixels at 255 in measured photos. A 0.8 vs 1.0
  coat changes that peak's intensity, which is exactly what clipping erases.
  Coat *roughness* changes the blob's size and falloff, which survive outside
  the clipped core; that is consistent with roughness being learned and weight
  not. (Coat roughness 0.01–0.1 is also far sharper than real paint — see
  audit — which makes the blob smaller and the clipping worse.)
- `peel_strength`: **the original explanation described an artefact.** Peel
  and flakes were summed into one normal applied to both layers, so peel was a
  small low-frequency term on top of large flake tilts. In real paint peel
  lives only on the coat. v3 routes it there; peel's observability is an open
  question until re-measured.

**Finding 3 — multi-task cost: not established.** Base-colour skill was ~85%
in Run 2 and ~55% here, but the two runs also differ in code path, data-shuffle
order (the extra head consumes RNG before the loader shuffles), and are one
seed each — while Run 2 alone showed a 40-point spread across colour channels.
The controlled comparison is the same code with `--scalar-weight 0` vs `1`,
ideally over 2–3 seeds.

---

## Not yet run

- **v3 controlled pair** (highest priority): `--scalar-weight 0` vs default on
  `dataset_v3`, same code, same seed. Replaces Runs 2–3 as the reportable
  result. v3 sample *i* is the same paint as v2 sample *i* except coat roughness
  and the layer routing, so v2 → v3 is also a paired comparison.
- extra seeds for that pair (`--seed 1`, `--seed 2`), to size run-to-run noise
- `--scalar-weight 0.5` — recover map accuracy while keeping layer parameters
- `--render-weight 1` — with/without the Deschaintre rendering-aware loss
  (implemented in `src/models/render.py`, off by default). Note its renderer is
  plain Cook-Torrance with no clearcoat or flake layer, so it approximates the
  appearance of the three-layer material it is supervising.
- widened `coat_weight` range, to test the appearance-invariance explanation
- photo pass rendered with `Standard` instead of `AgX`, to test whether the tone
  curve costs base-colour accuracy

## Known limitations

- Flake density is capped by resolution: below ~2 px per cell flakes average to
  grey regardless of how they are modelled, so 256×256 cannot represent
  physically realistic flake density. 512×512 would allow finer flakes.
- Flake tilts are applied as if texture space and world space agree. True for a
  flat plane facing +Z; curved geometry would need a proper tangent frame.
- Trained and evaluated entirely on synthetic Blender renders. The
  synthetic-to-real gap is unmeasured (the planned real paint-chip test set
  was not captured).
- Every result is a single seed.
- The rendering loss's own renderer (`src/models/render.py`) treats image row 0
  as y = −1, while Blender's image row 0 is +y. It renders prediction and
  target identically and samples lights symmetrically, so the loss is
  unaffected, but its renders are vertically mirrored relative to the photos.
- Photos are denoised (Cycles denoiser at 64 samples), which may soften the
  finest flake sparkle — though real phone cameras denoise too.

---

## Audit — 2026-09-27

Full re-check of code, data, metrics and notes before moving to the demo.

**Errors found**

1. **Clear coat carried the flake normals (v2).** The flake+peel normal went
   into the Principled BSDF's base `Normal`; `Coat Normal` was left
   unconnected. In Cycles an unconnected Coat Normal defaults to the base
   normal `N` (`stack_load_float3_default(stack, coat_normal_offset, N)`,
   `intern/cycles/kernel/svm/closure.h`), not to the smooth surface normal.
   Physically, flakes sit under a smooth coat and only orange peel ripples the
   coat — Günther 2005 samples flake normals from the glitter lobe and keeps a
   separate sharp clear-coat lobe; Kneiphof & Klein 2022, Fig. 1 show the base
   material underneath the coat. **Fixed in `generate_dataset_v3.py`**: flakes
   → `Normal`, peel → `Coat Normal`; `normal_*.png` is now the flake normal and
   a new `coat_normal_*.png` holds the peel normal.
2. **Coat roughness far sharper than real paint.** Blender maps roughness to
   microfacet α = roughness² (both layers). v2's 0.01–0.1 is α 0.0001–0.01;
   Günther's measured clear coats have m₃ = 0.013–0.034 (Table 1), i.e. Blender
   roughness ≈ 0.11–0.18. The design doc's "ranges anchored to Günther" was not
   true for the coat. **Fixed in v3**: 0.08–0.22.
3. **Run 2 → Run 3 attributed to multi-task cost without a control.** See
   Finding 3. Fix: controlled pair on v3.
4. **`train.py` defaulted `--out` to `runs/v3`**, so a run without `--out`
   would have overwritten Run 3. **Fixed**: default is a new timestamped folder,
   and it refuses to write into a folder that already holds a run.
5. **Scalar ranges were copied by hand** between generator and loader, and had
   already drifted once. **Fixed**: v3 writes `meta.json`; the loader reads the
   ranges from it, and `train.py` records them in each run's `config.json`.
6. Minor: v1 cell size misstated (1–2.5 px, not "under 1.5"); v2 base colour
   could slightly exceed 1 (clamped in v3); the final-epoch errors `train.py`
   prints are not the `best.pt` numbers (now labelled; use `eval_baseline.py`).

**Checked and fine**

- Every self-test re-run: renderer (hotspot, roughness ordering, gradients),
  loader (sRGB decode, scalar round-trip), model (both heads reach layer 1,
  global track active), training with `--render-weight 1` and
  `--scalar-weight 0`, evaluation.
- The mean-predictor baselines match what the generator's distributions
  predict (e.g. base colour ≈ 0.205 simulated vs 0.19–0.20 measured), so the
  evaluation is measuring what it claims.
- Using the mean rather than the L1-optimal median as the baseline inflates
  skill by < 1 point here (distributions are near-symmetric). Negligible.
- Reading notes checked against all six PDFs: every specific claim tested held
  (Deschaintre's 9 channels, 155 graphs, 200k samples, log-L1 loss, clear-coat
  failure case; MaterialGAN's 3–7 inputs, ~2 min for 2000 iterations, stated
  limitations; Günther's 8 paints in Table 1 and black-felt studio; Kneiphof's
  K = 3 lobes, cubemap array and SVD; Ershov's gloss/glitter/shade).