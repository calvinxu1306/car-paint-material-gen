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
> recorded, but their findings are annotated. **Runs 4–5 (corrected v3 data),
> Runs 6–8 (moving flash) and Run 9 (moving flash, trained to convergence —
> the best model) are the results to report.**

> **v2 (from 2026-10-07):** runs on the multi-light v5 data will be logged
> here as Runs 10 onward. Their predictions were written down before
> running, in `v2_plan.md` §4.

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
- `coat_weight` (sampled 0.8–1.0): **supported at the time, but not
  sufficient — see Finding 5.** The coat's highlight is a
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
ideally over 2–3 seeds. (Done as Runs 4–5 below.)

---

## Runs 4 and 5 — v3 data, controlled pair (the reportable result)

Dataset `dataset_v3` (2000 samples): flakes on the base layer, orange peel on
the clear coat, coat roughness 0.08–0.22. Same code, same seed (0), 50 epochs.
The only difference between the two runs is the scalar loss weight.
Both best checkpoints came from late epochs (48 and 47 of 50), so both were
probably still improving slightly.

### Per-pixel maps

| channel | mean pred | Run 4: maps only | Run 5: maps + layer head | change |
|---|---|---|---|---|
| basecolor_r | 0.2000 | 81.6% | 77.7% | −3.9 |
| basecolor_g | 0.1925 | 79.0% | **50.4%** | **−28.6** |
| basecolor_b | 0.1869 | 77.4% | 79.2% | +1.8 |
| roughness | 0.0732 | 80.3% | 79.0% | −1.3 |
| metallic | 0.0724 | 73.6% | 58.8% | −14.8 |
| normal_x | 0.0467 | 12.3% | 12.2% | −0.1 |
| normal_y | 0.0467 | 10.5% | 7.6% | −2.9 |
| normal_z | 0.0075 | 28.0% | 34.8% | (unreliable) |
| **overall** | **0.1032** | **70.8%** | **62.1%** | **−8.7** |

### Layer parameters

| parameter | Run 4 skill | Run 5 skill | Run 5 error (real units) | v2 skill (Run 3) |
|---|---|---|---|---|
| coat_weight | 0.0% | 0.4% | 0.051 (range 0.8–1.0) | −0.5% |
| coat_roughness | −2.0% | **71.0%** | 0.010 (range 0.08–0.22) | 84.6% |
| flake_scale | −1.2% | **81.7%** | 3.3 (range 75–150) | 85.5% |
| flake_strength | 0.4% | **88.8%** | 0.0086 (range 0.12–0.40) | 90.9% |
| peel_strength | −1.1% | 1.2% | 0.019 (range 0.02–0.09) | 0.7% |

Run 4's layer scores near 0% are the control working as designed: with the
scalar loss off, the head never trains, so it does no better than the mean.

**Finding 4 — the layer parameters survive physically correct paint.** With
flakes under a smooth coat, flake strength (88.8%), flake scale (81.7%) and
clear-coat roughness (71.0%) are still recovered from a single flash photo.
This is the result to report. The drops from v2 match the audit's prediction
that v2 was inflated by coat glints: flakes barely moved (−2 to −4 points),
while coat roughness, whose signal in v2 was smeared across every flake glint,
fell 14 points. (Coat roughness's range also changed, so the comparison is
indicative, not exact.)

**Finding 5 — `coat_weight` and `peel_strength` are unobservable in this
capture setup, and the clipping explanation was not sufficient.** v3's
highlight no longer clips (0–7 pixels at 255, versus 32–68 in v1/v2), yet
`coat_weight` is still at 0%. So clipping was not the whole story. Revised
explanation, for both parameters: the scene is a black room lit by one flash
at the camera, so the clear coat reflects *only* the flash. Everything the coat
does is confined to the central highlight — a few hundred pixels whose
brightness the AgX tone curve also compresses. Orange peel is normally seen as
a wobble in reflected surroundings, and this scene has no surroundings to
reflect. Prediction: an environment map, or several lights, would make both
learnable. Untested.

**Finding 6 — the layer head's cost to the maps is modest, apart from one
unstable channel.** Under a controlled comparison the overall cost is −8.7
points, but most of it is `basecolor_g` (−28.6), and green has now collapsed in
**two of five runs**: Run 2 (maps only, v2) and Run 5 (joint, v3), but not
Run 3 (joint, v2) or Run 4 (maps only, v3). So the green collapse happens with
and without the layer head and cannot be pinned on it from one seed. Metallic
(−14.8) is the cleaner candidate for a real cost; the other channels move by
≤ 4 points. Settling it needs 2–3 seeds per configuration.

**Open problem — green channel instability.** Green carries most of an image's
luminance (≈ 72% under Rec. 709 weights), so one hypothesis is that the network
sometimes falls into predicting green from brightness rather than colour. Cheap
check: on the validation set, correlate predicted green with true green and
with photo luminance, for a collapsed run (5) against a healthy one (4).

---

## Runs 6–8 — moving the flash (lighting generalization)

**Why.** v2 and v3 always lit the sample from exactly the camera position, so
the specular hotspot sat dead centre in every training photo. (In a real phone
photo the flash is about 1 cm from the lens, so the hotspot moves off-centre
mainly when the phone is *tilted*. v4 instead keeps the camera square to the
sample and moves the light sideways — Deschaintre's setup. It breaks the
fixed-position shortcut, but it is not the same geometry as a tilted phone:
see Known limitations.) In the demo, the
Run 5 model's base-colour maps carried a residual blob at their centre, which
tiling repeated across the sphere — a sign the network had learned where the
hotspot *always is* rather than how to find it. Deschaintre et al. 2018 avoid
this by placing the light "in a plane parallel to the material sample, at a
random offset from the camera center".

**Data: `dataset_v4`.** Same 2000 paints as v3 (sample *i* is the identical
paint; the ground-truth maps are copied from v3 after a per-sample check), but
the light moves up to ±1.6 units sideways at the camera's height, so the
hotspot lands anywhere within the central ~75% of the frame (only 1.6% of
samples have it near the centre).

**Runs.** Run 6 = the Run 5 model (trained centred) evaluated on v4 photos.
Run 7 = a new model trained on v4 (`runs/v4_joint`, 50 epochs, same settings
as Run 5), evaluated on v4. Run 8 = the same v4 model evaluated on v3 photos.
The validation paints are identical across all four cells; only the lighting
differs, so the table is a paired comparison.

| trained on → tested on | maps overall | base colour (avg) | roughness | metallic | coat roughness | flake scale | flake strength |
|---|---|---|---|---|---|---|---|
| centred → centred (Run 5) | 62.1% | 69.1% | 79.0% | 58.8% | 71.0% | 81.7% | **88.8%** |
| centred → moved (Run 6) | 40.0% | 43.8% | 61.0% | 35.5% | 60.9% | 64.2% | **33.1%** |
| moved → moved (Run 7) | 43.2% | 50.6% | 75.3% | 6.7% | 67.1% | 80.9% | **86.8%** |
| moved → centred (Run 8) | 39.1% | 44.8% | 75.0% | 7.2% | 68.1% | 82.4% | **81.5%** |

`coat_weight` and `peel_strength` stay at ~0% in every cell (Finding 5).

**Finding 7 — the centred-flash model had learned a positional shortcut.**
Moving the flash costs it 22 points on the maps and collapses flake strength
from 88.8% to 33.1% (−56). Flake scale (−18), roughness (−18), metallic (−23)
and coat roughness (−10) all drop too. The model was reading the paint relative
to where the hotspot always was, not where it is.

**Finding 8 — training with a moving flash makes the layer parameters
robust.** The v4 model recovers flake strength (86.8%), flake scale (80.9%) and
coat roughness (67.1%) with a moving flash — essentially matching the centred
model on its own easy case — and keeps them when tested on centred photos
(81.5 / 82.4 / 68.1%). Its layer-parameter skill changes by about 5 points
or less between lighting conditions, against 56 for the centred model. This is the
result that matters for real photos, where the flash is never exactly centred.

**Finding 9 — the per-pixel maps got harder and are not yet solved.** The v4
model's maps sit at ~40%: roughness holds (75%), base colour is ~45–50%, and
two channels collapsed to predicting the mean — metallic (6.7%) and the flake
normals (x/y ≈ 0%). Its best checkpoint was the **final** epoch (50 of 50), so
validation loss was still falling when training stopped: it is under-trained
for the harder task, not shown to be incapable of it. Next step: train longer
with a decaying learning rate before drawing conclusions about the maps.
(Done as Run 9: mostly under-training, except the flake normals.)

---

## Run 9 — moving flash, trained longer

`runs/v4_long`: same data and settings as Run 7, but 150 epochs with a cosine
learning-rate schedule (2e-4 decaying to 4e-6; `--schedule cosine`).
Evaluated on v4 (moving flash). Re-evaluating Run 7's checkpoint in the same
session reproduced its table to the last digit, so the evaluation is
deterministic and the comparison is clean.

| channel | Run 7 (50 epochs) | Run 9 (150, cosine) | Run 5 (centred model, centred test) |
|---|---|---|---|
| maps overall | 43.2% | **57.1%** | 62.1% |
| base colour R / G / B | 54.8 / 53.9 / 43.2% | 72.7 / 57.5 / 53.9% | 77.7 / 50.4 / 79.2% |
| roughness | 75.3% | 80.1% | 79.0% |
| metallic | 6.7% | **74.7%** | 58.8% |
| normal x / y | 0.0 / −0.1% | **−0.1 / −0.1%** | 12.2 / 7.6% |
| coat roughness | 67.1% | 68.2% (err 0.011) | 71.0% |
| flake scale | 80.9% | **86.1%** (err 2.5) | 81.7% |
| flake strength | 86.8% | **91.1%** (err 0.0068) | 88.8% |
| coat weight / peel | ~0% | ~0% | ~0% |

**Finding 10 — Finding 9 was mostly under-training.** With longer training,
metallic recovers from 6.7% to 74.7%, base colour and roughness rise, and the
maps overall reach 57.1%. Flake strength (91.1%) and flake scale (86.1%) are
now the best of any run. Run 9 is the best model in this project.

*Audit caveat (2026-10-07):* comparing Run 9 with Run 5 ("within 5 points of
the centred model", "metallic better than the centred model's 58.8%") is not a
fair fight — Run 9 had three times the training plus a decaying learning rate,
and the centred model never got that. A centred model trained the same way
might also improve. And Run 9's best epoch was not recorded (the header line
of that evaluation was not captured), so "trained to convergence" is
unverified; if best.pt is again the final epoch, it was still improving.

**Finding 11 — flake statistics are recoverable; individual flakes are not.**
The per-pixel flake normals stay at exactly the mean-predictor level (x/y skill
≈ 0%) even after 150 epochs, while flake *strength* and *scale* — the
statistics of the same flakes — are recovered at 91% and 86%. A flake's
brightness depends only on the *angle* between its normal and the half-way
vector between light and camera at that pixel, so one brightness value fits a
whole ring of tilts: even with the lighting known exactly, the direction a
single flake leans is not recoverable from one photo. With the light always at
the camera (v3), the half-way vector at each pixel points back towards the
image centre, so bright flakes tend to lean towards the centre — a positional
prior the network could exploit for its ~10% skill, the same kind of shortcut
as Finding 7. Once the light moves, that prior disappears, and an L1 loss is
minimised by the median tilt, which for symmetric flakes is flat.
Interpretation: from one photo the network can tell *how* flaky the paint is
but not *which way each flake points*. A loss that rewards plausible texture over
per-pixel agreement (the perceptual/adversarial term of Guo et al. 2021, or the
rendering-aware loss) is the standard response; alternatively, the recovered
statistics can drive a procedural flake model, which only needs the
statistics.

---

## Runs 10 and 11 — v5 data: one flash photo vs. several lights

The first v2 runs. `dataset_v5` was rendered with the Phase 2 commands in
`v2_plan.md` (4000 samples, all five paint types, a flash photo plus five
side-lit photos each). Both runs: 150 epochs, cosine schedule, batch 4, full
precision, on an RTX 4070. They are identical except for `--photos`:

- **Run 10** (`runs/v5_flash`, `--photos flash`): the flash photo only.
  ~148 s per epoch. Best epoch not recorded.
- **Run 11** (`runs/v5_multi`, `--photos random`): a random 1–6 of each
  sample's photos every batch. ~250 s per epoch. Best epoch 117 (validation
  loss 0.2489 with all photos).

Both scored on the held-out test set `dataset_v5_test` (400 samples, seeds
from 100000) with `eval_multi.py`. Run 10 has only a flash row: the evaluation
also scored it on side-lit photo sets it never trained with, which is
meaningless (a bug in `eval_multi.py`, now fixed — it took the photo limit
from `config.json`, where the dataset's photo count is recorded even for a
flash-only run).

| photos shown | maps | coat wt | coat rgh | tint (RGB) | flk size | flk str | peel | film d | film n | pigment acc. |
|---|---|---|---|---|---|---|---|---|---|---|
| **Run 10**, flash | **64.8%** | 3.1% | **62.5%** | 32–34% | **79.3%** | 81.1% | −2.5% | 77.1% | 8.1% | **86%** |
| Run 11, flash | 54.5% | 2.5% | 16.4% | 31–33% | 45.6% | 40.3% | −3.1% | 45.9% | −3.0% | 69% |
| Run 11, side1 | 48.9% | 1.7% | 16.6% | 31–33% | 41.9% | 78.9% | −5.0% | 54.8% | −1.6% | 75% |
| Run 11, flash+side1 | 59.9% | 3.6% | 17.4% | 31–32% | 56.0% | 80.2% | −3.5% | 75.3% | 3.4% | 82% |
| Run 11, flash+3 sides | 61.3% | 3.5% | 17.8% | 31–32% | 58.3% | 86.3% | −3.5% | 79.1% | 6.4% | 83% |
| Run 11, all 6 | 61.6% | 3.6% | 17.4% | 31–32% | 60.2% | **86.7%** | −3.5% | 78.7% | 6.4% | 83% |

Pigment accuracy is type accuracy, not skill; always guessing the commonest
type scores 26%. Real-unit errors, Run 10 (flash) vs. Run 11 (all 6): coat
roughness 0.045 vs. 0.100 (range 0.08–0.7), flake scale 4.0 vs. 7.6 (75–150),
flake strength 0.019 vs. 0.014 (0–0.4), film thickness 36 vs. 34 nm (0–700).

**Finding 12 — the multi-light model is not better than a flash-only model.**
Run 10, shown one flash photo, beats Run 11 shown all six on the maps, coat
roughness (62.5 vs. 17.4%), flake size (79.3 vs. 60.2%) and paint type (86 vs.
83%). Run 11 wins only flake strength (86.7 vs. 81.1%) and film thickness
(78.7 vs. 77.1%, which Finding 13 discounts). Within Run 11 the side lights
do add a lot (flash → all 6: flake strength 40 → 87%, film thickness 46 →
79%, paint type 69 → 83%), but mostly because Run 11 reads the flash photo
far worse than Run 10 does: given the same single flash photo, it scores
16.4% on coat roughness against 62.5%, and 40.3% on flake strength against
81.1%. Side photos do carry flake information of their own (side1 alone:
78.9% on flake strength), consistent with side lights making flakes glint.

**Finding 13 — two overall scores are mostly the paint type.** Film thickness
scores 78.7% overall, but within each type with a film it is −8.8% (pearl)
and +5.1% (colour-shift) in Run 11, and −9.1% / +1.4% in Run 10. Pearl and
colour-shift films are drawn from different ranges (100–400 vs. 250–700 nm),
so recognising the type earns nearly all of the skill; the thickness itself
is not being measured. Coat tint scores 31–34% overall in every row of both
runs, yet −163 to −187% within candy, the only paint with a tint: the network
correctly predicts "no tint" for the other four types and gets candy's tint
badly wrong. A silver base under a tinted coat looks much like a coloured
metallic base, and the confusions show it: candy is called metallic in 17 of
88 test samples (Run 10) and 29 of 88 (Run 11, all 6), and about a quarter of
metallic paints are called candy in both runs. The colour presumably ends up
in the base-colour map instead of the tint. The one tint-related signal that
does come through is candy's coat weight (36.9% skill within candy in Run 11,
20.5% in Run 10), which sets how strong the tint looks.

**Finding 14 — the predictions in `v2_plan.md` §4 mostly failed.**

| prediction | outcome |
|---|---|
| film_ior ~0% from the flash, clearly > 0 with side lights | 8.1% (Run 10) vs. 6.4% (Run 11, all 6); within type −14 to +11%. Not learned either way. |
| coat_weight > 0 with side lights | 3.6%. Not confirmed, even with the brightness fix (Audit 3); only candy's is learnable (Finding 13). |
| peel_strength small gain at most | −3.5%. As predicted (still no environment to reflect). |
| pearl/colour-shift vs. metallic confusions drop with side lights | Partly: in Run 11, colour-shift is identified in 22 of 54 test samples from the flash and 51 of 54 with all six. But Run 10 already gets 47 of 54 from the flash alone, so the side lights mostly make up for Run 11's weak reading of the flash photo. Candy ↔ metallic stays the main confusion. |
| flake size / strength, coat roughness "as v1 (70–90%)" | Run 10: yes for flakes (79 / 81%), coat roughness 62.5% (v1: 68%). Run 11: no (Finding 12). |

**Finding 15 (open) — why Run 11 reads the flash photo badly.** Not
under-training or over-fitting: its best epoch was 117 of 150, and at the end
train and validation loss were close (0.226 vs. 0.251) and flat. (Run 10 does
over-fit slightly: 0.158 vs. 0.295, validation loss creeping up over the last
epochs; `best.pt` keeps its best epoch.) Run 11's coat-roughness validation
error at the final epoch was 0.170 with all photos and 0.172 with the flash
alone (normalised),
against 0.077 for Run 10: it never learned to read coat roughness from any
input. Two explanations, not yet separated:

- **A: the flash photo was often missing in training.** `--photos random`
  leaves the flash photo out of ~42% of training samples. A side photo can
  still show the glossy coat's highlight, but off-centre: simulated with the
  generator's geometry, 42–47% of side photos have it in frame (a light high
  above the sample puts the coat's mirror reflection inside the picture), so ~12–14% of Run 11's
  training samples had no coat highlight in any photo. In the rest without
  the flash, the highlight's position and shape depend on a light the network
  is never told about. If learning from those samples is what held Run 11
  back, always including the flash photo should fix it. The ~12–14% figure
  is small, which makes A less likely than it first looked.
- **B: pooling hides which photo is the flash.** The layer-parameter head
  reads the max and the mean over photos of each photo's global vector, so the
  flash photo's evidence is mixed with up to five side photos', in
  proportions that change every batch, and nothing marks which photo is the
  flash. Giving the flash photo its own slot in the pooled vector would test
  this (`train_multi.py --flash-slot`, added 2026-10-09). Deschaintre et al.
  2019 report the same symptom — roughness "independent of the number of
  images" — for their max-pooled network (`reading_notes_v2.md` §4).

**Run 12 (next, predictions written before running):** `--photos
flash+random` — the flash photo in every sample plus a random 0–5 side
photos. That is the same photo-count spread as Run 11 (1–6 photos), and
matches a real capture, which always includes the flash photo. Everything
else as Run 11. If A holds, Run 12's flash row approaches Run 10's: coat
roughness ≥ 50% (Run 11: 16.4%), flake size ≥ 70%, flake strength ≥ 75%;
with all six photos, flake strength stays ≥ 85% and colour-shift
identification ≥ 50 of 54. If coat roughness from the flash stays below
~25%, A is ruled out and B is next. Either way, coat weight (except candy),
peel, film IOR and candy's tint are expected to stay near 0%: Findings 13–14
point to the capture, not the training, for those.

---

## Not yet run

- **Run 12** (v2, highest priority): `train_multi.py --photos flash+random`,
  everything else as Run 11 (but with `--amp`); predictions under Runs 10 and
  11. Then, depending on its outcome: if coat roughness recovers, add
  `--coords` (pixel coordinates as input channels, as Deschaintre et al.
  2019); if it does not, `--flash-slot` (Finding 15, B; Boss et al. 2020
  keep the flash photo in its own pathway). Then Runs 13–14 from
  `v2_plan.md` §4 (v1 initialisation; v1 code on v5 flash photos).
- v2 capture changes suggested by the literature (`reading_notes_v2.md`;
  each needs a re-render): HDR photos fed in log space, for `coat_weight`
  (Kaltheuner et al. 2021); fixed side-light slots instead of five random
  lights (same paper); one photo reflecting stripes, for `peel_strength`
  (how industry measures orange peel).
- **Held-out test sets** (highest priority for the report): 300 fresh samples
  per lighting condition at indices 2000+, scored with `--test-root`, so the
  reported numbers come from samples never used for training *or* for
  choosing checkpoints. Re-score Runs 5–9 on them; that table replaces the
  validation-split numbers above as the reportable result.
- Run 9's model tested on centred photos, to complete its row of the lighting
  table (fold into the held-out re-scoring)
- a centred model trained like Run 9 (150 epochs, cosine), to make the Run 9 vs
  Run 5 comparison budget-matched
- procedural flakes from predicted parameters (Finding 11): rebuild the flake
  normal map from predicted `flake_scale` / `flake_strength` with the same
  per-cell model the generator uses
- extra seeds for the Run 4 / Run 5 pair (`--seed 1`, `--seed 2`), to size
  run-to-run noise and settle Finding 6 and the green instability
- green-channel diagnostic (see open problem above)
- environment-lit capture (HDRI or multiple lights), to test Finding 5's
  explanation for `coat_weight` and `peel_strength`
- longer training (best checkpoints landed at epochs 45–48 of 50)
- `--scalar-weight 0.5` — recover map accuracy while keeping layer parameters
- `--render-weight 1` — with/without the Deschaintre rendering-aware loss
  (implemented in `src/models/render.py`, off by default). Note its renderer is
  plain Cook-Torrance with no clearcoat or flake layer, so it approximates the
  appearance of the three-layer material it is supervising.
- widened `coat_weight` range (0–1), a cheaper partial test of Finding 5
- photo pass rendered with `Standard` instead of `AgX`, to test whether the tone
  curve costs base-colour accuracy (and whether it affects the green collapse)

## Known limitations

- Flake density is capped by resolution: below ~2 px per cell flakes average to
  grey regardless of how they are modelled, so 256×256 cannot represent
  physically realistic flake density. 512×512 would allow finer flakes.
- Flake tilts are applied as if texture space and world space agree. True for a
  flat plane facing +Z; curved geometry would need a proper tangent frame.
- Trained and evaluated entirely on synthetic Blender renders. The
  synthetic-to-real gap is unmeasured (the planned real paint-chip test set
  was not captured).
- Every result is a single seed. Across runs trained on the same kind of data,
  the layer-parameter scores agree within ~5 points (flake strength 88.8 /
  86.8 / 91.1%), so their large effects are trustworthy; base colour, green
  especially, has ranged over 48 points, so base-colour differences of less
  than ~25 points between single runs should not be interpreted.
- Checkpoints are chosen on the same validation split the scores are reported
  on, which makes every number in Runs 1–9 slightly optimistic (see the
  held-out test sets under Not yet run).
- Camera tilt is never varied. Every photo is taken square to the sample. A
  handheld phone is rarely square, and tilting gives an oblique, perspective
  view with the flash still beside the lens — a condition none of the
  datasets contain.
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

---

## Audit 2 — 2026-10-07

Re-check of everything added since audit 1: the v3 and v4 data, Runs 4–9,
Findings 4–11, the cosine schedule and the demo.

**Errors and overclaims found**

1. **Checkpoint selection and reporting used the same samples.** `train.py`
   keeps the epoch with the lowest loss on the validation split, and
   `eval_baseline.py` reported skill on that same split. Every number in Runs
   1–9 is therefore slightly optimistic. **Fixed**: `eval_baseline.py` now
   takes `--test-root` to score a separate held-out folder, warns if its
   samples overlap the training data, and refuses if its scalar ranges differ.
   Held-out test sets are the top item under Not yet run.
2. **Finding 10 compared unequal training budgets** (Run 9: 150 epochs +
   cosine; Run 5: 50 epochs, constant) and claimed convergence without a
   recorded best epoch. **Annotated** in Finding 10.
3. **v4's stated motivation was physically off.** It said real photos have an
   off-centre hotspot because "the flash sits off the lens axis and the phone
   is held off-centre". A phone flash is ~1 cm from the lens, and holding a
   square-on phone off-centre does not move the hotspot; *tilting* does, which
   v4 does not model. **Fixed** in the v4 script's docstring and the Runs 6–8
   "Why"; camera tilt added to Known limitations.
4. **Finding 11's mechanism was imprecise.** It implied a known light makes
   flake direction recoverable. Brightness only fixes the angle to the
   half-way vector (a ring of tilts), so direction is ambiguous even with
   known lighting; v3's ~10% is better explained as a positional shortcut.
   **Rewritten.**
5. **The public demo page overclaimed.** It said the network "recovers its
   three layers", stated Finding 5's untested explanation as fact, and said
   "flakes come through the normal map" — false for the Run 9 model, whose
   normal map is flat. **Fixed** in `demo/index.html`.
6. Minor: Finding 8 said "at most 5 points" for a 5.3-point change (reworded).

**Checked and fine**

- Every number in Runs 6–9 matches the pasted evaluation output. Run 6's table
  arrived without its header line; it is identified by command order and is
  consistent with that (its metallic skill of 35.5% is impossible for the v4
  model, which scored 6.7–7.2% in the other two tables).
- The 2×2 lighting comparison is fair: Runs 5–8 all used 50 epochs at a
  constant learning rate, and every cell scores the same 200 paints.
- No leakage: the validation paints (indices 1800–1999) are unseen by every
  model in every cell.
- v4's hotspot geometry (halfway between camera and light, for equal heights)
  and its v3 pairing (2000/2000 paints identical) were re-verified.
- Full regression on mock data: loader, training with the cosine schedule
  plus rendering loss plus scalar head, evaluation, held-out evaluation,
  demo export, overwrite guard.

---

## Audit 3 — 2026-10-08

**Errors found**

1. **The network cannot see how bright a photo is.** Every `conv_block` ends
   in InstanceNorm, which removes each channel's image-wide mean and scale,
   and the global track averaged the features *after* that norm. So the
   whole network gives the same output for a photo and a uniformly darker
   copy of it (verified: the mean predicted base colour changes by < 2e-5
   between a photo and the same photo ×0.05; it holds for any weights). Absolute brightness, and therefore how dark or light a
   paint is, can only be inferred indirectly from the AgX tone curve's
   shape and from clipping. This applies to every run so far (1–9) and is a
   plausible contributor to base colour's modest, unstable skill (the green
   collapse). For v2 it is worse: each photo is normalised on its own, so
   `MultiLightPaintNet` could not compare the flash photo's brightness with
   the side-lit ones — the cue v2 relies on for the coat. **Fixed for v2**:
   `prenorm_global` makes the global track average each block's first
   convolution *before* its norm, as in Deschaintre 2018. It is on in
   `MultiLightPaintNet`, recorded in each run's `config.json`, and off by
   default in `CarPaintNet`, so v1 checkpoints and the demo behave exactly
   as trained (verified bit-identical). Same parameters either way, so
   `--init-from` still works. Check on random images whose only signal is
   brightness, scored on unseen images: error 0.166 without the fix (no
   better than guessing, 0.150), 0.006 with it.
2. **Ground-truth maps were dithered.** Blender adds ±~1 code value of noise
   to 8-bit output by default, the same size as the flake-normal errors
   being measured. **Fixed in v5**: maps render with dither 0, photos keep
   the default; recorded in `meta.json`, so a v5 folder started before this
   fix refuses to resume. v3/v4 left unchanged, so their held-out test sets
   match their training data.
3. **`export_demo.py` deleted `--out` unconditionally.** A mistyped
   `--out demo` would have wiped the site. **Fixed**: it refuses a non-empty
   folder without the `manifest.json` it writes.
4. Two unused car models (32 MB, one without a licence credit) were deployed
   with the demo. **Removed.**

**Open, not fixed:** Cycles' default pixel filter (~1.5 px) blends flake
normals across cells only 2–4 px wide, so ground-truth normals are slightly
softened and not unit length. A narrower `filter_width` for the map passes
would reduce this, at the cost of aliasing.
