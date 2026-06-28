# Design Document

**Status:** Week 1 draft. This is the single source of truth for the design decisions the rest of the project commits to. Update it when a decision changes; don't let the code drift from it silently.

---

## 1. Problem statement

Given a single flash photograph (or, later, a text prompt) of automotive paint, predict a **disentangled, multilayer car-paint material** — base coat + metallic/pearlescent flake layer + clearcoat — as editable maps/parameters, and render it live on a 3D car in the browser.

This differs from standard SVBRDF capture in that the output is **not** a flat 4-map isotropic material; it explicitly separates the three physical layers that the literature handles only by hand-fitting (Günther 2005, Kneiphof 2022) or physical simulation (Ershov 2001).

## 2. Output representation (the key decision)

Start from the standard 4-map Cook-Torrance/GGX SVBRDF that all three learning papers share (Deschaintre 2018, Guo 2021, MaterialGAN 2020), then **extend** it with car-paint-specific channels grounded in the AxF model (Kneiphof 2022) and the composition model (Ershov 2001):

| Group | Channels | Source / justification |
|---|---|---|
| Base coat | base color (RGB), roughness, metallic | standard SVBRDF (Deschaintre) |
| Geometry | normal (xy) | standard SVBRDF |
| Flake layer | flake density, flake size, flake orientation spread, flake color/variance | Günther's procedural flake-normal texture; Ershov's flake density D, mean area ⟨S⟩, orientation distribution P(β) |
| Clearcoat | clearcoat intensity/strength, clearcoat roughness | AxF clear coat (Kneiphof); Günther's dedicated specular lobe |

**Decision to make explicit:** clearcoat is represented as **its own predicted quantity**, not folded into the base specular. Three papers independently say clearcoat is a distinct physical layer (B as the highlight problem, D as a separate lobe, E as a Fresnel top layer), so letting it pollute the base coat is exactly the documented failure mode of Deschaintre 2018. This is the project's main modeling stance.

## 3. Architecture plan

- **Week 3 baseline:** Deschaintre's U-Net + global-features track, predicting the standard 4 maps. Reuse their architecture as-is to confirm the pipeline before adding complexity.
- **Week 4 extension:** add output heads for the flake + clearcoat channels. Use a **staged/chained decomposition** (predict base coat first, condition flake/clearcoat prediction on it) to reduce the multilayer ambiguity, and ablate staged vs. joint prediction.
- **Optional robustness upgrade:** swap standard convolutions in the encoder for the highlight-aware (HA) convolution of Guo 2021, since clearcoat creates the saturated highlights that motivated it. Treat as a stretch/ablation, not a core dependency.
- **Week 5 generation:** fine-tune a latent-diffusion generator for text/image→maps; MaterialGAN's latent-optimization route is the documented alternative to compare against.

## 4. Loss

Adopt Deschaintre's **rendering-aware loss** as the backbone: re-render predicted and GT maps under randomly sampled light/view directions (cosine-weighted + seeded mirror configs for highlights), compare in log space. This is parameterization-independent, which matters because the extended channels won't have a clean per-pixel pixel-loss interpretation. Add direct map losses on the channels that do (base color, normal). Consider Guo 2021's perceptual/adversarial term in Week 5 for texture sharpness.

**Open question:** the rendering loss needs an in-network renderer that understands the *extended* (clearcoat+flake) model, not just plain Cook-Torrance. Building that differentiable multilayer renderer is the technical crux of Week 4 — flag it as the highest-risk item.

## 5. Data plan

No public car-paint SVBRDF dataset exists, so follow the synthetic-procedural consensus (Deschaintre, MaterialGAN both train on procedural Substance data):

- **Synthetic (primary):** parametric car-paint shader in Blender, randomized over the parameters in §2, rendered to (flash photo → GT maps) pairs. **Seed the parameter ranges using Günther 2005 Table 1** (8 real measured car paints) so the synthetic distribution is anchored to real materials, not guessed.
- **Real (validation only):** a small set of paint chips photographed with on-camera flash (PhotoMat-style protocol) to measure the synthetic-to-real gap. Not for training.

## 6. Evaluation metrics

- Per-map PSNR/SSIM on held-out synthetic data (where GT exists).
- **Re-rendered perceptual error** (LPIPS) under novel light/view — MaterialGAN's evaluation approach; matters because raw map error is a weak proxy for appearance (Deschaintre make this point explicitly).
- Synthetic-vs-real gap, quantified and reported honestly rather than hidden.
- Ablation tables: staged vs. joint decomposition (Week 4); with/without HA convolution; with/without flake layer.

## 7. What "done" looks like per milestone

- **W3:** baseline reproduces plausible 4-map SVBRDF on synthetic + real test, metrics logged.
- **W4:** extended model predicts separable clearcoat/flake; ablation shows the staged decomposition helps.
- **W5:** text/image → novel paint generation, gallery of finishes.
- **W6:** maps fit to an editable procedural graph; before/after re-render comparison.
- **W7:** live three.js viewer, type-a-finish → see it on a car.
- **W8:** report + honest limitations + the synthetic-real gap discussion.

## 8. Biggest risks (watch these)

1. **The differentiable multilayer renderer (W4)** — without it the rendering loss can't supervise the new channels. Highest technical risk.
2. **Blender synthetic pipeline (W2)** — bottleneck for everything downstream.
3. **Synthetic-to-real gap** — the real failure mode of this whole family of methods; plan to measure it, not be surprised by it.
