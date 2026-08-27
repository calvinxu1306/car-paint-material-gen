# Reading Notes

Per-paper notes in a fixed format: **What it predicts/produces**, **Training/fitting signal**, **Main limitation**, **How this project uses it**. Kept deliberately terse — this is a working document, not prose.

---

## A. Deschaintre et al. 2018 — *Single-Image SVBRDF Capture with a Rendering-Aware Deep Network* (SIGGRAPH)

**Predicts:** 4 per-pixel maps from one flash photo — normal, diffuse albedo, specular albedo, specular roughness (9 channels total). Cook-Torrance + GGX BRDF.

**Training signal:** The headline idea — a **rendering-aware loss**. Instead of (only) comparing predicted maps to ground-truth maps pixel-wise, it re-renders both predicted and GT maps under randomly sampled light/view directions and compares the *renders* (L1 in log space). Two sampling regimes: cosine-weighted random directions, plus deliberately seeded mirror configurations so highlights actually appear. Architecture: U-Net + a parallel "global features" track that propagates whole-image information (fixes the U-Net's inability to produce even a constant-color output). Trained on ~200k procedural SVBRDFs (Substance Share, 155 curated graphs → augmented by parameter perturbation + pairwise map mixing).

**Main limitation (this is the gap the project targets):** 256×256, LDR input, isotropic Cook-Torrance only — **no clearcoat, no flakes, no anisotropy, no layering.** The authors explicitly note it *fails on materials with a clear coat over a textured diffuse base* and tends to bake correlated structure across maps. Fronto-parallel only, so no real Fresnel.

**How this project uses it:** This is the **Week 3 baseline** and the source of the rendering loss reused everywhere. The cloned repo is this paper. The car-paint extension (Week 4) is essentially "take this and add the layers it can't represent."

---

## B. Guo et al. 2021 — *Highlight-Aware Two-Stream Network for Single-Image SVBRDF Acquisition* (SIGGRAPH)

**Predicts:** Same 4-map Cook-Torrance + GGX SVBRDF from one casual flash image, but engineered to survive blown-out specular highlights.

**Training signal:** Two architectural ideas + a loss idea. (1) **Highlight-aware (HA) convolution** — a convolution variant that learns a soft mask of over-exposed pixels and down-weights features coming from them (conceptually related to partial/gated convolutions from inpainting). (2) **Two-stream network** — one stream of HA convolutions (extracts highlight-free features), one of standard convolutions (keeps the specular cues the highlight carries), fused by an attention module. (3) A **two-scale perceptual adversarial loss** for sharper texture detail. Trained on the same kind of synthetic procedural SVBRDF data.

**Main limitation:** Still planar, still the standard 4-map isotropic model — no clearcoat/flake layer. It improves *robustness* to highlights rather than *expanding* the material model.

**How this project uses it:** Directly relevant because **clearcoat produces exactly the saturated-highlight problem** this paper attacks. Two options: borrow the HA-convolution as a drop-in robustness upgrade to the baseline encoder, or cite it as motivation for handling the clearcoat specular peak as a separate predicted quantity rather than letting it pollute the base coat. Note: one author is affiliated with OPPO (a mobile OEM) — useful "industry cares about this" signal for the write-up.

---

## C. Guo et al. 2020 — *MaterialGAN: Reflectance Capture using a Generative SVBRDF Model* (SIGGRAPH Asia)

**Predicts/produces:** A StyleGAN2-based generator trained to synthesize plausible 4-map SVBRDFs. Capture is done by **optimizing in the GAN latent space** (W+ and noise N) so re-renders match 3–7 input flash photos — inverse rendering through a differentiable renderer, not a feed-forward pass.

**Training signal:** GAN training on Deschaintre's 200k dataset for the generator; at capture time, L2 + VGG perceptual loss on renders, optimized via alternating updates to W+ and noise. Key conceptual contribution: a **learned generative prior** that keeps solutions on the "manifold of realistic materials," and a latent space that supports semantic editing / interpolation / morphing.

**Main limitation:** Isotropic BRDF, flat samples, slow (~2 min/material optimization), not resolution-independent. Same model gap — **explicitly lists layering (e.g. book covers), fiber scattering, and anisotropy as unsupported**, which is precisely the multilayer-paint regime.

**How this project uses it:** The **Week 5 generation reference** and the contrast case for the write-up: feed-forward prediction (Deschaintre) vs. latent-space optimization (MaterialGAN). The latent interpolation/morphing is a compelling demo feature (smoothly morph between two paint finishes). Reinforces the synthetic-procedural-data strategy.

---

## D. Günther, Chen, Goesele, Wald, Seidel 2005 — *Efficient Acquisition and Realistic Rendering of Car Paint* (VMV)

**Predicts/produces:** Not learning — a measurement + fitting pipeline. Captures a car-paint-coated sphere (turntable, point light stepped in 1° increments, HDR), then fits a **multi-lobe Cook-Torrance BRDF** (2–3 lobes, one lobe dedicated to the sharp clearcoat specular peak) via constrained Levenberg-Marquardt with a physical-plausibility penalty. Adds a **procedural flake-normal texture** for sparkle, seeded by (u,v) so it's spatially/temporally coherent. Renders interactively under HDR env maps via ray tracing.

**Training/fitting signal:** Levenberg-Marquardt fit of analytic BRDF to measured HDR BRDF samples in a halfway-vector (Rusinkiewicz) parameterization. Sparkle: sample flake normals from the Beckmann distribution implied by the fitted roughness.

**Main limitation:** Manual acquisition, only 8 measured paints, sparkle is a heuristic add-on rather than learned, no generative/inverse component.

**How this project uses it:** The single most *operationally* useful paper. It gives (1) a concrete **flake = procedural-normal-texture** representation → directly the Week 6 procedural layer; (2) **Table 1 of fitted parameters for 8 real car paints** (Ford F8, Polaris Silber, Opel Titan, BMW 339, plus solid paints) — these are real numbers I can use to seed/validate the Blender synthetic generator in Week 2; (3) the "one Cook-Torrance lobe = clearcoat peak" decomposition that justifies treating clearcoat as its own predicted lobe in Week 4.

---

## E. Kneiphof & Klein 2022 — *Real-time Image-Based Lighting of Metallic and Pearlescent Car Paints* (Computers & Graphics)

**Predicts/produces:** A real-time rendering technique, not estimation. Takes the industry-standard **AxF (X-Rite) car paint model** and makes its angular colour shift (goniochromatism / "flop") cheap to render under image-based lighting by prefiltering a colour table into a small cubemap array (+ optional SVD compression).

**Model structure (the useful part):** AxF car paint = perfectly-smooth **clear coat** (Fresnel) + **base material** (Lambertian diffuse + K=3 Beckmann microfacet lobes) + a 2D **colour table** χ(θ_h, θ_d) for the angular colour variation + a **flake BTF** for sparkle (which they deliberately ignore, treating the material as spatially homogeneous).

**Main limitation:** Ignores spatial flake/glint effects; requires per-material *and* per-environment precomputation; assumes homogeneous material and isotropy around the reflection direction.

**How this project uses it:** The **Week 7 rendering reference.** Its clear-coat + multi-lobe-base + flake decomposition is essentially the parameterization I want to predict, validated by an industry format (AxF). For the three.js viewer I won't replicate the prefiltering math, but the *layer structure* and the "angular colour shift is a first-class effect" point shape how the live shader is built. Good citation for "this maps onto formats industry actually ships."

---

## F. Ershov, Kolchin, Myszkowski 2001 — *Rendering Pearlescent Appearance Based on Paint-Composition Modelling* (Eurographics)

**Predicts/produces:** A physically-based multilayer paint BRDF derived from **paint composition** (flake density, mean flake area, orientation distribution, interference-coating thickness, pigment density, binder absorption, substrate albedo, per-layer structure), using adding/doubling math borrowed from atmospheric radiative transfer.

**Key decomposition:** BRDF splits into three interpretable terms — **gloss** (Fresnel boundary), **glitter** (flake reflection), **shade** (substrate seen through the flake layer). Plus a separate sparkle model for close-up luminance fluctuation.

**Main limitation:** Expensive, assumes ideal specular platelets and spherical Mie pigments; far too heavy to evaluate inside an ML training loop directly. Oldest paper here.

**How this project uses it:** The **physical grounding / vocabulary source.** It tells me what the flake and clearcoat parameters *physically mean*, which is what makes the Week 4 parameterization defensible rather than arbitrary. The gloss/glitter/shade split maps cleanly onto clearcoat/flake/base — a clean conceptual backbone for both the Blender shader (Week 2) and the procedural DSL (Week 6). Cite it as the "why these parameters" justification.

---

## Cross-cutting takeaways (feed these into design.md)

1. **Everyone predicts the same 4-map isotropic Cook-Torrance/GGX SVBRDF** (A, B, C). **Nobody** in the learning-based papers handles clearcoat + flakes. The car-paint papers (D, E, F) *do* model those layers but only by hand-fitting or physical simulation, not by learning. **That intersection — learning a clearcoat+flake-aware SVBRDF — is the project's contribution, and it's genuinely open.**
2. **The rendering-aware loss (A)** is the reusable backbone across the whole pipeline.
3. **Flakes as a procedural normal texture (D)** is the bridge between the neural maps and the Week 6 procedural representation — they're the same idea seen from two ends.
4. **Clearcoat = the saturated-highlight problem (B)** and **= a dedicated Cook-Torrance lobe (D)** and **= a Fresnel top layer (E, F)**: three papers describe the same physical thing three ways. The design needs to pick how clearcoat is represented in the output.
5. **Synthetic procedural training data is the consensus solution to "no ground-truth dataset exists"** (A, C) — validates the Week 2 Blender plan.
