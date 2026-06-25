# Car Paint Material Generation — Project Plan

**Project:** Text/image-conditioned generation of physically-plausible car paint materials (base coat + metallic flake + clearcoat), fit to an editable procedural representation, rendered live in an interactive web viewer.

**Why this project:** It touches dense estimation (SVBRDF prediction), generative modeling (diffusion), procedural/symbolic methods (graph fitting), and neural rendering — and it maps onto a real industry need (online car configurators) rather than a purely academic toy task.

**Suggested pace:** 8 weeks full-time, or ~12–14 weeks at a few focused evenings/weekends per week. Each week below has Goals / Tasks / Reading / Deliverable so you can compress or stretch as needed.

---

## Week 0 (optional buffer) — Setup

- Install: Python 3.10+, PyTorch, Blender (latest LTS), `diffusers`, `transformers`, basic CV stack (`opencv-python`, `numpy`, `Pillow`), Node.js (for the eventual three.js demo).
- Create the GitHub repo now (see the **Git Setup Guide** at the bottom) so every week's work has somewhere to land — recruiters skimming commit history is a feature, not a side effect.

---

## Week 1 — Literature Review & Problem Framing

**Goals:** Know the landscape well enough to make defensible design decisions, and write them down before you start coding.

**Tasks:**
- Read the papers listed below. Take notes per paper: *what they predict, what their loss/training signal is, what their main limitation is.*
- Decide your exact output representation. Recommended starting point: `base_color, normal, roughness, metallic` (standard SVBRDF) **plus** `flake_density, flake_orientation_perturbation, clearcoat_intensity` (car-paint-specific).
- Write a 1-page design doc: architecture sketch, data plan, eval metrics, and what "done" looks like for each milestone.

**Reading (foundational SVBRDF estimation):**
- Deschaintre et al., *"Single-Image SVBRDF Capture with a Rendering-Aware Deep Network,"* ACM TOG 2018 — the canonical single-image SVBRDF paper; the "rendering-aware loss" idea (re-render predicted maps and compare images, not just raw maps) is something you'll reuse.
- Guo et al., *"MaterialGAN: Reflectance Capture Using a Generative SVBRDF Model,"* arXiv:2010.00114 (2020).
- Guo et al., *"Highlight-Aware Two-Stream Network for Single-Image SVBRDF Acquisition"* (2021) — relevant because clearcoat highlights are exactly the hard case this addresses.

**Reading (car paint, classic graphics):**
- Günther et al., *"Efficient Acquisition and Realistic Rendering of Car Paint,"* VMV 2005 — measurement + Cook-Torrance based car paint model.
- *"Rendering Pearlescent Appearance Based on Paint-Composition Modelling"* (2001) — multilayer paint operator model.
- Look up "Real-time image-based lighting of metallic and pearlescent car paints" (ScienceDirect, ~2022) for a modern treatment of the same problem.

**Deliverable:** `docs/design.md`, `docs/reading_notes.md`, repo skeleton initialized.

---

## Week 2 — Synthetic Data Pipeline

**Goals:** A way to generate (photo, ground-truth-maps) pairs on demand, since real car-paint SVBRDF capture data essentially doesn't exist publicly.

**Tasks:**
- Build a parametric car-paint shader node graph in Blender exposing: `base_hue, base_value, flake_density, flake_size, flake_color_variance, clearcoat_roughness, clearcoat_ior`.
- Write a `bpy` script that randomly samples those parameters, renders the patch under 2–3 lighting setups, and exports the photo + each ground-truth map as separate image files.
- Generate a few thousand synthetic samples. Start small (a few hundred) to validate the pipeline before scaling up.
- Separately: source or capture a **small real test set** for domain-gap sanity checking later. Auto body / paint supply shops sell sample chips cheaply; photograph them under a consistent setup (fixed camera distance, on-camera flash, dark background) — this mirrors the capture protocol used in flash-photo material datasets.

**Reading:**
- Zhou et al., *"PhotoMat"* (2023) — describes a practical real flash-photo material capture pipeline; useful as a protocol reference even if you don't replicate it exactly.
- AmbientCG and Poly Haven — CC0-licensed PBR texture libraries, useful as a fallback/sanity-check dataset for the general (non-car-specific) material estimation baseline.

**Deliverable:** `data/blender_gen/` script + generated dataset (not committed directly to git — see `.gitignore` notes below), `data/real_test_set/` with a short README documenting the capture protocol and licensing of each source.

**Watch out for:** Don't photograph or scrape branded paint swatches and label them with the manufacturer's proprietary color name as if you're reproducing their exact formula — use descriptive names ("deep metallic red") rather than trademarked product names to avoid IP issues.

---

## Week 3 — Baseline Dense Estimation Model

**Goals:** A working single-image → SVBRDF baseline before adding car-paint-specific complexity.

**Tasks:**
- Implement a U-Net (or small ViT encoder + conv decoder) that takes one photo and predicts the standard 4 SVBRDF channels.
- Train with a rendering-aware loss: re-render the predicted maps under known lighting and compare against the ground-truth render, in addition to direct map loss.
- Evaluate on held-out synthetic data and on your small real test set; expect a real-data performance drop — quantify and note it, don't hide it.
- Log experiments properly (Weights & Biases free tier, or even a structured CSV + matplotlib) — this is the kind of rigor a research-oriented reviewer will specifically look for.

**Deliverable:** trained baseline checkpoint, `notebooks/01_baseline_eval.ipynb` with metrics table and qualitative renders.

---

## Week 4 — Car-Paint-Specific Extension

**Goals:** Move from generic SVBRDF to the disentangled base/flake/clearcoat representation that's the actual novelty of this project.

**Tasks:**
- Extend the model to predict the flake and clearcoat channels separately from the base coat.
- Try a **chained/staged decomposition**: predict base color first, then condition subsequent predictions on it — this reduces the multi-modal ambiguity that flat single-pass prediction struggles with (the same idea underlies several recent material-from-image models).
- Run an ablation: single-pass joint prediction vs. staged decomposition. This ablation table is good evidence of research rigor in a write-up.

**Reading:**
- Look for **CHORD** (Ubisoft La Forge, debuted SIGGRAPH Asia 2025) — an open-source diffusion model predicting material maps from a single RGB image using exactly this kind of chained decomposition; worth reading their writeup and, if released, their code as a reference implementation.
- *"ControlMat: A Controlled Generative Approach to Material Capture,"* arXiv:2309.01700 — conditioning strategies for controllable material capture.

**Deliverable:** v2 model, ablation results in `docs/ablations.md`.

---

## Week 5 — Diffusion-Based Generation (text/image → novel material)

**Goals:** Go beyond *estimating* a material from a photo to *generating* novel ones from a text description.

**Tasks:**
- Fine-tune a small latent diffusion model (cheaper: adapt a pretrained Stable Diffusion VAE/UNet backbone rather than training from scratch) conditioned on text and/or a reference photo, to output your material map set.
- Build a small automotive-paint text vocabulary for conditioning — terms like "pearlescent white," "satin gunmetal," "candy apple flake red" (generic descriptive terms, not brand-specific color names).
- Produce a gallery of generated materials spanning a range of finishes (matte, metallic, pearlescent, candy).

**Reading:**
- Rombach et al., *"High-Resolution Image Synthesis with Latent Diffusion Models,"* CVPR 2022 — the LDM foundation your fine-tune will build on.
- Vecchio et al., *"MatFuse"* and *"MatGen"* (2024) — latent-diffusion material generators trained with multiple conditioning encoders (text/image/sketch).
- *"DreamPBR: Text-Driven Generation of High-Resolution SVBRDF with Multi-Modal Guidance,"* arXiv:2404.14676.
- *"ReflectanceFusion: Diffusion-Based Text to SVBRDF Generation,"* arXiv:2406.14565 (and the earlier Text2Mat it compares against).
- Optional stretch reading: *"HiMat: DiT-Based Ultra-High-Resolution SVBRDF Generation,"* arXiv:2508.07011, if you want to explore higher-resolution generation later.

**Deliverable:** text/image-conditioned generation notebook + sample gallery in `results/gallery/`.

---

## Week 6 — Procedural Layer / Symbolic Regression

**Goals:** Convert raster map outputs into an *editable, resolution-independent* procedural representation — this is your strongest differentiator and the most novel piece of the project.

**Tasks:**
- Define a small car-paint procedural node DSL mirroring your Week 2 Blender graph: a flake-generator node (Voronoi/cellular noise based), a clearcoat node, a base-tint node.
- Fit the DSL's parameters to match a generated/predicted material. Two viable approaches, pick one to start:
  - **Differentiable proxy nodes**: approximate each procedural node with a small differentiable function so you can backprop a rendering loss into the node parameters directly.
  - **Gradient-free search**: CMA-ES or similar, optimizing node parameters against a rendering-loss objective — simpler to implement, slower to run.
- Compare: raster prediction vs. fitted procedural graph, re-rendered side by side.

**Reading:**
- Hu et al., *"Node Graph Optimization Using Differentiable Proxies,"* SIGGRAPH 2022 — the differentiable-proxy approach.
- Li et al., *"End-to-End Procedural Material Capture with Proxy-Free Mixed-Integer Optimization,"* ACM TOG 2023.
- Li et al., *"Procedural Material Generation with Reinforcement Learning,"* ACM TOG 2024.
- Li et al., *"VLMaterial: Procedural Material Generation with Large Vision-Language Models,"* arXiv:2501.18623 (2025) — a quite different, interesting approach: prompting a VLM to write the procedural graph code directly. Worth reading even if you don't implement this version, since it's a good talking point for "what other approaches exist."

**Deliverable:** procedural-fitting module, before/after comparison figure.

---

## Week 7 — Real-Time Demo

**Goals:** A link a recruiter can click and play with, not just a folder of images.

**Tasks:**
- Build a three.js web viewer (browser-based, free, trivially shareable) with a CC0-licensed car mesh (check Poly Haven or Sketchfab CC0 listings carefully — avoid branded/trademarked car models).
- Wire it up: a user enters a description (or picks from your generated gallery) → maps stream in as a PBR material → live re-render on the mesh.
- Deploy via GitHub Pages so it's a real URL, not a video screen-recording.

**Deliverable:** live demo link.

---

## Week 8 — Evaluation, Write-Up, Polish

**Tasks:**
- Final quantitative eval: per-map PSNR/SSIM **and** a re-rendered perceptual comparison (raw map error is a known weak proxy for material quality — note this explicitly in your write-up, it shows awareness of the metric's limitations). If feasible, compare against an existing open baseline (e.g., run CHORD's released weights on your test set if available) for a head-to-head table.
- Write a 4–6 page report in a SIGGRAPH/arXiv-style structure: Problem → Related Work → Method → Results → Limitations → Future Work. Hosting it as a PDF in the repo is enough; posting to arXiv is a nice-to-have if you want it citable.
- Polish the README: demo GIF, architecture diagram, links to the live demo and the report.
- Optional stretch: draft a 1-page invention-disclosure note framing the procedural-graph-fitting step as the patentable mechanism (novel mechanism, not novel application, is what patents reward).

**Deliverable:** final report PDF, polished repo, demo GIF, optional arXiv submission.

---

## Consolidated Bibliography (for quick reference / citing later)

**Foundational SVBRDF estimation**
- Deschaintre et al., 2018 — Single-Image SVBRDF Capture with a Rendering-Aware Deep Network
- Guo et al., 2020 — MaterialGAN (arXiv:2010.00114)
- Henzler et al., 2021 — Generative Modelling of BRDF Textures from Flash Images (arXiv:2102.11861)
- Guo et al., 2021 — Highlight-Aware Two-Stream Network for Single-Image SVBRDF Acquisition
- Zhou et al., 2023 — PhotoMat

**Diffusion-based material generation**
- Rombach et al., 2022 — Latent Diffusion Models (LDM)
- Vecchio et al., 2024a/b — MatFuse, MatGen
- He et al., 2023 — Text2Mat
- Xue et al., 2024 — ReflectanceFusion (arXiv:2406.14565)
- DreamPBR, arXiv:2404.14676
- HiMat, arXiv:2508.07011
- RealMat, arXiv:2509.01134
- ControlMat, arXiv:2309.01700
- CHORD — Ubisoft La Forge, SIGGRAPH Asia 2025 (open-source prototype)

**Procedural material generation**
- Hu et al., 2019 — A Novel Framework for Inverse Procedural Texture Modeling
- Hu et al., 2022 — An Inverse Procedural Modeling Pipeline for SVBRDF Maps
- Hu et al., 2022 — Node Graph Optimization Using Differentiable Proxies
- Hu et al., 2023 — Generating Procedural Materials from Text or Image Prompts (SIGGRAPH 2023)
- Guerrero et al., 2022 — MatFormer
- Li et al., 2023 — End-to-End Procedural Material Capture with Proxy-Free Mixed-Integer Optimization
- Li et al., 2024 — Procedural Material Generation with Reinforcement Learning
- Li et al., 2025 — VLMaterial (arXiv:2501.18623)

**Car paint specific**
- Günther et al., 2005 — Efficient Acquisition and Realistic Rendering of Car Paint
- "Rendering Pearlescent Appearance Based on Paint-Composition Modelling," 2001
- "Real-Time Image-Based Lighting of Metallic and Pearlescent Car Paints," ~2022, ScienceDirect

**Datasets / assets**
- AmbientCG (CC0 PBR textures)
- Poly Haven (CC0 PBR textures + CC0 3D models, including some vehicle assets)

> Note: keep this bibliography updated as you read — `docs/reading_notes.md` is the place for your own summaries; this file should stay as a clean citation list.

---

## Git Repository Setup Guide

Goal: a repo that looks professional and trustworthy to a recruiter or hiring manager skimming it for 90 seconds, and that actually demonstrates engineering hygiene (not just research output).

### 1. Initialize locally

```bash
mkdir car-paint-material-gen && cd car-paint-material-gen
git init
git branch -M main
```

### 2. Folder structure

```
car-paint-material-gen/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt          # or pyproject.toml / environment.yml
├── data/
│   ├── blender_gen/          # synthetic data generation scripts
│   └── real_test_set/        # small real captures + README on licensing/protocol
├── src/
│   ├── models/                # network architectures
│   ├── train.py
│   ├── eval.py
│   └── procedural/            # node-graph fitting code
├── notebooks/                 # exploratory + eval notebooks
├── demo/                       # three.js web viewer
├── docs/
│   ├── design.md
│   ├── reading_notes.md
│   ├── ablations.md
│   └── PROJECT_PLAN.md        # this file
└── results/
    └── gallery/
```

### 3. `.gitignore` essentials

```gitignore
# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# Data & model weights — too large for git, handled via Releases or a download script
data/blender_gen/output/
*.ckpt
*.pt
*.safetensors

# Notebooks checkpoints
.ipynb_checkpoints/

# OS/editor cruft
.DS_Store
.vscode/
```

Large files (datasets, model checkpoints) should **not** be committed directly. Two good options:
- Attach them as binary assets on a **GitHub Release** (works well for a handful of files up to a few GB total) and provide a small `download_assets.sh` script in the repo that fetches them.
- Use **Git LFS** if you want them versioned alongside code — heavier-weight, only worth it if you expect to iterate on the data itself.

### 4. `requirements.txt` (or better, `pyproject.toml`)
Pin major versions so anyone cloning the repo six months from now can still reproduce your environment:
```
torch>=2.2
diffusers>=0.27
transformers>=4.40
numpy
opencv-python
pillow
```

### 5. License
- Add an MIT or Apache-2.0 `LICENSE` for your code.
- Add a short `ASSETS.md` or section in the README clarifying licensing for non-code assets (Blender shader files you wrote are yours; any third-party textures/meshes — list their license and source explicitly, e.g. "car_proxy.glb — Poly Haven, CC0").

### 6. README structure
A strong README, top to bottom:
1. One-sentence project description
2. A GIF or screenshot of the live demo right at the top (this is what gets attention in the first 5 seconds)
3. Link to the live demo (GitHub Pages URL)
4. Short architecture diagram (even a simple draw.io/Excalidraw export is fine)
5. Quickstart: `pip install -r requirements.txt`, how to run training/eval/demo
6. Results summary table
7. Key references (link to `docs/PROJECT_PLAN.md` bibliography for the full list)
8. License note

### 7. Push to GitHub

```bash
# create the repo on GitHub first via the web UI or `gh repo create`, then:
git remote add origin https://github.com/<your-username>/car-paint-material-gen.git
git add .
git commit -m "Initial commit: repo skeleton, design doc"
git push -u origin main
```

If you have the GitHub CLI installed, this is even faster:
```bash
gh repo create car-paint-material-gen --public --source=. --remote=origin --push
```

### 8. Commit hygiene
Aim for one meaningful commit per work session, with clear messages. A lightweight convention worth adopting (it signals process maturity to anyone browsing history):
```
feat: add baseline SVBRDF U-Net
data: blender synthetic generation script
docs: week 1 reading notes
fix: rendering-aware loss normalization bug
```

### 9. GitHub Pages for the live demo
Once your `demo/` three.js app is ready:
```bash
# if using a simple static build:
git subtree push --prefix demo origin gh-pages
```
Then enable Pages in repo Settings → Pages → source: `gh-pages` branch. You'll get a URL like `https://<username>.github.io/car-paint-material-gen/` — put this front and center in the README.

### 10. Optional CI polish
A minimal GitHub Actions workflow (`.github/workflows/lint.yml`) that just runs `flake8` or `ruff` on push is cheap to set up and signals engineering discipline:
```yaml
name: Lint
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ruff
      - run: ruff check .
```

### 11. Final visibility polish
- Add repo **topics/tags** on GitHub (`computer-vision`, `diffusion-models`, `material-generation`, `pytorch`) so it's discoverable and instantly legible.
- Set a **social preview image** (Settings → General → Social preview) — use a still from your demo.
- **Pin the repo** on your GitHub profile.
- Add badges to the README top (build status, license, Python version) — small but signals polish.

---

*This document is meant to be a living plan — update the checkboxes/notes as weeks progress, and keep the bibliography section as your running citation list for the eventual write-up.*
