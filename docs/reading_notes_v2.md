# Reading Notes — v2 (every paint type, several lights)

Companion to `reading_notes.md` (v1) and `v2_plan.md`. Grouped by the decision
each source supports. Every link was opened while compiling this list
(2026-10-07); the three that carry the most weight in the novelty argument
(Kaltheuner 2021, Guo 2020, Guo 2018) were checked a second time against
their own abstract or PDF. Deschaintre 2019, Kaltheuner 2021 and Boss 2020
were read in full on 2026-10-09, after Runs 10–11; their entries below say
what each means for the open problems in `ablations.md` (Findings 12–15).

ASTM standards are cited from their public scope pages only (ASTM's store
terms prohibit AI use of the standards themselves).

---

## 1. Why one flash photo is not enough (the motivation)

- **ASTM E2539** — *Multiangle Color Measurement of Interference Pigments.*
  [scope](https://store.astm.org/e2539-14r21.html). Interference pigments
  "require measurement at multiple angles of illumination and detection".
  The one-line justification for v2.
- **BYK-mac i** instrument family — [anodized-parts version](https://www.gardco.com/Products/Color-Testers/Spectrophotometers/Multi-angle-color-%26-effect-measurement%3A-anodized-parts/c/p-42847).
  Colour at six angles (−15° to 110°), sparkle under directional light,
  graininess under diffuse light. Industry's own decomposition of an effect
  finish; the same instrument is sold for anodized consumer parts (see §7).
- **Perales et al. 2020**, *Evaluating the Graininess Attribute by Visual
  Scaling for Coatings with Special-Effect Pigments*, Coatings 10(4) —
  [MDPI](https://www.mdpi.com/2079-6412/10/4/316). Graininess is judged under
  diffuse light and is independent of viewing angle; colour and sparkle are
  the angle-dependent parts.
- **Kitaguchi 2008**, *Modelling Texture Appearance of Gonioapparent Objects*,
  PhD thesis, Leeds — [White Rose](https://etheses.whiterose.ac.uk/11325).
  Coarseness under diffuse light, glint under directional light: supports
  combining a flash photo with side-lit photos.
- **Filip & Maile 2018**, *In the Search of an Ideal Measurement Geometry for
  Effect Coatings* — [record](https://asep.lib.cas.cz/arl-cav/en/detail-cav_un_epca-0495060-In-the-search-of-an-ideal-measurement-geometry-for-effect-coatings).
  A few well-chosen directions capture most of the variation; useful when
  choosing where to hold the torch in the real capture.
- **Filip & Vávra 2019**, *Image-based Appearance Acquisition of Effect
  Coatings*, Computational Visual Media 5(1) —
  [Springer](https://link.springer.com/article/10.1007/s41095-019-0134-3).
  A compact angular parameterisation needs far fewer images than a full
  BTF; supports a few-photo capture.

- **How industry measures orange peel: the distortion of a reflected
  pattern.** Pattern projection imaged by a 2D sensor
  ([Konica Minolta](https://sensing.konicaminolta.asia/wp-content/uploads/2022/09/Image-Clarity-Orange-Peel-and-Surface-Roughness-Measurement.pdf)),
  the waviness of a reflected light stripe
  ([Perceptron, Photonics.com](https://photonics.com/Article.aspx?AID=23155)),
  laser wave-scan with 1,250 points over 10 cm
  ([BYK, PCI Magazine](https://pcimag.com/articles/83827-controlling-orange-peel-s-impact-on-today-s-brilliant-auto-finishes)),
  or a scanned beam's specular reflection
  ([US5153445A](https://patents.google.com/patent/US5153445A/en)). A dark
  room lit by point lights gives the coat no pattern to distort, which fits
  `peel_strength` at ~0% in every run (Findings 5, 14). One photo of the
  sample reflecting stripes (a striped emissive plane in Blender; a phone
  screen in a real capture) should make it observable.

## 2. The paint model (what the generator renders)

- **Günther et al. 2005**, *Efficient Acquisition and Realistic Rendering of
  Car Paint*, VMV — [PDF](https://www.sci.utah.edu/~wald/Publications/2005/CarPaint/carpaint.pdf).
  Source of the gloss clear-coat roughness range (v3 onwards).
- **Ergun, Önel, Ozturk 2016**, *A General Micro-flake Model for Predicting
  the Appearance of Car Paint*, EGSR E&I — [diglib](https://diglib.eg.org/handle/10.2312/sre20161211).
  Physically based flake + coat layers; justifies the layer split.
- **Guo, Chen, Guo, Pan 2018**, *A Physically-based Appearance Model for
  Special Effect Pigments*, CGF — [diglib](https://diglib.eg.org/xmlui/handle/10.1111/cgf13476).
  Glints **and** thin-film interference with a thickness distribution: the
  forward model closest to v2's parameter set. Future upgrade: a film
  thickness *spread*, not one value.
- **Guillén et al. 2020**, *A General Framework for Pearlescent Materials*,
  ACM TOG 39(6) — [record](https://zaguan.unizar.es/record/130000).
  Platelet thickness variation and orientation correlation change pearl
  appearance; v5's single film per sample is the simplest member of this family.
- **Belcour & Barla 2017**, *A Practical Extension to Microfacet Theory for
  the Modeling of Varying Iridescence*, SIGGRAPH —
  [archive](https://history.siggraph.org/?p=102374). The thin-film model
  behind Blender's Thin Film and glTF's iridescence extension.
- **Blender 4.2 / 5.0 release notes** — thin film in the Principled BSDF
  ([4.2, dielectrics](https://projects.blender.org/blender/blender-developer-docs/pulls/64/files);
  [5.0, metals](https://developer.blender.org/docs/release_notes/5.0/cycles/)).
- **Khronos KHR_materials_iridescence** —
  [spec](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_iridescence/README.md).
  Film IOR + thickness range, the same two numbers v5 predicts, so a predicted
  material exports to glTF / three.js directly.
- **OpenPBR**, *Novel Features and Implementation Details* (2025) —
  [arXiv](https://arxiv.org/abs/2512.23696v1). Industry uber-shader: base,
  thin film, coat with absorption (tint), fuzz. No flake layer — the flakes
  stay this project's own procedural part.
- **Blender manual, Principled BSDF** (4.2) —
  [docs](https://docs.blender.org/manual/en/4.2/render/shader_nodes/shader/principled.html),
  and the Cycles source (`bsdf_coat_setup` / `slab_color_at_angle` in
  `intern/cycles/kernel/closure/bsdf_microfacet.h`). Coat Tint is absorption
  inside the coat: "saturation increases at shallower angles", depending on
  the coat IOR. Coat Weight scales both the coat's reflection *and* its tint
  (weight 0 removes the tint too), which is why `coat_weight` is learnable
  for candy only (Finding 13). How Cycles does it: the tint is raised to the
  power 1/cos(refracted angle) of the **viewing** direction only
  (`cosNI = dot(sd->wi, N)`; its comment says it has no access to the light
  direction and assumes both paths through the coat are equal). For v5, with
  the camera fixed overhead, that is tint^1.00 at the image centre and
  ~tint^1.03 in the corners, identical in all six photos of a sample: the
  side lights carry **no** information about candy's tint, so candy vs. a
  coloured metallic base is ambiguous in this data by construction.
  (Corrected 2026-10-09: an earlier version of this note worked out a
  light-dependent path of 2.00–2.28 coat thicknesses; real coats behave
  roughly like that, but Blender's renders do not.)
- **Sung et al. 2002**, *Optical Reflectance of Metallic Coatings: Effect of
  Aluminum Flake Orientation* — [paint.org](https://www.paint.org/ct-archives/optical-reflectance-of-metallic-coatings-effect-of-aluminum-flake-orientation/jctsept02-sung).
  Measured flake tilts have heavier tails than a Gaussian; a candidate change
  to the flake-tilt distribution.

## 3. Thin film: can it be recovered at all?

- **Kobayashi et al. 2013**, *BRDF Estimation of Structural Color Object by
  Using Hyper Spectral Image*, ICCV-W —
  [CVF](https://openaccess.thecvf.com/content_iccv_workshops_2013/W25/html/Kobayashi_BRDF_Estimation_of_2013_ICCV_paper.html).
  Film thickness and IOR from spectra — possible with a spectrometer.
- **Kobayashi et al. 2016**, *Reconstructing Shapes and Appearances of Thin
  Film Objects Using RGB Images*, CVPR —
  [CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Kobayashi_Reconstructing_Shapes_and_CVPR_2016_paper.html).
  RGB thickness lookup indexed by thickness **and angle** — works only when
  the angle varies/is known. Direct support for side-lit photos.
- **Kitagawa 2013**, *Thin-film Thickness Profile Measurement by
  Three-wavelength Interference Color Analysis*, Applied Optics —
  [Optica](https://opg.optica.org/spotlight/summary.cfm?id=251596).
  From RGB, thickness repeats periodically and needs known dispersion and a
  starting estimate. Why v5 keeps thickness ranges short and expects film
  IOR to be the hardest parameter.

## 4. Combining several photos (the network)

- **Deschaintre et al. 2019**, *Flexible SVBRDF Capture with a Multi-Image
  Deep Network*, EGSR — [arXiv](https://arxiv.org/abs/1906.11557),
  [project](https://team.inria.fr/graphdeco/projects/multi-materials/).
  Shared per-image network, order-independent pooling, any number of photos.
  The template for `MultiLightPaintNet`. *Read in full (2026-10-09):* each
  photo runs through its own copy of the single-image network (U-Net +
  global track); the 256×256×64 feature maps are fused by a per-pixel,
  per-channel **max** (the global vectors too), then decoded by 3 conv
  layers. Trained with 1–5 photos, tested with 1–10, lights uncalibrated
  ("we do not provide the network with any explicit knowledge of the light
  and view position"). Two points for this project: (1) "the quality of the
  roughness prediction seems on average independent of the number of
  images, suggesting that the method struggles to exploit additional
  information for this quantity" — the same pattern as Run 11's coat
  roughness (16–18% for every photo set), so a known weakness of pooled
  multi-image networks, not only a bug here; (2) they "provide pixel
  coordinates as extra channels to the input to help the convolutional
  network reason about spatial information", which v2 did not —
  `train_multi.py --coords` adds them. Loss: rendering loss plus L1 on each
  map.
- **Kaltheuner, Bode, Klein 2021**, *Capturing Anisotropic SVBRDFs*, VMV —
  [PDF](https://diglib7.eg.org/bitstream/handle/10.2312/vmv20211372/063-070.pdf).
  U-Net **with a global feature track**, max-pooled over a variable number
  of images — the closest published architecture to this project's.
  *Read in full (2026-10-09):* the capture is **designed, not random**: "not
  all features are visible under every input view- and
  light-configuration", so training uses five photos in fixed roles (one
  Fresnel configuration, two for anisotropy, two random). For v5 that
  suggests fixed side-light slots (e.g. one light high enough to put the
  coat reflection in frame, one raking light for tint and film) instead of
  five random ones, which would also tell the network which photo is which.
  Their input photos are HDR, mapped to log space, `(log(x + 0.01) −
  log 0.01) / (log 1.01 − log 0.01)`, where v5 feeds 8-bit AgX-tone-mapped
  PNGs that compress or clip the coat highlight's peak; `coat_weight` mostly
  changes that peak's height, so HDR input is a candidate fix for it. They
  then fine-tune the decoder per material with a rendering loss under the
  known configurations, which needs a renderer for every layer (v2 has none
  for coat, flakes and film).
- **Zaheer et al. 2017**, *Deep Sets* — [arXiv](https://arxiv.org/abs/1703.06114).
  Why max/mean pooling makes the output independent of photo order and lets
  one network take any number of photos (max-pooled features still grow with
  the count, so train on the full range you'll test on).
- **Lee et al. 2019**, *Set Transformer* — [arXiv](https://arxiv.org/pdf/1810.00825).
  Attention pooling: the upgrade path from max/mean pooling.
- **Boss et al. 2020**, *Two-shot SVBRDF and Shape Estimation*, CVPR —
  [arXiv](https://arxiv.org/abs/2004.00403). Flash + no-flash pair from a
  phone; the lightest multi-image capture. *Read in full (2026-10-09):* the
  two photos are **not** pooled symmetrically. "Merge convolution" blocks (4
  in the encoder, 4 in the decoder) give each photo its own pathway and
  exchange information through a third, "to keep the features from each of
  the images intact"; the no-flash photo supplies pixels where the flash
  photo is saturated. The precedent for treating the flash photo specially —
  `train_multi.py --flash-slot` is the lightweight version: the flash
  photo's features and global vector get their own slot next to the max and
  mean over all photos.
- **Gao et al. 2019**, *Deep Inverse Rendering for High-resolution SVBRDF
  Estimation from an Arbitrary Number of Images*, TOG —
  [NSF PAR](https://par.nsf.gov/biblio/10167644). Optimisation-based
  alternative / refinement stage.
- **Wiersma et al. 2024**, *Fast and Uncertainty-Aware SVBRDF Recovery from
  Multi-View Capture* — [arXiv](https://arxiv.org/abs/2406.17774v1).
  Reports where extra images help — the same question as v2's headline table.
- **Kavoosighafi et al. 2024**, *Deep SVBRDF Acquisition and Modelling: A
  Survey*, CGF STAR — [diglib](https://diglib.eg.org/handle/10.1111/cgf15199).

## 5. Glints and flakes

- **Jakob et al. 2014**, *Discrete Stochastic Microfacet Models*, TOG —
  [RGL](https://rgl.epfl.ch/publications/Jakob2014Discrete). Flake count /
  density parameterisation.
- **Zirr & Kaplanyan 2016**, *Real-time Rendering of Procedural Multiscale
  Materials* — [NVIDIA](https://research.nvidia.com/publication/real-time-rendering-procedural-multiscale-materials).
- **Deliot & Belcour 2023**, *Real-Time Rendering of Glinty Appearances using
  Distributed Binomial Laws on Anisotropic Grids* — [arXiv](https://arxiv.org/abs/2306.05051).
  Drives glints from statistics — what the demo could do with predicted
  flake strength/size.
- **Kneiphof & Klein 2025**, *Real-time Image-based Lighting of Glints* —
  [arXiv](https://arxiv.org/abs/2507.02674v1). Glints under environment maps.
- **Baba et al. 2005**, *Reflectance Estimation of Sparkle in Metallic
  Paints*, SIGGRAPH poster —
  [archive](https://history.siggraph.org/learning/reflectance-estimation-of-sparkle-in-metallic-paints-by-baba-miura-mukunoki-and-asada/).
  Sparkle parameters fitted to real images, without learning — precedent.
- **Filip et al. 2021**, *Assessment of Sparkle and Graininess in Effect
  Coatings…*, J. Coatings Technol. Res. —
  [Springer](https://link.springer.com/10.1007/s11998-021-00518-5).
  Computational sparkle/graininess predictors validated against observers.

## 6. Evaluation

- **Zhang et al. 2018**, *LPIPS* — [arXiv](https://arxiv.org/abs/1801.03924).
- **Sharma, Wu, Dalal 2005**, *CIEDE2000 Implementation Notes* —
  [page](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/). Has test
  pairs to validate a ΔE00 implementation.
- **Gómez et al. 2016**, *Visual and Instrumental Assessments of Color
  Differences in Automotive Coatings*, Color Res. Appl. —
  [UPC](https://upcommons.upc.edu/entities/publication/4eac39e7-0482-4d18-bbf9-b1c201f4ed44).
  Angle-by-angle colour difference for metallic and pearl paints: the
  model for a per-angle colour error.
- **Ummenhofer et al. 2024**, *Objects With Lighting* — [arXiv](https://arxiv.org/abs/2401.09126v2)
  and **Kuang et al. 2023**, *Stanford-ORB* — [arXiv](https://arxiv.org/pdf/2310.16044).
  Relighting under a held-out light as the real-photo metric when there is
  no ground truth.
- **Tobin et al. 2017**, *Domain Randomization* — [arXiv](https://arxiv.org/abs/1703.06907v1).

## 7. Device finishes (the planned extension)

- **Aggerbeck et al. 2014**, *Appearance of Anodised Aluminium…*, Surface &
  Coatings Technology — [AAU](https://vbn.aau.dk/en/publications/appearance-of-anodised-aluminium-effect-of-alloy-composition-and-).
  Roughness ranges for etched/anodized aluminium.
- **Shih, Wei, Huang 2008**, *Optical Properties of Anodic Aluminum Oxide
  Films on Al1050 Alloys* — [NCU](https://scholars.ncu.edu.tw/zh/publications/optical-properties-of-anodic-aluminum-oxide-films-on-al1050-alloy/).
  Interference appears above ~100 nm of oxide: the thin-film head transfers.
- **Walter et al. 2007**, *Microfacet Models for Refraction through Rough
  Surfaces*, EGSR — [Cornell](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.html).
  Frosted (anti-glare) glass.
- **Apple, US11448801B2**, *Textured Glass Layers in Electronic Devices* —
  [patent](https://patents.google.com/patent/US11448801). Textured glass plus
  an interference stack, neutral head-on and coloured off-axis — invisible to
  a collocated flash, exactly the v2 argument.

## 8. Where this project sits (novelty check)

- **Guo, Hašan, Yan, Zhao 2020**, *A Bayesian Inference Framework for
  Procedural Material Parameter Estimation*, CGF —
  [arXiv](https://arxiv.org/abs/1912.01067v3). Fits **layered metallic
  paint** parameters to photos, by optimisation and MCMC (no thin film, no
  feed-forward network). The closest prior work; cite it.
- **Hu et al. 2019**, *A Novel Framework for Inverse Procedural Texture
  Modeling*, TOG — [Yale](https://graphics.cs.yale.edu/node/199);
  **Shi et al. 2020**, *MATch* — [MIT](https://match.csail.mit.edu/);
  **Hu et al. 2022**, *Node Graph Optimization Using Differentiable Proxies* —
  [Yale](https://graphics.cs.yale.edu/publications/node-graph-optimization-using-differentiable-proxies);
  **MatFormer** — [arXiv](https://arxiv.org/pdf/2207.01044).
- **Axalta, US11574420B2** — [patent](https://patents.google.com/patent/US11574420).
  A network on phone/tablet images predicts flake *type* for paint
  formulation — not rendering parameters.

A search for a learned, feed-forward method that estimates flake, thin-film
and clear-coat parameters together from phone photos found none. Queries
used (2026-10): "deep learning estimate car paint appearance parameters flakes
from images", "car paint neural network flake parameters photograph",
"smartphone metallic effect coating sparkle neural network", "learned
estimation thin-film iridescence parameters", "goniochromatic SVBRDF thin-film
estimation network", "layered material clear coat SVBRDF estimation", "glint
parameter estimation from photographs". Defensible claim: *a learned, feed-forward joint
estimate of flake, thin-film and clear-coat parameters from a few
consumer-capturable photos*, with Guo 2020 and the Axalta patent as the
nearest prior work. "Not found" is not "doesn't exist" — re-check before
publishing anything.

## 9. Datasets for real-world testing

- **RGL-EPFL Material Database** — [link](https://rgl.epfl.ch/pages/lab/material-database).
  Measured spectral BRDFs including car paints. CC0.
- **X-Rite AxF sample library** — [link](https://xrite.com/axf/sample-library).
  Includes car paints with flakes; free for testing under X-Rite's terms.
- **OpenSVBRDF** — [link](https://opensvbrdf.github.io/). 1,000+ measured
  SVBRDFs with raw photos (licence not stated on the page — check first).
- **MatSynth** — [Hugging Face](https://huggingface.co/datasets/gvecchio/MatSynth).
  CC0/CC-BY synthetic PBR materials (training variety, not real tests).
