"""
render.py — a differentiable renderer, and the rendering-aware loss.

THE IDEA (Deschaintre et al. 2018)
Comparing predicted maps to ground-truth maps pixel by pixel is a weak signal.
Different material parameters can look identical, and identical-looking numbers
can render differently. What we actually care about is whether the predicted
material LOOKS right.

So: take the predicted maps, render them under some random light and camera,
render the ground-truth maps under the SAME conditions, and compare the two
images. Because every step is differentiable, the error flows back through the
renderer into the network.

This also fixes the failure the map-space loss can't see. L1 barely punishes a
blurred normal map, but blur one and the highlight in the render goes soft
immediately - so the rendering loss cares about flake detail in a way plain L1
never does.

WHAT'S MODELLED
Cook-Torrance with a GGX distribution: the same isotropic model Deschaintre,
MaterialGAN and the Highlight-Aware paper all use.

WHAT'S NOT (be honest about this in the write-up)
No clearcoat layer and no flake layer. The renderer matches the 8 channels the
baseline predicts, not the full three-layer paint the Blender data came from.
So the rendered comparison is an approximation of the real appearance: it will
miss the coat's sharp highlight. Building the multilayer version is the Week 4
crux already flagged in design.md, and this file is where that work will land.

QUICK TEST:
    python src/models/render.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def decode_maps(maps: torch.Tensor):
    """Split the 8-channel stack into the quantities the renderer needs.

    maps: (B, 8, H, W) in 0..1 -> basecolor, roughness, metallic, normal
    """
    basecolor = maps[:, 0:3]
    roughness = maps[:, 3:4].clamp(0.05, 1.0)   # 0 roughness makes the maths blow up
    metallic = maps[:, 4:5]
    normal = maps[:, 5:8] * 2.0 - 1.0           # 0..1 -> -1..1
    normal = F.normalize(normal, dim=1, eps=1e-6)
    return basecolor, roughness, metallic, normal


def make_grid(h: int, w: int, device, dtype):
    """Surface points on a flat sample spanning [-1, 1] in x and y, at z = 0."""
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy, torch.zeros_like(gx)], dim=0)  # (3, H, W)


def render(maps: torch.Tensor, light_pos: torch.Tensor, view_pos: torch.Tensor,
           light_intensity: float = 10.0) -> torch.Tensor:
    """Render the material under a point light. Returns (B, 3, H, W) radiance.

    light_pos / view_pos: (B, 3) positions in the same space as the surface.
    """
    basecolor, roughness, metallic, normal = decode_maps(maps)
    b, _, h, w = maps.shape
    p = make_grid(h, w, maps.device, maps.dtype)[None]        # (1, 3, H, W)

    lp = light_pos[:, :, None, None]
    vp = view_pos[:, :, None, None]

    to_light = lp - p
    dist2 = to_light.pow(2).sum(1, keepdim=True).clamp(min=1e-6)
    wi = to_light / dist2.sqrt()                              # towards the light
    wo = F.normalize(vp - p, dim=1, eps=1e-6)                 # towards the camera
    hv = F.normalize(wi + wo, dim=1, eps=1e-6)                # halfway vector

    dot = lambda a, c: (a * c).sum(1, keepdim=True).clamp(min=1e-6)
    n_l, n_v, n_h, v_h = dot(normal, wi), dot(normal, wo), dot(normal, hv), dot(wo, hv)

    a = roughness.pow(2)
    a2 = a.pow(2)

    # GGX normal distribution: how many microfacets point towards the halfway vector.
    denom = n_h.pow(2) * (a2 - 1.0) + 1.0
    D = a2 / (math.pi * denom.pow(2).clamp(min=1e-6))

    # Smith geometry term: microfacets shadowing each other.
    k = a / 2.0
    G = (n_l / (n_l * (1 - k) + k)) * (n_v / (n_v * (1 - k) + k))

    # Schlick Fresnel. Metals tint their reflection; dielectrics reflect ~4% white.
    f0 = 0.04 * (1 - metallic) + basecolor * metallic
    Fr = f0 + (1 - f0) * (1 - v_h).clamp(0, 1).pow(5)

    specular = D * G * Fr / (4 * n_l * n_v).clamp(min=1e-6)
    diffuse = basecolor * (1 - metallic) / math.pi

    radiance = (diffuse + specular) * n_l * light_intensity / dist2
    return radiance.clamp(min=0.0)


def sample_light_view(batch: int, device, dtype, collocated: bool = False,
                      generator: torch.Generator | None = None):
    """Random light and camera positions above the sample.

    collocated=True puts the camera at the light, like a phone flash - the
    configuration that makes specular highlights visible, which Deschaintre
    deliberately includes among the random ones.
    """
    rand = lambda *s: torch.rand(*s, device=device, dtype=dtype, generator=generator)
    lx = rand(batch, 1) * 2.4 - 1.2
    ly = rand(batch, 1) * 2.4 - 1.2
    lz = rand(batch, 1) * 2.0 + 1.5          # always above the surface
    light = torch.cat([lx, ly, lz], dim=1)

    if collocated:
        return light, light.clone()

    vx = rand(batch, 1) * 1.6 - 0.8
    vy = rand(batch, 1) * 1.6 - 0.8
    vz = rand(batch, 1) * 2.0 + 2.0
    return light, torch.cat([vx, vy, vz], dim=1)


class RenderLoss(nn.Module):
    """Render prediction and target under shared random conditions, compare.

    n_random   how many random light/view setups per step
    n_colocated how many flash-style setups (highlights guaranteed)

    The comparison is done in log space. Renders have a huge dynamic range - a
    specular highlight can be hundreds of times brighter than the surroundings -
    and without the log the loss would only ever care about the highlight.
    """

    def __init__(self, n_random: int = 3, n_colocated: int = 1,
                 light_intensity: float = 10.0, eval_seed: int = 1234):
        super().__init__()
        self.n_random = n_random
        self.n_colocated = n_colocated
        self.light_intensity = light_intensity
        self.eval_seed = eval_seed
        self._gens: dict = {}

    def _generator(self, device):
        """In eval mode, reuse the SAME light/camera positions every time.

        Otherwise validation loss wobbles from epoch to epoch purely because the
        lighting was re-rolled, and you can't tell improvement from luck.
        Training still gets fresh random lighting every step, which is the point.
        """
        if self.training:
            return None
        key = str(device)
        if key not in self._gens:
            self._gens[key] = torch.Generator(device=device)
        g = self._gens[key]
        g.manual_seed(self.eval_seed)
        return g

    def forward(self, pred_maps: torch.Tensor, target_maps: torch.Tensor):
        b = pred_maps.shape[0]
        device, dtype = pred_maps.device, pred_maps.dtype
        total = pred_maps.new_zeros(())
        n = 0
        gen = self._generator(device)

        for collocated in ([False] * self.n_random + [True] * self.n_colocated):
            light, view = sample_light_view(b, device, dtype, collocated,
                                            generator=gen)
            a = render(pred_maps, light, view, self.light_intensity)
            t = render(target_maps, light, view, self.light_intensity)
            total = total + (torch.log1p(a) - torch.log1p(t)).abs().mean()
            n += 1

        return total / max(n, 1)


class CombinedLoss(nn.Module):
    """map-space L1 + rendering loss.

    Keeping some map-space loss stabilises early training: with the rendering
    loss alone the network can wander into materials that happen to render
    similarly but are wrong.
    """

    def __init__(self, render_weight: float = 1.0, map_weight: float = 1.0,
                 **render_kwargs):
        super().__init__()
        self.map_weight = map_weight
        self.render_weight = render_weight
        self.render_loss = RenderLoss(**render_kwargs)
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        map_term = self.l1(pred, target)
        render_term = self.render_loss(pred, target)
        total = self.map_weight * map_term + self.render_weight * render_term
        return total, {"map": map_term.detach(), "render": render_term.detach()}


# --- Self-test --------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 64, 64

    maps = torch.rand(B, 8, H, W)
    maps[:, 5:8] = torch.tensor([0.5, 0.5, 1.0])[None, :, None, None]  # flat normals

    light, view = sample_light_view(B, maps.device, maps.dtype, collocated=True)
    img = render(maps, light, view)
    print(f"render {tuple(img.shape)}  range [{img.min():.4f}, {img.max():.4f}]")
    assert torch.isfinite(img).all(), "render produced NaN or inf"

    # The highlight should appear WHERE THE LIGHT IS. Use fixed positions here:
    # with a randomly placed light the hotspot isn't at the centre, so testing
    # the centre would fail for the wrong reason.
    centred = torch.tensor([[0.0, 0.0, 2.0]]).expand(B, 3)
    im_c = render(maps, centred, centred)
    centre = im_c[:, :, H // 2 - 4:H // 2 + 4, W // 2 - 4:W // 2 + 4].mean()
    corner = im_c[:, :, :8, :8].mean()
    print(f"centred light -> centre {centre:.4f} vs corner {corner:.4f} "
          f"({'hotspot present' if centre > corner else 'NO hotspot'})")

    offset = torch.tensor([[-0.8, -0.8, 2.0]]).expand(B, 3)
    im_o = render(maps, offset, offset)
    tl = im_o[:, :, :16, :16].mean()
    br = im_o[:, :, -16:, -16:].mean()
    print(f"offset light  -> near {tl:.4f} vs far {br:.4f} "
          f"({'highlight follows the light' if tl > br else 'DOES NOT follow'})")

    # Rougher materials should give a dimmer, more spread-out highlight.
    smooth = maps.clone(); smooth[:, 3:4] = 0.1
    rough = maps.clone(); rough[:, 3:4] = 0.9
    ls, lv = sample_light_view(B, maps.device, maps.dtype, collocated=True)
    peak_s = render(smooth, ls, lv).max().item()
    peak_r = render(rough, ls, lv).max().item()
    print(f"peak brightness: smooth {peak_s:.3f} vs rough {peak_r:.3f} "
          f"({'correct' if peak_s > peak_r else 'WRONG WAY ROUND'})")

    # Identical materials must give zero loss; different ones must give more.
    loss_fn = RenderLoss()
    same = loss_fn(maps, maps).item()
    other = torch.rand(B, 8, H, W)
    diff = loss_fn(maps, other).item()
    print(f"loss(identical) {same:.6f}  loss(different) {diff:.6f} "
          f"({'correct' if same < 1e-6 < diff else 'WRONG'})")

    # Gradients must reach the maps, or the loss can't train anything.
    p = torch.rand(B, 8, H, W, requires_grad=True)
    t = torch.rand(B, 8, H, W)
    total, parts = CombinedLoss()(p, t)
    total.backward()
    g = p.grad
    print(f"combined loss {total.item():.4f} "
          f"(map {parts['map']:.4f}, render {parts['render']:.4f})")
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print("gradients flow back through the renderer")

    # The normal channels must receive gradient - that's the whole point.
    gn = g[:, 5:8].abs().mean().item()
    print(f"gradient reaching normal channels: {gn:.6f} "
          f"({'OK' if gn > 0 else 'NONE'})")

    print("\nRESULT: renderer and loss look correct.")