"""
model.py — U-Net + global-features track, with two output heads.

INPUT   photo  (B, 3, 256, 256)
OUTPUT  dict:
    "maps"     (B, 8, 256, 256)  basecolor RGB, roughness, metallic, normal XYZ
    "scalars"  (B, 5)            coat weight/roughness, flake scale/strength,
                                 peel strength - normalised 0..1

WHY A U-NET
Predicting maps is a per-pixel job. The encoder halves the image repeatedly
(more context, less detail), the decoder scales back up, and skip connections
hand the decoder the fine detail the encoder recorded on the way down. Without
the skips you get blurry maps with no flake detail.

WHY THE GLOBAL TRACK (Deschaintre 2018)
A plain U-Net is convolutional, so each output pixel sees only a limited
neighbourhood. That's a real problem here: the flash makes the middle of the
photo bright and the corners dark, yet the true base colour is the SAME
everywhere. Through a small window the network cannot tell "bright because the
flash is here" from "bright because the paint is lighter". So a parallel track
carries whole-image information: at each level we average the features over the
entire image, push that through a small MLP, and add it back to every pixel.

WHY A SECOND HEAD (this project's contribution)
Clear coat and flakes are properties of a LAYER, not of a pixel: one coat
roughness, one flake size per sample. Forcing them through the per-pixel decoder
would spend capacity learning to emit a constant. Instead they are read off the
global vector - which already holds exactly the whole-image evidence they depend
on (how tight the highlight is, how dense the sparkle). No learning-based SVBRDF
paper in the reading list predicts these at all.

Both heads end in a sigmoid: map channels are stored 0..1, and scalars are
normalised to 0..1 by the dataset.

QUICK TEST:
    python src/models/model.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two convolutions at the same resolution. The workhorse of the U-Net."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.InstanceNorm2d(c_out, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.InstanceNorm2d(c_out, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
    )


class GlobalTrack(nn.Module):
    """Carries whole-image context alongside the convolutions.

    At each level: average the feature map over space, mix it into a running
    global vector, then add a per-channel offset back to every pixel.
    """

    def __init__(self, c_feat: int, c_global: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Linear(c_global + c_feat, c_global),
            nn.SELU(inplace=True),
        )
        self.to_features = nn.Linear(c_global, c_feat)

    def forward(self, feat: torch.Tensor, g: torch.Tensor):
        pooled = feat.mean(dim=(2, 3))                 # (B, C) image-wide average
        g = self.mix(torch.cat([g, pooled], dim=1))    # update the global vector
        bias = self.to_features(g)[:, :, None, None]   # (B, C, 1, 1)
        return feat + bias, g                          # same value added everywhere


class CarPaintNet(nn.Module):
    """U-Net with a global track and a layer-parameter head."""

    def __init__(self, in_ch: int = 3, out_ch: int = 8, n_scalars: int = 5,
                 base: int = 32, depth: int = 5, c_global: int = 128):
        super().__init__()
        self.depth = depth
        self.c_global = c_global
        self.n_scalars = n_scalars

        chans = [min(base * 2 ** i, 256) for i in range(depth)]

        # --- Encoder (down) ---
        self.enc = nn.ModuleList()
        self.enc_global = nn.ModuleList()
        self.downs = nn.ModuleList()
        c_prev = in_ch
        for c in chans:
            self.enc.append(conv_block(c_prev, c))
            self.enc_global.append(GlobalTrack(c, c_global))
            self.downs.append(nn.Conv2d(c, c, 4, stride=2, padding=1))
            c_prev = c

        # --- Bottleneck ---
        self.bottleneck = conv_block(c_prev, c_prev)
        self.bottleneck_global = GlobalTrack(c_prev, c_global)

        # --- Decoder (up) ---
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.dec_global = nn.ModuleList()
        for c in reversed(chans):
            self.ups.append(nn.ConvTranspose2d(c_prev, c, 4, stride=2, padding=1))
            self.dec.append(conv_block(c * 2, c))  # *2 because of the skip connection
            self.dec_global.append(GlobalTrack(c, c_global))
            c_prev = c

        self.head = nn.Conv2d(c_prev, out_ch, 1)

        # --- Layer-parameter head: reads the global vector, not the pixels ---
        self.scalar_head = nn.Sequential(
            nn.Linear(c_global, 128),
            nn.SELU(inplace=True),
            nn.Linear(128, 64),
            nn.SELU(inplace=True),
            nn.Linear(64, n_scalars),
        ) if n_scalars else None

    def forward(self, x: torch.Tensor) -> dict:
        b = x.shape[0]
        g = x.new_zeros(b, self.c_global)  # the global vector starts empty
        skips = []

        for block, gtrack, down in zip(self.enc, self.enc_global, self.downs):
            h = block(x)
            h, g = gtrack(h, g)
            skips.append(h)          # keep the full-resolution features for later
            x = down(h)              # then halve the resolution

        x = self.bottleneck(x)
        x, g = self.bottleneck_global(x, g)

        for up, block, gtrack, skip in zip(self.ups, self.dec, self.dec_global,
                                           reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)   # the skip connection
            x = block(x)
            x, g = gtrack(x, g)

        out = {"maps": torch.sigmoid(self.head(x))}
        if self.scalar_head is not None:
            out["scalars"] = torch.sigmoid(self.scalar_head(g))
        return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Self-test --------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    net = CarPaintNet()
    print(f"parameters: {count_params(net):,}")

    x = torch.rand(2, 3, 256, 256)
    out = net(x)
    print(f"input   {tuple(x.shape)}")
    print(f"maps    {tuple(out['maps'].shape)}  "
          f"range [{out['maps'].min():.3f}, {out['maps'].max():.3f}]")
    print(f"scalars {tuple(out['scalars'].shape)}  "
          f"range [{out['scalars'].min():.3f}, {out['scalars'].max():.3f}]")

    assert out["maps"].shape == (2, 8, 256, 256), "map output shape is wrong"
    assert out["scalars"].shape == (2, 5), "scalar output shape is wrong"

    # Gradients must reach layer 1 from BOTH heads, or a head is disconnected.
    for name in ("maps", "scalars"):
        net.zero_grad()
        o = net(x)
        (o[name] - torch.rand_like(o[name])).abs().mean().backward()
        gr = net.enc[0][0].weight.grad
        assert gr is not None and gr.abs().sum() > 0, f"no gradient from {name}"
        print(f"backward via {name:<8} OK - gradients reach the first layer")

    # The global track should let a change in one corner affect distant pixels.
    net.eval()
    with torch.no_grad():
        a = torch.rand(1, 3, 256, 256)
        b = a.clone()
        b[:, :, :16, :16] = 0.0
        d = (net(a)["maps"] - net(b)["maps"]).abs()
        far = d[:, :, -16:, -16:].mean().item()
    print(f"far-corner response to a local change: {far:.6f} "
          f"({'global track is working' if far > 1e-6 else 'NO global influence'})")

    print("\nRESULT: model looks correct.")