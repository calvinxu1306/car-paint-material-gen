"""
model.py — the baseline network: U-Net + a global-features track.

INPUT   photo  (B, 3, 256, 256)
OUTPUT  maps   (B, 8, 256, 256)   basecolor RGB, roughness, metallic, normal XYZ

WHY A U-NET
Predicting maps is a per-pixel job: every output pixel corresponds to an input
pixel. A U-Net is the standard shape for that. The encoder halves the image
repeatedly (seeing more context but losing detail), the decoder scales back up,
and skip connections hand the decoder the fine detail the encoder recorded on
the way down. Without the skips you'd get blurry maps with no flake detail.

WHY THE GLOBAL TRACK (this is the Deschaintre 2018 idea)
A plain U-Net is convolutional, so every output pixel only sees a limited
neighbourhood of the input. That's a real problem here: the flash makes the
middle of the photo bright and the corners dark, yet the true base colour is
the SAME everywhere. Looking through a small window, the network can't tell
"bright because the flash is here" from "bright because the paint is lighter".
Deschaintre's fix is a parallel track carrying whole-image information: at each
level we average the features over the entire image, push that through a small
MLP, and add it back to every pixel. Now every pixel knows what the image looks
like globally.

The output is squashed with a sigmoid because every target channel lives in
0..1 (that's how the maps were stored).

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
    """U-Net with a global track. ~11M parameters at the default width."""

    def __init__(self, in_ch: int = 3, out_ch: int = 8,
                 base: int = 32, depth: int = 5, c_global: int = 128):
        super().__init__()
        self.depth = depth
        self.c_global = c_global

        # Channel widths per level, capped so the deepest layers stay affordable.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return torch.sigmoid(self.head(x))    # every channel lives in 0..1


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Self-test --------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    net = CarPaintNet()
    print(f"parameters: {count_params(net):,}")

    x = torch.rand(2, 3, 256, 256)
    y = net(x)
    print(f"input  {tuple(x.shape)}")
    print(f"output {tuple(y.shape)}  range [{y.min():.3f}, {y.max():.3f}]")

    assert y.shape == (2, 8, 256, 256), "output shape is wrong"
    assert 0.0 <= y.min() and y.max() <= 1.0, "output escaped 0..1"

    # A backward pass, to prove gradients actually reach the first layer.
    loss = (y - torch.rand_like(y)).abs().mean()
    loss.backward()
    first = net.enc[0][0].weight.grad
    assert first is not None and first.abs().sum() > 0, "no gradient reached layer 1"
    print("backward pass OK - gradients reach the first layer")

    # The global track should let a change in one corner affect distant pixels,
    # which a plain convolutional net could not do.
    net.eval()
    with torch.no_grad():
        a = torch.rand(1, 3, 256, 256)
        b = a.clone()
        b[:, :, :16, :16] = 0.0            # change only the top-left corner
        delta = (net(a) - net(b)).abs()
        far = delta[:, :, -16:, -16:].mean().item()   # look at the far corner
    print(f"far-corner response to a local change: {far:.6f} "
          f"({'global track is working' if far > 1e-6 else 'NO global influence'})")

    print("\nRESULT: model looks correct.")