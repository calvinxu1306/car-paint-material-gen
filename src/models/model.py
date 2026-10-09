"""
model.py — U-Net + global-features track, with two output heads.

Two models live here:
    CarPaintNet         v1: one flash photo in (this docstring)
    MultiLightPaintNet  v2: any number of photos of the same paint in, sharing
                        CarPaintNet's trunk (see its own docstring below)

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

    `pooled` overrides what gets averaged in: CarPaintNet passes the block's
    PRE-InstanceNorm average when prenorm_global is on (see CarPaintNet).
    """

    def __init__(self, c_feat: int, c_global: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Linear(c_global + c_feat, c_global),
            nn.SELU(inplace=True),
        )
        self.to_features = nn.Linear(c_global, c_feat)

    def forward(self, feat: torch.Tensor, g: torch.Tensor,
                pooled: torch.Tensor | None = None):
        if pooled is None:
            pooled = feat.mean(dim=(2, 3))             # (B, C) image-wide average
        g = self.mix(torch.cat([g, pooled], dim=1))    # update the global vector
        bias = self.to_features(g)[:, :, None, None]   # (B, C, 1, 1)
        return feat + bias, g                          # same value added everywhere


class CarPaintNet(nn.Module):
    """U-Net with a global track and a layer-parameter head.

    prenorm_global (off by default, so v1 checkpoints behave as trained):
    every conv_block ends in InstanceNorm, which removes each channel's
    image-wide mean and scale. With the global track averaging AFTER the norm,
    the whole network gives the same output for a photo and a uniformly darker
    copy of it - it cannot see absolute brightness, only the tone curve's
    shape. With prenorm_global on, the global track averages the block's first
    convolution BEFORE the norm (as in Deschaintre 2018), so brightness and
    colour level reach the global vector and, through it, every pixel.
    Same parameters either way: v1 weights load into both settings.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 8, n_scalars: int = 5,
                 base: int = 32, depth: int = 5, c_global: int = 128,
                 prenorm_global: bool = False):
        super().__init__()
        self.prenorm_global = prenorm_global
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

    def _block(self, block: nn.Sequential, gtrack: GlobalTrack,
               x: torch.Tensor, g: torch.Tensor):
        """conv_block then global track; see prenorm_global in the class doc."""
        if not self.prenorm_global:
            return gtrack(block(x), g)
        pre = block[0](x)                          # first conv, before its norm
        return gtrack(block[1:](pre), g, pooled=pre.mean(dim=(2, 3)))

    def features(self, x: torch.Tensor):
        """The U-Net trunk: full-resolution features + the global vector.
        Split out so the multi-photo model (below) can reuse it per photo."""
        b = x.shape[0]
        g = x.new_zeros(b, self.c_global)  # the global vector starts empty
        skips = []

        for block, gtrack, down in zip(self.enc, self.enc_global, self.downs):
            h, g = self._block(block, gtrack, x, g)
            skips.append(h)          # keep the full-resolution features for later
            x = down(h)              # then halve the resolution

        x, g = self._block(self.bottleneck, self.bottleneck_global, x, g)

        for up, block, gtrack, skip in zip(self.ups, self.dec, self.dec_global,
                                           reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)   # the skip connection
            x, g = self._block(block, gtrack, x, g)
        return x, g

    def forward(self, x: torch.Tensor) -> dict:
        x, g = self.features(x)
        out = {"maps": torch.sigmoid(self.head(x))}
        if self.scalar_head is not None:
            out["scalars"] = torch.sigmoid(self.scalar_head(g))
        return out


class MultiLightPaintNet(nn.Module):
    """Any number of photos of the same paint in, one material out.

    INPUT   photos (B, K, 3, H, W) - K photos of the same sample, same camera,
            different lights. K can change from batch to batch.
    OUTPUT  dict:
        "maps"     (B, 8, H, W)   as CarPaintNet
        "scalars"  (B, S)         layer parameters, normalised 0..1
        "pigment"  (B, T)         logits over pigment types (solid, metallic, ...)

    HOW THE PHOTOS ARE COMBINED
    Every photo goes through the SAME U-Net trunk (shared weights - one network,
    run K times). That gives K feature maps and K global vectors. They are
    then pooled across photos with max AND mean, which don't care how many
    photos there are or in what order (the Deep Sets result, Zaheer et al.
    2017). Deschaintre et al. 2019 and Kaltheuner et al. 2021 fuse multi-image
    SVBRDF networks the same way, with max-pooling.

    Max keeps the strongest evidence any photo has (the side photo that caught
    a flake's colour); mean keeps what they agree on. Two convs after pooling
    let the network compare the two, added on top of the mean (residual).

    With K = 1, max = mean = that photo's features, so this is v1's network
    plus a slightly wider head - and v1 checkpoints can initialise the trunk
    (see load_trunk_from).

    TWO OPTIONS (off by default; recorded in config.json):
    coords      every photo gets two extra input channels, the pixel's x and y
                in -1..1 (Deschaintre et al. 2019 do this). Convolutions
                can't tell where in the picture they are, and here position
                matters: the flash hotspot sits at the centre, a side light's
                coat highlight wherever that light puts it.
    flash_slot  photos[:, 0] must be the flash photo. Its features and global
                vector get their own slot next to the max and mean over all
                photos, so the network always knows what the flash photo saw
                (Boss et al. 2020 keep the flash photo in its own pathway for
                the same reason). Order still doesn't matter among the side
                photos. Needs a photo policy that always includes the flash.
    """

    def __init__(self, out_ch: int = 8, n_scalars: int = 10, n_pigments: int = 5,
                 base: int = 32, depth: int = 5, c_global: int = 128,
                 prenorm_global: bool = True, coords: bool = False,
                 flash_slot: bool = False):
        super().__init__()
        self.coords = coords
        self.flash_slot = flash_slot
        n_pool = 3 if flash_slot else 2      # max, mean (, flash photo)
        # prenorm_global ON here (see CarPaintNet): without it every photo is
        # normalised on its own, so the network cannot compare how bright the
        # flash photo is with the side-lit ones - a direct cue for the coat.
        self.trunk = CarPaintNet(in_ch=5 if coords else 3, out_ch=out_ch,
                                 n_scalars=0, base=base, depth=depth,
                                 c_global=c_global, prenorm_global=prenorm_global)
        del self.trunk.head          # the trunk's own map head isn't used
        c = base                     # channels of the trunk's final features
        # NO InstanceNorm here. InstanceNorm subtracts each image's own mean
        # from every channel, and most of a paint's map IS that mean (one base
        # colour, one roughness). The trunk puts it back with its global track;
        # a norm after the trunk would strip it out again. (The first version
        # had conv_block here and failed the overfit test: map loss stuck at
        # 0.16 where v1 reached 0.017.) So: plain convs, a residual path from
        # the mean-pooled features (with one photo this is v1's head input),
        # and the pooled global vector added back as a per-channel bias.
        self.fuse = nn.Sequential(
            nn.Conv2d(n_pool * c, c, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.fuse_global = nn.Linear(n_pool * c_global, c)
        self.head = nn.Conv2d(c, out_ch, 1)

        def mlp(n_out):
            return nn.Sequential(nn.Linear(n_pool * c_global, 128), nn.SELU(inplace=True),
                                 nn.Linear(128, 64), nn.SELU(inplace=True),
                                 nn.Linear(64, n_out))
        self.scalar_head = mlp(n_scalars) if n_scalars else None
        self.pigment_head = mlp(n_pigments) if n_pigments else None

    def forward(self, photos: torch.Tensor) -> dict:
        if photos.dim() == 4:                 # a single photo per sample
            photos = photos[:, None]
        b, k = photos.shape[:2]
        x = photos.flatten(0, 1)                              # (B*K, 3, H, W)
        if self.coords:
            h, w = x.shape[-2:]
            ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
            xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            x = torch.cat([x, torch.stack([gx, gy]).expand(x.shape[0], -1, -1, -1)], 1)
        feat, g = self.trunk.features(x)                      # (B*K, C, H, W)
        feat = feat.unflatten(0, (b, k))
        g = g.unflatten(0, (b, k))

        f_mean = feat.mean(dim=1)
        pooled = [feat.amax(dim=1), f_mean]
        g_pooled = [g.amax(dim=1), g.mean(dim=1)]
        if self.flash_slot:                  # photos[:, 0] is the flash photo
            pooled.append(feat[:, 0])
            g_pooled.append(g[:, 0])
        pooled, g_pooled = torch.cat(pooled, dim=1), torch.cat(g_pooled, dim=1)

        x = (f_mean + self.fuse(pooled)
             + self.fuse_global(g_pooled)[:, :, None, None])
        out = {"maps": torch.sigmoid(self.head(x))}
        if self.scalar_head is not None:
            out["scalars"] = torch.sigmoid(self.scalar_head(g_pooled))
        if self.pigment_head is not None:
            out["pigment"] = self.pigment_head(g_pooled)
        return out

    def load_trunk_from(self, ckpt_path: str, map_location="cpu") -> int:
        """Initialise the shared trunk from a v1 CarPaintNet checkpoint.
        Returns how many tensors were copied."""
        state = torch.load(ckpt_path, map_location=map_location)
        state = state.get("model", state)
        own = self.trunk.state_dict()
        copied = {k: v for k, v in state.items()
                  if k in own and own[k].shape == v.shape}
        first = "enc.0.0.weight"                 # the first convolution
        if (self.coords and first in state and first in own
                and own[first].shape[1] == state[first].shape[1] + 2
                and own[first].shape[2:] == state[first].shape[2:]):
            # v1 saw RGB only: copy its RGB weights and start the two
            # coordinate channels at zero, so the trunk starts out as v1's.
            w = torch.zeros_like(own[first])
            w[:, :state[first].shape[1]] = state[first]
            copied[first] = w
        if len(copied) < len(own):
            missing = sorted(set(own) - set(copied))
            raise ValueError(
                f"{ckpt_path} is not a matching v1 CarPaintNet checkpoint: only "
                f"{len(copied)} of {len(own)} trunk tensors fit (first missing: "
                f"{missing[:3]}). --init-from expects a checkpoint from train.py.")
        own.update(copied)
        self.trunk.load_state_dict(own)
        return len(copied)


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

    # --- Multi-photo model ---
    print("\nMultiLightPaintNet")
    multi = MultiLightPaintNet()
    print(f"parameters: {count_params(multi):,}")
    photos = torch.rand(2, 4, 3, 128, 128)
    o = multi(photos)
    assert o["maps"].shape == (2, 8, 128, 128)
    assert o["scalars"].shape == (2, 10) and o["pigment"].shape == (2, 5)
    print(f"4 photos -> maps {tuple(o['maps'].shape)}, scalars "
          f"{tuple(o['scalars'].shape)}, pigment logits {tuple(o['pigment'].shape)}")
    multi.eval()
    with torch.no_grad():
        # Order of the photos must not matter...
        a = multi(photos)["scalars"]
        b = multi(photos[:, [2, 0, 3, 1]])["scalars"]
        print(f"shuffled photo order changes output by {(a - b).abs().max():.2e} "
              f"(should be ~0)")
        assert (a - b).abs().max() < 1e-4, "output depends on photo order"
        # ...and any number of photos must work.
        for k in (1, 2, 6):
            multi(torch.rand(1, k, 3, 64, 64))
        print("1, 2 and 6 photos all run")
    # A v1 checkpoint can initialise the trunk.
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "v1.pt")
        torch.save({"model": net.state_dict()}, path)
        n = multi.load_trunk_from(path)
        print(f"copied {n} tensors from a v1 checkpoint into the trunk")
        assert n > 0

    # The two options: coordinate channels and a flash slot.
    for opts in ({"coords": True}, {"flash_slot": True},
                 {"coords": True, "flash_slot": True}):
        m = MultiLightPaintNet(**opts).eval()
        with torch.no_grad():
            a = m(photos)
            assert a["maps"].shape == (2, 8, 128, 128) and a["scalars"].shape == (2, 10)
            m(torch.rand(1, 1, 3, 64, 64))                    # flash photo alone
            if opts.get("flash_slot"):
                # Side photos (1..K-1) are interchangeable; the flash photo is not.
                b = m(photos[:, [0, 3, 1, 2]])["scalars"]
                c = m(photos[:, [1, 0, 2, 3]])["scalars"]
                assert (a["scalars"] - b).abs().max() < 1e-4, "side-photo order matters"
                assert (a["scalars"] - c).abs().max() > 1e-6, "flash slot has no effect"
        print(f"options {opts}: OK" + (" - side photos interchangeable, flash photo "
                                         "is not" if opts.get("flash_slot") else ""))
    # A v1 checkpoint loads into the coords model: RGB weights copied, the two
    # coordinate channels zero, every other trunk tensor as in the plain model.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "v1.pt")
        torch.save({"model": net.state_dict()}, path)
        plain, withc = MultiLightPaintNet(), MultiLightPaintNet(coords=True)
        plain.load_trunk_from(path)
        withc.load_trunk_from(path)
        w = withc.trunk.enc[0][0].weight
        assert torch.equal(w[:, :3], net.enc[0][0].weight) and not w[:, 3:].any()
        sp, sc = plain.trunk.state_dict(), withc.trunk.state_dict()
        assert all(torch.equal(sp[k], sc[k]) for k in sp if k != "enc.0.0.weight")
        print("v1 checkpoint -> coords trunk: RGB weights copied, coordinate "
              "weights start at zero")

    print("\nRESULT: model looks correct.")
