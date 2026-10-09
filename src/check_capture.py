"""
check_capture.py — is a real-paint capture good enough to use? Check it
before you take the setup down.

Phase 5 of docs/v2_plan.md photographs real paint swatches with a phone on a
stand: one flash photo, then 4-5 photos lit by a second phone's torch from
different directions (docs/capture_kit.md is the printable checklist). The
network was trained on renders where every photo lines up pixel-for-pixel,
the exposure never changes and the side lights sit 20-65 deg above the
surface, all the way round. A real capture can break any of these without it
showing on the phone's screen, and most of them can only be fixed by
re-shooting. So this script checks, for every swatch:

    photos     every photo the same size and orientation (after the EXIF
               rotation tag, as a photo viewer shows them)
    alignment  how far each torch photo is shifted from the flash photo
               (phase correlation); more than 3 px = the phone or swatch moved
    exposure   shutter time, ISO and f-number the same in every torch photo
               (EXIF), i.e. the exposure really was locked
    clipping   how much of the swatch is blown out (any channel >= 250)
    sharpness  variance of the Laplacian over the swatch (relative to a
               blurred copy, see sharpness()), against the flash photo's
               (camera shake or focus)
    lights     where the torch was, from the coat's mirror highlight (below)
    coverage   at least 4 torch photos, no gap of more than 120 deg between
               their directions, at least one low (< 35 deg) and one high
               (> 50 deg) light
then prints a verdict per swatch and one for the whole folder.

FOLDER LAYOUT (docs/capture_kit.md)
    swatches/<swatch name>/flash.jpg      flash on
    swatches/<swatch name>/torch_1.jpg    flash off, torch at position 1
    swatches/<swatch name>/torch_2.jpg    ... (torch1.jpg, torch-1.jpg also work)
Pass the folder of swatches, or one swatch folder. JPEG, PNG or TIFF; HEIC
must be exported as JPEG first (Pillow can't read it). Other files (notes,
RAW .dng files kept alongside the JPEGs) are ignored.

WHERE THE TORCH WAS: THE COAT'S MIRROR HIGHLIGHT
Put the camera (a pinhole) at C = (0, 0, H) above the swatch centre, looking
straight down; the swatch is the plane z = 0, the torch is at L = (Lx, Ly, Lz).
A flat glossy coat shows the torch where the mirror angle holds: where the
line from L to the camera's mirror image C' = (0, 0, -H) crosses z = 0,

    P = (Lx, Ly) * H / (Lz + H)

So the highlight sits on the same side of the centre as the torch (its
azimuth is the torch's azimuth), at distance d = |P|. Seen from P the camera
and the torch are mirror images, so both are at the same elevation:

    tan(elev_P) = H / d = (Lz + H) / |(Lx, Ly)|,   elev_P = 90 deg - atan(d / H)

d comes from the photo: a camera looking straight down maps the plane
linearly, at (2 H tan(hfov / 2)) / (photo's long side in px) cm per pixel.
hfov is the field of view across the photo's LONG side: taken from the EXIF
35 mm-equivalent focal length when the photo has one, else 70 deg (a typical
phone main camera, 24-26 mm equivalent), or set with --hfov-deg.

elev_P is the elevation seen from the HIGHLIGHT, not from the swatch, and it
is always higher (tan(elev_P) = (Lz + H) / |Lxy| > Lz / |Lxy|): a torch 30 cm
from the swatch and 40 deg above it, with the camera 28 cm up, puts its
highlight 13.6 cm out, where it is seen 64 deg up. v5's side lights are
20-65 deg above the SAMPLE CENTRE, so the script also walks back up the
reflected ray (from P, away from the centre, elev_P above the surface) until
it is --torch-distance-cm (default 30, as in the capture kit) from the swatch
centre Q, and reports the direction of that point as seen from Q:

    L = P + s (cos(elev_P) u, sin(elev_P)),  u = P / |P|,  |L - Q| = R, s >= 0

(the worked example above walks back to 40.0 deg; --self-test checks the
round trip for many random torches). The azimuth hardly depends on R; the
elevation does: a torch 25 cm away read as 30 cm comes out 3-6 deg too high,
so the kit has you hold the torch at a measured distance.

No highlight in the frame means the mirror point lies beyond the frame edge
(d > d_edge), so the torch was LOWER than the elevation computed at the edge
(reported as "< X deg"). The azimuth then comes from the brightness gradient
across the swatch: a torch 30 cm away lights its own side ~15-35% more
(inverse square law, cosine of incidence). No bound is given for a photo too
bright to search, nor when no torch photo of a swatch shows a reflection:
then the mirror points most likely fell off the glossy surface (which is why
the kit puts the swatch on glossy black). A bright spot only counts as a
reflection if it is far brighter than a lit object there would be, judged
from the flash photo: a speck of dust near a low torch clips too.

Azimuths use v5's convention (params_*.json "azimuth_deg"): 0 = towards the
photo's right edge, 90 = towards its top edge, counter-clockwise. The clock
position (12 = top of the photo) is the same direction for the paper table.
In the flash photo the flash's own highlight marks the point straight below
the camera; its distance from the centre gives the camera's tilt.

--export DIR writes every swatch without errors to DIR/<swatch name>/: the
same square cropped from every photo (default: a 3 cm square at the centre,
--crop-cm; or --crop x,y,size in pixels of the photo as a viewer shows it),
resized to 256 px like v5's renders, named like v5's photos: photo.png = the
flash photo, side1.png, side2.png ... = the torch photos in their numbered
order. capture.json next to them holds the light estimates and the checks,
for a later prediction script. Note for that script: a 3 cm crop seen from
28 cm spans ~6 deg; a v5 render spans ~40 deg (the whole frame is paint), so
the crop sees a much narrower range of angles than the renders did.

RUN (from the repo root; PowerShell is happy with / in paths)
    python src/check_capture.py swatches
    python src/check_capture.py "swatches/deep red pearl" --camera-height-cm 27
    python src/check_capture.py swatches --export data/real_capture
    python src/check_capture.py --self-test
Dependencies: numpy and Pillow only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Phone photos are large (48-200 MP modes exist). These are the user's own
# files, so lift Pillow's decompression-bomb guard well above that.
Image.MAX_IMAGE_PIXELS = 400_000_000

# --- Capture geometry (defaults match docs/capture_kit.md) -------------------
CAMERA_HEIGHT_CM = 28.0
HFOV_DEG = 70.0          # across the long side: a typical phone main camera
TORCH_DISTANCE_CM = 30.0
CROP_CM = 3.0
EXPORT_RES = 256         # v5's render size
DIAG_35MM = math.hypot(36.0, 24.0)

# --- Thresholds -------------------------------------------------------------
MAX_SHIFT_PX = 3.0       # torch photo vs flash photo, in photo pixels
MIN_MATCH = 8.0          # phase-correlation peak-to-sidelobe ratio
MAX_CLIPPED = 0.02       # fraction of the swatch at >= 250 in any channel
MAX_CLIPPED_FLASH = 0.10 # the flash's own hotspot on a gloss coat is allowed some
MIN_SHARPNESS = 0.5      # torch photo's sharpness() / flash photo's
MIN_BRIGHTNESS = 0.03    # median of the swatch below this = torch missed it
SAME_PHOTO = 0.004       # mean abs difference below this = the same photo
MAX_TILT_DEG = 5.0
MIN_TORCH_PHOTOS = 4
MAX_AZIMUTH_GAP = 120.0
LOW_ELEV, HIGH_ELEV = 35.0, 50.0
MIN_GRADIENT = 0.04      # relative brightness change across the swatch

# Highlight search, on the downsampled grey copy (values 0..1)
ANALYSIS_SIZE = 1024     # long side of the copy used for alignment/highlights
HL_MIN_PEAK = 0.85       # a torch's mirror image in a gloss coat is (nearly) clipped
HL_MIN_CONTRAST = 0.3
HL_MAX_AREA = 0.005      # of the frame; a larger bright patch isn't a mirror image
HL_MAX_ASPECT = 4.0      # a long bright band (a lit card edge) isn't one either
HL_MAX_BRIGHT = 0.03     # more of the frame than this near the peak = give up
HL_VS_OBJECT = 2.0       # a reflection is this much brighter than a lit object would be

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
HEIC_EXTS = {".heic", ".heif", ".hif"}
RAW_EXTS = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf",
            ".srw", ".pef"}
SKIP_FILES = {"thumbs.db", "desktop.ini", ".ds_store", "capture.json"}
TORCH_RE = re.compile(r"torch[ _-]?(\d+)", re.IGNORECASE)

EXIF_IFD = 0x8769
TAG_ORIENTATION, TAG_MAKE, TAG_MODEL = 0x0112, 0x010F, 0x0110
TAG_EXPOSURE_TIME, TAG_FNUMBER, TAG_ISO = 0x829A, 0x829D, 0x8827
TAG_FLASH, TAG_WHITE_BALANCE, TAG_FOCAL_35MM = 0x9209, 0xA403, 0xA405


# --- Small helpers ----------------------------------------------------------
def to_clock(az: float) -> str:
    """Azimuth (0 = photo's right edge, 90 = top) -> clock position, nearest half hour."""
    half = round(((90.0 - az) % 360.0) / 15.0) / 2.0 % 12.0
    return f"{int(half) or 12}:{30 if half % 1 else 0:02d}"


def fmt_exposure(t: float | None) -> str:
    if t is None:
        return "?"
    return f"1/{round(1 / t)} s" if 0 < t < 1 else f"{t:g} s"


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def hfov_from_35mm(f35: float, width: int, height: int) -> float:
    """Field of view across the long side from a 35 mm-equivalent focal
    length, which is defined on the 43.3 mm diagonal of a 36x24 mm frame."""
    a = max(width, height) / min(width, height)
    long_mm = DIAG_35MM * a / math.hypot(a, 1.0)
    return math.degrees(2 * math.atan(long_mm / 2 / f35))


# --- Geometry (see the docstring) -------------------------------------------
def walk_back(p, elev_p_deg, q, distance):
    """The torch on the reflected ray: start at the mirror point p = (x, y)
    (cm, from the point below the camera), go away from the centre at
    elev_p above the surface, stop `distance` from q. Returns (azimuth,
    elevation) seen from q, or None if the ray never gets that far from q."""
    d = math.hypot(*p)
    ux, uy = (p[0] / d, p[1] / d) if d > 1e-9 else (1.0, 0.0)
    e = math.radians(elev_p_deg)
    v = (math.cos(e) * ux, math.cos(e) * uy, math.sin(e))
    pq = (p[0] - q[0], p[1] - q[1], 0.0)
    b = sum(vi * wi for vi, wi in zip(v, pq))
    c = sum(wi * wi for wi in pq) - distance ** 2
    disc = b * b - c
    if disc < 0:
        return None
    s = -b + math.sqrt(disc)
    if s < 0:
        return None
    lx, ly, lz = (pq[i] + s * v[i] for i in range(3))
    return (math.degrees(math.atan2(ly, lx)) % 360.0,
            math.degrees(math.atan2(lz, math.hypot(lx, ly))))


def light_from_offset(off_cm, height, q, distance):
    """Highlight at off_cm = (x right, y up) from the frame centre ->
    the light's direction, as in the docstring."""
    d = math.hypot(*off_cm)
    elev_p = math.degrees(math.atan2(height, d))
    seen = walk_back(off_cm, elev_p, q, distance)
    return {"highlight_cm": d,
            "highlight_azimuth_deg": math.degrees(math.atan2(off_cm[1], off_cm[0])) % 360.0,
            "elevation_at_highlight_deg": elev_p,
            "azimuth_deg": seen[0] if seen else None,
            "elevation_deg": seen[1] if seen else None}


def edge_offset_px(az, width, height):
    """Distance from the frame centre to its edge along azimuth az (px)."""
    c, s = math.cos(math.radians(az)), math.sin(math.radians(az))
    tx = (width / 2) / abs(c) if abs(c) > 1e-9 else math.inf
    ty = (height / 2) / abs(s) if abs(s) > 1e-9 else math.inf
    return min(tx, ty)


# --- Image measurements -----------------------------------------------------
def analysis_grey(img: Image.Image):
    """Downsampled grey copy (0..1) + the full-res pixels per copy pixel (x, y)."""
    w, h = img.size
    k = max(1.0, max(w, h) / ANALYSIS_SIZE)
    size = (max(8, round(w / k)), max(8, round(h / k)))
    g = img.convert("L").resize(size, Image.Resampling.BOX)
    return np.asarray(g, dtype=np.float32) / 255.0, (w / size[0], h / size[1])


def edges(g: np.ndarray) -> np.ndarray:
    """Gradient magnitude. Lighting can flip an edge's contrast between two
    photos (a card brighter than the backing in one, darker in the other);
    its gradient magnitude stays put."""
    gx = g[1:-1, 2:] - g[1:-1, :-2]
    gy = g[2:, 1:-1] - g[:-2, 1:-1]
    return np.hypot(gx, gy)


def phase_shift(ref: np.ndarray, img: np.ndarray):
    """How far img's content is shifted from ref's, (dx, dy) in pixels of
    these arrays, and how clear the match is (peak-to-sidelobe ratio:
    ~10+ for the same view, ~5 or less for unrelated images)."""
    a, b = edges(ref), edges(img)
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    fa = np.fft.rfft2((a - a.mean()) * win)
    fb = np.fft.rfft2((b - b.mean()) * win)
    cross = fb * np.conj(fa)
    cross /= np.abs(cross) + 1e-9 * np.abs(cross).max()
    # Damp the highest frequencies, which are mostly sensor noise.
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    cross *= np.exp(-(fx ** 2 + fy ** 2) / (2 * 0.15 ** 2))
    corr = np.fft.irfft2(cross, s=(h, w))
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak = corr[iy, ix]

    def refine(m1, c0, p1):
        # Gaussian fit through three samples (parabola on the log), else parabola.
        if m1 > 0 and c0 > 0 and p1 > 0:
            m1, c0, p1 = math.log(m1), math.log(c0), math.log(p1)
        den = m1 - 2 * c0 + p1
        return 0.5 * (m1 - p1) / den if den < 0 else 0.0

    dy = iy + refine(corr[(iy - 1) % h, ix], peak, corr[(iy + 1) % h, ix])
    dx = ix + refine(corr[iy, (ix - 1) % w], peak, corr[iy, (ix + 1) % w])
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx

    side = np.ones_like(corr, dtype=bool)
    rows = np.arange(iy - 5, iy + 6) % h
    cols = np.arange(ix - 5, ix + 6) % w
    side[np.ix_(rows, cols)] = False
    rest = corr[side]
    psr = float((peak - rest.mean()) / (rest.std() + 1e-12))
    return float(dx), float(dy), psr


def laplacian(g: np.ndarray) -> np.ndarray:
    return g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]


def box3(g: np.ndarray) -> np.ndarray:
    return sum(g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
               for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0


def sharpness(grey: np.ndarray) -> float:
    """Variance of the Laplacian, divided by the same after a 3x3 blur.

    The plain variance of the Laplacian can't be compared across photos lit
    from different sides: it measures how much fine detail there is, and a
    low torch makes far fewer flakes sparkle than the flash (on synthetic
    swatches it ranged 0.04-1.3x the flash photo's with nothing blurred).
    Dividing by the blurred copy's asks instead how much of the detail is
    finer than 3 px, which blur removes whatever the light. The swatch is
    first box-averaged by 2 or more (to about 2x the export size), so sensor
    noise, which no blur can remove after the fact, counts less."""
    g = grey.astype(np.float32)
    k = max(2, round(min(g.shape) / (2 * EXPORT_RES)))
    h, w = g.shape[0] // k * k, g.shape[1] // k * k
    if h < 16 * k or w < 16 * k:
        return float("nan")
    g = g[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    soft = float(laplacian(box3(g)).var())
    return float(laplacian(g).var()) / soft if soft > 1e-12 else float("nan")


def brightness_gradient(crop: Image.Image):
    """Fit brightness = a + b x + c y over the swatch (linear light, x right,
    y up, both -0.5..0.5). Returns (azimuth of the brighter side, relative
    change across the swatch |(b, c)| / a)."""
    small = np.asarray(crop.resize((64, 64), Image.Resampling.BOX),
                       dtype=np.float32) / 255.0
    lin = srgb_to_linear(small)
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    t = (np.arange(64) + 0.5) / 64 - 0.5
    yy, xx = np.meshgrid(-t, t, indexing="ij")          # row 0 = top = +y
    A = np.stack([np.ones(lum.size), xx.ravel(), yy.ravel()], axis=1)
    (a, b, c), *_ = np.linalg.lstsq(A, lum.ravel(), rcond=None)
    if a <= 1e-6:
        return None, 0.0
    return math.degrees(math.atan2(c, b)) % 360.0, float(math.hypot(b, c) / a)


def components(mask: np.ndarray):
    """8-connected regions of a sparse mask, as arrays of (row, col)."""
    label = np.full(mask.shape, -1, dtype=np.int32)
    h, w = mask.shape
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if label[y0, x0] >= 0:
            continue
        idx = len(out)
        label[y0, x0] = idx
        stack, pts = [(y0, x0)], []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if (0 <= ny < h and 0 <= nx < w and mask[ny, nx]
                            and label[ny, nx] < 0):
                        label[ny, nx] = idx
                        stack.append((ny, nx))
        out.append(np.array(pts))
    return out


def find_highlight(grey: np.ndarray, scale, ref: np.ndarray | None = None):
    """The torch's (or flash's) mirror image in a glossy surface: the
    brightest small, compact spot that stands well clear of its
    surroundings. Returns its centre in full-res pixels (x, y) and how it
    was found, or None plus the reason.

    ref (the flash photo, for a torch photo) rules out lit objects: a speck
    of dust or a label near a low torch can clip too, but it is also there
    in the flash photo, and a diffuse object's brightness scales with the
    light falling on it (its surroundings show by how much). A reflection is
    far brighter than that prediction."""
    g = grey.copy()
    g[1:-1, 1:-1] = box3(grey)
    if ref is not None:
        lin_g = srgb_to_linear(g)
        lin_r = srgb_to_linear(ref)
        lin_r[1:-1, 1:-1] = box3(lin_r)
    top = float(g.max())
    if top < HL_MIN_PEAK:
        return None, "nothing bright enough"
    thr = max(HL_MIN_PEAK, 0.5 * (top + float(np.median(g))))
    mask = g >= thr
    if mask.mean() > HL_MAX_BRIGHT:
        return None, f"too bright: {100 * mask.mean():.0f}% of the photo is near its peak"
    best, objects = None, 0
    for pts in components(mask):
        ys, xs = pts[:, 0], pts[:, 1]
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if (len(pts) > HL_MAX_AREA * g.size
                or max(y1 - y0, x1 - x0) > HL_MAX_ASPECT * min(y1 - y0, x1 - x0)):
            continue
        # Background: a ring around the spot's box, one to two pads out.
        pad = max(4, (y1 - y0 + x1 - x0) // 2)
        oy, ox = max(0, y0 - 2 * pad), max(0, x0 - 2 * pad)
        box = (slice(oy, y1 + 2 * pad), slice(ox, x1 + 2 * pad))
        ring = np.ones(g[box].shape, dtype=bool)
        ring[max(0, y0 - pad - oy):y1 + pad - oy, max(0, x0 - pad - ox):x1 + pad - ox] = False
        if not ring.any():
            continue
        bg = float(np.median(g[box][ring]))
        peak = float(g[ys, xs].max())
        if peak - bg < HL_MIN_CONTRAST:
            continue
        if ref is not None:
            # What a diffuse object would look like here: its contrast in the
            # flash photo, times how much brighter the torch lights this spot.
            t_bg, r_bg = float(np.median(lin_g[box][ring])), float(np.median(lin_r[box][ring]))
            as_object = (max(0.0, float(lin_r[ys, xs].mean()) - r_bg)
                         * t_bg / max(r_bg, 1e-3))
            if float(lin_g[ys, xs].mean()) - t_bg < HL_VS_OBJECT * as_object:
                objects += 1
                continue
        wts = g[ys, xs] - thr + 1e-3
        cand = {"x": (float((xs * wts).sum() / wts.sum()) + 0.5) * scale[0],
                "y": (float((ys * wts).sum() / wts.sum()) + 0.5) * scale[1],
                "peak": peak, "contrast": peak - bg, "area": len(pts) / g.size}
        if best is None or (peak, cand["contrast"]) > (best["peak"], best["contrast"]):
            best = cand
    if best is None:
        return None, (f"only lit objects ({objects}), no reflection" if objects
                      else "no small bright spot")
    return best, "found"


# --- Reading a swatch folder ------------------------------------------------
def list_photos(folder: str):
    """flash + numbered torch photos. Returns (flash path, [(n, path)], issues)."""
    issues, by_role, raws = [], {}, []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isdir(path) or name.startswith(".") or name.lower() in SKIP_FILES:
            continue
        stem, ext = os.path.splitext(name)
        ext = ext.lower()
        m = TORCH_RE.fullmatch(stem)
        role = "flash" if stem.lower() == "flash" else (int(m.group(1)) if m else None)
        known = ext in PHOTO_EXTS | HEIC_EXTS | RAW_EXTS
        if role is None:
            if known:
                issues.append(("NOTE", f"{name}: not named flash.* or torch_N.*, ignored"))
            continue
        if not known:
            issues.append(("NOTE", f"{name}: not a photo format this script reads, ignored"))
            continue
        by_role.setdefault(role, []).append(name)

    chosen = {}
    for role, names in by_role.items():
        label = "flash" if role == "flash" else f"torch_{role}"
        usable = [n for n in names if os.path.splitext(n)[1].lower() in PHOTO_EXTS]
        heic = [n for n in names if os.path.splitext(n)[1].lower() in HEIC_EXTS]
        raw = [n for n in names if os.path.splitext(n)[1].lower() in RAW_EXTS]
        if len(usable) > 1:
            issues.append(("ERROR", f"two photos for {label}: {', '.join(usable)}. Keep one."))
        elif usable:
            chosen[role] = os.path.join(folder, usable[0])
            raws += raw
        elif heic:
            issues.append(("ERROR", f"{heic[0]} is HEIC, which Pillow can't read. Export it "
                                    f"as JPEG (iPhone: Settings > Camera > Formats > Most "
                                    f"Compatible for future shots, or share it as JPEG)."))
        else:
            issues.append(("ERROR", f"{raw[0]} is a RAW file only. Save a JPEG next to it "
                                    f"(RAW+JPEG in the camera app, or export one)."))
    if raws:
        issues.append(("NOTE", f"RAW files kept but not checked: {', '.join(raws)}"))

    flash = chosen.pop("flash", None)
    if flash is None and "flash" not in by_role:
        issues.append(("ERROR", "no flash photo (flash.jpg)"))
    torch = sorted(chosen.items())
    numbers = sorted(n for n in by_role if n != "flash")
    if numbers:
        missing = [n for n in range(min(1, numbers[0]), numbers[-1] + 1) if n not in by_role]
        if missing:
            issues.append(("WARN", "torch photo numbering skips "
                                   + ", ".join(f"torch_{n}" for n in missing)
                                   + " - deleted, renamed or never taken?"))
    return flash, torch, issues


def read_exif(img: Image.Image) -> dict:
    exif = img.getexif()
    sub = exif.get_ifd(EXIF_IFD) if exif else {}

    def num(tag):
        v = sub.get(tag, exif.get(tag))
        if isinstance(v, (tuple, list)):
            v = v[0] if v else None
        try:
            f = float(v) if v is not None else None
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        return None if f is None or not math.isfinite(f) or f <= 0 else f

    def text(tag):
        v = exif.get(tag)
        return str(v).strip("\x00 ") if v else ""

    flash = sub.get(TAG_FLASH)
    wb = sub.get(TAG_WHITE_BALANCE)
    return {"exposure_time": num(TAG_EXPOSURE_TIME), "iso": num(TAG_ISO),
            "f_number": num(TAG_FNUMBER), "focal_35mm": num(TAG_FOCAL_35MM),
            "flash_fired": (int(flash) & 1 == 1) if isinstance(flash, int) else None,
            "white_balance": ("auto" if wb == 0 else "manual" if wb == 1 else None),
            "orientation": int(exif.get(TAG_ORIENTATION, 1) or 1),
            "camera": " ".join(t for t in (text(TAG_MAKE), text(TAG_MODEL)) if t)}


def open_photo(path: str, rotate: bool):
    """-> (RGB image as a viewer shows it, size as stored, EXIF dict), or an error message."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
        if head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"mif1", b"msf1",
                                                     b"heim", b"heis", b"hevc"):
            return f"{os.path.basename(path)} is really a HEIC file with a JPEG name. " \
                   f"Export it as JPEG."
        with Image.open(path) as im:
            if im.mode.startswith("I") or im.mode == "F":
                return (f"{os.path.basename(path)} is a 16-bit/float image; export an 8-bit "
                        f"JPEG or PNG")
            exif = read_exif(im)
            stored = im.size
            out = ImageOps.exif_transpose(im) if rotate else im.copy()
            return out.convert("RGB"), stored, exif
    except UnidentifiedImageError:
        return f"can't read {os.path.basename(path)}: not a JPEG/PNG/TIFF image"
    except OSError as e:
        return f"can't read {os.path.basename(path)}: {e}"


# --- One swatch -------------------------------------------------------------
def check_swatch(folder: str, opts) -> dict:
    """Run every check on one swatch folder. Returns the report as a dict;
    'crops' holds the cropped photos for --export (not JSON)."""
    rep = {"name": os.path.basename(os.path.normpath(folder)), "folder": folder,
           "issues": [], "photos": [], "crops": {}}
    issue = lambda level, msg: rep["issues"].append((level, msg))
    flash_path, torch, file_issues = list_photos(folder)
    rep["issues"] += file_issues
    if flash_path is None:
        return rep
    if not torch:
        issue("ERROR", "no torch photos (torch_1.jpg, torch_2.jpg, ...)")

    loaded = open_photo(flash_path, opts.rotate)
    if isinstance(loaded, str):
        issue("ERROR", loaded)
        return rep
    img, stored, exif = loaded
    W, H = img.size
    rep["size"] = [W, H]
    rep["camera"] = exif["camera"]

    # Geometry: cm per pixel on the swatch plane.
    if opts.hfov_deg:
        hfov, hfov_src = opts.hfov_deg, "--hfov-deg"
    elif exif["focal_35mm"]:
        hfov = hfov_from_35mm(exif["focal_35mm"], W, H)
        hfov_src = f"EXIF: {exif['focal_35mm']:g} mm equivalent"
    else:
        hfov, hfov_src = HFOV_DEG, "default, typical phone main camera"
    cm_per_px = 2 * opts.camera_height_cm * math.tan(math.radians(hfov) / 2) / max(W, H)
    rep["geometry"] = {"camera_height_cm": opts.camera_height_cm, "hfov_deg": hfov,
                       "hfov_source": hfov_src, "torch_distance_cm": opts.torch_distance_cm,
                       "cm_per_px": cm_per_px}

    # The swatch region: --crop, or a --crop-cm square at the centre.
    if opts.crop:
        cx0, cy0, side = opts.crop
        crop_src = "--crop"
        if cx0 < 0 or cy0 < 0 or cx0 + side > W or cy0 + side > H:
            issue("ERROR", f"--crop {cx0},{cy0},{side} is outside the {W}x{H} photo")
            return rep
    else:
        side = min(W, H, max(16, round(opts.crop_cm / cm_per_px)))
        cx0, cy0 = (W - side) // 2, (H - side) // 2
        crop_src = f"{side * cm_per_px:.1f} cm square at the centre"
    box = (cx0, cy0, cx0 + side, cy0 + side)
    q = ((cx0 + side / 2 - W / 2) * cm_per_px, -(cy0 + side / 2 - H / 2) * cm_per_px)
    rep["crop"] = {"x": cx0, "y": cy0, "size": side, "size_cm": side * cm_per_px,
                   "source": crop_src,
                   "view_angle_deg": math.degrees(2 * math.atan(
                       side * cm_per_px / 2 / opts.camera_height_cm))}

    def measure(name, img, exif, stored):
        crop = img.crop(box)
        rgb = np.asarray(crop)
        grey_crop = np.asarray(crop.convert("L"), dtype=np.float32)
        grey, scale = analysis_grey(img)
        return {"file": name, "stored_size": list(stored), "exif": exif,
                "clipped": float((rgb.max(axis=2) >= 250).mean()),
                "brightness": float(np.median(grey_crop)) / 255.0,
                "sharpness_raw": sharpness(grey_crop),
                "crop": crop, "grey": grey, "scale": scale}

    flash = measure(os.path.basename(flash_path), img, exif, stored)
    flash["role"] = "photo"
    del img

    # The flash's own highlight sits straight below the camera: its offset
    # from the centre is the camera's tilt.
    hl, why = find_highlight(flash["grey"], flash["scale"])
    if hl:
        off = math.hypot(hl["x"] - W / 2, hl["y"] - H / 2) * cm_per_px
        tilt = math.degrees(math.atan2(off, opts.camera_height_cm))
        rep["flash_highlight"] = {"x": hl["x"], "y": hl["y"], "offset_cm": off, "tilt_deg": tilt}
        if tilt > MAX_TILT_DEG:
            issue("WARN", f"the flash highlight is {off:.1f} cm from the centre: the camera "
                          f"is tilted ~{tilt:.0f} deg. Point it straight down (use the "
                          f"camera's level).")
    else:
        rep["flash_highlight"] = None
        issue("NOTE", f"no flash highlight in {flash['file']} ({why}): fine for matte "
                      f"paint; otherwise check the phone points straight down")
    if exif["flash_fired"] is False:
        issue("NOTE", f"EXIF says the flash did not fire in {flash['file']} (fine if the "
                      f"phone's own light was on as a torch)")
    rep["photos"].append(flash)

    # Torch photos, one at a time (phone photos are big).
    for n, path in torch:
        name = os.path.basename(path)
        loaded = open_photo(path, opts.rotate)
        if isinstance(loaded, str):
            issue("ERROR", loaded)
            continue
        img, stored, exif = loaded
        if img.size != (W, H):
            same_stored = tuple(stored) == tuple(flash["stored_size"])
            hint = (" They are the same size as stored, so the phone changed its rotation "
                    "tag between shots (common when it points straight down): rerun with "
                    "--no-rotate." if same_stored and opts.rotate else
                    " Same phone, same camera, same zoom for every photo.")
            issue("ERROR", f"{name} is {img.size[0]}x{img.size[1]} but {flash['file']} is "
                           f"{W}x{H}.{hint}")
            continue
        ph = measure(name, img, exif, stored)
        del img
        ph["role"] = f"side{len([p for p in rep['photos'] if p['role'] != 'photo']) + 1}"
        ph["torch_number"] = n

        dx, dy, psr = phase_shift(flash["grey"], ph["grey"])
        ph["shift_px"] = [dx * ph["scale"][0], dy * ph["scale"][1]]
        ph["match"] = psr
        ph["sharpness"] = (ph["sharpness_raw"] / flash["sharpness_raw"]
                           if flash["sharpness_raw"] > 0 else float("nan"))
        ph["light"] = estimate_light(ph, flash, W, H, cm_per_px, q, opts)
        rep["photos"].append(ph)

    torch_ph = [p for p in rep["photos"] if p["role"] != "photo"]
    if torch_ph and not any(p["light"]["source"] == "highlight" for p in torch_ph):
        # No reflection in any photo: most likely nothing glossy around the
        # swatch, and then a missing reflection says nothing about elevation.
        for p in torch_ph:
            p["light"]["elevation_max_deg"] = p["light"]["elevation_at_edge_deg"] = None
    judge_photos(rep, flash, torch_ph, issue)
    judge_exposure(flash, torch_ph, issue)
    judge_coverage(rep, torch_ph, issue)
    rep["crops"] = {p["role"]: p.pop("crop") for p in rep["photos"]}
    for p in rep["photos"]:
        p.pop("grey"), p.pop("scale")
    return rep


def estimate_light(ph, flash, W, H, cm_per_px, q, opts) -> dict:
    hl, why = find_highlight(ph["grey"], ph["scale"], ref=flash["grey"])
    grad_az, grad = brightness_gradient(ph["crop"])
    light = {"gradient_azimuth_deg": grad_az if grad >= MIN_GRADIENT else None,
             "gradient_strength": grad}
    if hl:
        off = ((hl["x"] - W / 2) * cm_per_px, -(hl["y"] - H / 2) * cm_per_px)
        light.update(light_from_offset(off, opts.camera_height_cm, q, opts.torch_distance_cm))
        light.update(source="highlight", highlight_px=[hl["x"], hl["y"]])
        if light["azimuth_deg"] is None:     # torch nearer than assumed
            light["azimuth_deg"] = light["highlight_azimuth_deg"]
        return light
    # No mirror image in the frame: the mirror point lies beyond the edge, so
    # the torch was lower than it would be with the highlight right at the edge
    # - unless the photo was too bright to find one at all.
    az = light["gradient_azimuth_deg"]
    # The gradient's azimuth is rough, so take the nearest edge within 30 deg
    # of it (any direction if there's no gradient): the bound then still holds.
    near = range(-30, 31, 5) if az is not None else range(0, 360, 5)
    edge_cm = min(edge_offset_px((az or 0.0) + a, W, H) for a in near) * cm_per_px
    edge = ((edge_cm * math.cos(math.radians(az)), edge_cm * math.sin(math.radians(az)))
            if az is not None else (edge_cm, 0.0))
    bound = light_from_offset(edge, opts.camera_height_cm, q, opts.torch_distance_cm)
    bounded = not why.startswith("too bright")
    light.update(source="gradient" if az is not None else None, no_highlight=why,
                 azimuth_deg=az, elevation_deg=None,
                 elevation_max_deg=bound["elevation_deg"] if bounded else None,
                 elevation_at_edge_deg=bound["elevation_at_highlight_deg"] if bounded else None)
    return light


def judge_photos(rep, flash, torch_ph, issue):
    if flash["clipped"] > MAX_CLIPPED_FLASH:
        issue("WARN", f"{flash['file']}: {100 * flash['clipped']:.0f}% of the swatch is "
                      f"clipped to white (a small hotspot is normal for gloss paint, this "
                      f"much is not): lower the exposure for the flash photo")
    for ph in torch_ph:
        name = ph["file"]
        sx, sy = ph["shift_px"]
        shift = math.hypot(sx, sy)
        if ph["match"] < MIN_MATCH:
            issue("WARN", f"{name} could not be matched to {flash['file']} (match "
                          f"{ph['match']:.1f}): rotated, zoomed or moved a lot? Re-shoot it.")
        elif shift > MAX_SHIFT_PX:
            issue("WARN", f"{name} is shifted {shift:.1f} px from {flash['file']} (more than "
                          f"{MAX_SHIFT_PX:g}): the phone or the swatch moved. Re-shoot it.")
        if ph["clipped"] > MAX_CLIPPED:
            issue("WARN", f"{name}: {100 * ph['clipped']:.0f}% of the swatch is clipped to "
                          f"white: torch too close/bright, or the exposure too high")
        if ph["brightness"] < MIN_BRIGHTNESS:
            issue("WARN", f"{name}: the swatch is nearly black (median "
                          f"{255 * ph['brightness']:.0f}/255): torch off or pointing away?")
        if ph["sharpness"] < MIN_SHARPNESS:
            issue("WARN", f"{name} is blurry: sharpness {ph['sharpness']:.2f} of the flash "
                          f"photo's. Camera shake (use the timer) or focus moved.")
        if ph["exif"]["flash_fired"]:
            issue("WARN", f"EXIF says the flash fired in {name}: torch photos need the flash OFF")
        for other in [flash] + torch_ph[:torch_ph.index(ph)]:
            if float(np.abs(ph["grey"] - other["grey"]).mean()) < SAME_PHOTO:
                issue("WARN", f"{name} looks the same as {other['file']}: same file twice, "
                              f"or the torch didn't move?")
                break
        light = ph["light"]
        if (light.get("source") == "highlight" and light["gradient_azimuth_deg"] is not None):
            diff = abs((light["azimuth_deg"] - light["gradient_azimuth_deg"] + 180) % 360 - 180)
            if diff > 90:
                issue("NOTE", f"{name}: the highlight and the brightness gradient point "
                              f"{diff:.0f} deg apart; is the bright spot really the torch?")
    if torch_ph:
        torch_sharp = float(np.median([p["sharpness"] for p in torch_ph]))
        if torch_sharp > 1 / MIN_SHARPNESS:
            issue("WARN", f"{flash['file']} looks blurrier than the torch photos (they are "
                          f"{torch_sharp:.1f}x sharper): re-shoot the flash photo")
    cams = {p["exif"]["camera"] for p in rep["photos"] if p["exif"]["camera"]}
    if len(cams) > 1:
        issue("WARN", f"photos from different cameras: {', '.join(sorted(cams))}")
    orients = {p["exif"]["orientation"] for p in rep["photos"]}
    if len(orients) > 1:
        # Only a note: if applying the tags rotated one photo against the
        # others, the alignment check above has already failed for it.
        issue("NOTE", f"EXIF rotation tags differ between photos ({sorted(orients)}): a "
                      f"phone pointing straight down often flips them. They were applied; "
                      f"if a photo didn't match, try --no-rotate.")


def judge_exposure(flash, torch_ph, issue):
    if not torch_ph:
        return
    fields = [("exposure_time", "shutter time", fmt_exposure),
              ("iso", "ISO", lambda v: f"{v:g}"), ("f_number", "f-number", lambda v: f"f/{v:g}")]
    missing = []
    for key, label, show in fields:
        vals = [(p["file"], p["exif"][key]) for p in torch_ph]
        have = [v for _, v in vals if v is not None]
        if len(have) < len(vals):
            missing.append(label)
        if len(have) >= 2 and max(have) > min(have) * 1.02:
            issue("WARN", f"{label} changed between torch photos ("
                          + ", ".join(f"{n} {show(v)}" for n, v in vals if v is not None)
                          + "): the exposure was not locked. Use Pro/manual mode or AE lock.")
    if missing:
        issue("NOTE", f"no EXIF {', '.join(missing)} in some torch photos, so the exposure "
                      f"lock can't be checked (an editing app may have stripped EXIF)")
    if any(p["exif"]["white_balance"] == "auto" for p in torch_ph + [flash]):
        issue("NOTE", "white balance was automatic (EXIF): colours can shift between "
                      "photos. Use a fixed white balance (Pro mode) next time.")


def judge_coverage(rep, torch_ph, issue):
    distance = rep["geometry"]["torch_distance_cm"]
    lights = [p["light"] for p in torch_ph]
    n = len(lights)
    if 0 < n < MIN_TORCH_PHOTOS:
        issue("WARN", f"only {n} torch photo{'s' if n > 1 else ''}; take at least "
                      f"{MIN_TORCH_PHOTOS} (5 is best: v5 has 5 side lights)")
    az = sorted(l["azimuth_deg"] for l in lights if l["azimuth_deg"] is not None)
    cov = {"azimuths_deg": az, "largest_gap_deg": None, "low": False, "high": False}
    if len(az) >= 2:
        gaps = [b - a for a, b in zip(az, az[1:])] + [az[0] + 360 - az[-1]]
        cov["largest_gap_deg"] = max(gaps)
        if max(gaps) > MAX_AZIMUTH_GAP:
            i = int(np.argmax(gaps))
            mid = (az[i] + gaps[i] / 2) % 360
            issue("WARN", f"no torch photo between azimuth {az[i]:.0f} and "
                          f"{az[(i + 1) % len(az)]:.0f} deg (a {max(gaps):.0f} deg gap): add one "
                          f"from about {to_clock(mid)} o'clock")
    if n and len(az) < n:
        issue("NOTE", f"{n - len(az)} torch photo(s) without a direction estimate")
    cov["low"] = any((l["elevation_deg"] is not None and l["elevation_deg"] < LOW_ELEV)
                     or (l.get("elevation_max_deg") is not None
                         and l["elevation_max_deg"] < LOW_ELEV) for l in lights)
    cov["high"] = any(l["elevation_deg"] is not None and l["elevation_deg"] > HIGH_ELEV
                      for l in lights)
    if n and not any(l["source"] == "highlight" for l in lights):
        issue("WARN", "no torch reflection in any photo, so the elevations are unknown: put "
                      "the swatch on glossy black (docs/capture_kit.md), or check your table "
                      f"has a torch below {LOW_ELEV:g} deg and one above {HIGH_ELEV:g} deg")
    else:
        assumed = (f" (elevations assume the torch was {distance:g} cm from the swatch; "
                   f"set --torch-distance-cm if it wasn't)")
        if n and not cov["low"]:
            issue("WARN", f"no light confirmed below {LOW_ELEV:g} deg: add a low, raking "
                          f"torch position (about 10 cm above the table)" + assumed)
        if n and not cov["high"]:
            issue("WARN", f"no light confirmed above {HIGH_ELEV:g} deg: add a high torch "
                          f"position (about 25 cm up, close to the phone)" + assumed)
    rep["coverage"] = cov


# --- Printing ---------------------------------------------------------------
def verdict(issues) -> str:
    levels = {lvl for lvl, _ in issues}
    return "ERROR" if "ERROR" in levels else "WARN" if "WARN" in levels else "OK"


def describe_light(l: dict) -> str:
    if l.get("source") == "highlight":
        az = f"{l['azimuth_deg']:5.0f} deg ({to_clock(l['azimuth_deg']):>5})"
        el = (f"{l['elevation_deg']:4.0f} deg" if l["elevation_deg"] is not None
              else "   ? (torch nearer than --torch-distance-cm?)")
        return (f"{az}  {el}   [highlight {l['highlight_cm']:.1f} cm out, "
                f"{l['elevation_at_highlight_deg']:.0f} deg seen from there]")
    az = (f"{l['azimuth_deg']:5.0f} deg ({to_clock(l['azimuth_deg']):>5})"
          if l["azimuth_deg"] is not None else "    ?            ")
    el = (f"< {l['elevation_max_deg']:.0f} deg" if l.get("elevation_max_deg") is not None
          else "?")
    how = ("azimuth from the brightness gradient" if l["azimuth_deg"] is not None
           else "no clear brightness gradient either")
    return f"{az}  {el:>8}   [no highlight in frame; {how}]"


def print_report(rep: dict) -> None:
    print(f"\n=== {rep['name']}  ({rep['folder']})")
    photos = rep["photos"]
    if photos:
        g, c = rep["geometry"], rep["crop"]
        n_torch = len(photos) - 1
        print(f"  {photos[0]['file']} + {n_torch} torch photo{'s' if n_torch != 1 else ''}, "
              f"{rep['size'][0]}x{rep['size'][1]}"
              + (f", {rep['camera']}" if rep["camera"] else ""))
        print(f"  geometry  camera {g['camera_height_cm']:g} cm up, {g['hfov_deg']:.0f} deg "
              f"across the long side ({g['hfov_source']}) -> {g['cm_per_px'] * 10:.3f} mm per px")
        print(f"  swatch    x={c['x']} y={c['y']} size={c['size']} px ({c['source']})")
        fh = rep.get("flash_highlight")
        if fh:
            print(f"  flash     highlight {fh['offset_cm']:.1f} cm from the centre -> camera "
                  f"tilt ~{fh['tilt_deg']:.0f} deg")
        print(f"\n  {'photo':<16}{'shift px':>9}{'match':>7}{'sharp':>7}{'clipped':>9}   "
              f"light: azimuth (clock)  elevation at the swatch")
        for p in photos:
            if p["role"] == "photo":
                print(f"  {p['file']:<16}{'-':>9}{'-':>7}{1.0:>7.2f}"
                      f"{100 * p['clipped']:>8.1f}%   (flash)")
                continue
            print(f"  {p['file']:<16}{math.hypot(*p['shift_px']):>9.1f}{p['match']:>7.1f}"
                  f"{p['sharpness']:>7.2f}{100 * p['clipped']:>8.1f}%   "
                  f"{describe_light(p['light'])}")
        cov = rep.get("coverage")
        if cov and cov["azimuths_deg"]:
            gap = (f"largest gap {cov['largest_gap_deg']:.0f} deg"
                   if cov["largest_gap_deg"] is not None else "one direction only")
            print(f"\n  coverage  azimuths {', '.join(f'{a:.0f}' for a in cov['azimuths_deg'])} "
                  f"-> {gap}; low (< {LOW_ELEV:g} deg): {'yes' if cov['low'] else 'no'}; "
                  f"high (> {HIGH_ELEV:g} deg): {'yes' if cov['high'] else 'no'}")
    if rep["issues"]:
        print()
    order = {"ERROR": 0, "WARN": 1, "NOTE": 2}
    for level, msg in sorted(rep["issues"], key=lambda i: order[i[0]]):
        print(f"  {level:<5} {msg}")
    print(f"  verdict: {verdict(rep['issues'])}")


# --- Export -----------------------------------------------------------------
def finite(x):
    """NaN/inf -> None, so capture.json is plain JSON."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return x


def jsonable(rep: dict) -> dict:
    keep = ("file", "role", "torch_number", "shift_px", "match", "sharpness", "clipped",
            "brightness", "light", "exif")
    return {
        "written_by": "src/check_capture.py",
        "swatch": rep["name"],
        "source_folder": os.path.abspath(rep["folder"]),
        "photos": [p["role"] for p in rep["photos"]],
        "resolution": EXPORT_RES,
        "image_size": rep["size"],
        "crop": rep["crop"],
        "geometry": rep["geometry"],
        "conventions": "azimuth_deg: 0 = towards the photo's right edge, 90 = towards "
                       "its top edge, counter-clockwise (as v5's params files). "
                       "elevation_deg: above the surface, seen from the swatch (crop) "
                       "centre, assuming the torch was geometry.torch_distance_cm away; "
                       "elevation_max_deg: an upper bound when no highlight was found.",
        "flash_highlight": rep.get("flash_highlight"),
        "coverage": rep.get("coverage"),
        "per_photo": [{k: p[k] for k in keep if k in p} for p in rep["photos"]],
        "issues": [{"level": lvl, "message": msg} for lvl, msg in rep["issues"]],
        "verdict": verdict(rep["issues"]),
    }


def export_swatch(rep: dict, out_root: str) -> str:
    folder = os.path.join(out_root, rep["name"])
    # The folder is replaced, so make sure this script wrote it: a mistyped
    # --export (say, the swatches folder itself) must not delete photos.
    if os.path.isdir(folder) and os.listdir(folder) and (
            not os.path.exists(os.path.join(folder, "capture.json")) or has_photos(folder)):
        return (f"not exported: {folder} is not empty and wasn't written by this script "
                f"(no capture.json, or it holds flash/torch photos). Refusing to replace it.")
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    for role, crop in rep["crops"].items():
        crop.resize((EXPORT_RES, EXPORT_RES), Image.Resampling.LANCZOS).save(
            os.path.join(folder, f"{role}.png"))
    with open(os.path.join(folder, "capture.json"), "w") as f:
        json.dump(finite(jsonable(rep)), f, indent=2)
    return f"exported {len(rep['crops'])} photos + capture.json to {folder}"


# --- Main -------------------------------------------------------------------
def has_photos(folder: str) -> bool:
    for name in os.listdir(folder):
        stem = os.path.splitext(name)[0]
        if stem.lower() == "flash" or TORCH_RE.fullmatch(stem):
            return True
    return False


def find_swatches(path: str) -> list[str]:
    if has_photos(path):
        return [path]
    return [os.path.join(path, d) for d in sorted(os.listdir(path))
            if not d.startswith(".") and os.path.isdir(os.path.join(path, d))]


def parse_crop(text: str):
    try:
        x, y, s = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("--crop wants x,y,size in pixels, e.g. 1700,1200,600")
    if s < 16 or x < 0 or y < 0:
        raise argparse.ArgumentTypeError("--crop: x, y >= 0 and size >= 16 px")
    return x, y, s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check a real-paint capture (docs/capture_kit.md).")
    ap.add_argument("folder", nargs="?", help="a folder of swatch folders, or one swatch folder")
    ap.add_argument("--camera-height-cm", type=float, default=CAMERA_HEIGHT_CM,
                    help="lens to swatch surface (default %(default)g)")
    ap.add_argument("--hfov-deg", type=float, default=None,
                    help="field of view across the photo's long side (default: from EXIF "
                         "35 mm-equivalent focal length, else %g, a typical phone main "
                         "camera)" % HFOV_DEG)
    ap.add_argument("--torch-distance-cm", type=float, default=TORCH_DISTANCE_CM,
                    help="torch to swatch centre, for the elevation (default %(default)g)")
    ap.add_argument("--crop-cm", type=float, default=CROP_CM,
                    help="side of the centre square used as the swatch (default %(default)g)")
    ap.add_argument("--crop", type=parse_crop, default=None,
                    help="x,y,size in pixels of the photo as a viewer shows it; overrides "
                         "--crop-cm (applies to every swatch checked)")
    ap.add_argument("--no-rotate", dest="rotate", action="store_false",
                    help="ignore the EXIF rotation tag and use the photos as stored")
    ap.add_argument("--export", default=None, help="write 256 px crops + capture.json here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1
    if not args.folder:
        ap.error("give a folder to check (or --self-test)")
    if min(args.camera_height_cm, args.torch_distance_cm, args.crop_cm) <= 0:
        ap.error("--camera-height-cm, --torch-distance-cm and --crop-cm must be > 0")
    if args.hfov_deg is not None and not 0 < args.hfov_deg < 180:
        ap.error("--hfov-deg must be between 0 and 180")
    if not os.path.isdir(args.folder):
        print(f"{args.folder} is not a folder")
        return 1

    swatches = find_swatches(args.folder)
    print(f"checking {len(swatches)} swatch folder{'s' if len(swatches) != 1 else ''} in "
          f"{args.folder}")
    results = []
    for folder in swatches:
        if not has_photos(folder):
            print(f"\n=== {os.path.basename(folder)}: no flash.* or torch_N.* photos, skipped")
            continue
        rep = check_swatch(folder, args)
        print_report(rep)
        if args.export:
            if verdict(rep["issues"]) == "ERROR" or not rep["crops"]:
                print("  not exported: fix the errors first")
            else:
                print("  " + export_swatch(rep, args.export))
        results.append(verdict(rep["issues"]))

    n_err, n_warn = results.count("ERROR"), results.count("WARN")
    print(f"\nRESULT: {len(results)} swatch{'es' if len(results) != 1 else ''} checked: "
          f"{results.count('OK')} OK, {n_warn} with warnings, {n_err} with errors.")
    if not results:
        print("No swatches found. Expected <folder>/<swatch name>/flash.jpg, torch_1.jpg, ...")
    elif n_err or n_warn:
        print("Fix or re-shoot those before taking the setup down; NOTEs are for information.")
    return 1 if n_err or not results else 0


# --- Self-test (geometry and alignment, no files needed) --------------------
def self_test() -> bool:
    ok = True
    rng = np.random.default_rng(0)
    H, R = CAMERA_HEIGHT_CM, TORCH_DISTANCE_CM

    # 1. The docstring's worked example.
    e = math.radians(40.0)
    L = (R * math.cos(e), 0.0, R * math.sin(e))
    p = (L[0] * H / (L[2] + H), 0.0)
    got = light_from_offset(p, H, (0.0, 0.0), R)
    print(f"worked example: highlight {got['highlight_cm']:.2f} cm out, "
          f"{got['elevation_at_highlight_deg']:.2f} deg there, "
          f"{got['elevation_deg']:.2f} deg at the centre (expect 13.61, 64.08, 40.00)")
    ok &= (abs(got["highlight_cm"] - 13.61) < 0.01
           and abs(got["elevation_at_highlight_deg"] - 64.075) < 0.01
           and abs(got["elevation_deg"] - 40.0) < 1e-6)

    # 2. Round trip: random torch -> mirror point -> back, also from an
    # off-centre swatch. The mirror point is checked by the law of reflection.
    worst = 0.0
    for _ in range(2000):
        az, el = rng.uniform(0, 360), rng.uniform(10, 85)
        dist, q = rng.uniform(15, 50), tuple(rng.uniform(-5, 5, 2))
        a, e = math.radians(az), math.radians(el)
        L = np.array([q[0] + dist * math.cos(e) * math.cos(a),
                      q[1] + dist * math.cos(e) * math.sin(a), dist * math.sin(e)])
        P = np.array([L[0], L[1], 0.0]) * H / (L[2] + H)
        to_cam = np.array([0, 0, H]) - P
        to_light = L - P
        mirrored = to_cam / np.linalg.norm(to_cam) * [-1, -1, 1]
        refl_err = np.linalg.norm(mirrored - to_light / np.linalg.norm(to_light))
        got = light_from_offset((P[0], P[1]), H, q, dist)
        d_az = abs((got["azimuth_deg"] - az + 180) % 360 - 180)
        worst = max(worst, d_az, abs(got["elevation_deg"] - el), refl_err)
    print(f"round trip over 2000 random torches: worst error {worst:.2e} (deg / unit vector)")
    ok &= worst < 1e-6

    # 3. Clock positions.
    clocks = [to_clock(a) for a in (0, 90, 180, 270, 15, 100)]
    print(f"clock: 0/90/180/270/15/100 deg -> {clocks}")
    ok &= clocks == ["3:00", "12:00", "9:00", "6:00", "2:30", "11:30"]

    # 4. Phase correlation recovers a known sub-pixel shift on a textured scene.
    base = rng.random((300, 400)).astype(np.float32)
    big = np.asarray(Image.fromarray((base * 255).astype(np.uint8)).resize(
        (1600, 1200), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    sx, sy = 12, -8                                   # in the big image = 3, -2 px at 1/4
    moved = np.roll(big, (sy, sx), axis=(0, 1))
    small = lambda a: np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
        (400, 300), Image.Resampling.BOX), dtype=np.float32) / 255.0
    dx, dy, psr = phase_shift(small(big), small(moved))
    print(f"phase correlation: shift ({dx:.2f}, {dy:.2f}), expected (3.00, -2.00), "
          f"match {psr:.0f}")
    ok &= abs(dx - 3) < 0.15 and abs(dy + 2) < 0.15 and psr > MIN_MATCH
    _, _, psr0 = phase_shift(small(big), rng.random((300, 400)).astype(np.float32))
    print(f"phase correlation on unrelated images: match {psr0:.1f} (below {MIN_MATCH:g})")
    ok &= psr0 < MIN_MATCH

    print("\nRESULT:", "self-test passed." if ok else "SELF-TEST FAILED.")
    return ok


if __name__ == "__main__":
    sys.exit(main())
