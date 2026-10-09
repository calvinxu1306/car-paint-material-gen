# Real-paint capture kit (Phase 5)

Printable checklist for `v2_plan.md` §6: per swatch, one flash photo and 5
torch-lit photos. Run `src/check_capture.py` **before you take the setup
down**: most problems can only be fixed by re-shooting.

## 1. Materials

- [ ] **Swatches.** Nail polish or model-kit paint, one of each finish: solid,
      solid + matte top coat, metallic, pearl, candy (clear colour over
      silver), colour-shift ("chameleon"). 3–4 per type is about 20.
- [ ] **Flat cards** to paint on, at least 5 × 5 cm of paint (the checker
      uses the middle 3 × 3 cm). Trim the card close to the paint or use
      black card: the torch's reflection lands 8–17 cm from the centre and
      must fall on the black sheet. No spoons: the estimate assumes a flat
      swatch.
- [ ] **Descriptive names**, not product names ("deep red pearl"), written on
      the back of each card.
- [ ] **Glossy black sheet**, about 40 × 30 cm, filling the photo: black
      acrylic, a glossy tray or tile, or picture-frame glass on black paper.
      The torch's reflection in it shows the checker where the torch was.
- [ ] Overhead phone stand, the camera phone, a second phone (the torch), a
      ruler, a strip of tape, a remote shutter or the self-timer.

## 2. Setup

- [ ] **Dark room** (lights off, curtains shut, screens dimmed); black sheet
      on the table, swatch card in its middle.
- [ ] Camera phone on the stand, **lens 25–30 cm above the paint** (measure
      and note it; the checker assumes 28 cm), pointing **straight down**
      (iPhone: Settings > Camera > Level). Swatch in the middle of the frame.
      Main camera (1×): no zoom, portrait or macro mode.
- [ ] **Lock focus and exposure.** Best: Pro/manual mode with fixed ISO
      (100–400), shutter (about 1/30–1/60 s), white balance (daylight) and
      focus. Otherwise tap and hold the swatch until AE/AF lock shows.
- [ ] **Off:** HDR, Night mode, beauty filters, scene optimiser / AI
      enhance, Live Photo. Flash **On** (not Auto) for the flash photo,
      **Off** for the torch photos.
- [ ] **JPEG, not HEIC** (iPhone: Settings > Camera > Formats > Most
      Compatible). RAW + JPEG if offered; keep the RAW files.
- [ ] **Mark 12 o'clock**: tape on the table at the top edge of the picture
      (look at the screen). 12 = the photo's top edge, 3 = its right edge.
      Shoot with the remote or a 3 s timer: touching the phone moves it.

## 3. Shot list, per swatch

Don't touch the camera phone or the swatch between shots.

1. **`flash.jpg`**: flash on, torch off.
2. **`torch_1.jpg` … `torch_5.jpg`**: flash off; the second phone's torch at
   full brightness, **30 cm from the swatch centre**, pointing at it. These
   match v5's side lights (20–65° above the surface, all around).

| photo | clock | elevation | torch height above table | out from swatch centre |
|---|---|---|---|---|
| `torch_1` | 12:00 | 20° | 10 cm | 28 cm |
| `torch_2` | 2:30 | 60° | 26 cm | 15 cm |
| `torch_3` | 5:00 | 40° | 19 cm | 23 cm |
| `torch_4` | 7:30 | 30° | 15 cm | 26 cm |
| `torch_5` | 10:00 | 50° | 23 cm | 19 cm |

(Height = 30 cm × sin(elevation), out = 30 cm × cos(elevation).) Keep your
hand and the torch phone out of the picture, its screen dark.

**Fill in per swatch** (or type it into `notes.txt` in the swatch folder):

Swatch: ______________________  Camera height: ____ cm  Date: __________

| photo | clock | height (cm) | distance (cm) | elevation | notes |
|---|---|---|---|---|---|
| torch_1 | | | | | |
| torch_2 | | | | | |
| torch_3 | | | | | |
| torch_4 | | | | | |
| torch_5 | | | | | |

## 4. Files

Copy the photos off the phone unedited (editing apps drop EXIF and change
the size) and rename them in shooting order (IMG_0101 = flash, IMG_0102 =
torch_1, …). One folder per swatch: `swatches\deep red pearl\flash.jpg`,
`torch_1.jpg` … `torch_5.jpg`, optional `notes.txt`.

## 5. Check before you pack up (PowerShell, in the repo root)

```
python src/check_capture.py swatches
python src/check_capture.py "swatches\deep red pearl" --camera-height-cm 27
python src/check_capture.py swatches --export data\real_capture
```

Options: `--camera-height-cm` (28) and `--torch-distance-cm` (30) as
measured; `--hfov-deg` (field of view across the long side: from EXIF, else
70°); `--crop-cm 2` or `--crop x,y,size` (pixels) for a small or off-centre
swatch. `--export` writes 256 px crops named like v5's photos (`photo.png`,
`side1.png` …) plus `capture.json` (`data\real_capture` is not git-ignored).

What it checks, and the fix:

- [ ] **Same size and orientation** → same phone, lens and zoom. If only the
      rotation tag flipped (it says so), add `--no-rotate`.
- [ ] **Shift ≤ 3 px** from the flash photo → re-shoot; use the timer.
- [ ] **Same shutter, ISO, f-number** in every torch photo → lock exposure.
- [ ] **Clipped**: < 2% of the swatch in torch photos, < 10% with the flash
      → move the torch back or lower the exposure.
- [ ] **Sharpness** ≥ 0.5 × the flash photo's → shake or focus moved.
- [ ] **Tilt** < 5° (from the flash's reflection) → level the phone.
- [ ] **Coverage**: ≥ 4 torch photos, no gap over 120° between directions,
      one light below 35° and one above 50°. The directions (from the
      reflection in the black sheet) should match your table; "< X deg" =
      reflection out of the picture, so trust your table there.
- [ ] **By eye** (not checked): no shadows on the swatch, no room light, no
      dust or fingerprints, the swatch fills the middle 3 × 3 cm.
