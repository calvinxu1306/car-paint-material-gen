# Handoff — where the project stands (2026-10-09)

Read this first in a new session, then `docs/ablations.md` (results and
findings, newest at the bottom of the Runs sections) and `docs/v2_plan.md`
(the v2 plan and its run table). Everything below is already in `main`
unless marked otherwise.

## State in one paragraph

v1 (one flash photo → metallic paint layers) is finished; best model Run 9
(`runs/v4_long`), live demo at `demo/index.html`. v2 extends it to five paint
types (solid, metallic, pearl, colour-shift, candy) with a flash photo plus
five side-lit photos (`generate_dataset_v5.py`, `train_multi.py`,
`eval_multi.py`, `MultiLightPaintNet`). Runs 10–12 showed the multi-light
model reads flakes and film thickness well once the flash photo is in every
training sample (Run 12, `--photos flash+random`), but **coat roughness
stays at ~18% skill versus 62.5% for a flash-only model** (Finding 16).
**Run 13 (`--flash-slot`) is training now** to test whether giving the flash
photo its own slot in the pooling fixes it.

## Run 13 — in progress

```powershell
python src/train_multi.py --root data/blender_gen/dataset_v5 --out runs/v5_flashslot --photos flash+random --flash-slot 2>&1 | Tee-Object -FilePath runs\v5_flashslot_console.txt
```

- Epoch 1 had a one-off validation spike (2.29); epoch 2 recovered (0.49).
- Predictions, written before it finishes (`ablations.md`, under Run 12):
  coat roughness from the flash **≥ 50%** skill (error ≤ ~0.06), with flake
  size ≥ 70%, flake strength ≥ 75% (flash) / ≥ 85% (all six) and the maps at
  Run 12's level. Below ~25% → explanation B ruled out too.
- When it finishes, the new session needs: the end of the training output
  (`best validation loss` + error table), the `<- best` epoch line, and the
  full output of
  ```powershell
  python src/eval_multi.py --run runs/v5_flashslot --test-root data/blender_gen/dataset_v5_test --json runs/v5_flashslot/test.json
  ```

## Decision after Run 13

| Run 13 coat roughness from the flash | Next |
|---|---|
| ≥ 50% | Run 14: add `--coords` (pixel coordinates as input channels) |
| < 25% | look at the layer-parameter head and its loss (how coat roughness is weighted against the other nine targets) |
| in between | Run 14 with `--coords --flash-slot` |

Then Runs 15–16 from `v2_plan.md` §4 (v1 initialisation; v1 code on v5).

## Ways of working that matter here

- **Predictions before results.** Every run's predictions go into
  `ablations.md` before its numbers come in; results are checked against
  them, failures recorded as failures.
- **Report on the held-out test set** (`dataset_v5_test`, seeds 100000+), not
  the validation split that picked `best.pt`.
- **Within-type skill matters.** Overall film-thickness and coat-tint scores
  are mostly paint-type recognition (Finding 13); check the per-pigment table.
- **Single seeds everywhere**; differences of a few points are noise.

## Known pitfalls

- **No `--amp`**: it turned training to NaN (Run 12's first attempt) and was
  only ~10% faster. Training now stops itself at a NaN validation loss.
- **Long runs**: run in a standalone PowerShell window with `Tee-Object`, or
  set VS Code's `terminal.integrated.gpuAcceleration` to `off`; the VS Code
  terminal display froze during Run 12.
- **Run time**: ~250 s per epoch, ~10 h per 150-epoch run on the RTX 4070;
  slower means something else is using the machine.
- **Demo**: anything under `demo/` on `main` is deployed to GitHub Pages.
  Don't commit `demo/assets/samples_v2/` until the assets are final.
- **Cloud sessions**: arxiv.org, diglib.eg.org and openaccess.thecvf.com are
  only reachable if allowed in the environment's network settings; the three
  multi-image papers are already summarised in `reading_notes_v2.md` §4.

## Open items for the user (not blocked on Run 13)

- [ ] Push the `v1.0` tag on commit `9188e78` (the session could not push
      tags): `git tag -a v1.0 9188e78 -m "v1.0" ; git push origin v1.0`.
- [ ] **HDR check** (one assumption untested): render 4 samples with
      `generate_dataset_v5.py -- --count 4 --out dataset_v5_hdr --hdr` after
      `pip install OpenEXR`, then `python src/data/dataset_multi.py --root
      data/blender_gen/dataset_v5_hdr`; its HDR section must not say
      `MISMATCH` (that would mean the EXRs went through the AgX tone curve).
- [ ] **v2 demo**: `python src/export_demo_multi.py --run runs/v5_flashplus
      --test-root data/blender_gen/dataset_v5_test`, then serve `demo/` and
      open `v2.html`.
- [ ] **Capture kit**: print `docs/capture_kit.md`, photograph one swatch,
      run `python src/check_capture.py "C:\capture\swatches\<name>"`.

## Later

Extra seeds; capture changes that need a re-render (HDR training data, fixed
side-light slots, a stripe-reflection photo for orange peel); Phase 5 real-
paint prediction (needs a prediction script and a v5 render set up to match
the phone's framing — see `capture_kit.md` §5). Candy's tint is not
recoverable from these renders by construction (`reading_notes_v2.md` §2):
Cycles applies it by viewing angle only, and the camera never moves.
