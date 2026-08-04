# Flake Depth

Thickness of 2D-material flakes from a single color micrograph, by inverting
thin-film optics. No training data, no per-lab retraining.

## Install

    pip install jax numpy scipy matplotlib pillow

Check the physics (expect peaks near 87/276 nm, C_max ~0.142):

    python core/dtmm.py

## Use it on YOUR images

    python flakedepth.py --images IMG_DIR --masks MASK_DIR \
                         --material graphene --na 0.45 --oxide 90

* `--masks`: one file per image, same filename stem. 0 = bare substrate, >0 = flake.
* `--na`: your OBJECTIVE's numerical aperture, engraved on the barrel next to the
  magnification (e.g. "20x/0.45"). It is a property of the objective, not the camera.
  REQUIRED. A wrong value silently rescales every thickness and is NOT caught by the
  color-fit verdict. Add `--verify-na` to have the tool fit NA from the images and
  warn if it disagrees with what you supplied.
* `--oxide`: SiO2 thickness in nm from your wafer spec. Don't know it? Use
  `--scan-oxide` and it will recover it from the images (no labels needed).
* Add `--labels-are-layers` if your mask values are layer counts, to score agreement.

The transfer function (sRGB vs linear) is detected automatically.
Per-image white balance is solved exactly from the bare substrate.

## Read the verdict first

`summary.txt` reports a color-fit residual and one of three verdicts:

    < 0.08   USABLE - clean
    < 0.15   USABLE - with caution
    >= 0.15  UNUSABLE

UNUSABLE means no oxide value explains your images: they likely carry contrast
enhancement or a tone curve, or the material is too low-contrast. Re-acquire with
FIXED white balance and no auto-contrast. This check needs no ground truth.

## Acquisition guidance

Fixed (not auto) white balance, no auto-contrast, no tone curve beyond sRGB,
uncompressed if possible, and record the objective NA and wafer oxide spec.

## Layout

    flakedepth.py   the tool
    core/           forward model, estimator, baselines
    experiments/    every benchmark in the paper, with seeds
    paper/          manuscript source
    tools/          browser demo
