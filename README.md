# Flake Depth

Thickness estimation for 2D-material flakes from a single color micrograph, by
inverting thin-film optics at inference time. No training data.

## Install
    pip install -r requirements.txt

## Quick check (verifies the physics)
    python core/dtmm.py
Expect contrast peaks near 87/276 nm oxide, C_max ~0.142, and a bare-substrate
linear RGB of about [0.110, 0.098, 0.177] at 90 nm oxide.

## Two non-negotiable inputs

**The objective's numerical aperture.** A normal-incidence approximation is fine
at NA <= 0.55 and fails badly at NA 0.9. NA cannot be recovered from the image,
and it is engraved on the objective. Pass `--na`.

**The transfer function.** Whether the file holds linear radiance or an
sRGB-encoded image. Getting this wrong compresses contrast by roughly a factor
of two and rescales recovered thickness by about the same, while leaving the
per-image residual clean, so nothing warns you. The MaskTerial datasets are
linear, and their documentation does not say so.

`maskterial_eval.py` and `core/maskterial_calib.py` default to **linear**.
Override with:

    set FLAKEDEPTH_GAMMA=srgb        # Windows
    export FLAKEDEPTH_GAMMA=srgb     # macOS / Linux

`flakedepth.py --gamma auto` scores both and reports which it chose.

## Usage

    python flakedepth.py --images IMG_DIR --masks MASK_DIR --material graphene --na 0.45 --oxide 90

Read the color-fit verdict first: below 0.08 is clean, above 0.15 means the
physics cannot reproduce the observed ratios at any oxide and the thicknesses
should not be trusted.

See `RUN.md` for the MaskTerial workflow, including partitioning a dataset by
exfoliation run before fitting a global oxide. See `DATA.md` for data sources.

## Interpreting the self-consistency check
Flakes on one wafer must yield the same fitted oxide. That check is reported as
median and MAD rather than mean and standard deviation, because 10-40% of flakes
per acquisition land in a joint thickness-nuisance ridge; a mean would be
dominated by them and would void otherwise correct runs. The ridge failures are
almost all unimodal, so the ambiguity flag does not catch them.

## Layout
    core/           forward model (dtmm), scene model, estimator, baselines,
                    restricted-mode calibration
    experiments/    every benchmark in the paper, with seeds
    paper/          manuscript source and figures
    flakedepth.py           unified tool: run this on new data
    flakedepth_eval.py      AFM-registered evaluation from a manifest
    maskterial_eval.py      full joint evaluation on a MaskTerial dataset
    set_map.py              which acquisitions a dataset contains
    build_acq_corpus.py     re-partition datasets by exfoliation run
    per_set.py              contrast and substrate color per acquisition
    collect_runs.py         assemble a per-acquisition table from a batch
    stability.py            bootstrap CIs, subsample curves, pairwise tests

## Reproducibility note
`maskterial_eval.py` and `core/maskterial_calib.py` previously hard-coded sRGB
decoding with no way to change it, and `flakedepth_eval.py` defaulted its
per-row transfer function to sRGB. Any result produced with those versions on
linear imagery is affected. The self-consistency check also previously reported
mean and standard deviation of the fitted oxide, which the ridge failures
dominate, so it could declare correct runs void.
