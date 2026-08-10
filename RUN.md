# Running the MaskTerial evaluation

TERMINAL commands (Command Prompt / bash), NOT Python.
If your prompt shows `>>>` you are inside Python: type `exit()` first.

You do NOT need MaskTerial's code, PyTorch, detectron2, CUDA, or a GPU.
You only need the dataset zips and this repo.

---

## READ THIS FIRST: two things that will silently ruin a run

**1. Transfer function.** The MaskTerial images are radiometrically LINEAR, not
sRGB-encoded, and the dataset documentation does not say so. Decoding them as
sRGB compresses contrast by roughly a factor of two and inflates recovered
thickness by about the same, while leaving the per-image residual clean, so
nothing warns you. On one graphene exfoliation run this was the difference
between 0% and 90% exact layer agreement.

`maskterial_eval.py` and `core/maskterial_calib.py` now default to **linear**.
To process imagery that really is sRGB-encoded, or to reproduce the previous
behavior:

```
set FLAKEDEPTH_GAMMA=srgb          # Windows
export FLAKEDEPTH_GAMMA=srgb       # macOS / Linux
```

If you are unsure which a dataset is, `flakedepth.py --gamma auto` scores both
and reports which it chose.

**2. A MaskTerial "dataset" is not one wafer.** `GrapheneL`, `GrapheneM` and
`GrapheneH` are low, medium and high substrate-variance samplings drawn from a
shared pool of named exfoliation runs, spanning roughly 5, 10 and 20 nm of oxide
about a nominal 90 nm wafer. They overlap: L and M share 162 images,
byte-identical where they overlap. `meta_data/{split}_set_name_to_uuid.json`
inside each dataset maps images to their exfoliation run.

`core/maskterial_calib.py` fits ONE global oxide per dataset, which is only
appropriate for a single wafer. Partition by exfoliation run first with
`build_acq_corpus.py`, or expect the fitted oxide to be a mixture-weighted
compromise.

---

## 1. Environment (one time)

Windows (Command Prompt, in the folder containing this file):

```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python core\dtmm.py
```

macOS / Linux / WSL:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python core/dtmm.py
```

The last command should print contrast peaks near 87/276 nm, C_max ~0.142, and a
bare-substrate linear RGB of about [0.110, 0.098, 0.177] at 90 nm oxide. If it
does, the physics is working.

## 2. Get a dataset

From https://zenodo.org/records/15765514 download and unzip, e.g.
`Real_GrapheneL.zip`. You should end up with a folder containing
`test_images/`, `test_semantic_masks/`, `train_images/`,
`train_semantic_masks/` and `meta_data/`.

## 3. See what acquisitions it contains, and partition by them

```
python set_map.py --dataset L=.\GrapheneL M=.\GrapheneM H=.\GrapheneH --split test
python build_acq_corpus.py --dataset L=.\GrapheneL M=.\GrapheneM H=.\GrapheneH ^
       --out .\by_acquisition --dry-run
```

Drop `--dry-run` to write. Images are deduplicated across datasets, so an
acquisition appearing in two of them is written once. `--hardlink` avoids a
second copy on the same volume.

## 4. Find the wafer oxide

Coarse first, then refine around the winner:

```
python core\maskterial_calib.py --dataset .\by_acquisition\<ACQ> --material graphene ^
       --calibrate --oxide-range "60,320,5" --limit 60 --out cal_coarse
python core\maskterial_calib.py --dataset .\by_acquisition\<ACQ> --material graphene ^
       --calibrate --oxide-range "LOW,HIGH,0.5" --limit 60 --out cal_fine
```

Select on **color-fit**, not on layer-consistency. If the reported best sits at
either end of the range, the true minimum is outside it: widen and re-run. A
value pinned at a range boundary is not a measurement.

Graphene wafers in this corpus sit between about 88 and 96 nm, so `85,105,0.5`
is usually enough for the fine pass. hBN is exfoliated onto ~70 nm wafers, not
285 nm.

## 5. Evaluate

Full joint (illuminant free, per-flake nuisances, no per-dataset assumption):

```
python maskterial_eval.py --dataset .\by_acquisition\<ACQ> --material graphene ^
       --na 0.45 --oxide <FITTED> --limit 40 --out fj_<ACQ>
```

Restricted mode (illuminant pinned, one global oxide; more reliable for
per-flake layer counting):

```
python core\maskterial_calib.py --dataset .\by_acquisition\<ACQ> --material graphene ^
       --oxide <FITTED> --limit 200 --out res_<ACQ>
```

Then collect across acquisitions:

```
python collect_runs.py --dir . --prefix fj_ --csv per_acquisition.csv
```

Materials: graphene, hbn, mos2, mose2, ws2, wse2.
`--na 0.45` is correct for this corpus: MaskTerial images are captured with a
CFI TU Plan Fluor EPI 20x objective at NA 0.45.

For AFM-registered evaluation on your own data, use `flakedepth_eval.py` with
`manifest_template.csv`. Its per-row `gamma` column now defaults to `linear`.

## Notes

- ~12-15 s per flake on CPU with the NA-averaged forward model. 40 flakes is
  about 9 minutes. Run a `--limit 20` smoke test first.
- The fitted-oxide scatter in `summary.txt` is the real validity check: flakes on
  ONE wafer must agree. It is reported as median and MAD, which is robust to the
  ridge failures the method is known to produce; a mean and standard deviation
  would be dominated by them and would void correct runs.
- Roughly 10-40% of flakes per acquisition land in a spurious basin near 35-41 nm
  oxide. These are joint thickness-nuisance ridge failures, they are almost all
  unimodal, and the ambiguity flag does not catch them. Judge a run by the median
  and MAD, not by the mean.
- On real images the overall exposure is arbitrary and fits around 4, against a
  log-gain prior centered at 1 with width 0.1. Widening that prior tightens the
  fitted-oxide MAD but does not fix the ridge.
- Windows uses backslashes in paths (`core\dtmm.py`, not `core/dtmm.py`).
