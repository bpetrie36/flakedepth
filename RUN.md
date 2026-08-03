# Running the MaskTerial evaluation

These are TERMINAL commands (Command Prompt / bash), NOT Python.
If your prompt shows `>>>` you are inside Python: type `exit()` first.

You do NOT need MaskTerial's code, PyTorch, detectron2, CUDA, or a GPU.
You only need the dataset zips and this repo.

## 1. Environment (one time)

Windows (Command Prompt, in the folder containing this file):

```
py -m venv .venv
.venv\Scripts\activate
pip install jax numpy scipy matplotlib pillow
python core\dtmm.py
```

macOS / Linux / WSL:

```
python3 -m venv .venv
source .venv/bin/activate
pip install "jax[cpu]" numpy scipy matplotlib pillow
python core/dtmm.py
```

The last command should print contrast peaks near 87/276 nm and C_max ~0.142.
If it does, the physics is working.

## 2. Get one dataset

From https://zenodo.org/records/15765514 download and unzip, e.g. Real_hBN_Thin.zip.
You should end up with a folder containing test_images/ and test_semantic_masks/.

## 3. Find their wafer oxide (a few minutes)

```
python maskterial_eval.py --dataset .\hBN_Thin --material hbn --oxide-scan
```

Pick the value with the SMALLEST fitted scatter, not the closest mean.

## 4. Smoke test, then the full run

```
python maskterial_eval.py --dataset .\hBN_Thin --material hbn --oxide VALUE --na 0.45 --limit 30 --out smoke\
python maskterial_eval.py --dataset .\hBN_Thin --material hbn --oxide VALUE --na 0.45 --out results_hbn\
```

Materials: graphene, hbn, mos2, mose2, ws2, wse2

## 5. Send back

results_*/summary.txt, results_*/confusion.txt, results_*/per_flake.csv

## Notes

- ~1-2 s per flake on CPU. A full test split is hundreds of flakes, so expect
  tens of minutes. Always run the --limit 30 smoke test first.
- Read diagnostics.txt first: it reports conditions under which the numbers are void.
- The fitted-oxide scatter in summary.txt is the real validity check. Flakes on one
  wafer must agree; large scatter means the model is absorbing something unmodelled.
- Windows uses backslashes in paths (core\dtmm.py, not core/dtmm.py).
