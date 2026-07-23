# CA-EviNet

**Conflict-aware Evidential Aggregation for Multi-view TEE Characterization of Mitral Regurgitation Etiology**

Official PyTorch implementation of the paper *"Conflict-aware Evidential Aggregation for
Multi-view TEE Characterization of Mitral Regurgitation Etiology"*.

CA-EviNet characterizes the **functional-versus-degenerative mitral regurgitation etiology**
from three routine grayscale TEE views (2CH/3CH/4CH). Each view is encoded by a shared
per-frame ViT and a temporal Transformer, and a per-view **evidential head** parameterizes a
Dirichlet distribution that gives a per-view probability and uncertainty. A **conflict-aware
Reliable Aggregation Layer (RAL)** then measures the inter-view conflict with a total-variation
distance decoupled from confidence and attenuates a discordant view's evidence *before* the
evidence is aggregated, so a conflicting view contributes less and the fused decision stays
**well calibrated** under view disagreement. In parallel, an **auxiliary branch** regresses four
clinically used valve measurements (leaflet-to-annulus angles, tenting area, flail width) as
clinically grounded, verifiable outputs.

## Repository structure
```
config.py            central configuration (paths via env vars, hyperparameters)
data_split.py        patient-level stratified 5-fold split
dataset.py           multi-view keyframe loader
model.py             CA-EviNet (FrameViT + temporal enc + evidential heads + RAL + quant)
losses.py            evidential (Bayes-risk + capped KL) + masked-Huber quantification
metrics.py           AUC/AP/ACC/F1, ECE, AURC, per-measurement MAE/RMSE/r
train.py             train/evaluate one fold (records history, checkpoints, analysis)
aggregate.py         cross-fold mean +/- std
gradcam.py           Grad-CAM for the aggregated evidence
(ablations)          run train.py with CAE_ABLATE / CAE_VIEWS flags (see below)
comparison/          baselines.py + trainer + runners (SOTA comparison)
```

## Installation
```bash
conda create -n caevinet python=3.10 -y && conda activate caevinet
pip install -r requirements.txt
```

## Data
The clinical dataset is **not released** for patient-privacy reasons. To run on your own
cohort, provide the following layout (paths configurable via the `MVPROJ_DATA` /
`MVPROJ_LABELS` environment variables):
```
data/dataset/<patient_id>/{2,3,4}/frame_00.png ... frame_15.png   # 224x224 grayscale, ECG-gated
data/usable_cohort.csv
```
`usable_cohort.csv` columns: `patient_id`, `Functional MR` (1=functional, 0=degenerative),
and the four measurements `Aα, Pα, Tent_A, flail_width`.

## Usage

All commands are plain Python. Hyperparameters default to the paper configuration
and can be overridden via `CAE_*` environment variables (see `config.py`).

**Step 1 — Train the main model (5-fold).** Trains one fold at a time and saves
per-fold metrics, checkpoints, and analysis arrays under `output/`.
```bash
for f in 0 1 2 3 4; do python train.py --fold $f; done
python aggregate.py                 # prints and saves cross-fold mean +/- std
```

**Step 2 — Ablation study.** Module ablations (each removes one component) and
single-view ablations, over all 5 folds:
```bash
# module ablations
for a in noquant noview noconflict noevidential; do
  for f in 0 1 2 3 4; do CAE_ABLATE=$a CAE_TAG=abl_$a python train.py --fold $f; done
done
# view ablations (single chamber view)
for v in 2 3 4; do
  for f in 0 1 2 3 4; do CAE_VIEWS=$v CAE_TAG=view_$v python train.py --fold $f; done
done
```

**Step 3 — SOTA baselines.** Trains each baseline under the identical protocol,
then aggregates:
```bash
for m in resnet mvcnn samil swin convnext; do
  for f in 0 1 2 3 4; do python comparison/train_baseline.py --model $m --fold $f; done
done
python comparison/aggregate_baselines.py
```

**Step 4 — Grad-CAM (qualitative check).**
```bash
python gradcam.py --fold 0
```

## License
Released under the MIT License (see `LICENSE`).
