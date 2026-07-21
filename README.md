# CA-EviNet

**Conflict-aware Evidential Learning for Reliable and Interpretable Multi-view Transesophageal Echocardiographic Characterization of Mitral Regurgitation Etiology**

Official PyTorch implementation of the paper *"Conflict-aware Evidential Learning for
Reliable and Interpretable Multi-view TEE Characterization of Mitral Valve Pathology"*.

CA-EviNet characterizes the **functional-vs-degenerative mitral regurgitation etiology**
from three routine grayscale TEE views (2CH/3CH/4CH). Each view is encoded by a shared
per-frame ViT + temporal Transformer; a per-view **evidential head** estimates a Dirichlet
uncertainty; a **conflict-aware Reliable Aggregation Layer (RAL)** fuses the views by their
uncertainty and inter-view conflict; and an **auxiliary branch** regresses four clinically
used valve measurements (leaflet-to-annulus angles, tenting area, flail width) for
interpretability.

## Repository structure
```
config.py            central configuration (paths via env vars, hyperparameters)
data_split.py        patient-level stratified 5-fold split
dataset.py           multi-view keyframe loader
model.py             CA-EviNet (FrameViT + temporal enc + evidential heads + RAL + quant)
losses.py            evidential (Bayes-risk + capped KL) + masked-Huber quantification
metrics.py           AUC/AP/ACC/F1, ECE/Brier, AURC, per-measurement MAE/RMSE/r
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
for a in noquant notemporal noview noconflict noevidential; do
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

**Step 4 — Grad-CAM (qualitative interpretability).**
```bash
python gradcam.py --fold 0
```

## Results (etiology, patient-level 5-fold, mean +/- std %)
| Method | AUC | AP | ACC | F1 | ECE ↓ |
|---|---|---|---|---|---|
| ResNet   | 77.8 | 77.6 | 66.1 | 61.9 | 26.2 |
| MVCNN    | 76.7 | 74.9 | 70.6 | 64.3 | 20.5 |
| SAMIL    | 79.6 | 75.6 | 72.9 | 69.3 | 22.3 |
| Swin     | 78.0 | 74.5 | 69.3 | 67.9 | 25.4 |
| ConvNeXt | 81.1 | 79.3 | 75.1 | 70.8 | 22.7 |
| **CA-EviNet (ours)** | **82.6** | **81.5** | 73.3 | 69.4 | **10.0** |

## Citation
```bibtex
@article{caevinet2026,
  title   = {Conflict-aware Evidential Learning for Reliable and Interpretable
             Multi-view TEE Characterization of Mitral Valve Pathology},
  author  = {Zhao, Jianfeng and others},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2026}
}
```

## License
Released under the MIT License (see `LICENSE`).
