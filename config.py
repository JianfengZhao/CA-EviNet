"""Central configuration for CA-EviNet.

CA-EviNet characterizes the mitral-regurgitation etiology (functional vs.
degenerative) from three grayscale TEE views, plus an auxiliary regression of the
clinical valve measurements. All hyperparameters and paths are defined here.
"""
import os

# ---------------- paths ----------------
# Paths are configurable via environment variables for portability.
CODE       = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.environ.get("MVPROJ_DATA",   os.path.join(CODE, "data", "dataset"))        # <pid>/<view>/frame_XX.png
LABELS_CSV = os.environ.get("MVPROJ_LABELS", os.path.join(CODE, "data", "usable_cohort.csv"))
OUT        = os.environ.get("MVPROJ_OUT",    os.path.join(CODE, "output"))
CKPT_DIR  = f"{OUT}/checkpoints"
LOG_DIR   = f"{OUT}/logs"
GRADCAM_DIR = f"{OUT}/gradcam"
PRED_DIR  = f"{OUT}/predictions"
for d in (CKPT_DIR, LOG_DIR, GRADCAM_DIR, PRED_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- data ----------------
VIEWS_ALL  = ["2", "3", "4"]          # 2CH / 3CH / 4CH
# view subset for the view-ablation study; "234" = all three (default).
VIEW_SUBSET = os.environ.get("CAE_VIEWS", "234")
VIEWS      = [v for v in VIEWS_ALL if v in VIEW_SUBSET]
T          = 16                        # keyframes per view
IMG_SIZE   = 224
IN_CH      = 1                         # grayscale
NUM_CLASSES = 2                        # per task (K)

# Classification target: etiology (functional vs. degenerative).
# pos_label = minority class taken as positive for sensitivity/specificity/F1/AP.
_ETIOLOGY = {"name": "etiology", "col": "Functional MR", "pos_label": 0}  # 0 = degenerative
TASKS = [_ETIOLOGY]
TASK_NAMES = [t["name"] for t in TASKS]
N_TASKS = len(TASKS)

# The four clinically used measurements regressed by the auxiliary branch.
QUANT_COLS = ["Aα", "Pα", "Tent_A", "flail_width"]
M = len(QUANT_COLS)
ID_COL    = "patient_id"

# ---------------- model ----------------
# per-frame ViT backbone. "vit_s_16" = ImageNet ViT-S/16 (timm, D=384) is the
# setting used for real training / the paper. "small" is a tiny MONAI ViT for
# fast crash-testing only.
BACKBONE   = os.environ.get("CAE_BACKBONE", "vit_s_16")   # {"vit_s_16","vit_b_16","small"}
VIT_CONFIGS = {
    "vit_b_16": dict(hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12),
    "small":    dict(hidden_size=192, mlp_dim=384,  num_layers=4,  num_heads=3),
}
PATCH_SIZE = 16
# temporal transformer over T frame features
TEMPORAL_LAYERS = 2
TEMPORAL_HEADS  = 4
# quantification head (MLP)
QUANT_HIDDEN = 256

# ---------------- ablation switches (leave-one-out from the full model) ----------------
# "" = full model. Others match the paper's Table II rows:
#   noquant     : beta=0 (drop the auxiliary quantification)
#   notemporal  : replace the temporal Transformer with mean pooling over T frames
#   noview      : drop the light per-view evidential supervision (per-view weight -> 0)
#   noconflict  : uniform average of views instead of conflict/uncertainty weighting
#   noevidential: softmax cross-entropy + mean view fusion (no Dirichlet, no RAL)
ABLATE = os.environ.get("CAE_ABLATE", "")

# ---------------- loss / RAL ----------------
GAMMA      = 1.0        # conflict penalty in RAL (exp(-gamma * delta_v))
BETA       = float(os.environ.get("CAE_BETA", "1.0"))    # weight of quantification loss
PER_VIEW_W = float(os.environ.get("CAE_PERVIEW", "0.3")) # light per-view evidential weight
KL_ANNEAL_EPOCHS = 10   # ramp fraction: min(1, epoch / KL_ANNEAL_EPOCHS)
KL_MAX     = float(os.environ.get("CAE_KLMAX", "0.1"))   # cap on KL weight (1.0 -> evidence collapse)
RAL_EPS    = 1e-8       # numerical stability in weight normalization

# ---------------- training ----------------
N_FOLDS    = 5
EPOCHS     = int(os.environ.get("CAE_EPOCHS", "40"))
LR         = float(os.environ.get("CAE_LR", "3e-5"))     # 1e-4 destroys the pretrained ViT
WARMUP_EPOCHS = int(os.environ.get("CAE_WARMUP", "10"))
BALANCED_SAMPLER = os.environ.get("CAE_BALANCED", "1") == "1"   # balance classes per batch
FREEZE_BACKBONE = os.environ.get("CAE_FREEZE", "0") == "1"      # freeze ViT (anti-overfit)
DROPOUT = float(os.environ.get("CAE_DROPOUT", "0.3"))
WEIGHT_DECAY = float(os.environ.get("CAE_WD", "0.05"))
BATCH_SIZE = int(os.environ.get("CAE_BATCH", "4"))      # micro-batch that fits memory
ACCUM      = int(os.environ.get("CAE_ACCUM", "2"))      # grad-accum steps; effective batch = BATCH_SIZE*ACCUM
NUM_WORKERS = 4
SEED       = 42
DEVICE     = os.environ.get("CAE_DEVICE", "cuda")       # "cuda" or "cpu"
CKPT_KEEP  = int(os.environ.get("CAE_CKPT_KEEP", "2"))  # keep best-K checkpoints per fold

# which label drives class-balanced batch sampling when BALANCED_SAMPLER is on.
SAMPLER_TASK = os.environ.get("CAE_SAMPLER_TASK", "etiology")

# debug mode: tiny subset + few epochs for a quick sanity run
DEBUG           = os.environ.get("CAE_DEBUG", "0") == "1"
DEBUG_N_PATIENTS = 8
DEBUG_EPOCHS     = 2

# ---------------- run identity (for recording, not ablation) ----------------
RUN_TAG = os.environ.get("CAE_TAG", "joint")   # names this run's output files


def vit_cfg():
    return VIT_CONFIGS[BACKBONE]


def snapshot():
    """All UPPER_CASE config values, for recording exactly what a run used."""
    import json
    d = {}
    for k, v in globals().items():
        if k.isupper():
            try:
                json.dumps(v); d[k] = v
            except TypeError:
                d[k] = str(v)
    return d
