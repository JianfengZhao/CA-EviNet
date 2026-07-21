"""Grad-CAM for CA-EviNet's frame ViT.

We take the aggregated evidence as the target score, back-propagate to the
per-frame patch tokens, and form a class-discriminative heatmap for each view. This
supports the interpretable pillar together with the clinical quantification.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import config as C


def _colorize(cam):                                  # cam: HxW in [0,1] -> RGB uint8
    import matplotlib.cm as cm
    rgb = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def gradcam_for_patient(model, batch, device, out_dir, tag="patient", task=None):
    """Grad-CAM on the patch tokens that enter the LAST transformer block (those tokens
    influence the final class-token feature, so gradients flow to them). `task` selects
    which task's aggregated evidence is used as the target score (default: first task)."""
    os.makedirs(out_dir, exist_ok=True)
    task = task or C.TASK_NAMES[0]
    model.eval()
    fv = model.frame_vit
    if fv.kind == "tv":
        target = fv.tv.encoder.layers[-1]
    elif fv.kind == "timm":
        target = fv.timm.blocks[-1]
    else:
        target = fv.vit.blocks[-1]
    cap = {}

    def pre_hook(module, inp):
        t = inp[0]
        t.retain_grad()
        cap["t"] = t

    h = target.register_forward_pre_hook(pre_hook)
    views = batch["views"].to(device)                # (1,V,T,1,H,W)
    out = model(views)
    to = out["tasks"][task]
    score_t = to["alpha"][:, 1] if "alpha" in to else to["prob"][:, 1]
    score = score_t.sum()                            # aggregated task evidence/prob
    model.zero_grad(); score.backward()
    h.remove()

    Np = (C.IMG_SIZE // C.PATCH_SIZE) ** 2            # 196 patch tokens
    tok = cap["t"][:, -Np:]                           # drop CLS if present -> (V*T,Np,hidden)
    grad = cap["t"].grad[:, -Np:]
    alpha = grad.mean(dim=1)                          # (V*T, hidden) channel weights
    cam = F.relu((tok * alpha.unsqueeze(1)).sum(-1))  # (V*T, Np)
    gh = gw = int(round(Np ** 0.5))                   # 14 for 224/16
    cam = cam.reshape(len(C.VIEWS), C.T, gh, gw)      # (V,T,gh,gw)

    saved = []
    for vi, view in enumerate(C.VIEWS):
        t = C.T // 2                                  # representative (mid) frame
        m = cam[vi, t]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        m = F.interpolate(m[None, None], size=(C.IMG_SIZE, C.IMG_SIZE),
                          mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        frame = views[0, vi, t, 0].detach().cpu().numpy()
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        base = (np.stack([frame] * 3, -1) * 255).astype(np.uint8)
        heat = _colorize(m)
        overlay = (0.55 * base + 0.45 * heat).astype(np.uint8)
        fn = os.path.join(out_dir, f"{tag}_view{view}_gradcam.png")
        Image.fromarray(overlay).save(fn)
        saved.append(fn)
    return saved


if __name__ == "__main__":
    # quick self-test on one patient using a trained checkpoint (if present)
    import argparse
    from data_split import load_labels, get_folds, quant_normalizer
    from dataset import CAEDataset
    from model import CAEviNet
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
    df = load_labels(); ids = get_folds(df)[a.fold]
    mean, std = quant_normalizer(df, ids["train"])
    ds = CAEDataset(df, ids["test"][:1], mean, std, train=False)
    batch = torch.utils.data.default_collate([ds[0]])
    model = CAEviNet().to(device)
    ckpt = f"{C.CKPT_DIR}/caevinet_{C.RUN_TAG}_fold{a.fold}.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    print("saved:", gradcam_for_patient(model, batch, device, C.GRADCAM_DIR,
                                        tag=f"fold{a.fold}_p{int(batch['pid'][0])}"))
