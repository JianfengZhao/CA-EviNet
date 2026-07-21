"""Train one baseline classifier for one fold, under the SAME protocol as CA-EviNet
(same patient-level 5-fold split, same 3-view 16-frame grayscale input, same
metrics). Classification only. Saves test metrics into comparison/<model>/.

Usage: CAE_TASK2=none python comparison/train_baseline.py --model resnet --fold 0
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from data_split import load_labels, get_folds, quant_normalizer
from dataset import CAEDataset
import metrics as MET
from baselines import build

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCHS = int(os.environ.get("BL_EPOCHS", "25"))
LR = float(os.environ.get("BL_LR", "1e-4"))
BATCH = int(os.environ.get("BL_BATCH", "4"))
WORKERS = int(os.environ.get("BL_WORKERS", "4"))


def make_loader(df, ids, mean, std, train):
    ds = CAEDataset(df, ids, mean, std, train=train)
    sampler, shuffle = None, train
    if train:
        lab = df.set_index(C.ID_COL)["label_etiology"]
        y = np.array([int(lab.loc[int(i)]) for i in ids])
        w = 1.0 / np.clip(np.bincount(y, minlength=2), 1, None).astype(float)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(w[y], dtype=torch.double), len(ids), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, sampler=sampler,
                      num_workers=WORKERS, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); P, Y = [], []
    for b in loader:
        logit = model(b["views"].to(device))
        P.append(F.softmax(logit, 1)[:, 1].cpu().numpy())
        Y.append(b["label_etiology"].numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    m = MET.classification_metrics(Y, P, pos_label=0)      # degenerative = positive
    return {k: m[k] for k in ["auc", "ap", "acc", "f1", "ece"]}


def run(model_name, fold):
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    outdir = os.path.join(HERE, model_name); os.makedirs(outdir, exist_ok=True)
    device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
    df = load_labels(); folds = get_folds(df); ids = folds[fold]
    mean, std = quant_normalizer(df, ids["train"])
    tl = make_loader(df, ids["train"], mean, std, True)
    vl = make_loader(df, ids["val"], mean, std, False)
    te = make_loader(df, ids["test"], mean, std, False)
    model = build(model_name).to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    steps = max(1, len(tl)) * EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    best, best_state, t0 = -1, None, time.time()
    for ep in range(EPOCHS):
        model.train()
        for b in tl:
            logit = model(b["views"].to(device))
            loss = F.cross_entropy(logit, b["label_etiology"].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        v = evaluate(model, vl, device)
        auc = v["auc"] if not math.isnan(v["auc"]) else 0.0
        if ep + 1 >= 5 and auc > best:
            best = auc; best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        print(f"[{model_name} f{fold}] ep{ep+1}/{EPOCHS} loss={loss.item():.3f} val_auc={v['auc']:.3f}", flush=True)
    if best_state:
        model.load_state_dict(best_state)
    test = evaluate(model, te, device)
    test["params_M"] = round(n, 2); test["train_sec"] = round(time.time() - t0)
    json.dump(test, open(os.path.join(outdir, f"test_fold{fold}.json"), "w"), indent=2)
    print(f"[{model_name} f{fold}] TEST {test}", flush=True)
    return test


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args(); run(a.model, a.fold)
