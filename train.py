"""Train / evaluate the JOINT CA-EviNet for one patient-level fold.

Trains and evaluates CA-EviNet for one fold (etiology classification + auxiliary
quantification). Records everything a paper needs: per-epoch history (losses/lr/KL/metrics/time), per-fold config snapshot,
best-K checkpoints, per-task test metrics, per-measurement quantification, and the
raw test-time arrays for all post-hoc figures.

Usage:
  python train.py --fold 0
  CAE_DEBUG=1 CAE_BACKBONE=small python train.py --fold 0   # fast crash test
"""
import os, sys, json, time, argparse, math, logging, csv
import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from data_split import load_labels, get_folds, quant_normalizer
from dataset import CAEDataset
from model import CAEviNet
from losses import total_loss
import metrics as MET


def get_logger(fold):
    os.makedirs(C.LOG_DIR, exist_ok=True)
    lg = logging.getLogger(f"{C.RUN_TAG}_fold{fold}")
    lg.setLevel(logging.INFO); lg.handlers.clear()
    fh = logging.FileHandler(f"{C.LOG_DIR}/train_{C.RUN_TAG}_fold{fold}.log")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(sh)
    return lg


def write_history(path, history):
    if not history:
        return
    cols = list(history[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(history)


def make_loader(df, ids, mean, std, train):
    ds = CAEDataset(df, ids, mean, std, train=train)
    sampler, shuffle = None, train
    if train and C.BALANCED_SAMPLER:
        # class-balanced sampling (on the SAMPLER_TASK label) to prevent collapse
        col = f"label_{C.SAMPLER_TASK}"
        lab = df.set_index(C.ID_COL)[col]
        y = np.array([int(lab.loc[int(i)]) for i in ids])
        cls_count = np.bincount(y, minlength=C.NUM_CLASSES).astype(float)
        w_cls = 1.0 / np.clip(cls_count, 1.0, None)
        weights = torch.as_tensor(w_cls[y], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(ids), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=shuffle, sampler=sampler,
                      num_workers=C.NUM_WORKERS, pin_memory=True, drop_last=False)


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, mean, std, return_arrays=False):
    """Evaluate both tasks. Returns {task: metrics_dict, 'quant_mae': float} and,
    if requested, the raw per-task arrays for the figures."""
    model.eval()
    per = {t["name"]: {"P": [], "U": [], "Y": [], "W": [], "Cf": [], "UV": []} for t in C.TASKS}
    Q, QT, QM, FEAT = [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["views"])
        for t in C.TASKS:
            to = out["tasks"][t["name"]]
            per[t["name"]]["P"].append(to["prob"][:, 1].cpu().numpy())
            per[t["name"]]["U"].append(to["uncertainty"].cpu().numpy())
            per[t["name"]]["Y"].append(batch[f"label_{t['name']}"].cpu().numpy())
            if return_arrays:
                per[t["name"]]["W"].append(to["weights"].cpu().numpy())
                per[t["name"]]["Cf"].append(to["conflict"].cpu().numpy())
                per[t["name"]]["UV"].append(to["u_views"].cpu().numpy())
        Q.append(out["quant"].cpu().numpy())
        QT.append(batch["quant"].cpu().numpy())
        QM.append(batch["quant_mask"].cpu().numpy())
        if return_arrays:
            FEAT.append(out["feat"].cpu().numpy())
    Q, QT, QM = np.concatenate(Q), np.concatenate(QT), np.concatenate(QM)
    res = {}
    arrays = {"quant_pred": Q, "quant_target": QT, "quant_mask": QM,
              "quant_mean": mean, "quant_std": std}
    if return_arrays:
        arrays["feat"] = np.concatenate(FEAT)
    for t in C.TASKS:
        P = np.concatenate(per[t["name"]]["P"]); U = np.concatenate(per[t["name"]]["U"])
        Y = np.concatenate(per[t["name"]]["Y"])
        m = MET.classification_metrics(Y, P, pos_label=t["pos_label"])
        m["aurc"] = MET.selective_prediction_aurc(Y, P, U)
        m["unc_err_auc"] = MET.uncertainty_error_auc(Y, P, U)
        res[t["name"]] = m
        if return_arrays:
            arrays[f"prob_{t['name']}"] = P
            arrays[f"label_{t['name']}"] = Y
            arrays[f"uncertainty_{t['name']}"] = U
            arrays[f"weights_{t['name']}"] = np.concatenate(per[t["name"]]["W"])
            arrays[f"conflict_{t['name']}"] = np.concatenate(per[t["name"]]["Cf"])
            arrays[f"u_views_{t['name']}"] = np.concatenate(per[t["name"]]["UV"])
    res["quant_mae"], _ = MET.quant_mae(Q, QT, QM, mean, std)
    if return_arrays:
        return res, arrays
    return res


def build_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs):
    total = steps_per_epoch * epochs
    warm = steps_per_epoch * warmup_epochs
    def fn(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def mean_val_auc(val):
    aucs = [val[t["name"]]["auc"] for t in C.TASKS]
    aucs = [a for a in aucs if not math.isnan(a)]
    return float(np.mean(aucs)) if aucs else 0.0


def run_fold(fold):
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    lg = get_logger(fold)
    device = torch.device(C.DEVICE if torch.cuda.is_available() or C.DEVICE == "cpu" else "cpu")
    epochs = C.DEBUG_EPOCHS if C.DEBUG else C.EPOCHS

    df = load_labels()
    folds = get_folds(df)
    ids = folds[fold]
    tr, va, te = ids["train"], ids["val"], ids["test"]
    if C.DEBUG:
        tr, va, te = tr[:C.DEBUG_N_PATIENTS], va[:4], te[:4]
    mean, std = quant_normalizer(df, tr)

    lg.info(f"[{C.RUN_TAG}][fold {fold}] tasks={C.TASK_NAMES} views={C.VIEWS} ablate='{C.ABLATE}' "
            f"backbone={C.BACKBONE} device={device} debug={C.DEBUG} "
            f"train={len(tr)} val={len(va)} test={len(te)} epochs={epochs}")

    cfg_snap = C.snapshot()
    cfg_snap["fold"] = fold
    cfg_snap["n_train"], cfg_snap["n_val"], cfg_snap["n_test"] = len(tr), len(va), len(te)
    cfg_snap["quant_mean"] = [float(x) for x in mean]
    cfg_snap["quant_std"] = [float(x) for x in std]
    json.dump(cfg_snap, open(f"{C.LOG_DIR}/config_{C.RUN_TAG}_fold{fold}.json", "w"), indent=2)

    train_loader = make_loader(df, tr, mean, std, train=True)
    val_loader   = make_loader(df, va, mean, std, train=False)
    test_loader  = make_loader(df, te, mean, std, train=False)

    model = CAEviNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    accum = max(1, C.ACCUM)
    opt_steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    sched = build_scheduler(opt, opt_steps_per_epoch, epochs, C.WARMUP_EPOCHS)
    lg.info(f"[{C.RUN_TAG}][fold {fold}] trainable_params={n_params/1e6:.2f}M")

    hist_path = f"{C.LOG_DIR}/history_{C.RUN_TAG}_fold{fold}.csv"
    history = []
    kept = []                      # list of {"score","epoch","path"}  (best-K)
    smooth_hist = []
    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        lam_t = C.KL_MAX * min(1.0, (epoch + 1) / C.KL_ANNEAL_EPOCHS)
        t0, running, gnorm, n_opt = time.time(), {}, 0.0, 0
        n_micro = len(train_loader)
        opt.zero_grad()
        for it, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            out = model(batch["views"])
            loss, parts = total_loss(out, batch, lam_t, C.BETA)
            (loss / accum).backward()                        # scale for grad accumulation
            parts["loss_total"] = float(loss.item())
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            if (it + 1) % accum == 0 or (it + 1) == n_micro:  # optimizer step
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); sched.step(); opt.zero_grad()
                gnorm += float(gn); n_opt += 1
        nb = max(1, n_micro)
        running = {k: v / nb for k, v in running.items()}
        gnorm = gnorm / max(1, n_opt)
        lr_now = opt.param_groups[0]["lr"]
        val = evaluate(model, val_loader, device, mean, std)
        elapsed = time.time() - t0

        rec = {"epoch": epoch + 1, "lr": lr_now, "lambda_kl": lam_t,
               "loss_total": running.get("loss_total", float("nan")),
               "l_quant": running.get("l_quant", float("nan")),
               "grad_norm": gnorm, "sec": elapsed}
        for t in C.TASKS:
            rec[f"l_{t['name']}_agg"] = running.get(f"l_{t['name']}_agg", float("nan"))
            rec[f"l_{t['name']}_view"] = running.get(f"l_{t['name']}_view", float("nan"))
            for k, v in val[t["name"]].items():
                rec[f"val_{t['name']}_{k}"] = float(v)
        rec["val_quant_mae"] = float(val["quant_mae"])
        history.append(rec)
        write_history(hist_path, history)

        aucs = " ".join(f"{t['name'][:4]}={val[t['name']]['auc']:.3f}" for t in C.TASKS)
        lg.info(f"  epoch {epoch+1}/{epochs} lr={lr_now:.2e} loss={running.get('loss_total',float('nan')):.4f} "
                f"quant={running.get('l_quant',0):.3f} val[{aucs}] mae={val['quant_mae']:.3f} ({elapsed:.0f}s)")

        # robust checkpoint selection: SMOOTHED mean-task val AUC (last 3), skip early
        mva = mean_val_auc(val)
        smooth_hist.append(mva)
        smooth = float(np.mean(smooth_hist[-3:]))
        if epoch + 1 >= max(C.WARMUP_EPOCHS, 5):
            if len(kept) < C.CKPT_KEEP or smooth > min(k["score"] for k in kept):
                path = f"{C.CKPT_DIR}/caevinet_{C.RUN_TAG}_fold{fold}_e{epoch+1}.pt"
                torch.save({"model": model.state_dict(), "fold": fold, "epoch": epoch + 1,
                            "score": smooth, "quant_mean": mean, "quant_std": std}, path)
                kept.append({"score": smooth, "epoch": epoch + 1, "path": path})
                if len(kept) > C.CKPT_KEEP:                    # evict the worst
                    worst = min(kept, key=lambda k: k["score"])
                    kept.remove(worst)
                    if os.path.exists(worst["path"]):
                        os.remove(worst["path"])

    total_train_sec = time.time() - t_start
    # best checkpoint = highest smoothed score (fallback: save last if none kept)
    if not kept:
        path = f"{C.CKPT_DIR}/caevinet_{C.RUN_TAG}_fold{fold}_e{epochs}.pt"
        torch.save({"model": model.state_dict(), "fold": fold, "epoch": epochs,
                    "quant_mean": mean, "quant_std": std}, path)
        kept.append({"score": 0.0, "epoch": epochs, "path": path})
    best = max(kept, key=lambda k: k["score"])
    model.load_state_dict(torch.load(best["path"], map_location=device, weights_only=False)["model"])

    test, arr = evaluate(model, test_loader, device, mean, std, return_arrays=True)
    # per-measurement quantification table (MAE/RMSE/r)
    qdet = MET.quant_metrics(arr["quant_pred"], arr["quant_target"], arr["quant_mask"], mean, std)
    qdet["measurements"] = C.QUANT_COLS
    json.dump(qdet, open(f"{C.PRED_DIR}/quant_{C.RUN_TAG}_fold{fold}.json", "w"), indent=2)
    # raw test-time arrays for the figures (t-SNE, calibration, risk-coverage,
    # uncertainty, conflict-weight, Bland-Altman) -- both tasks + shared features
    np.savez(f"{C.PRED_DIR}/analysis_{C.RUN_TAG}_fold{fold}.npz", **arr)

    test_flat = {"quant_mae": test["quant_mae"]}
    for t in C.TASKS:
        for k, v in test[t["name"]].items():
            test_flat[f"{t['name']}_{k}"] = v
    lg.info(f"[{C.RUN_TAG}][fold {fold}] best_epoch={best['epoch']} best_score={best['score']:.3f} "
            f"train_sec={total_train_sec:.0f} TEST {test_flat}")

    summary = {"run_tag": C.RUN_TAG, "fold": fold, "backbone": C.BACKBONE,
               "ablate": C.ABLATE, "views": C.VIEWS, "epochs": epochs,
               "best_epoch": best["epoch"], "best_score": best["score"],
               "kept_checkpoints": [{"epoch": k["epoch"], "score": k["score"]} for k in kept],
               "n_params_M": n_params / 1e6, "train_sec": total_train_sec,
               "test": test, "history_csv": os.path.basename(hist_path)}
    json.dump(summary, open(f"{C.PRED_DIR}/summary_{C.RUN_TAG}_fold{fold}.json", "w"), indent=2)
    json.dump(test_flat, open(f"{C.PRED_DIR}/test_{C.RUN_TAG}_fold{fold}.json", "w"), indent=2)
    return test_flat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    run_fold(args.fold)
