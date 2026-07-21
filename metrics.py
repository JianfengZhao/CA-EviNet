"""Evaluation metrics for the classification and quantification tasks.

  (i)   discrimination : ACC, AUC, AP, sensitivity, specificity, F1
  (ii)  calibration    : ECE, Brier
  (iii) uncertainty    : AURC (selective prediction), uncertainty->error AUC
  (iv)  quantification : per-measurement MAE, RMSE, Pearson r

The minority class is taken as positive for sensitivity/specificity/F1/AP; AUC is
computed on the class-1 probability and is invariant to this choice.
"""
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             average_precision_score, f1_score)


def expected_calibration_error(y, p_pos, n_bins=10):
    """p_pos: predicted probability of the positive class."""
    conf = np.where(p_pos >= 0.5, p_pos, 1 - p_pos)
    pred = (p_pos >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(y)
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() > 0:
            ece += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(y, p_pos):
    return float(np.mean((p_pos - y) ** 2))


def classification_metrics(y, p1, pos_label=0):
    """y: (N,) 0/1 labels; p1: (N,) predicted prob of class 1.
    `pos_label` is the class treated as positive for sens/spec/F1/AP (the minority
    class of the task (degenerative = 0 for etiology). AUC is
    computed on the class-1 probability and is invariant to this choice."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p1).astype(float)
    yhat = (p >= 0.5).astype(int)                       # predicted class 1
    # positive-class view
    y_pos = (y == pos_label).astype(int)
    p_pos = p if pos_label == 1 else 1.0 - p
    pred_pos = (p_pos >= 0.5).astype(int)
    tp = int(((pred_pos == 1) & (y_pos == 1)).sum())
    fn = int(((pred_pos == 0) & (y_pos == 1)).sum())
    fp = int(((pred_pos == 1) & (y_pos == 0)).sum())
    tn = int(((pred_pos == 0) & (y_pos == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0         # positive-class recall
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    two_class = len(np.unique(y)) > 1
    try:
        auc = roc_auc_score(y, p) if two_class else float("nan")
    except ValueError:
        auc = float("nan")
    try:
        ap = average_precision_score(y_pos, p_pos) if two_class else float("nan")
    except ValueError:
        ap = float("nan")
    f1 = f1_score(y_pos, pred_pos, zero_division=0)
    return {
        "acc": accuracy_score(y, yhat),
        "auc": auc,
        "ap": float(ap),
        "sensitivity": sens,
        "specificity": spec,
        "f1": float(f1),
        "ece": expected_calibration_error(y, p),
        "brier": brier_score(y, p),
    }


def selective_prediction_aurc(y, p_pos, uncertainty):
    """Area under the risk-coverage curve. Lower is better: deferring the most
    uncertain cases should leave a lower error rate on the retained ones.
    uncertainty: (N,) higher = less confident."""
    y = np.asarray(y).astype(int)
    err = ((np.asarray(p_pos) >= 0.5).astype(int) != y).astype(float)
    order = np.argsort(np.asarray(uncertainty))          # most confident first
    err_sorted = err[order]
    n = len(y)
    if n == 0:
        return float("nan")
    cum_risk = np.cumsum(err_sorted) / np.arange(1, n + 1)  # risk at each coverage
    return float(cum_risk.mean())


def uncertainty_error_auc(y, p_pos, uncertainty):
    """AUC of using uncertainty to detect misclassifications (higher is better)."""
    y = np.asarray(y).astype(int)
    err = ((np.asarray(p_pos) >= 0.5).astype(int) != y).astype(int)
    if len(np.unique(err)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(err, np.asarray(uncertainty)))
    except ValueError:
        return float("nan")


def quant_mae(pred, target, mask, mean, std):
    """De-normalize and compute masked MAE per target and overall (kept for the
    training loop / backward compatibility). pred/target/mask: (N,M) normalized."""
    pred = np.asarray(pred) * std + mean
    target = np.asarray(target) * std + mean
    mask = np.asarray(mask)
    ae = np.abs(pred - target) * mask
    denom = mask.sum(0).clip(min=1)
    per_target = ae.sum(0) / denom
    overall = float(ae.sum() / mask.sum().clip(min=1))
    return overall, per_target


def quant_metrics(pred, target, mask, mean, std):
    """Per-measurement quantification metrics for the final table: MAE, RMSE, and
    Pearson r (on de-normalized values, masked). Each measurement is reported on
    its own; no cross-measurement aggregate is computed (units differ)."""
    pred = np.asarray(pred) * std + mean
    target = np.asarray(target) * std + mean
    mask = np.asarray(mask).astype(bool)
    M = pred.shape[1]
    mae, rmse, corr, n = (np.full(M, np.nan), np.full(M, np.nan),
                          np.full(M, np.nan), np.zeros(M, dtype=int))
    for j in range(M):
        m = mask[:, j]
        n[j] = int(m.sum())
        if m.sum() == 0:
            continue
        pj, tj = pred[m, j], target[m, j]
        mae[j] = np.mean(np.abs(pj - tj))
        rmse[j] = np.sqrt(np.mean((pj - tj) ** 2))
        if m.sum() >= 2 and pj.std() > 0 and tj.std() > 0:
            corr[j] = float(np.corrcoef(pj, tj)[0, 1])
    return {"mae": mae.tolist(), "rmse": rmse.tolist(), "r": corr.tolist(),
            "n": n.tolist()}
