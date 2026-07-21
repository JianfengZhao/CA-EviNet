"""Patient-level stratified K-fold split for the JOINT two-task setting.

Splitting is done on patient IDs so that no patient's frames appear in more than
one split (prevents leakage). We keep only patients whose BOTH task labels are a
clean 0/1 (the 446-patient intersection: 4 etiology-ambiguous patients dropped),
so neither task needs label masking. Folds are stratified on the joint stratum
(etiology, MAC) so both class ratios are preserved across folds.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import config as C


def load_labels():
    """Return the 446-patient joint cohort with columns label_<task> in {0,1}."""
    df = pd.read_csv(C.LABELS_CSV)
    df[C.ID_COL] = df[C.ID_COL].astype(int)
    # coerce quantification columns to numeric (some cells are strings)
    for col in C.QUANT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # binarize each task label and keep only clean 0/1 rows for BOTH tasks
    keep = pd.Series(True, index=df.index)
    for t in C.TASKS:
        lab = pd.to_numeric(df[t["col"]], errors="coerce")
        keep &= lab.isin([0.0, 1.0])
    df = df[keep].copy()
    for t in C.TASKS:
        df[f"label_{t['name']}"] = (pd.to_numeric(df[t["col"]], errors="coerce") >= 0.5).astype(int)
    # joint stratum = etiology*2 + mac  (4 strata) for balanced folds
    strat = np.zeros(len(df), dtype=int)
    for i, t in enumerate(C.TASKS):
        strat = strat * 2 + df[f"label_{t['name']}"].values
    df["_stratum"] = strat
    return df.reset_index(drop=True)


def get_folds(df, n_folds=C.N_FOLDS, seed=C.SEED, val_frac=0.15):
    """Return list of dicts {train, val, test} of patient-id arrays, one per fold.
    Stratified on the joint (etiology, MAC) stratum."""
    ids = df[C.ID_COL].values
    y = df["_stratum"].values.astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, te_idx in skf.split(ids, y):
        test_ids = ids[te_idx]
        tr_ids_all, tr_y = ids[tr_idx], y[tr_idx]
        # carve a stratified validation set out of the training portion
        rng = np.random.RandomState(seed)
        val_ids = []
        for cls in np.unique(tr_y):
            cls_ids = tr_ids_all[tr_y == cls]
            n_val = max(1, int(round(len(cls_ids) * val_frac)))
            val_ids.extend(rng.choice(cls_ids, size=n_val, replace=False).tolist())
        val_ids = np.array(val_ids)
        train_ids = np.array([i for i in tr_ids_all if i not in set(val_ids.tolist())])
        folds.append(dict(train=train_ids, val=val_ids, test=test_ids))
    return folds


def quant_normalizer(df, train_ids):
    """Per-measurement mean/std computed on the TRAIN patients only (no leakage)."""
    sub = df[df[C.ID_COL].isin(set(train_ids.tolist()))]
    mean = sub[C.QUANT_COLS].mean(skipna=True).values.astype("float32")
    std = sub[C.QUANT_COLS].std(skipna=True).replace(0, 1).values.astype("float32")
    return mean, std
