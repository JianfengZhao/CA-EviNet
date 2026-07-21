"""Aggregate baseline per-fold results -> mean+/-std, and print LaTeX rows for Table I."""
import os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = [("resnet","ResNet~\\cite{He2016ResNet}"),
         ("nnunet","nnU-Net~\\cite{Isensee2021nnUNet}"),
         ("r2plus1d","R(2+1)D~\\cite{Tran2018R2plus1D}"),
         ("timesformer","TimeSformer~\\cite{Bertasius2021TimeSformer}"),
         ("mvcnn","MVCNN~\\cite{Su2015MVCNN}"),
         ("samil","SAMIL~\\cite{Huang2023SAMIL}")]
COLS = ["auc","ap","acc","f1","ece"]
allagg = {}
print("\n==== baseline aggregate (mean+/-std %, over folds) ====")
for tag, disp in NAMES:
    fs = sorted(glob.glob(os.path.join(HERE, tag, "test_fold*.json")))
    if not fs:
        print(f"  {tag}: 0/5"); continue
    rows = [json.load(open(f)) for f in fs]
    cells = []
    agg = {}
    for c in COLS:
        v = np.array([r[c] for r in rows if r.get(c) is not None and not (isinstance(r[c],float) and np.isnan(r[c]))])*100
        agg[c] = (v.mean(), v.std())
        cells.append(f"{v.mean():.1f}$\\pm${v.std():.1f}")
    allagg[tag] = agg
    print(f"  {disp:40s} & " + " & ".join(cells) + f" \\\\   % n={len(rows)}")
json.dump(allagg, open(os.path.join(HERE,"aggregate_baselines.json"),"w"), indent=2)
