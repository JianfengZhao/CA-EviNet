"""Aggregate baseline per-fold results into mean +/- std and print a summary table."""
import os, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["resnet", "mvcnn", "samil", "swin", "convnext"]
COLS = ["auc", "ap", "acc", "f1", "ece"]


def main():
    print("\n==== baseline aggregate (mean +/- std %, over folds) ====")
    print(f"  {'model':10s}" + "".join(f"{c.upper():>13s}" for c in COLS))
    for m in MODELS:
        fs = sorted(glob.glob(os.path.join(HERE, m, "test_fold*.json")))
        if not fs:
            print(f"  {m:10s} (no results)"); continue
        rows = [json.load(open(f)) for f in fs]
        cells = []
        for c in COLS:
            v = np.array([r[c] for r in rows if r.get(c) is not None]) * 100
            cells.append(f"{v.mean():5.1f}+/-{v.std():4.1f}")
        print(f"  {m:10s}" + "".join(f"{x:>13s}" for x in cells) + f"   (n={len(rows)})")


if __name__ == "__main__":
    main()
