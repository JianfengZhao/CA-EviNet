"""Aggregate per-fold JOINT results into mean +/- std for the paper.

Reads output/predictions/test_{tag}_fold*.json (flat keys like etiology_auc,
mac_auc, quant_mae) and quant_{tag}_fold*.json, writes aggregate_{tag}.json and
prints a readable per-task summary. Safe to run any time (uses whatever folds
have finished).
"""
import os, json, glob
import numpy as np
import config as C

TAG = C.RUN_TAG
METRICS = ["auc", "ap", "acc", "sensitivity", "specificity", "f1",
           "ece", "brier", "aurc", "unc_err_auc"]      # per task, reported in %


def _clean(vals):
    return [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]


def main():
    tests = sorted(glob.glob(f"{C.PRED_DIR}/test_{TAG}_fold*.json"))
    rows = [json.load(open(t)) for t in tests]
    agg = {"tag": TAG, "n_folds": len(rows)}
    if rows:
        for k in rows[0]:
            vals = _clean([r.get(k) for r in rows])
            if vals:
                agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    quants = sorted(glob.glob(f"{C.PRED_DIR}/quant_{TAG}_fold*.json"))
    qrows = [json.load(open(q)) for q in quants]
    if qrows:
        qagg = {"measurements": qrows[0]["measurements"]}
        for metric in ("mae", "rmse", "r"):
            arr = np.array([qr[metric] for qr in qrows], dtype=float)
            qagg[f"{metric}_mean"] = np.nanmean(arr, 0).tolist()
            qagg[f"{metric}_std"] = np.nanstd(arr, 0).tolist()
        agg["quant"] = qagg

    json.dump(agg, open(f"{C.PRED_DIR}/aggregate_{TAG}.json", "w"), indent=2)

    print(f"\n===== Aggregate over {len(rows)} fold(s)  [tag={TAG}] =====")
    for t in C.TASK_NAMES:
        print(f"\n  -- {t} --")
        for m in METRICS:
            k = f"{t}_{m}"
            if k in agg:
                print(f"    {m:14s} {agg[k]['mean']*100:6.2f} +/- {agg[k]['std']*100:5.2f} %")
    if "quant_mae" in agg:
        print(f"\n  quant_mae (overall) {agg['quant_mae']['mean']:.3f} +/- {agg['quant_mae']['std']:.3f}")
    if "quant" in agg:
        print("\n  -- quantification (per measurement: MAE / RMSE / r) --")
        q = agg["quant"]
        for i, name in enumerate(q["measurements"]):
            print(f"    {name:18s} MAE {q['mae_mean'][i]:7.3f}  RMSE {q['rmse_mean'][i]:7.3f}  r {q['r_mean'][i]:5.2f}")
    print(f"\n  saved: {C.PRED_DIR}/aggregate_{TAG}.json\n")


if __name__ == "__main__":
    main()
