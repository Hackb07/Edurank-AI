"""Honest evaluation — Polyglot port of src/evaluate.js.

Scores the FROZEN artifacts (thresholds tuned on April) once each on:
  - test: May 2025 (temporal holdout, never touched in training)
  - future: June 2025 Day-2 update file (distribution shift + schema drift)
Plus anti-references: random-split optimism, leakage demo, flag bounds.

Run: uv run python -m parcelrank.evaluate
"""
import math
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from .artifacts import load_models
from .clean import load_carriers, load_capacity, load_shipments
from .csv_io import parse_csv, read_text
from .features import fit_preprocessing, transform_all
from .metrics import (
    COST_FN, COST_FP, apply_daily_policy, default_prioritize, prf, wilson, tune_threshold,
)

TRANSIT_GOLDEN = 1e-12


def report(name, labels, scores, threshold):
    m = prf(labels, scores, threshold)
    cost = m["fn"] * COST_FN + m["fp"] * COST_FP
    print(
        f"  {name}: roc={roc_auc_score(labels, scores):.3f} pr={average_precision_score(labels, scores):.3f} "
        f"@{threshold:.2f} P={m['precision']:.3f} R={m['recall']:.3f} "
        f"tp={m['tp']} fp={m['fp']} fn={m['fn']} cost=${cost}"
    )
    return {**m, "cost": cost}


def segments(name, rows, scores, threshold, key, min_n=50):
    print(f"  -- {name} by {key} (min n={min_n}) --")
    groups = {}
    for i, r in enumerate(rows):
        if r["label"] is None:
            continue
        k = r["ship_ymd"][:7] if key == "month" else r[key]
        groups.setdefault(k, []).append(i)
    for k in sorted(groups):
        idxs = groups[k]
        if len(idxs) < min_n:
            continue
        lab = [rows[i]["label"] for i in idxs]
        sc = [scores[i] for i in idxs]
        m = prf(lab, sc, threshold)
        lo, hi = wilson(m["tp"], m["tp"] + m["fn"])
        rate = sum(lab) / len(idxs)
        print(
            f"    {k}: n={len(idxs)} lateRate={rate:.2f} "
            f"P={m['precision']:.2f} R={m['recall']:.2f} [{lo:.2f}-{hi:.2f}]"
        )


def calibration(name, labels, scores):
    idx = sorted(range(len(scores)), key=lambda i: scores[i])
    bins = 5
    parts = []
    for b in range(bins):
        sl = idx[(b * len(idx)) // bins : ((b + 1) * len(idx)) // bins]
        mean_p = sum(scores[i] for i in sl) / len(sl)
        mean_y = sum(labels[i] for i in sl) / len(sl)
        parts.append(f"{mean_p:.2f}->{mean_y:.2f}")
    print(f"  calibration {name} (pred->actual by quintile): {' '.join(parts)}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    artifacts = load_models(f"{root}/models")
    preproc = artifacts["preproc"]
    logreg_w = np.array(artifacts["logreg"]["weights"])
    logreg_b = artifacts["logreg"]["bias"]
    forest = artifacts["forest"]
    th = artifacts["thresholds"]

    carriers = load_carriers(f"{root}/data/carriers.csv")
    cap = load_capacity(f"{root}/data/warehouse_capacity.csv")
    ctx = dict(capacity=cap["table"], known_carriers=carriers)
    day1 = load_shipments(f"{root}/data/shipments_day1.csv", **ctx)["rows"]
    june = load_shipments(f"{root}/data/shipments_day2_update.csv", **ctx)["rows"]

    train_rows = [r for r in day1 if r["ship_ymd"] < "2025-04-01"]
    test = [r for r in day1 if r["ship_ymd"] >= "2025-05-01" and r["label"] is not None]
    fut = [r for r in june if r["label"] is not None]
    print(f"test May: n={len(test)}, future Jun: n={len(fut)}")

    for split, rows in (("TEST May", test), ("FUTURE Jun", fut)):
        print(f"\n== {split} (frozen April thresholds) ==")
        labels = [r["label"] for r in rows]
        days = len(set(r["ship_ymd"] for r in rows))
        X = np.array(transform_all(rows, preproc), dtype=float)
        s_log = list(1.0 / (1.0 + np.exp(-(X @ logreg_w + logreg_b))))
        s_for = [_forest_p(x, forest["trees"]) for x in X.tolist()]
        n_late = sum(labels)
        print(f"  flag-none cost=${n_late * COST_FN} (${(n_late * COST_FN) / days:.2f}/day)")
        print(f"  flag-all  cost=${(len(labels) - n_late) * COST_FP} (${((len(labels) - n_late) * COST_FP) / days:.2f}/day)")
        r_log = report("logreg ", labels, s_log, th["logreg"])
        r_for = report("forest ", labels, s_for, th["forest"])
        print(f"  logreg ${r_log['cost'] / days:.2f}/day over {days} days")
        print(f"  forest ${r_for['cost'] / days:.2f}/day over {days} days")
        flags = apply_daily_policy(
            rows, s_log, top_k=25, floor=0.10,
            threshold=th["logreg"], tiers=default_prioritize(),
        )
        m_k = prf(labels, [float(f) for f in flags], 0.5)
        cost_k = m_k["fn"] * COST_FN + m_k["fp"] * COST_FP
        print(
            f"  top-25/day (tiered {','.join(default_prioritize())}): "
            f"P={m_k['precision']:.3f} R={m_k['recall']:.3f} cost=${cost_k} (${cost_k / days:.2f}/day)"
        )
        calibration(split, labels, s_log)

    print("\n== segment analysis (logreg, TEST May) ==")
    X = np.array(transform_all(test, preproc), dtype=float)
    s_log_t = list(1.0 / (1.0 + np.exp(-(X @ logreg_w + logreg_b))))
    for key in ("carrier_id", "warehouse_id", "service_level", "dest_region"):
        segments("test", test, s_log_t, th["logreg"], key)

    print("\n== segment analysis (logreg, FUTURE Jun) ==")
    X = np.array(transform_all(fut, preproc), dtype=float)
    s_log_f = list(1.0 / (1.0 + np.exp(-(X @ logreg_w + logreg_b))))
    for key in ("carrier_id", "warehouse_id", "service_level", "dest_region"):
        segments("future", fut, s_log_f, th["logreg"], key)

    # --- Anti-reference 1: random split optimism -------------------------------
    print("\n== anti-reference: random split vs temporal split ==")
    pool = [r for r in day1 if r["ship_ymd"] < "2025-05-01" and r["label"] is not None]
    from .rng import make_rng

    rng = make_rng(999)
    idx = rng["shuffle"](list(range(len(pool))))
    cut = int(len(idx) * 0.8)
    r_tr = [pool[i] for i in idx[:cut]]
    r_te = [pool[i] for i in idx[cut:]]
    r_pre = fit_preprocessing(r_tr)
    m = LogisticRegression(C=100.0, solver="lbfgs", max_iter=2000)
    m.fit(np.array(transform_all(r_tr, r_pre), dtype=float), [r["label"] for r in r_tr])
    r_scores = list(m.predict_proba(np.array(transform_all(r_te, r_pre), dtype=float))[:, 1])
    print(
        f"  random 80/20 (same-period rows): roc={roc_auc_score([r['label'] for r in r_te], r_scores):.3f} "
        f"pr={average_precision_score([r['label'] for r in r_te], r_scores):.3f}"
    )
    print("  temporal May (above): trained Jan-Mar, tested May — lower because the future drifts (seasonality, new rows).")
    print("  A random split mixes April rows into both sides, so seasonal and operational patterns leak across.")

    # --- Anti-reference 2: leakage demo ----------------------------------------
    print("\n== anti-reference: LEAKAGE DEMO (invalid model, do not ship) ==")
    print("  (loads raw days_in_transit as a feature to show what fooling yourself looks like)")
    _, raw_day1 = parse_csv(read_text(f"{root}/data/shipments_day1.csv"))
    transit_by_id = {}
    for r in raw_day1:
        v = str(r.get("days_in_transit") or "").strip()
        if v != "":
            try:
                transit_by_id[str(r["parcel_id"]).strip()] = float(v)
            except ValueError:
                pass
    leak_tr = [r for r in train_rows if r["label"] is not None]
    leak_te = test
    t_vals = [transit_by_id[r["parcel_id"]] for r in leak_tr if r["parcel_id"] in transit_by_id]
    t_mean = sum(t_vals) / len(t_vals)
    t_sd = math.sqrt(sum((v - t_mean) ** 2 for v in t_vals) / len(t_vals)) or 1.0

    def with_leak(rows_, X_):
        out = []
        for vec, r in zip(X_, rows_):
            v = transit_by_id.get(r["parcel_id"], t_mean)
            out.append(list(vec) + [(v - t_mean) / t_sd])
        return np.array(out, dtype=float)

    ml = LogisticRegression(C=100.0, solver="lbfgs", max_iter=2000)
    Xltr = with_leak(leak_tr, np.array(transform_all(leak_tr, preproc), dtype=float))
    ml.fit(Xltr, [r["label"] for r in leak_tr])
    Xlte = with_leak(leak_te, np.array(transform_all(leak_te, preproc), dtype=float))
    leak_scores = list(ml.predict_proba(Xlte)[:, 1])
    print(
        f"  logreg + days_in_transit on TEST May: roc={roc_auc_score([r['label'] for r in leak_te], leak_scores):.3f} "
        "(INVALID — uses post-outcome data)"
    )


def _forest_p(x, trees):
    def leaf(tree, xx):
        while not tree["leaf"]:
            tree = tree["left"] if xx[tree["feature"]] <= tree["threshold"] else tree["right"]
        return tree["p"]

    return sum(leaf(t, x) for t in trees) / len(trees)


if __name__ == "__main__":
    main()
