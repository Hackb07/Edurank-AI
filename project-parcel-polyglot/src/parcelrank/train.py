"""Day-2 training — Polyglot port of src/train.js.

Temporal discipline (unchanged): preprocessing + models see ONLY the train
window (Jan-Mar 2025); April validation tunes lambda and the costed
threshold; May (test) and June (future) are NEVER touched here.

Models: scikit-learn LogisticRegression (L2, lambda grid on validation
log-loss) and RandomForestClassifier 60 trees d/3 max features. Artifacts
are exported in the SAME JSON layout the Node pipeline uses, so the Rust
scorer (rust/parcel-scorer) can consume them directly.

Run: uv run python -m parcelrank.train
"""
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .artifacts import save_models
from .clean import load_carriers, load_capacity, load_shipments
from .features import fit_preprocessing, transform_all
from .metrics import COST_FN, COST_FP, apply_daily_policy, prf, roc_auc, pr_auc, tune_threshold

LAMBDAS = [0.01, 0.1, 1.0, 10.0]


def deciles(vals):
    s = sorted(v for v in vals if v is not None)
    if not s:
        return []
    n = len(s)
    return [s[min(n - 1, (k * n) // 10)] for k in range(11)]


def freq(rows, key):
    counts = {}
    for r in rows:
        counts[r[key]] = counts.get(r[key], 0) + 1
    return {k: v / len(rows) for k, v in counts.items()}


def _export_sklearn_forest(rf, model_meta):
    """Serialize a sklearn forest into the Node pipeline's recursive JSON:
    {leaf, p} or {feature, threshold, left, right}; p = P(late) at leaf."""
    trees_out = []
    for est in rf.estimators_:
        t = est.tree_

        def build(i):
            if t.children_left[i] == -1:  # sklearn marks leaves with -1 children
                return {"leaf": True, "p": float(t.value[i][0][1]) / float(t.value[i][0][0] + t.value[i][0][1])}
            return {
                "leaf": False,
                "feature": int(t.feature[i]),
                "threshold": float(t.threshold[i]),
                "left": build(int(t.children_left[i])),
                "right": build(int(t.children_right[i])),
            }

        trees_out.append(build(0))
    return {"trees": trees_out, **model_meta}


def train(root):
    known_carriers = load_carriers(f"{root}/data/carriers.csv")
    cap = load_capacity(f"{root}/data/warehouse_capacity.csv")
    rows = load_shipments(
        f"{root}/data/shipments_day1.csv",
        capacity=cap["table"],
        known_carriers=known_carriers,
    )["rows"]

    train = [r for r in rows if r["ship_ymd"] < "2025-04-01" and r["label"] is not None]
    val = [r for r in rows if "2025-04-01" <= r["ship_ymd"] < "2025-05-01" and r["label"] is not None]
    print(f"train Jan-Mar: n={len(train)}, val Apr: n={len(val)}")

    preproc = fit_preprocessing(train)
    import numpy as np

    Xtr = np.array(transform_all(train, preproc))
    ytr = np.array([r["label"] for r in train])
    Xva = np.array(transform_all(val, preproc))
    yva = np.array([r["label"] for r in val])

    # Logistic regression: lambda grid on validation log-loss.
    best_m = None
    best_lambda = None
    best_val_loss = None
    for lam in LAMBDAS:
        m = LogisticRegression(C=1.0 / lam, solver="lbfgs", max_iter=2000)
        m.fit(Xtr, ytr)
        loss = float(log_loss(yva, m.predict_proba(Xva)[:, 1], labels=[0, 1]))
        val_scores = m.predict_proba(Xva)[:, 1]
        print(
            f"  logreg lambda={lam}: val logloss={loss:.4f} "
            f"roc={roc_auc(list(yva), list(val_scores)):.3f} "
            f"pr={pr_auc(list(yva), list(val_scores)):.3f}"
        )
        if best_val_loss is None or loss < best_val_loss:
            best_m, best_lambda, best_val_loss = m, lam, loss
    print(f"  -> chosen lambda={best_lambda}")

    logreg = {
        "weights": [float(x) for x in best_m.coef_[0]],
        "bias": float(best_m.intercept_[0]),
        "lambda": best_lambda,
        "valLogLoss": best_val_loss,
    }

    # Random forest: single honest configuration (no tuning grid).
    mtry = max(1, Xtr.shape[1] // 3)
    rf = RandomForestClassifier(
        n_estimators=60,
        max_depth=8,
        min_samples_split=20,
        max_features=mtry,
        random_state=123,
        n_jobs=1,
    )
    rf.fit(Xtr, ytr)
    nodes = sum(est.tree_.node_count for est in rf.estimators_)
    print(f"  forest trained in-bound, {nodes} total nodes")

    val_log_scores = [float(x) for x in best_m.predict_proba(Xva)[:, 1]]
    val_for_scores = [float(x) for x in rf.predict_proba(Xva)[:, 1]]
    print(
        f"  val logreg: roc={roc_auc(list(yva), val_log_scores):.3f} pr={pr_auc(list(yva), val_log_scores):.3f}"
    )
    print(f"  val forest: roc={roc_auc(list(yva), val_for_scores):.3f} pr={pr_auc(list(yva), val_for_scores):.3f}")

    # Costed threshold per model on validation ($25 miss / $4 false alarm).
    th_log = tune_threshold(list(yva), val_log_scores)
    th_for = tune_threshold(list(yva), val_for_scores)
    for name, scores, th in (
        ("logreg", val_log_scores, th_log),
        ("forest", val_for_scores, th_for),
    ):
        m = prf(list(yva), scores, th["t"])
        print(
            f"  val {name} @{th['t']:.2f}: P={m['precision']:.3f} R={m['recall']:.3f} "
            f"cost=${th['total']} (${th['total'] / 30:.2f}/day)"
        )

    models_dir = f"{root}/models"
    save_models(
        models_dir,
        (
            preproc,
            logreg,
            {**_export_sklearn_forest(rf, {"nTrees": 60, "maxDepth": 8, "minSplit": 20, "mtry": mtry, "seed": 123})},
            {"primary": "logreg", "logreg": th_log["t"], "forest": th_for["t"], "costFn": COST_FN, "costFp": COST_FP},
            _baseline_stats(train, val_log_scores, th_log["t"]),
        ),
    )
    print(f"artifacts -> {os.path.abspath(models_dir)} (primary=logreg)")
    return models_dir


def _baseline_stats(train, val_scores, threshold):
    decile_edges = {
        "weight_kg": deciles([r["weight_kg"] for r in train]),
        "distance_km": deciles([r["distance_km"] for r in train]),
        "promised_lead_days": deciles([r["promised_lead_days"] for r in train]),
    }
    from .metrics import bin_index

    baseline_props = {}
    for name, edges in decile_edges.items():
        vals = [r[name] for r in train if r[name] is not None]
        counts = [0] * 10
        for v in vals:
            counts[bin_index(edges, v)] += 1
        baseline_props[name] = [c / len(vals) if vals else 0.0 for c in counts]

    utils = [r["utilization"] for r in train if r["utilization"] is not None]
    return json.loads(json.dumps({
        "fitted_on": "train Jan-Mar 2025 (labelable only)",
        "n_train": len(train),
        "numeric_deciles": decile_edges,
        "numeric_baseline_props": baseline_props,
        "over_capacity_share": sum(1 for v in utils if v >= 1) / len(utils) if utils else 0.0,
        "cat_freqs": {
            "carrier_id": freq(train, "carrier_id"),
            "warehouse_id": freq(train, "warehouse_id"),
            "service_level": freq(train, "service_level"),
            "dest_region": freq(train, "dest_region"),
        },
        "missing_rates": {
            "weight_kg": sum(1 for r in train if r["weight_kg"] is None) / len(train),
            "distance_km": sum(1 for r in train if r["distance_km"] is None) / len(train),
            "dest_region": sum(1 for r in train if r["dest_region"] == "MISSING") / len(train),
            "utilization": sum(1 for r in train if r["utilization"] is None) / len(train),
        },
        "val_positive_rate_at_threshold": sum(1 for s in val_scores if s >= threshold) / len(val_scores),
    }))


if __name__ == "__main__":
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) if len(sys.argv) < 2 else sys.argv[1]
    train(root)
