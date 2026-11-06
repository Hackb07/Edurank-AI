"""Explanations, two levels — port of src/explain.js.

1. Per-parcel (explain_parcel): EXACT linear contributions for the primary
   logreg model — contribution_j = weight_j * x_j, which sum (plus bias) to
   the log-odds. Top risk-increasing factors render in plain language; the
   forest challenger only gets global importances (approximate, labeled so).
2. Global (permutation_importance + `uv run python -m parcelrank.explain`):
   log-loss increase when a feature column is shuffled (seeded).
"""
import math

from sklearn.metrics import log_loss

from .artifacts import load_models
from .clean import load_carriers, load_capacity, load_shipments
from .features import transform_all

MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


def weekday_js(ms):
    from datetime import datetime, timezone

    return (datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).weekday() + 1) % 7


def raw_of(row, name):
    return row.get(name)


def describe(feature, row, preproc):
    kind, name = feature.split(":", 1)
    if kind == "num":
        if name == "carrier_hist_rate":
            rate = preproc["carrierRates"].get(row["carrier_id"])
            base = rate if rate is not None else preproc["globalRate"]
            return (
                f"carrier {row['carrier_id']} was late {pct(base)} of the time in Jan-Mar "
                f"(overall {pct(preproc['globalRate'])})"
            )
        if name == "utilization":
            return (
                "warehouse load that week is unknown"
                if row.get("utilization") is None
                else f"warehouse {row['warehouse_id']} was at {pct(row['utilization'])} of capacity that week"
            )
        if name == "util_over_1":
            return f"warehouse {row['warehouse_id']} was OVER capacity that week"
        if name == "is_friday":
            return "shipped on a Friday (parcels sit over the weekend)"
        if name == "ship_weekday":
            return f"shipped on a {WEEKDAYS[weekday_js(row['ship_ms'])]}"
        if name == "ship_month":
            return f"shipped in {MONTHS[int(row['ship_ymd'][5:7])]}"
        if name == "promised_lead_days":
            return f"promised delivery in {row['promised_lead_days']} day(s)"
        if name == "lead_tight":
            return f"only {row['promised_lead_days']} day(s) promised — a tight promise"
        if name == "distance_km":
            return "distance unknown" if row.get("distance_km") is None else f"long haul ({round(row['distance_km'])} km)"
        if name == "weight_kg":
            return "weight unknown" if row.get("weight_kg") is None else f"parcel weighs {row['weight_kg']:.1f} kg"
        if name == "items_count":
            return f"{row['items_count']} item(s) in the parcel"
        if name == "priority_flag":
            return "marked as priority handling" if row.get("priority_flag") else "not marked as priority handling"
        return f"{name} = {raw_of(row, name)!r}"

    if kind == "cat":
        col, val = name.split("=", 1)
        if val == "OTHER":
            return f"{col} has an unusual value ({row[col]})"
        if col == "service_level":
            return f"{val} shipping (tighter promise than standard)"
        if col == "dest_region":
            return f"headed to {val}"
        if col == "origin_region":
            return f"shipping from {val}"
        if col == "lane":
            o, d = str(row["lane"]).split("|")
            return f"the {o} to {d} lane has a poor record"
        if col == "carrier_id":
            return f"carrier {val}"
        if col == "warehouse_id":
            return f"warehouse {val}"
        if col == "parcel_type":
            return f"parcel type: {val}"
        return f"{col} = {val}"

    if kind == "flag":
        texts = {
            "weight_missing": "weight was missing, so the system guessed",
            "weight_unit_suspect": "weight unit looked wrong (grams labeled as kg)",
            "weight_invalid": "weight was impossible, so the system guessed",
            "distance_missing": "distance was missing or impossible, so the system guessed",
            "distance_invalid": "distance was missing or impossible, so the system guessed",
            "dest_missing": "destination was missing",
            "promised_missing": "promised date was missing",
            "carrier_unknown": "carrier is not in our reference list",
            "warehouse_unknown": "warehouse is not in our reference list",
            "util_missing": "warehouse load that week is unknown",
            "items_invalid": "item count was invalid",
            "is_new_carrier": f"carrier {row['carrier_id']} is new (no history yet)",
            "is_new_service": f"service {row['service_level']} is new (no history yet)",
        }
        return texts.get(name, name)

    return feature


def explain_parcel(row, vec, preproc, logreg, top_k=3):
    """Exact per-parcel explanation for logreg."""
    scored = [
        {
            "feature": f,
            "contribution": logreg["weights"][j] * vec[j],
            "active": vec[j] != 0,
        }
        for j, f in enumerate(preproc["featureNames"])
    ]
    risks = sorted(
        (s for s in scored if s["contribution"] > 0 and s["active"]),
        key=lambda s: -s["contribution"],
    )[:top_k]
    protectors = sorted(
        (s for s in scored if s["contribution"] < 0 and s["active"]),
        key=lambda s: s["contribution"],
    )[:1]
    reasons = [{**s, "text": describe(s["feature"], row, preproc)} for s in risks]
    if protectors:
        reasons.append({**protectors[0], "text": describe(protectors[0]["feature"], row, preproc), "protective": True})
    z = logreg["bias"] + sum(w * x for w, x in zip(logreg["weights"], vec))
    return {"probability": 1 / (1 + math.exp(-z)), "reasons": reasons}


def reason_line(expl):
    risks = [r["text"] for r in expl["reasons"] if not r.get("protective")]
    prot = next((r for r in expl["reasons"] if r.get("protective")), None)
    s = f"Flagged ({pct(expl['probability'])} late risk) because: {'; '.join(risks)}."
    if prot:
        s += f" Working in its favor: {prot['text']}."
    return s


def log_loss_of(y, p):
    return float(log_loss(list(y), [min(1 - 1e-12, max(1e-12, x)) for x in list(p)], labels=[0, 1]))


def permutation_importance(X, y, predict_fn, seed=2025):
    """Model-agnostic: shuffle column j (seeded Fisher-Yates), measure
    log-loss increase. Returns sorted [(j, importance)] (higher = relied on)."""
    import numpy as np
    from .rng import make_rng

    rng = make_rng(seed)
    X = np.asarray(X, dtype=float)
    base = log_loss_of(y, predict_fn(X))
    n, d = X.shape
    n_pos = sum(1 for v in y if v == 1)
    out = []
    for j in range(d):
        col = X[:, j].copy()
        for i in range(n - 1, 0, -1):
            k = int(rng["next"]() * (i + 1))
            col[i], col[k] = col[k], col[i]
        Xs = X.copy()
        Xs[:, j] = col
        out.append((j, log_loss_of(y, predict_fn(Xs)) - base))
    return sorted(out, key=lambda t: -t[1])


def aggregate_to_columns(ranked, feature_names):
    """Aggregate dummy-level importances to column level (cat:col=val -> col)."""
    agg = {}
    for j, importance in ranked:
        f = feature_names[j]
        if f.startswith("num:"):
            col = f[4:]
        elif f.startswith("flag:"):
            col = f[5:]
        else:  # cat:
            col = f[4:].split("=", 1)[0]
        agg[col] = agg.get(col, 0.0) + max(0.0, importance)
    return sorted(agg.items(), key=lambda kv: -kv[1])


def main():
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    artifacts = load_models(f"{root}/models")
    preproc, logreg, forest = artifacts["preproc"], artifacts["logreg"], artifacts["forest"]
    carriers = load_carriers(f"{root}/data/carriers.csv")
    cap = load_capacity(f"{root}/data/warehouse_capacity.csv")
    rows = load_shipments(
        f"{root}/data/shipments_day1.csv", capacity=cap["table"], known_carriers=carriers
    )["rows"]
    sample = [r for r in rows if "2025-04-01" <= r["ship_ymd"] < "2025-05-01" and r["label"] is not None]
    import numpy as np

    X = np.array(transform_all(sample, preproc))
    y = [r["label"] for r in sample]
    print(f"permutation importance on April validation (n={len(y)}), by column:")
    # Model predict functions operating on numpy arrays.
    w = np.array(logreg["weights"])
    b = logreg["bias"]

    def predict_logreg(M):
        return 1.0 / (1.0 + np.exp(-(M @ w + b)))

    def forest_leaf_p(tree, x):
        while not tree["leaf"]:
            tree = tree["left"] if x[tree["feature"]] <= tree["threshold"] else tree["right"]
        return tree["p"]

    def predict_forest(M):
        out = []
        for x in M:
            out.append(sum(forest_leaf_p(t, x) for t in forest["trees"]) / len(forest["trees"]))
        return np.array(out)

    for name, fn in (("logreg", predict_logreg), ("forest", predict_forest)):
        ranked = permutation_importance(X, y, fn)
        print(f"-- {name} --")
        for col, imp in aggregate_to_columns(ranked, preproc["featureNames"])[:15]:
            print(f"  {col}: +{imp:.4f} logloss")


if __name__ == "__main__":
    main()
