"""Decay monitor — port of src/monitor.js with identical bands and exit codes.

  uv run python -m parcelrank.monitor --current <csv> [--model logreg] [--threshold auto|<0-1>]

Checks (bands fixed from five in-distribution months Jan-May, verified quiet
there and firing on the June disruption — docs/10):
  - PSI > 0.25 BREACH / > 0.10 warn, per numeric + categorical input
    (utilization excluded: decile-PSI pages on healthy months; overload is
    watched via the share below instead)
  - over-capacity share > 40% BREACH / > 30% warn (train 13.8%, June 63%)
  - predicted-positive-rate shift > 25pp BREACH ONLY (breakage guard)
  - missing-rate increase > 5pp warn
  - quarantine rate > 20% BREACH / > 5% warn
Exit 0 when clean-or-warnings (warnings print), 1 on any breach, 2 on bad args.
"""
import math
import os
import sys

from .artifacts import load_models
from .clean import load_carriers, load_capacity, load_shipments
from .features import transform_all
from .metrics import bin_index, psi

PSI_BREACH = 0.25
PSI_WARN = 0.1
RATE_BREAKAGE_PP = 0.25
OVERCAP_WARN = 0.3
OVERCAP_BREACH = 0.4


def psi_numeric(edges, base_props, values):
    cur = [0] * 10
    vals = [v for v in values if v is not None]
    if not vals:
        return {"psi": 0.0, "empty": True}
    for v in vals:
        cur[bin_index(edges, v)] += 1
    cur = [c / len(vals) for c in cur]
    return {"psi": psi(base_props, cur), "empty": False}


def psi_categorical(base_freq, values):
    keys = sorted(set(list(base_freq.keys()) + [v for v in values if v is not None]))
    cur = {}
    for v in values:
        cur[v] = cur.get(v, 0) + 1
    if not values:
        return 0.0
    expected = [base_freq.get(k, 0.0) for k in keys]
    actual = [_approx((cur.get(k, 0.0)) / len(values)) for k in keys]
    return psi(expected, actual)


def _approx(x):
    return x if x > 0 else 1e-4  # matches Node: psi() self-floors anyway


def check_batch(root, current, model="logreg", threshold=None):
    artifacts = load_models(f"{root}/models")
    t = threshold if threshold is not None else artifacts["thresholds"][model]
    carriers = load_carriers(f"{root}/data/carriers.csv")
    cap = load_capacity(f"{root}/data/warehouse_capacity.csv")
    res = load_shipments(current, capacity=cap["table"], known_carriers=carriers)
    rows, stats = res["rows"], res["stats"]
    base = artifacts["baseline_stats"]
    breaches = []
    warnings = []
    info = []
    import numpy as np

    # 1. Input distribution drift (PSI).
    for name, edges in base["numeric_deciles"].items():
        out = psi_numeric(edges, base["numeric_baseline_props"][name], [r[name] for r in rows])
        if out["empty"]:
            warnings.append(f"{name}: no non-missing values in current batch")
            continue
        line = f"{name}: PSI={out['psi']:.3f}"
        if out["psi"] > PSI_BREACH:
            breaches.append(f"BREACH {line} (> {PSI_BREACH})")
        elif out["psi"] > PSI_WARN:
            warnings.append(f"warn {line} (> {PSI_WARN})")
        else:
            info.append(f"ok {line}")
    for name, base_freq in base["cat_freqs"].items():
        v = psi_categorical(base_freq, [r[name] for r in rows])
        line = f"{name}: PSI={v:.3f}"
        if v > PSI_BREACH:
            breaches.append(f"BREACH {line} (> {PSI_BREACH})")
        elif v > PSI_WARN:
            warnings.append(f"warn {line} (> {PSI_WARN})")
        else:
            info.append(f"ok {line}")

    # 2. Warehouse overload.
    utils = [r["utilization"] for r in rows if r["utilization"] is not None]
    over_cap = (sum(1 for v in utils if v >= 1) / len(utils)) if utils else 0.0
    over_line = (
        f"over-capacity share {100 * over_cap:.1f}% vs baseline {100 * base['over_capacity_share']:.1f}%"
    )
    if not utils:
        warnings.append("utilization: no non-missing values in current batch")
    elif over_cap > OVERCAP_BREACH:
        breaches.append(f"BREACH {over_line} (> {round(100 * OVERCAP_BREACH)}%)")
    elif over_cap > OVERCAP_WARN:
        warnings.append(f"warn {over_line} (> {round(100 * OVERCAP_WARN)}%)")
    else:
        info.append(f"ok {over_line}")

    # 3. Prediction breakage guard.
    preproc, logreg, forest = artifacts["preproc"], artifacts["logreg"], artifacts["forest"]
    if model == "logreg":
        import numpy as np

        w = np.array(logreg["weights"])
        b = logreg["bias"]
        scores = list(1.0 / (1.0 + np.exp(-(np.array(transform_all(rows, preproc), dtype=float) @ w + b))))
    else:
        scores = []
        for x in transform_all(rows, preproc):
            scores.append(_prob_forest(x, forest["trees"]))
    pos_rate = sum(1 for s in scores if s >= t) / len(scores) if scores else 0.0
    base_rate = base["val_positive_rate_at_threshold"]
    d_rate = abs(pos_rate - base_rate)
    rate_line = f"positive rate {pos_rate:.3f} vs baseline {base_rate:.3f} (delta {100 * d_rate:.1f}pp @{t})"
    if d_rate > RATE_BREAKAGE_PP:
        breaches.append(f"BREACH {rate_line} (breakage band +/-{round(100 * RATE_BREAKAGE_PP)}pp)")
    else:
        info.append(f"ok {rate_line}")

    # 4. Missingness drift.
    miss_now = {
        "weight_kg": sum(1 for r in rows if r["weight_kg"] is None) / len(rows),
        "distance_km": sum(1 for r in rows if r["distance_km"] is None) / len(rows),
        "dest_region": sum(1 for r in rows if r["dest_region"] == "MISSING") / len(rows),
        "utilization": sum(1 for r in rows if r["utilization"] is None) / len(rows),
    }
    for k, v in miss_now.items():
        d = v - base["missing_rates"][k]
        if d > 0.05:
            warnings.append(
                f"warn {k} missingness {100 * v:.1f}% vs {100 * base['missing_rates'][k]:.1f}% baseline"
            )
        else:
            info.append(f"ok {k} missingness {100 * v:.1f}%")

    # 5. Quarantine volume.
    q_rate = (stats["rawRows"] - stats["kept"]) / stats["rawRows"] if stats["rawRows"] else 1
    if q_rate > 0.2:
        breaches.append(f"BREACH quarantine rate {100 * q_rate:.1f}% (> 20%)")
    elif q_rate > 0.05:
        warnings.append(f"warn quarantine rate {100 * q_rate:.1f}% (> 5%)")
    else:
        info.append(f"ok quarantine rate {100 * q_rate:.1f}%")

    # 6. Unseen categories (informational).
    for name in ("carrier_id", "service_level"):
        known = base["cat_freqs"][name]
        unseen = sorted({r[name] for r in rows} - set(known))
        if unseen:
            info.append(f"info new {name} values: {', '.join(unseen)}")

    return {
        "breaches": breaches, "warnings": warnings, "info": info,
        "pos_rate": pos_rate, "n": len(rows), "threshold": t, "model": model,
    }


def _prob_forest(x, trees):
    def leaf(tree, xx):
        while not tree["leaf"]:
            tree = tree["left"] if xx[tree["feature"]] <= tree["threshold"] else tree["right"]
        return tree["p"]

    return sum(leaf(t, x) for t in trees) / len(trees)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--current" not in argv:
        print(
            "usage: uv run python -m parcelrank.monitor --current <csv> "
            "[--model logreg|forest] [--threshold auto|<0-1>]",
            file=sys.stderr,
        )
        return 2
    ci = argv.index("--current")

    def after(flag, default):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        return default

    model = after("--model", "logreg")
    if model not in ("logreg", "forest"):
        print(f"ERROR: --model must be logreg or forest (got {model})", file=sys.stderr)
        return 2
    current = argv[ci + 1] if ci + 1 < len(argv) else None
    if not current or not os.path.exists(current):
        print(f"ERROR: file not found: {current}", file=sys.stderr)
        return 2
    t_arg = after("--threshold", "auto")
    threshold = None if t_arg == "auto" else _float_or(t_arg)

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r = check_batch(root, current, model, threshold)
    print(f"monitor: n={r['n']} model={r['model']} threshold={r['threshold']}")
    for l in [*r["breaches"], *r["warnings"], *r["info"]]:
        print(f"  {l}")
    print("  note: true labels lag ~2 weeks; this watches inputs + predictions, not accuracy. Pair with a lagged backtest.")
    if r["breaches"]:
        print(f"RESULT: BREACH ({len(r['breaches'])} failing check(s)) — investigate before trusting flags.")
        return 1
    print(f"RESULT: OK ({len(r['warnings'])} warning(s)).")
    return 0


def _float_or(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    sys.exit(main())
