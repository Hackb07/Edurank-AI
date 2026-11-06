"""Shipped CLI: score new parcels from the command line — port of
src/predict.js with identical behavior and exit codes.

  uv run python -m parcelrank.cli --input new_parcels.csv --output out/flags.csv \
      [--top-k 25] [--floor 0.10] [--threshold auto|<0-1>] [--model logreg|forest] \
      [--prioritize a,b] [--explain]

Behavior:
 - Same cleaning pipeline as training (no train/serve skew by construction).
 - Strict validation: required columns must exist; rows with unparseable ship
   dates or missing promised dates are quarantined, never silently scored.
   Post-outcome columns in the input are IGNORED (features can't read them).
 - Daily policy: per ship-date, rank by (priority tier, probability), flag
   top K rows above --floor. --top-k 0 disables the cap (pure threshold).
 - Refusal: if more than 20% of rows quarantine — exit 2 without trusting
   the output. Exit 0 on success, 2 on input/validation failure.

Output: parcel_id, ship_ymd, late_probability, flag (+ reason with --explain).
Quarantine: <output>.quarantine.csv with parcel_id + reason.
"""
import math
import os
import sys

from .artifacts import load_models
from .clean import load_carriers, load_capacity, load_shipments, QUARANTINE_SAMPLE
from .csv_io import parse_csv, to_csv, read_text
from .features import FORBIDDEN, transform_all
from .metrics import apply_daily_policy, default_prioritize

REQUIRED_COLUMNS = [
    "parcel_id", "ship_date", "promised_date", "carrier_id", "warehouse_id",
    "origin_region", "dest_region", "service_level", "parcel_type",
    "items_count", "weight_value", "weight_unit", "distance_km",
]
QUARANTINE_REFUSAL_RATE = 0.2


def _prob_logreg(x, w, b):
    z = b
    for wj, xj in zip(w, x):
        z += wj * xj
    if z >= 0:
        e = math.exp(-z)
        return 1 / (1 + e)
    e = math.exp(z)
    return e / (1 + e)


def _prob_forest(x, trees):
    def leaf(tree, xx):
        while not tree["leaf"]:
            tree = tree["left"] if xx[tree["feature"]] <= tree["threshold"] else tree["right"]
        return tree["p"]

    return sum(leaf(t, x) for t in trees) / len(trees)


def run_predict(root, *, input, output, top_k=25, floor=0.10, threshold=None,
                model="logreg", prioritize=None, explain=False):
    prioritize = prioritize if prioritize is not None else default_prioritize()
    errors = []
    if not os.path.exists(input):
        errors.append(f"input file not found: {input}")
    if model not in ("logreg", "forest"):
        errors.append(f"--model must be logreg or forest (got {model})")
    if not (top_k >= 0):
        errors.append(f"--top-k must be >= 0 (got {top_k})")
    if not (floor >= 0 and floor <= 1):
        errors.append(f"--floor must be in [0,1] (got {floor})")
    if threshold is not None and not (threshold >= 0 and threshold <= 1):
        errors.append(f"--threshold must be in [0,1] (got {threshold})")
    if errors:
        return {"ok": False, "errors": errors}

    text = read_text(input)
    header, _ = parse_csv(text)
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        return {"ok": False, "errors": [f"missing required columns: {', '.join(missing)}"]}
    ignored = [c for c in FORBIDDEN if c in header]

    carriers = load_carriers(f"{root}/data/carriers.csv")
    cap = load_capacity(f"{root}/data/warehouse_capacity.csv")
    res = load_shipments(input, capacity=cap["table"], known_carriers=carriers)
    rows, quarantine, stats = res["rows"], [dict(q) for q in res["quarantine"]], res["stats"]

    # Prediction-time quarantine: no promise => no assessable risk.
    scorable = []
    extra_q = []
    for r in rows:
        if not r.get("promised_ymd"):
            extra_q.append({
                "parcel_id": r["parcel_id"],
                "reason": "missing promised_date (cannot assess promise risk)",
            })
        else:
            scorable.append(r)

    all_q = list(quarantine) + extra_q
    q_rate = all_q.__len__() / stats["rawRows"] if stats["rawRows"] else 1
    if stats["rawRows"] == 0 or not scorable:
        return {"ok": False, "errors": ["no scorable rows: input is empty or every row quarantined"], "quarantine": all_q}
    if q_rate > QUARANTINE_REFUSAL_RATE:
        return {
            "ok": False,
            "errors": [
                f"refusing to score: quarantine rate {100 * q_rate:.1f}% exceeds 20% — "
                "fix the input, see quarantine file"
            ],
            "quarantine": all_q,
        }

    models = load_models(f"{root}/models")
    t = threshold if threshold is not None else models["thresholds"][model]
    X = transform_all(scorable, models["preproc"])
    if model == "logreg":
        w, b = models["logreg"]["weights"], models["logreg"]["bias"]
        scores = [_prob_logreg(x, w, b) for x in X]
    else:
        trees = models["forest"]["trees"]
        scores = [_prob_forest(x, trees) for x in X]

    flags = apply_daily_policy(scorable, scores, top_k=top_k, floor=floor, threshold=t, tiers=prioritize)
    by_day = {}
    for r in scorable:
        by_day[r["ship_ymd"]] = by_day.get(r["ship_ymd"], 0) + 1

    out_header = ["parcel_id", "ship_ymd", "late_probability", "flag"] + (["reason"] if explain else [])
    out_rows = []
    for i, r in enumerate(scorable):
        base = {
            "parcel_id": r["parcel_id"],
            "ship_ymd": r["ship_ymd"],
            "late_probability": f"{scores[i]:.4f}",
            "flag": flags[i],
        }
        if explain:
            if model == "logreg":
                if flags[i]:
                    expl = explain_parcel_from(scorable[i], X[i], models["preproc"], models["logreg"])
                    base["reason"] = reason_line(expl)
                else:
                    base["reason"] = f"Not flagged ({round(100 * scores[i])}% late risk)."
            else:
                base["reason"] = (
                    (
                        f"Flagged by forest ({round(100 * scores[i])}% of trees vote late). "
                        "Top risk factors unavailable exactly for trees; see global importance."
                    )
                    if flags[i]
                    else f"Not flagged ({round(100 * scores[i])}% late risk)."
                )
        out_rows.append(base)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        f.write(to_csv(out_header, out_rows))
    q_path = f"{output}.quarantine.csv"
    with open(q_path, "w", encoding="utf-8", newline="") as f:
        f.write(to_csv(["parcel_id", "reason"], all_q))

    return {
        "ok": True,
        "errors": [],
        "summary": {
            "input": input, "output": output, "quarantineFile": q_path, "model": model,
            "threshold": t, "topK": top_k, "floor": floor,
            "rawRows": stats["rawRows"], "scored": len(scorable),
            "flagged": sum(flags), "quarantined": len(all_q),
            "ignoredPostOutcomeColumns": ignored, "days": len(by_day),
            "meanProb": sum(scores) / len(scores),
        },
    }


def explain_parcel_from(row, vec, preproc, logreg):
    from .explain import explain_parcel

    return explain_parcel(row, vec, preproc, logreg)


def reason_line(expl):
    from .explain import reason_line

    return reason_line(expl)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    def arg(name, default=None):
        if name in argv:
            i = argv.index(name)
            if i + 1 < len(argv):
                return argv[i + 1]
        return default

    def has(name):
        return name in argv

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input = arg("--input")
    output = arg("--output")
    if not input or not output:
        print(
            "usage: uv run python -m parcelrank.cli --input <csv> --output <csv> "
            "[--top-k 25] [--floor 0.10] [--threshold auto|<0-1>] [--model logreg|forest] "
            "[--prioritize a,b] [--explain]",
            file=sys.stderr,
        )
        return 2
    t_arg = arg("--threshold", "auto")
    t_num = None if t_arg == "auto" else _float_or(t_arg)
    model = arg("--model", "logreg")
    explain = has("--explain")

    result = run_predict(
        root,
        input=input,
        output=output,
        top_k=int(_float_or(arg("--top-k", "25"))) if arg("--top-k", "25") is not None else 25,
        floor=_float_or(arg("--floor", "0.10")),
        threshold=t_num,
        model=model,
        prioritize=[s.strip() for s in (arg("--prioritize") or ",".join(default_prioritize())).split(",") if s.strip()],
        explain=explain,
    )
    if not result["ok"]:
        for e in result["errors"]:
            print(f"ERROR: {e}", file=sys.stderr)
        q = result.get("quarantine")
        if q:
            q_path = f"{output}.quarantine.csv"
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            with open(q_path, "w", encoding="utf-8", newline="") as f:
                f.write(to_csv(["parcel_id", "reason"], q))
            print(f"quarantine details -> {q_path}", file=sys.stderr)
        return 2

    s = result["summary"]
    print(
        f"scored {s['scored']}/{s['rawRows']} rows over {s['days']} day(s), flagged {s['flagged']} "
        f"(model={s['model']} threshold={s['threshold']} top-k={s['topK']} floor={s['floor']})"
    )
    print(f"mean late probability: {s['meanProb']:.3f}; quarantined: {s['quarantined']}")
    if s["ignoredPostOutcomeColumns"]:
        print(f"note: ignored post-outcome columns present in input: {', '.join(s['ignoredPostOutcomeColumns'])}")
    print(f"flags -> {s['output']}")
    print(f"quarantine -> {s['quarantineFile']}")
    return 0


def _float_or(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    sys.exit(main())
