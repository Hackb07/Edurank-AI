"""Feature engineering — exact port of the Node pipeline's src/features.js.

Golden rule: features are built ONLY from an explicit allowlist of
known-at-ship fields. Post-outcome columns are never read — not for values,
not for imputation, not for flags.

Pipeline: fit_preprocessing(train_rows) -> preproc dict (frozen, saved to
models/preprocessing.json) -> transform(row, preproc) -> list[float].
Unseen categories (new carrier C07, same_day service) fall into the OTHER
bucket plus is_new_* flags so schema drift cannot crash scoring.
"""
import math

FORBIDDEN = [
    "actual_delivery_date",
    "actual_ymd",
    "days_in_transit",
    "delivery_status",
    "carrier_delay_note",
    "address_correction_after_ship",
]

NUM = [
    "weight_kg",
    "distance_km",
    "items_count",
    "promised_lead_days",
    "utilization",
    "ship_month",
    "ship_weekday",
    "priority_flag",
    "carrier_hist_rate",
    # Binary indicators kept in NUM so they are standardized like the rest:
    # Friday dispatch (weekend dwell), capacity breach (overload), tight
    # promise (<= 1 day).
    "is_friday",
    "util_over_1",
    "lead_tight",
]

# Hard plausibility bounds: outside values are data errors that slipped past
# cleaning — clip, don't drop (destroys less evidence; *_suspect flags speak).
CLIP = {
    "weight_kg": (0.05, 100),
    "distance_km": (1, 5000),
    "items_count": (1, 20),
    "promised_lead_days": (-2, 14),
    "utilization": (0, 3),
}

CATS = [
    "carrier_id",
    "warehouse_id",
    "service_level",
    "parcel_type",
    "origin_region",
    "dest_region",
    "lane",  # origin|dest interaction (bad corridors live here)
]

FLAGS = [
    "weight_missing",
    "weight_unit_suspect",
    "weight_invalid",
    "distance_missing",
    "distance_invalid",
    "dest_missing",
    "promised_missing",
    "carrier_unknown",
    "warehouse_unknown",
    "util_missing",
    "items_invalid",
]


def month_of(ymd: str) -> int:
    return int(ymd[5:7])


def median(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def clip_val(name, v):
    b = CLIP.get(name)
    if b is None or v is None:
        return v
    return min(b[1], max(b[0], v))


def raw_numeric(row, name, carrier_rates, global_rate):
    if name == "weight_kg":
        return row.get("weight_kg")
    if name == "distance_km":
        return row.get("distance_km")
    if name == "items_count":
        return row.get("items_count")
    if name == "promised_lead_days":
        return row.get("promised_lead_days")
    if name == "utilization":
        return row.get("utilization")
    if name == "ship_month":
        return float(month_of(row["ship_ymd"]))
    if name == "ship_weekday":
        return float(weekday_js(row["ship_ms"]))
    if name == "priority_flag":
        return float(row.get("priority_flag") or 0)
    if name == "carrier_hist_rate":
        return carrier_rates.get(row["carrier_id"], global_rate)
    if name == "is_friday":
        return 1.0 if weekday_js(row["ship_ms"]) == 5 else 0.0
    if name == "util_over_1":
        u = row.get("utilization")
        return 1.0 if (u is not None and u > 1) else 0.0
    if name == "lead_tight":
        p = row.get("promised_lead_days")
        return 1.0 if (p is not None and p <= 1) else 0.0
    return None


def weekday_js(ms: int) -> int:
    """0=Sun..6=Sat (JavaScript getUTCDay equivalent), from epoch ms."""
    from datetime import datetime, timezone

    return (datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).weekday() + 1) % 7


def fit_preprocessing(train_rows):
    # Carrier history rates: TRAIN window only, frozen before validation.
    by_carrier = {}
    g_late = 0.0
    g_n = 0
    for r in train_rows:
        if r.get("label") is None:
            continue
        key = r["carrier_id"]
        ent = by_carrier.setdefault(key, [0.0, 0])
        ent[1] += 1
        ent[0] += r["label"]
        g_n += 1
        g_late += r["label"]
    carrier_rates = {c: (l / n) for c, (l, n) in by_carrier.items() if n}
    global_rate = (g_late / g_n) if g_n else 0.2

    # Medians for imputation (train only).
    buckets = {n: [] for n in NUM}
    for r in train_rows:
        for name in NUM:
            v = raw_numeric(r, name, carrier_rates, global_rate)
            if v is not None:
                buckets[name].append(clip_val(name, v))
    medians = {n: (median(buckets[n]) if buckets[n] else 0.0) for n in NUM}
    medians["utilization"] = 1.0  # fixed prior: unknown load reads as "full"

    # Standardization stats on imputed+clipped train values.
    means = {}
    stds = {}
    for name in NUM:
        vals = []
        for r in train_rows:
            v = raw_numeric(r, name, carrier_rates, global_rate)
            vals.append(clip_val(name, medians[name] if v is None else v))
        mean = sum(vals) / len(vals) if vals else 0.0
        var = sum((b - mean) ** 2 for b in vals) / len(vals) if vals else 0.0
        sd = math.sqrt(var) or 1.0
        means[name] = mean
        stds[name] = sd

    # Known categories per column (train only).
    categories = {c: sorted({r[c] for r in train_rows}) for c in CATS}

    feature_names = []
    for n in NUM:
        feature_names.append(f"num:{n}")
    for c in CATS:
        for v in categories[c]:
            feature_names.append(f"cat:{c}={v}")
        feature_names.append(f"cat:{c}=OTHER")
    for f in FLAGS:
        feature_names.append(f"flag:{f}")
    feature_names.extend(["flag:is_new_carrier", "flag:is_new_service"])

    return {
        "medians": medians,
        "means": means,
        "stds": stds,
        "categories": categories,
        "carrierRates": carrier_rates,
        "globalRate": global_rate,
        "featureNames": feature_names,
    }


def transform(row, preproc):
    vec = []
    for name in NUM:
        v = raw_numeric(row, name, preproc["carrierRates"], preproc["globalRate"])
        if v is None:
            v = preproc["medians"][name]
        v = clip_val(name, v)
        vec.append((v - preproc["means"][name]) / preproc["stds"][name])

    is_new_carrier = 0
    is_new_service = 0
    for c in CATS:
        cats = preproc["categories"][c]
        known = row[c] in cats
        if not known:
            if c == "carrier_id":
                is_new_carrier = 1
            if c == "service_level":
                is_new_service = 1
        for v in cats:
            vec.append(1.0 if row[c] == v else 0.0)
        vec.append(0.0 if known else 1.0)  # OTHER bucket
    flags = row.get("flags") or {}
    for f in FLAGS:
        vec.append(1.0 if flags.get(f) else 0.0)
    vec.append(float(is_new_carrier))
    vec.append(float(is_new_service))
    return vec


def transform_all(rows, preproc):
    return [transform(r, preproc) for r in rows]
