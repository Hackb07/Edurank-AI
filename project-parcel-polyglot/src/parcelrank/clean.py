"""Loading + cleaning pipeline shared by train/evaluate/predict/monitor.

Faithful port of the Node pipeline's src/clean.js. Design rules (unchanged):
  1. Cleaning never invents outcomes: unlabelable rows keep label=None.
  2. Cleaning never peeks at post-outcome columns to fix pre-outcome ones.
  3. Everything dropped or imputed is counted in `stats`, and the first N
     quarantined rows are returned with reasons.
"""
import re
from .csv_io import parse_csv
from .dates import parse_flexible, diff_days, week_start_ymd
from .labels import label_late

QUARANTINE_SAMPLE = 50
FLOW_SCALE = 10

DEFAULT_KNOWN_WAREHOUSES = frozenset(["W1", "W2", "W3", "W4"])


def num_or_null(s):
    if s is None:
        return None
    t = str(s).strip().replace(",", "")
    if t == "":
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def norm_upper(s):
    return str(s if s is not None else "").strip().upper()


def norm_lower(s):
    return str(s if s is not None else "").strip().lower()


def norm_region(s):
    t = str(s if s is not None else "").strip().lower()
    if t == "":
        return "MISSING"
    return t[0].upper() + t[1:]


def parse_boolish(s):
    t = norm_lower(s)
    if t in ("true", "1", "yes", "y", "t"):
        return 1
    return 0


# Reference tables ---------------------------------------------------------

def load_carriers(csv_path):
    _, rows = parse_csv(open(csv_path, encoding="utf-8").read())
    return {norm_upper(r.get("carrier_id")) for r in rows if norm_upper(r.get("carrier_id"))}


def load_capacity(csv_path):
    _, rows = parse_csv(open(csv_path, encoding="utf-8").read())
    table = {}
    stats = {"rows": len(rows), "dupWeeks": 0, "badValues": 0, "badWeeks": 0}
    for r in rows:
        wh = norm_upper(r.get("warehouse_id"))
        wk = parse_flexible(r.get("week_start"))
        cap = num_or_null(r.get("capacity_parcels"))
        if not wh or wk is None:
            stats["badWeeks"] += 1
            continue
        if cap is None or cap <= 0:
            stats["badValues"] += 1
            continue
        key = f"{wh}|{wk[1]}"
        if key in table:
            stats["dupWeeks"] += 1
            continue  # first wins
        table[key] = cap
    return {"table": table, "stats": stats}


# Shipments ----------------------------------------------------------------

_NEW_CARRIER = re.compile(r"^C\d+$")


def dedup(raw_rows, stats, quarantine):
    by_id = {}
    for i, r in enumerate(raw_rows):
        pid = str(r.get("parcel_id") or "").strip()
        if not pid:
            stats["blankId"] += 1
            if len(quarantine) < QUARANTINE_SAMPLE:
                quarantine.append({"parcel_id": f"(row {i + 2})", "reason": "blank parcel_id"})
            continue
        by_id.setdefault(pid, []).append((r, i))

    kept = []
    for pid, group in by_id.items():
        if len(group) == 1:
            kept.append(group[0][0])
            continue
        stats["dupIds"] += 1
        stats["dupExtraRows"] += len(group) - 1

        def sig(row):
            return "|".join(
                str(row.get(k) or "") for k in (
                    "promised_date", "carrier_id", "warehouse_id",
                    "delivery_status", "actual_delivery_date",
                )
            )

        first_sig = sig(group[0][0])
        if any(sig(row) != first_sig for row, _ in group):
            stats["dupConflicts"] += 1
        # Last-in-file wins: extracts are appended, later row is the more
        # recent correction (documented assumption, docs/02).
        kept.append(group[-1][0])
        if len(quarantine) < QUARANTINE_SAMPLE:
            quarantine.append({
                "parcel_id": pid,
                "reason": f"duplicate parcel_id x{len(group)}, kept last in file",
            })
    return kept


def clean_weight(raw_val, raw_unit, flags):
    v = num_or_null(raw_val)
    if v is None:
        flags["weight_missing"] = 1
        return None
    unit = norm_lower(raw_unit)
    if unit in ("g", "gram", "grams"):
        return v / 1000.0
    # Mislabelled-grams rescue: no plausible parcel > 500 kg. Deliberately
    # conservative (docs/02). Applied before the <=0/negative check, like the
    # Node pipeline.
    if v > 500:
        flags["weight_unit_suspect"] = 1
        return v / 1000.0
    if v <= 0:
        flags["weight_invalid"] = 1
        return None
    return v


def clean_distance(raw, flags):
    v = num_or_null(raw)
    if v is None:
        flags["distance_missing"] = 1
        return None
    if v <= 0 or v > 20000:
        flags["distance_invalid"] = 1
        return None
    return v


def clean_row(raw, ctx):
    flags = {}
    ship = parse_flexible(raw.get("ship_date"))
    if ship is None:
        return {"drop": "unparseable ship_date"}
    created = parse_flexible(raw.get("created_at"))
    promised = parse_flexible(raw.get("promised_date"))
    actual = parse_flexible(raw.get("actual_delivery_date"))
    if actual is not None and actual[1] < ship[1]:
        return {"drop": "actual_delivery_date before ship_date (impossible)"}

    carrier = norm_upper(raw.get("carrier_id")) or "UNKNOWN"
    if (
        carrier != "UNKNOWN"
        and carrier not in ctx["known_carriers"]
        and not _NEW_CARRIER.match(carrier)
    ):
        carrier = "UNKNOWN"
    if carrier == "UNKNOWN":
        flags["carrier_unknown"] = 1

    wh = norm_upper(raw.get("warehouse_id")) or "UNKNOWN"
    if wh != "UNKNOWN" and wh not in ctx["known_warehouses"]:
        flags["warehouse_unknown"] = 1
        wh = "UNKNOWN"

    service = norm_lower(raw.get("service_level")) or "MISSING"
    ptype = norm_lower(raw.get("parcel_type")) or "MISSING"
    origin = norm_region(raw.get("origin_region"))
    dest = norm_region(raw.get("dest_region"))
    if dest == "MISSING":
        flags["dest_missing"] = 1

    weight_kg = clean_weight(raw.get("weight_value"), raw.get("weight_unit"), flags)
    distance_km = clean_distance(raw.get("distance_km"), flags)

    items_raw = num_or_null(raw.get("items_count"))
    if items_raw is None:
        flags["items_invalid"] = 1
        items = 1
    else:
        import math

        items = int(math.floor(items_raw))
        if items < 1:
            flags["items_invalid"] = 1
            items = 1

    promised_lead_days = diff_days(promised[1], ship[1]) if promised is not None else None
    if promised_lead_days is None:
        flags["promised_missing"] = 1

    utilization = None
    if wh != "UNKNOWN":
        wk = week_start_ymd(ship[0])
        key = f"{wh}|{wk}"
        cap = ctx["capacity"].get(key)
        if cap is not None and key in ctx["week_volume"]:
            utilization = ctx["week_volume"][key] / cap
    if utilization is None:
        flags["util_missing"] = 1

    priority = 0 if "priority_flag" not in raw else parse_boolish(raw.get("priority_flag"))

    clean = {
        "parcel_id": str(raw.get("parcel_id", "")).strip(),
        "ship_ms": ship[0],
        "ship_ymd": ship[1],
        "created_ms": created[0] if created is not None else None,
        "promised_ymd": promised[1] if promised is not None else None,
        "actual_ymd": actual[1] if actual is not None else None,
        "carrier_id": carrier,
        "warehouse_id": wh,
        "origin_region": origin,
        "dest_region": dest,
        "lane": f"{origin}|{dest}",
        "service_level": service,
        "parcel_type": ptype,
        "items_count": items,
        "weight_kg": weight_kg,
        "distance_km": distance_km,
        "promised_lead_days": promised_lead_days,
        "utilization": utilization,
        "priority_flag": priority,
        "delivery_status": norm_lower(raw.get("delivery_status")),
        "flags": flags,
    }
    label, reason = label_late(clean)
    clean["label"] = label
    clean["unlabelable_reason"] = reason
    return {"clean": clean}


def load_shipments(ship_csv_path, *, capacity, known_carriers, known_warehouses=DEFAULT_KNOWN_WAREHOUSES):
    header, raw_rows = parse_csv(open(ship_csv_path, encoding="utf-8").read())
    stats = {
        "rawRows": len(raw_rows), "blankId": 0, "dupIds": 0, "dupExtraRows": 0,
        "dupConflicts": 0, "dropped": 0, "kept": 0, "labelable": 0, "unlabelable": 0,
        "dropReasons": {}, "unlabelableReasons": {},
        "hasPriorityCol": "priority_flag" in header,
        "hasTransitCol": "days_in_transit" in header,
    }
    quarantine = []
    deduped = dedup(raw_rows, stats, quarantine)

    # First pass: ship dates for volume counting (parse-only, no drops yet).
    with_ship = []
    for r in deduped:
        ship = parse_flexible(r.get("ship_date"))
        if ship is None:
            stats["dropped"] += 1
            stats["dropReasons"]["unparseable ship_date"] = stats["dropReasons"].get("unparseable ship_date", 0) + 1
            if len(quarantine) < QUARANTINE_SAMPLE:
                quarantine.append({"parcel_id": str(r.get("parcel_id", "")).strip(), "reason": "unparseable ship_date"})
            continue
        with_ship.append((r, ship[0]))

    week_volume = {}
    for r, ship_ms in with_ship:
        wh = norm_upper(r.get("warehouse_id"))
        if wh not in known_warehouses:
            continue
        key = f"{wh}|{week_start_ymd(ship_ms)}"
        week_volume[key] = week_volume.get(key, 0) + FLOW_SCALE

    ctx = {
        "capacity": capacity,
        "known_carriers": known_carriers,
        "known_warehouses": known_warehouses,
        "week_volume": week_volume,
    }
    cleaned = []
    for r, _ in with_ship:
        out = clean_row(r, ctx)
        if "drop" in out:
            stats["dropped"] += 1
            reason = out["drop"]
            stats["dropReasons"][reason] = stats["dropReasons"].get(reason, 0) + 1
            if len(quarantine) < QUARANTINE_SAMPLE:
                quarantine.append({"parcel_id": str(r.get("parcel_id", "")).strip(), "reason": reason})
            continue
        clean = out["clean"]
        stats["kept"] += 1
        if clean["label"] is None:
            stats["unlabelable"] += 1
            k = clean["unlabelable_reason"]
            stats["unlabelableReasons"][k] = stats["unlabelableReasons"].get(k, 0) + 1
        else:
            stats["labelable"] += 1
        cleaned.append(clean)
    return {"rows": cleaned, "quarantine": quarantine, "stats": stats, "header": header}


def load_reference_tables(root):
    """Standard loaders used by every entrypoint (matches the Node callers)."""
    carriers = load_carriers(f"{root}/data/carriers.csv")
    cap = load_capacity(f"{root}/data/warehouse_capacity.csv")
    return carriers, cap["table"]
