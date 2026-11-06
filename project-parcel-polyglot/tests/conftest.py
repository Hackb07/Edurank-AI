"""Shared test fixtures.

build_mini_root() creates a fully self-contained repo root with synthetic
data (deterministic, no RNG needed) and trains models into it, so tests can
exercise the train/CLI/monitor paths in ~1-3s per fixture.
"""
import datetime
from pathlib import Path

from parcelrank.csv_io import to_csv
from parcelrank.train import train

DAY = 86_400_000
CARRIER_IDS = ["C01", "C02", "C03"]
SHIP_HEADER = [
    "parcel_id", "created_at", "ship_date", "promised_date", "carrier_id",
    "warehouse_id", "origin_region", "dest_region", "service_level",
    "parcel_type", "items_count", "weight_value", "weight_unit", "distance_km",
    "actual_delivery_date", "delivery_status", "carrier_delay_note",
]


def utc(y, m, d):
    return int(datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc).timestamp() * 1000)


def iso(ms) -> str:
    dt = datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_date(ms) -> str:
    return datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d")


def make_rows(n, start_day=0, prefix="P"):
    """Deterministic synthetic rows: Jan-Mar (start_day=0) or April (91)."""
    base_ms = utc(2025, 1, 6)
    rows = []
    for i in range(n):
        ship_ms = base_ms + (start_day + (i % 57)) * DAY + 10 * 3600_000
        service = ["standard", "express", "overnight"][i % 3]
        lead = {"standard": 5, "express": 2, "overnight": 1}[service]
        carrier = CARRIER_IDS[i % 3] if (i % 13) else "CXx"  # malformed -> UNKNOWN
        promised_ms = ship_ms + lead * DAY
        late = (i % 11) < 4  # ~36% late
        missed = i % 29 == 0  # missing promised_date -> unlabelable
        cancelled = i % 23 == 0
        if late:
            actual_ms = promised_ms + 2 * DAY
        else:
            actual_ms = promised_ms - int(i % 3) * DAY
        row = {
            "parcel_id": f"{prefix}{i:05d}",
            "created_at": iso(ship_ms - 5 * 3600_000),
            "ship_date": iso(ship_ms),
            "promised_date": "" if missed else iso(promised_ms),
            "carrier_id": carrier,
            "warehouse_id": ["W1", "W2"][i % 2],
            "origin_region": "North" if i % 2 else "south",  # case variants
            "dest_region": "South" if late else "West",
            "service_level": service.upper() if (i % 5 == 0) else service,
            "parcel_type": "small" if i % 2 else "doc",
            "items_count": "2",
            "weight_value": "" if (i % 37 == 0) else f"{1.5 + (i % 10):.1f}",
            "weight_unit": "g" if (i % 17 == 0) else "kg",  # grams-rescue rows
            "distance_km": str(200 + (i % 40) * 30),
            "actual_delivery_date": iso(actual_ms),
            "delivery_status": "cancelled" if cancelled else ("late" if late else "on_time"),
            "carrier_delay_note": "",
        }
        rows.append(row)
    return rows


def write_mini_root(root: Path) -> Path:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "data" / "carriers.csv").write_text(
        to_csv(
            ["carrier_id", "carrier_name", "home_region"],
            [{"carrier_id": c, "carrier_name": c + " Ltd", "home_region": "North"} for c in CARRIER_IDS],
        ),
        encoding="utf-8",
    )
    cap_rows = []
    week = utc(2025, 1, 6)
    end = utc(2025, 5, 5)
    while week <= end:
        for i, wh in enumerate(("W1", "W2")):
            cap_rows.append({"warehouse_id": wh, "week_start": iso_date(week), "capacity_parcels": str(400 + i * 100)})
        week += 7 * DAY
    (root / "data" / "warehouse_capacity.csv").write_text(
        to_csv(["warehouse_id", "week_start", "capacity_parcels"], cap_rows), encoding="utf-8"
    )
    rows = make_rows(140, start_day=0, prefix="P") + make_rows(50, start_day=91, prefix="V")
    (root / "data" / "shipments_day1.csv").write_text(to_csv(SHIP_HEADER, rows), encoding="utf-8")
    return root


def build_mini_root(tmp_path, train_models=True) -> Path:
    root = write_mini_root(tmp_path / "mini")
    if train_models:
        train(str(root))
    return root


def real_root() -> Path:
    """The real polyglot root (committed data + trained models/)."""
    this = Path(__file__).resolve().parent.parent
    assert (this / "models" / "logreg.json").exists(), "run `uv run python -m parcelrank.train` first"
    return this
