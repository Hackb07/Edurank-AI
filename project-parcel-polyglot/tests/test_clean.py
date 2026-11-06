"""Cleaning tests — grams rescue, negatives, impossible rows, dedup,
normalization, and label behavior."""
import pytest

from parcelrank.clean import clean_row, load_shipments, dedup
from parcelrank.cli import REQUIRED_COLUMNS


def ctx():
    return {
        "capacity": {"W1|2025-01-06": 100.0, "W2|2025-01-06": 100.0},
        "known_carriers": {"C01", "C02"},
        "known_warehouses": {"W1", "W2"},
        "week_volume": {"W1|2025-01-06": 120, "W2|2025-01-06": 60},
    }


def base_row(**over):
    r = {
        "parcel_id": "P1",
        "created_at": "2025-01-06T05:00:00Z",
        "ship_date": "2025-01-06T10:00:00Z",
        "promised_date": "2025-01-09",
        "carrier_id": "C01",
        "warehouse_id": "W1",
        "origin_region": "North",
        "dest_region": "South",
        "service_level": "standard",
        "parcel_type": "doc",
        "items_count": "2",
        "weight_value": "10",
        "weight_unit": "kg",
        "distance_km": "300",
        "actual_delivery_date": "2025-01-08",
        "delivery_status": "on_time",
    }
    r.update(over)
    return r


def test_grams_to_kg():
    out = clean_row(base_row(weight_value="1200", weight_unit="g"), ctx())
    assert out["clean"]["weight_kg"] == pytest.approx(1.2)


def test_mislabeled_grams_above_500kg_rescued():
    # >500 "kg" is treated as grams (rescue heuristic), unit flag set.
    out = clean_row(base_row(weight_value="36000", weight_unit="kg"), ctx())
    assert out["clean"]["weight_kg"] == pytest.approx(36.0)
    assert out["clean"]["flags"]["weight_unit_suspect"] == 1


def test_negative_weight_nulled():
    out = clean_row(base_row(weight_value="-5"), ctx())
    assert out["clean"]["weight_kg"] is None
    assert out["clean"]["flags"]["weight_invalid"] == 1


def test_missing_weight_flag():
    out = clean_row(base_row(weight_value=""), ctx())
    assert out["clean"]["weight_kg"] is None
    assert out["clean"]["flags"]["weight_missing"] == 1


def test_distance_zero_and_negative_invalid():
    for bad in ("0", "-50"):
        out = clean_row(base_row(distance_km=bad), ctx())
        assert out["clean"]["distance_km"] is None
        assert out["clean"]["flags"]["distance_invalid"] == 1


def test_impossible_actual_before_ship_dropped():
    out = clean_row(base_row(actual_delivery_date="2025-01-05"), ctx())
    assert out == {"drop": "actual_delivery_date before ship_date (impossible)"}


def test_unparseable_ship_dropped():
    out = clean_row(base_row(ship_date="6-Mar-25"), ctx())  # 2-digit year not accepted
    assert out == {"drop": "unparseable ship_date"}


def test_new_wellformed_carrier_keeps_own_bucket():
    out = clean_row(base_row(carrier_id="C09"), ctx())
    assert out["clean"]["carrier_id"] == "C09"
    assert out["clean"]["flags"].get("carrier_unknown") is None
    assert out["clean"]["flags"].get("is_new_carrier") is None  # feature-time flag


def test_malformed_carrier_collapses_to_unknown():
    out = clean_row(base_row(carrier_id="CX9x"), ctx())
    assert out["clean"]["carrier_id"] == "UNKNOWN"
    assert out["clean"]["flags"]["carrier_unknown"] == 1


def test_unknown_warehouse_flagged_and_normalized():
    out = clean_row(base_row(warehouse_id="WX"), ctx())
    assert out["clean"]["warehouse_id"] == "UNKNOWN"
    assert out["clean"]["flags"]["warehouse_unknown"] == 1


def test_utilization_uses_capacity_join():
    out = clean_row(base_row(warehouse_id="W1"), ctx())  # 120/100 = 1.2 overload
    assert out["clean"]["utilization"] == pytest.approx(1.2)
    out2 = clean_row(base_row(warehouse_id="W2"), ctx())
    assert out2["clean"]["utilization"] == pytest.approx(0.6)


def test_items_below_one_become_one_with_flag():
    out = clean_row(base_row(items_count="0"), ctx())
    assert out["clean"]["items_count"] == 1
    assert out["clean"]["flags"]["items_invalid"] == 1


def test_dedup_last_in_file_wins_and_conflicts_counted():
    rows = [
        {"parcel_id": "A", "promised_date": "x", "delivery_status": "late", "carrier_id": "C01", "warehouse_id": "W1", "actual_delivery_date": "y"},
        {"parcel_id": "A", "promised_date": "x", "delivery_status": "on_time", "carrier_id": "C01", "warehouse_id": "W1", "actual_delivery_date": "y"},
        {"parcel_id": "B", "promised_date": "z", "delivery_status": "late", "carrier_id": "C01", "warehouse_id": "W1", "actual_delivery_date": "y"},
        {"parcel_id": "B", "promised_date": "z", "delivery_status": "late", "carrier_id": "C01", "warehouse_id": "W1", "actual_delivery_date": "y"},
    ]
    stats = {"blankId": 0, "dupIds": 0, "dupExtraRows": 0, "dupConflicts": 0}
    q = []
    kept = dedup(rows, stats, q)
    assert [r["parcel_id"] for r in kept] == ["A", "B"]
    assert kept[0]["delivery_status"] == "on_time"  # last wins
    assert stats["dupIds"] == 2 and stats["dupConflicts"] == 1


def test_labels_actual_after_promised_is_late_equal_is_ontime():
    out = clean_row(base_row(actual_delivery_date="2025-01-10"), ctx())  # promised 09
    assert out["clean"]["label"] == 1
    out2 = clean_row(base_row(actual_delivery_date="2025-01-09"), ctx())
    assert out2["clean"]["label"] == 0


def test_cancelled_lost_and_missing_dates_unlabelable():
    for over in (
        {"delivery_status": "cancelled"},
        {"delivery_status": "lost"},
        {"promised_date": ""},
        {"actual_delivery_date": ""},
    ):
        out = clean_row(base_row(**over), ctx())
        assert out["clean"]["label"] is None
        assert out["clean"]["unlabelable_reason"]
