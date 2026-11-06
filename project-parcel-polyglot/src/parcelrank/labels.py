"""Late-label definition (single source of truth; matches src/labels.js).

late = actual_delivery_date (UTC calendar date) is AFTER promised_date.
cancelled/lost, missing promised, missing actual -> unlabelable (None).
actual == promised -> on time (promise met, no grace).
delivery_status is deliberately IGNORED except to exclude cancelled/lost.
"""
from __future__ import annotations


def label_late(clean: dict):
    status = (clean.get("delivery_status") or "").strip().lower()
    if status in ("cancelled", "lost"):
        return (None, f"status={status}")
    if not clean.get("promised_ymd"):
        return (None, "missing promised_date")
    if not clean.get("actual_ymd"):
        return (None, "missing actual_delivery_date")
    late = 1 if clean["actual_ymd"] > clean["promised_ymd"] else 0
    return (late, None)


def unlabelable_reason(clean: dict) -> str | None:
    if clean.get("label") is None:
        return clean.get("unlabelable_reason")
    return None
