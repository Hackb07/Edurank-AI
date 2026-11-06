"""Flexible date parsing, mirroring the Node pipeline's lib/dates.js 1:1.

Accepts:
  - epoch seconds (10 digits) or epoch milliseconds (13 digits)
  - MM/DD/YYYY with optional time (US slash)
  - DD-Mon-YYYY (04-Mar-2025)
  - YYYY/MM/DD
  - ISO dates/datetimes; naive datetimes are read as UTC (documented
    assumption: results must be identical on any machine).

parse_flexible returns (ms, ymd) or None for blank/invalid.
"""
import calendar
import re
from datetime import datetime, timedelta, timezone

DAY_MS = 86_400_000
_MONTHS = {
    "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
    "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11,
}

_US_SLASH = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$")
_DAY_MON = re.compile(r"^(\d{1,2})-([A-Za-z]{3})-(\d{4})$")
_YYYY_SLASH = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$")
_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_NAIVE_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$")


def ymd_of(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def _ymd(y: int, mo: int, d: int):
    try:
        return datetime(y, mo, d, tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_flexible(raw):
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None

    if re.fullmatch(r"\d{10}", s):
        ms = int(s) * 1000
        return (ms, ymd_of(ms))
    if re.fullmatch(r"\d{13}", s):
        ms = int(s)
        return (ms, ymd_of(ms))

    m = _US_SLASH.match(s)
    if m:
        mo = int(m.group(1)) - 1
        d = int(m.group(2))
        y = int(m.group(3))
        hh = int(m.group(4) or 0)
        mi = int(m.group(5) or 0)
        ss = int(m.group(6) or 0)
        base = _ymd(y, mo + 1, d)
        if base is None:
            return None
        try:
            dt = base + timedelta(hours=hh, minutes=mi, seconds=ss)
        except OverflowError:
            return None
        return (int(dt.timestamp() * 1000), ymd_of(int(dt.timestamp() * 1000)))

    m = _DAY_MON.match(s)
    if m:
        mo = _MONTHS.get(m.group(2).lower())
        d = int(m.group(1))
        y = int(m.group(3))
        if mo is None:
            return None
        dt = _ymd(y, mo + 1, d)
        if dt is None:
            return None
        ms = int(dt.timestamp() * 1000)
        return (ms, ymd_of(ms))

    m = _YYYY_SLASH.match(s)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        d = int(m.group(3))
        dt = _ymd(y, mo, d)
        if dt is None:
            return None
        ms = int(dt.timestamp() * 1000)
        return (ms, ymd_of(ms))

    if _ISO_DAY.fullmatch(s) or s[:10].count("-") == 2 and _ISO_DAY.match(s[:10]):
        # Plain date (possibly followed by a time / zone).
        y, mo, d = (int(x) for x in s[:10].split("-"))
        day = _ymd(y, mo, d)
        if day is None:
            return None
        if _ISO_DAY.fullmatch(s):
            ms = int(day.timestamp() * 1000)
            return (ms, ymd_of(ms))
        iso = s
        if _ISO_NAIVE_TIME.fullmatch(s):
            iso = s + "Z"
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            return None
        ms = int(dt.timestamp() * 1000)
        return (ms, ymd_of(ms))

    return None


def diff_days(a_ymd: str, b_ymd: str) -> int:
    pa = [int(x) for x in a_ymd.split("-")]
    pb = [int(x) for x in b_ymd.split("-")]
    da = datetime(pa[0], pa[1], pa[2], tzinfo=timezone.utc)
    db = datetime(pb[0], pb[1], pb[2], tzinfo=timezone.utc)
    return round((da - db).total_seconds() / 86400)


def week_start_ymd(ms: int) -> str:
    """Monday (UTC) of the week containing the given ms."""
    d = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    back = d.weekday()  # Monday=0 .. Sunday=6 -> days since Monday
    return ymd_of(ms - back * DAY_MS)


def weekday_of_utc(ms: int) -> int:
    """0=Sun..6=Sat, matching JavaScript getUTCDay()."""
    d = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return (d.weekday() + 1) % 7


def max_ymd_of_dates(m):
    """None-safe max over ymd strings."""
    vals = [x for x in m if x]
    return max(vals) if vals else None
