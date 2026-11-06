"""Date parser tests: every format the flexible parser must accept, and the
ones it must reject."""
from parcelrank.dates import parse_flexible, diff_days, week_start_ymd, ymd_of
import datetime


def utc(y, m, d, hh=0, mi=0, ss=0):
    return int(datetime.datetime(y, m, d, hh, mi, ss, tzinfo=datetime.timezone.utc).timestamp() * 1000)


def test_epoch_seconds_and_millis():
    assert parse_flexible("1735776000")[1] == "2025-01-02"  # 10 digits -> seconds
    ms = utc(2025, 1, 2, 0, 0)
    assert parse_flexible(str(ms))[1] == "2025-01-02"  # 13 digits -> millis


def test_us_slash_with_and_without_time():
    assert parse_flexible("1/2/2025") == (utc(2025, 1, 2), "2025-01-02")
    assert parse_flexible("12/31/2024 6:30") == (utc(2024, 12, 31, 6, 30), "2024-12-31")


def test_day_mon_year():
    assert parse_flexible("04-Mar-2025") == (utc(2025, 3, 4), "2025-03-04")


def test_yyyy_slash():
    assert parse_flexible("2025/3/4") == (utc(2025, 3, 4), "2025-03-04")


def test_iso_naive_is_read_as_utc():
    naive = parse_flexible("2025-03-04T05:06:07")
    assert naive == (utc(2025, 3, 4, 5, 6, 7), "2025-03-04")


def test_iso_zoned():
    # 2025-03-04T10:30:00+05:30 == 05:00 UTC
    assert parse_flexible("2025-03-04T10:30:00+05:30")[1] == "2025-03-04"


def test_overflow_date_rejected():
    # Feb 30 does not exist: the parser must NOT roll it over to March.
    assert parse_flexible("2025-02-30") is None
    assert parse_flexible("2/30/2025") is None


def test_blank_and_junk_are_none():
    assert parse_flexible("") is None
    assert parse_flexible(None) is None
    assert parse_flexible("   ") is None
    assert parse_flexible("not a date") is None


def test_diff_days():
    assert diff_days("2025-03-10", "2025-03-04") == 6
    assert diff_days("2025-03-04", "2025-03-04") == 0


def test_week_start_is_monday():
    # 2025-01-05 is a Sunday; its week starts Monday 2024-12-30.
    assert week_start_ymd(utc(2025, 1, 5, 13, 0)) == "2024-12-30"
    assert week_start_ymd(utc(2025, 1, 6, 1, 0)) == "2025-01-06"


def test_ymd_of_roundtrip():
    assert ymd_of(utc(2025, 6, 15, 23, 59)) == "2025-06-15"
