"""Cost math + shipped daily-flagging policy tests."""
import pytest

from parcelrank.metrics import (
    COST_FN, COST_FP, confusion_at, prf, expected_cost, tune_threshold,
    wilson, roc_auc, pr_auc, apply_daily_policy,
)


def one_miss_one_fa():
    # 1 missed late (y=1, score< t) + 1 false alarm (y=0, score>=t)
    labels = [1, 0]
    scores = [0.3, 0.9]
    return labels, scores


def test_costs_25_miss_plus_4_false_alarm():
    labels, scores = one_miss_one_fa()
    ec = expected_cost(labels, scores, 0.5)
    assert ec["total"] == COST_FN + COST_FP  # $29, per the Day-1 pricing


def test_confusion_treats_equal_to_threshold_as_flagged():
    c = confusion_at([1], [0.14], 0.14)
    assert c["tp"] == 1  # >= comparison, matching the shipped CLI


def test_threshold_sweep_finds_cost_optimum():
    # 9 true late at 0.3, 1 false-alarm-prone negative at 0.2 -> no flag wins; but
    # put 1 late high and 3 negatives higher so flagging exactly that sliver is optimal.
    labels = [1, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.9, 0.8, 0.85, 0.75, 0.05, 0.04, 0.1, 0.2]
    best = tune_threshold(labels, scores)
    # t=0.5..0.7: fp=3, fn=4 -> 4*25+3*4=112; t=0.75..0.8: fp=2, fn=4 -> 108;
    # t=0.85..0.9: fp=1, fn=4 -> 104; t=0.9: fp=0, fn=4*25=100 -> best grid point 0.9
    assert best["t"] == 0.9
    assert best["total"] == 100


def test_prf_zero_division_safety():
    m = prf([0, 0], [0.1, 0.2], 0.5)
    assert m["precision"] == 0 and m["recall"] == 0 and m["f1"] == 0


def test_wilson_bounds():
    lo, hi = wilson(1, 1)
    assert 0 <= lo < hi <= 1
    lo2, hi2 = wilson(0, 0)
    assert (lo2, hi2) == (0, 1)


def test_roc_auc_perfect_and_inverted_with_ties():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4]) == 0.0
    assert roc_auc([1, 0], [0.5, 0.5]) == 0.5  # complete tie
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0


def _day(rows):
    return [{"ship_ymd": rows[0]} for _ in rows]


def test_policy_caps_at_top_k():
    rows = _day(["2025-06-01"] * 5)
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    flags = apply_daily_policy(rows, scores, top_k=3, floor=0.1, threshold=0.99, tiers=None)
    assert flags == [1, 1, 1, 0, 0]


def test_policy_tiers_beat_probability():
    # standard w/ 0.99 must lose scarce slots to same_day @ 0.5
    rows = [
        {"ship_ymd": "2025-06-01", "service_level": "standard"},
        {"ship_ymd": "2025-06-01", "service_level": "standard"},
        {"ship_ymd": "2025-06-01", "service_level": "same_day"},
    ]
    scores = [0.99, 0.98, 0.5]
    flags = apply_daily_policy(rows, scores, top_k=1, floor=0.1, threshold=0.99, tiers=["same_day", "overnight", "express"])
    assert flags == [0, 0, 1]


def test_policy_floor_excludes_low_scores_even_with_free_slots():
    rows = _day(["2025-06-01"] * 3)
    scores = [0.05, 0.02, 0.15]
    flags = apply_daily_policy(rows, scores, top_k=10, floor=0.10, threshold=0.99, tiers=None)
    assert flags == [0, 0, 1]


def test_policy_top_k_zero_is_pure_threshold():
    rows = _day(["2025-06-01"] * 3)
    scores = [0.9, 0.13, 0.14]
    flags = apply_daily_policy(rows, scores, top_k=0, floor=0.1, threshold=0.14, tiers=None)
    assert flags == [1, 0, 1]  # equal-to-threshold counts (>=)


def test_policy_caps_each_day_independently():
    rows = [{"ship_ymd": "2025-06-01"}] * 2 + [{"ship_ymd": "2025-06-02"}] * 2
    scores = [0.9, 0.8, 0.7, 0.6]
    flags = apply_daily_policy(rows, scores, top_k=1, floor=0.1, threshold=0.99, tiers=None)
    assert flags == [1, 0, 1, 0]
