"""Shared evaluation math — port of src/metrics.js plus the shipped daily
flagging policy from src/predict.js (one function so reported numbers always
describe the deployed behavior).
"""
from __future__ import annotations

import math

COST_FN = 25  # $ per missed late parcel (credit + support + churn risk)
COST_FP = 4   # $ per false alarm (message + amortized reroute/upgrade)


def confusion_at(labels, scores, threshold):
    tp = fp = tn = fn = 0
    for y, s in zip(labels, scores):
        pred = 1 if s >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1:
            fp += 1
        elif y == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def prf(labels, scores, threshold):
    c = confusion_at(labels, scores, threshold)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
    return {**c, "precision": precision, "recall": recall, "f1": f1}


def roc_auc(labels, scores):
    """Rank-based AUC with average ranks for ties."""
    n = len(labels)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = 0.0
    n_pos = 0
    for k in range(n):
        if labels[k] == 1:
            sum_pos += ranks[k]
            n_pos += 1
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc(labels, scores):
    """Average precision (step-interpolated area under the PR curve)."""
    n = len(labels)
    order = sorted(range(n), key=lambda i: -scores[i])
    tp = fp = 0
    ap = 0.0
    n_pos = sum(1 for y in labels if y == 1)
    if n_pos == 0:
        return 0.0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
            ap += tp / (tp + fp)
        else:
            fp += 1
    return ap / n_pos


def expected_cost(labels, scores, threshold, cost_fn=COST_FN, cost_fp=COST_FP):
    c = confusion_at(labels, scores, threshold)
    return {"total": c["fn"] * cost_fn + c["fp"] * cost_fp, "fp": c["fp"], "fn": c["fn"]}


def tune_threshold(labels, scores, cost_fn=COST_FN, cost_fp=COST_FP):
    """Cost-theory optimum on a 0.01-grid: t* = FP/(FN+FP) when costs are $1/$25... here swept."""
    best = {"t": 0.5, "total": math.inf}
    for k in range(1, 100):
        t = k / 100.0
        ec = expected_cost(labels, scores, t, cost_fn, cost_fp)
        if ec["total"] < best["total"]:
            best = {"t": t, "total": ec["total"]}
    return best


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + (z * z) / n
    center = p + (z * z) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def bin_index(edges, v):
    """Bin index for 10 bins defined by 11 decile edges (matches metrics.js so
    train-side binning and monitor-side binning can never disagree)."""
    for k in range(1, 10):
        if v < edges[k]:
            return k - 1
    return 9


def psi(expected, actual, eps=1e-4):
    """Population Stability Index between two proportion vectors. Empty bins
    are floored so a new value SPIKES the index instead of producing NaN."""
    total = 0.0
    for e, a in zip(expected, actual):
        e = max(eps, e)
        a = max(eps, a)
        total += (a - e) * math.log(a / e)
    return total


def default_prioritize():
    """Ops priority order (Day-3 memo): tight promises get the scarce
    slots first; standard fills leftovers."""
    return ["same_day", "overnight", "express"]


def apply_daily_policy(rows, scores, *, top_k, floor, threshold, tiers):
    """Per ship-date: rank candidates (score >= floor) by (priority tier,
    score) and flag the top K. top_k == 0 disables the cap (pure threshold
    mode: everything >= threshold is flagged)."""
    active = tiers if tiers else None

    def tier_of(r):
        if not active:
            return 0
        try:
            return active.index(r["service_level"])
        except ValueError:
            return len(active)

    by_day = {}
    for i, r in enumerate(rows):
        by_day.setdefault(r["ship_ymd"], []).append(i)

    flags = [0] * len(rows)
    for idxs in by_day.values():
        cands = [i for i in idxs if scores[i] >= floor]
        cands.sort(key=lambda i: (tier_of(rows[i]), -scores[i]))
        if top_k == 0:
            chosen = [i for i in cands if scores[i] >= threshold]
        else:
            chosen = cands[:top_k]
        for i in chosen:
            flags[i] = 1
    return flags
