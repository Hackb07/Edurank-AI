"""Deterministic seeded RNG (LCG) — exact port of the Node pipeline's
src/lib/rng.js. Everything random in this repo flows through here so data,
training, and evaluation are exactly reproducible from a seed.
"""
import math


def make_rng(seed: int):
    s = (seed % 4294967296) or 1
    state = {"s": s}
    spare = {"v": None}

    def next():
        state["s"] = (state["s"] * 1664525 + 1013904223) % 4294967296
        return state["s"] / 4294967296.0

    def int_in(lo, hi):
        return lo + math.floor(next() * (hi - lo + 1))

    def pick(arr):
        return arr[math.floor(next() * len(arr))]

    def boolean(p):
        return next() < p

    def normal(mean=0.0, sd=1.0):
        if spare["v"] is not None:
            v = spare["v"]
            spare["v"] = None
            return mean + sd * v
        u = v = 0.0
        while u == 0:
            u = next()
        while v == 0:
            v = next()
        mag = math.sqrt(-2 * math.log(u))
        spare["v"] = mag * math.sin(2 * math.pi * v)
        return mean + sd * mag * math.cos(2 * math.pi * v)

    def shuffle(arr):
        a = list(arr)
        for i in range(len(a) - 1, 0, -1):
            j = math.floor(next() * (i + 1))
            a[i], a[j] = a[j], a[i]
        return a

    return {"next": next, "int": int_in, "pick": pick, "boolean": boolean, "normal": normal, "shuffle": shuffle}
