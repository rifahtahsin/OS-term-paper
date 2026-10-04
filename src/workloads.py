"""Synthetic disk-request workloads. Each window = WINDOW requests (arrival order).

  sequential : a run of nearby, mostly monotone cylinders (file scan / log append)
  random     : uniform over the whole disk (database index lookups)
  bursty     : runs of requests clustered around 2-3 hot spots (hot files / metadata)
"""
from __future__ import annotations
import numpy as np
from schedulers import MAX_CYL

WINDOW = 20
TYPES = ["sequential", "random", "bursty"]


def gen_window(kind: str, rng: np.random.Generator, n: int = WINDOW) -> list[int]:
    if kind == "sequential":
        d = rng.choice([-1, 1])
        start = int(rng.integers(0, MAX_CYL + 1))
        steps = rng.integers(1, 7, size=n) * d
        reqs = np.clip(start + np.cumsum(steps), 0, MAX_CYL)
        noise = rng.random(n) < 0.10                        # 10 % stray requests
        reqs[noise] = rng.integers(0, MAX_CYL + 1, size=noise.sum())
    elif kind == "random":
        reqs = rng.integers(0, MAX_CYL + 1, size=n)
    elif kind == "bursty":
        hot = rng.integers(10, MAX_CYL - 9, size=int(rng.integers(2, 4)))
        reqs = []
        while len(reqs) < n:
            c = rng.choice(hot)
            run = int(rng.integers(3, 9))
            reqs += list(np.clip(np.round(rng.normal(c, 4, size=run)), 0, MAX_CYL))
        reqs = np.array(reqs[:n])
    else:
        raise ValueError(kind)
    return [int(r) for r in reqs]


def gen_dataset(n_per_type: int, seed: int):
    """Independent windows with a random head position/direction (training/testing)."""
    rng = np.random.default_rng(seed)
    rows = []
    for kind in TYPES:
        for _ in range(n_per_type):
            rows.append((kind, gen_window(kind, rng), int(rng.integers(0, MAX_CYL + 1)),
                         int(rng.choice([-1, 1]))))
    return rows


def gen_shift_timeline(seed: int, phases=(("sequential", 50), ("random", 50), ("bursty", 50))):
    """Timeline of windows whose workload type changes at fixed points."""
    rng = np.random.default_rng(seed)
    return [(kind, gen_window(kind, rng)) for kind, k in phases for _ in range(k)]
