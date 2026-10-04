"""BONUS: rule-based natural-language explanation + self-rated confidence.

No LLM API is used (the brief allows a rule-based template generator).
The explainer looks only at the window features, names the access pattern it
sees, explains the selector's pick in words, and rates its OWN confidence that
the pick is optimal. Base confidences were fixed a priori from textbook reasoning
(not tuned on test data):
  sequential / bursty + SSTF or SCAN -> 0.90 (nearest-first or one sweep is ideal)
  random + SSTF or SCAN              -> 0.70 (several near-equal orderings exist)
  FCFS / C-SCAN on sequential        -> 0.75 (a one-way run is already short)
  FCFS / C-SCAN on non-sequential    -> 0.40 (arrival order / wrap-jump wastes travel)
  ambiguous pattern                  -> 0.50
"""
from __future__ import annotations
import numpy as np

REASON = {
    "SSTF": "serving the nearest request first keeps each move short",
    "SCAN": "one sweep in the current direction covers the requests without back-tracking",
    "CSCAN": "a one-way sweep with a wrap-around gives uniform service",
    "FCFS": "the arrival order already runs in a single direction, so reordering gains nothing",
}


def detect_pattern(f: dict):
    mono, jump, burst = f["monotone_frac"], f["mean_jump"], f["burstiness"]
    if mono >= 0.70:
        return "sequential", min(1.0, (mono - 0.70) / 0.30 + 0.3)
    if burst >= 1.3 and jump < 0.20:
        return "bursty", min(1.0, (burst - 1.3) / 1.2 + 0.3)
    if jump >= 0.20 and burst < 1.3:
        return "random", min(1.0, (jump - 0.20) / 0.15 + 0.3)
    return "ambiguous", 0.0


def explain(f: dict, pick: str):
    pattern, clarity = detect_pattern(f)
    if pattern == "ambiguous":
        base = 0.50
    elif pick in ("SSTF", "SCAN"):
        base = 0.90 if pattern in ("sequential", "bursty") else 0.70
    elif pick in ("FCFS", "CSCAN") and pattern == "sequential":
        base = 0.75   # one-directional run: arrival order / one-way sweep is already short
    else:
        base = 0.40
    conf = 0.5 + (base - 0.5) * clarity if base >= 0.5 else base
    desc = {"sequential": f"requests move mostly one way ({f['monotone_frac']:.0%} of moves)",
            "bursty": f"requests are clustered (gap variation {f['burstiness']:.1f})",
            "random": f"requests are spread across the disk (mean jump {f['mean_jump']*199:.0f} cyl.)",
            "ambiguous": "the pattern is mixed"}[pattern]
    text = f"Chose {pick}: {desc}; {REASON[pick]}. Self-rated confidence {conf:.2f}."
    return text, float(np.clip(conf, 0, 1)), pattern
