"""Window features the selector sees (all computable by the OS from its request queue).

  std_tracks    : spread of requested cylinders (normalised)
  range_tracks  : max - min cylinder (normalised)
  mean_jump     : mean |difference| between consecutive arrivals (normalised)
  monotone_frac : fraction of consecutive moves in the dominant direction (sequentiality)
  burstiness    : coefficient of variation of gaps between sorted cylinders
                  (high = requests bunched into clusters)
  head_offset   : (head - mean cylinder) / MAX_CYL  (where the head sits vs. the load)
  dir_to_mass   : +1 if the current head direction points toward the load, else -1
"""
from __future__ import annotations
import numpy as np
from schedulers import MAX_CYL

FEATURES = ["std_tracks", "range_tracks", "mean_jump", "monotone_frac",
            "burstiness", "head_offset", "dir_to_mass"]


def window_features(reqs, head: int, direction: int) -> list[float]:
    r = np.asarray(reqs, dtype=float)
    diffs = np.diff(r)
    nz = diffs[diffs != 0]
    mono = max((nz > 0).mean(), (nz < 0).mean()) if len(nz) else 1.0
    gaps = np.diff(np.sort(r))
    burst = gaps.std() / gaps.mean() if gaps.mean() > 0 else 0.0
    mean = r.mean()
    dir_to_mass = 1.0 if np.sign(mean - head) in (0, np.sign(direction)) else -1.0
    return [r.std() / MAX_CYL, (r.max() - r.min()) / MAX_CYL,
            np.abs(diffs).mean() / MAX_CYL if len(diffs) else 0.0,
            mono, burst, (head - mean) / MAX_CYL, dir_to_mass]
