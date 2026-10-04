"""Ground truth: run all four schedulers on a window and find the best."""
from schedulers import SCHEDULERS, NAMES

# Tie-break: among schedulers with EQUAL seek, prefer the most starvation-resistant
# (FCFS never starves; C-SCAN gives the most uniform wait; SCAN bounded; SSTF can starve).
TIE_PREFERENCE = ["FCFS", "CSCAN", "SCAN", "SSTF"]


def evaluate_window(reqs, head, direction):
    """Return dict name -> (seek, final_head, final_dir), best label, set of optimal names."""
    out = {n: SCHEDULERS[n](reqs, head, direction)[1:] for n in NAMES}
    best_seek = min(v[0] for v in out.values())
    optimal = {n for n, v in out.items() if v[0] == best_seek}
    label = next(n for n in TIE_PREFERENCE if n in optimal)
    return out, label, optimal, best_seek
