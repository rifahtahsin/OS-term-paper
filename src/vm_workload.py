"""Part B: the SAME fixed workload, run unchanged under every VM configuration.

It is the Track-2 pipeline at a larger scale, split into two timed phases:

  Phase M (memory-bound, single process)
     Builds a large in-memory request trace (default 1024 MiB of float64 cylinder
     numbers), then makes 3 sequential passes over it (FCFS seek total + per-window
     spread) and 1 random-access pass (gathers 200,000 random positions).
     Its working set is fixed, so its speed depends on how much RAM the VM has:
     if the working set does not fit, pages are swapped and major faults appear.

  Phase C (CPU-bound, parallel)
     Runs all four schedulers on 240,000 request windows and labels the best one,
     split across one worker process per vCPU. Total work is fixed, so its speed
     depends on how many vCPUs the VM has (and on host-core contention).

The trace and windows are generated from fixed seeds; the printed checksum must be
identical in every run, which proves the workload never changed.
Output: one JSON line on stdout.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from workloads import gen_window, TYPES
from labels import evaluate_window

CHUNK = 8_000_000                # elements per block (61 MiB of float64; multiple of WINDOW)
WINDOW = 20


def phase_memory(size_mib: int, passes: int, random_reads: int, seed: int):
    n = size_mib * 1024 * 1024 // 8
    n -= n % WINDOW
    rng = np.random.default_rng(seed)
    trace = np.empty(n, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(0, n, CHUNK):                       # fill (touches every page once)
        m = min(CHUNK, n - i)
        trace[i:i + m] = rng.integers(0, 200, size=m)
    t_fill = time.perf_counter() - t0
    seek_total, spread = 0.0, 0.0
    for _ in range(passes):                            # sequential passes
        seek_total, spread, prev = 0.0, 0.0, None
        for i in range(0, n, CHUNK):
            blk = trace[i:i + CHUNK]
            seek_total += np.abs(np.diff(blk)).sum()
            if prev is not None:
                seek_total += abs(blk[0] - prev)
            prev = blk[-1]
            spread += blk.reshape(-1, WINDOW).std(axis=1).sum()
    t_seq = time.perf_counter() - t0 - t_fill
    idx_rng = np.random.default_rng(seed + 1)          # random-access pass
    gathered = 0.0
    for _ in range(random_reads // 100_000):
        gathered += trace[idx_rng.integers(0, n, size=100_000)].sum()
    t_rand = time.perf_counter() - t0 - t_fill - t_seq
    checksum = float(seek_total + spread + gathered)
    del trace
    return {"fill_s": t_fill, "seq_passes_s": t_seq, "random_pass_s": t_rand,
            "total_s": t_fill + t_seq + t_rand, "elements": n}, checksum


def _cpu_chunk(args):
    seed, n_windows = args
    rng = np.random.default_rng(seed)
    total, wins = 0, {}
    for k in range(n_windows):
        reqs = gen_window(TYPES[k % 3], rng, WINDOW)
        _, label, _, best = evaluate_window(reqs, int(rng.integers(0, 200)), 1)
        total += best
        wins[label] = wins.get(label, 0) + 1
    return total, wins


def phase_cpu(n_windows: int, workers: int, seed: int):
    n_chunks = 48                                    # fixed split -> same work for any worker count
    per = n_windows // n_chunks
    jobs = [(seed + c, per) for c in range(n_chunks)]
    t0 = time.perf_counter()
    with Pool(workers) as pool:
        results = pool.map(_cpu_chunk, jobs, chunksize=1)
    t = time.perf_counter() - t0
    best_total = sum(r[0] for r in results)
    return {"total_s": t, "workers": workers, "windows": per * n_chunks}, float(best_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-mib", type=int, default=1024)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--random-reads", type=int, default=200_000)
    ap.add_argument("--cpu-windows", type=int, default=240_000)
    ap.add_argument("--workers", type=int, default=0, help="0 = one per vCPU")
    ap.add_argument("--seed", type=int, default=307)
    a = ap.parse_args()
    workers = a.workers or os.cpu_count() or 1
    t0 = time.perf_counter()
    mem, c1 = phase_memory(a.size_mib, a.passes, a.random_reads, a.seed)
    cpu, c2 = phase_cpu(a.cpu_windows, workers, a.seed)
    print(json.dumps({"params": vars(a), "vcpus_seen": os.cpu_count(),
                      "memory_phase": mem, "cpu_phase": cpu,
                      "wall_s": time.perf_counter() - t0,
                      "checksum": round(c1 + c2, 3)}))


if __name__ == "__main__":
    main()
