# OS-term-paper — Learned Disk-Scheduler Selector (CSE-307, Track 2) + VM Resource Experiment

**Author:** Rifah Tahsin · ID 202414071 · Level 3, Section B · CSE-307 Operating Systems (Part B term paper)

This repository has two parts:

- **Part A — Track 2 (the brief):** FCFS, SSTF, SCAN and C-SCAN disk schedulers, a small learned
  selector that predicts which scheduler will give the minimum seek (with a confidence score),
  a workload shift test, a calibration check, and the bonus explanation track.
- **Part B — the announcement:** the same pipeline run as **one fixed workload** inside an Ubuntu VM
  (VMware Workstation Pro) under five RAM/vCPU configurations, plus an optional `fio` test of
  Linux's real I/O schedulers.

## Repository layout


    src/
      schedulers.py      FCFS, SSTF, SCAN, C-SCAN (conventions documented at the top)
      workloads.py       sequential / random / bursty request windows, shift timelines
      features.py        7 window features the selector uses
      labels.py          runs all 4 schedulers on a window -> best scheduler(s)
      selector.py        learned selector (decision tree / logistic regression) + calibration metrics
      explain.py         BONUS: rule-based explanation + self-rated confidence
      run_partA.py       Part A experiment -> results/partA/
      vm_workload.py     Part B fixed workload (memory phase + parallel CPU phase)
    scripts/
      setup.sh           one-time setup inside the VM (apt packages + Python venv + tests)
      run_vm_experiment.sh   runs the fixed workload 3x under the current VM config, logs everything
      fio_test.sh        optional: mq-deadline / bfq / none x sequential / random reads
      analyze_vm.py      raw VM + fio logs -> CSV tables + figures
    tests/test_schedulers.py   unit tests (textbook example: FCFS = 640, SSTF = 236)
    results/partA/       Part A tables (CSV) and figures
    results/vm/          Part B raw logs, summaries, figures, screenshots/
    report/figures/      Part B figures used in the report (the PDF report is submitted separately)

## Setup (Ubuntu VM)


    cd ~/OS-term-paper
    bash scripts/setup.sh          # installs python3-venv, time, fio; creates .venv; runs the tests

A virtual environment is used because recent Ubuntu releases block system-wide `pip install`.

## How to run

**Part A** (about 1–2 minutes, no special hardware needed):

    .venv/bin/python src/run_partA.py

**Part B** — one configuration at a time. Order chosen so each step changes only one setting:

| Order | Config | RAM | vCPU | Change in VMware (VM **powered off**: VM → Settings) |
|---|---|---|---|---|
| 1 | C5 | 4 GB | 4 | none (current setup) |
| 2 | C4 | 4 GB | 2 | Processors → 2 |
| 3 | C3 | 4 GB | 1 | Processors → 1 |
| 4 | C2 | 2 GB | 1 | Memory → 2048 MB |
| 5 | C1 | 1.5 GB | 1 | Memory → 1536 MB (1024 MB left the desktop unusable) |

For each configuration:

    bash scripts/run_vm_experiment.sh C5     # then C4, C3, C2, C1
Shut down from inside Ubuntu (`sync && systemctl poweroff -i`) before changing settings — **not** Suspend.
Close the browser and other apps during runs. Take two screenshots per config
(`C5_settings.png`, `C5_run.png`, …) into `results/vm/screenshots/`.

Optional `fio` test (once, any config): `bash scripts/fio_test.sh`

Then build the tables and figures:

    .venv/bin/python scripts/analyze_vm.py

## Results summary

### Part A (reproducible; seeds fixed)

Test set (1,500 windows): mean seek per window in cylinders, optimal-pick rate, mean regret
(seek above the per-window best).

| Policy | Sequential | Random | Bursty | Optimal % | Regret |
|---|---|---|---|---|---|
| FCFS | 374.1 | 1318.4 | 245.1 | 4.0 | 473.7 |
| SSTF | 163.2 | 256.3 | 114.0 | 88.4 | 5.7 |
| SCAN | 227.8 | 277.8 | 181.2 | 34.4 | 56.8 |
| C-SCAN | 323.2 | 379.5 | 271.0 | 20.4 | 152.5 |
| **Selector-DT** | 163.9 | 252.4 | 114.0 | **88.7** | **4.6** |
| Selector-LR | 164.4 | 256.7 | 114.0 | 88.2 | 6.2 |
| Oracle | 160.9 | 241.6 | 113.9 | 100 | 0 |

Key findings:
- **Under the shift**, FCFS degrades the most (3.6× more seek going from sequential to random).
  SCAN and C-SCAN fall furthest behind the best choice on bursty input. SSTF is the most
  shift-robust classical policy.
- **The selector is well calibrated:** ECE = 0.029, mean confidence 0.93 on correct picks vs.
  0.76 on wrong ones. Its total-seek gain over always-SSTF is within noise (better in 5 of 10
  shift timelines), with a clear gain only in the random phase (regret −32%).
- **A selector trained only on pre-shift (sequential) data** has 10× higher regret in the bursty
  phase than the full model, showing drift in the learned component itself.
- **Bonus:** the rule-based explainer is under-confident (ECE = 0.124).

Figures: `results/partA/fig1_seek_by_type.png`, `fig2_shift_timeline.png`, `fig3_reliability.png`.

### Part B (Ubuntu 26.04.1 guest, VMware Workstation Pro, host 8 GB RAM / 4 cores)

Mean of 3 runs per configuration. All 15 runs completed with an identical workload checksum.
Swap figures are in thousands of 4 KiB pages.

| Config | RAM / vCPU | Wall (s) | Memory phase (s) | CPU phase (s) | Major faults | Swap in/out (k pages) | I/O wait |
|---|---|---|---|---|---|---|---|
| C1 | 1.5 GB / 1 | 106.1 ± 8.3 | 70.9 | 34.1 | 192,562 | 872 / 306 | 26.4% |
| C2 | 2 GB / 1 | 102.1 ± 6.5 | 64.6 | 36.3 | 151,810 | 748 / 297 | 24.3% |
| C3 | 4 GB / 1 | 47.8 ± 10.5 | 13.9 | 32.9 | 36 | 0 / 0 | 0.1% |
| C4 | 4 GB / 2 | 30.6 ± 8.5 | 12.7 | 17.0 | 36 | 0 / 0 | 0.2% |
| C5 | 4 GB / 4 | 19.8 ± 4.9 | 10.1 | 9.1 | 36 | 0 / 0 | 0.3% |

C1 was planned at 1 GB, but the Ubuntu desktop was unusable at that size, so 1.5 GB (1536 MB) was used.
C3 was re-run after a fresh boot, because an earlier C3 attempt had been run a second time without rebooting
(every configuration now follows the same procedure: boot, then 3 runs).

Key findings:
- **Thrashing.** Once guest RAM is below the ~1.16 GiB working set plus the OS footprint (C1, C2),
  the memory phase is about 5× slower on average, major faults exceed 150k per run, and ~3 GiB is swapped in.
  The CPU phase is unaffected.
- **vCPU scaling.** The CPU phase speeds up 1.93× with 2 vCPUs and 3.63× with 4 vCPUs.
  Involuntary context switches fall from 1,835 to 245.
- **First-run slowdown.** In C3, C4 and C5 the first run after boot had a 2.5–3.9× slower memory phase,
  converging to 6.1–6.6 s by the third run. The cause is unconfirmed (possibly the host backing guest memory
  only on first touch).
- **fio.** The guest I/O scheduler (`mq-deadline`, `bfq`, `none`) made little reliable difference,
  and random reads reached 73–78% of sequential IOPS.

Raw logs: `results/vm/C*/`. Tables: `vm_summary.csv`, `vm_runs.csv`, `fio_summary.csv`.
Figures: `results/vm/fig4_vm_configs.png`, `fig5_fio.png`. Screenshots: `results/vm/screenshots/`.

## Design notes (conventions)

- **Disk model:** cylinders 0–199; seek = cylinders travelled.
- **SCAN:** goes to the disk edge before reversing, but only if requests remain behind the head.
- **C-SCAN:** the 199 → 0 return jump is counted in the seek total.
- **Ties:** many windows have more than one optimal scheduler. A pick counts as correct whenever
  it achieves the minimum seek. Exact probability ties go to the more starvation-resistant
  scheduler (FCFS → C-SCAN → SCAN → SSTF).
- **Confidence:** the selector's estimated probability that its pick is optimal.

## AI assistance disclosure

Claude (Anthropic) was used to help write the code, scripts, analysis tooling and the report
draft. The experimental design choices (tracks, configurations, window size, tie-break rule) were
discussed and approved by me, the VM experiments were run by me, and I reviewed the results and
analysis.

## References

See the PDF report (Silberschatz et al. 2018; Teorey & Pinkerton 1972; Worthington et al. 1994;
Guo et al. 2017; Denning 1968; Barham et al. 2003; Waldspurger 2002; and others).
