#!/usr/bin/env bash
# Part B: run the fixed workload REPS times under the CURRENT VM configuration.
# Usage (from repo root):  bash scripts/run_vm_experiment.sh C1      (C1 ... C5)
# Change RAM / vCPUs in VMware (VM powered OFF) before each configuration.
set -uo pipefail
CFG=${1:?usage: bash scripts/run_vm_experiment.sh <C1|C2|C3|C4|C5>}
REPS=${REPS:-3}
TIMEOUT=${TIMEOUT:-900}                     # seconds per run; a run that exceeds it is logged as TIMEOUT
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=results/vm/$CFG
mkdir -p "$OUT"
[ -x "$PY" ] || { echo "Run 'bash scripts/setup.sh' first."; exit 1; }
[ -x /usr/bin/time ] || { echo "Missing /usr/bin/time: sudo apt install -y time"; exit 1; }
sudo -v

{ echo "config: $CFG"; date -Is
  echo "== nproc"; nproc
  echo "== free -m"; free -m
  echo "== swap"; swapon --show
  echo "== lscpu"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|Hypervisor'
  echo "== kernel"; uname -r
  echo "== python"; $PY --version
} > "$OUT/sysinfo.txt" 2>&1

echo "== $CFG: $(nproc) vCPU, $(free -m | awk '/Mem:/{print $2}') MiB RAM, swap: $(swapon --show --noheadings | awk '{print $3}' | paste -sd+ -)"
for r in $(seq 1 "$REPS"); do
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true
  sleep 3
  grep -E '^(pswpin|pswpout|pgmajfault|pgfault) ' /proc/vmstat > "$OUT/procvmstat_before_r$r.txt"
  vmstat -n 1 > "$OUT/vmstat_r$r.log" &
  VPID=$!
  /usr/bin/time -v -o "$OUT/time_r$r.txt" timeout "$TIMEOUT" "$PY" src/vm_workload.py \
      > "$OUT/workload_r$r.json" 2> "$OUT/stderr_r$r.txt"
  EC=$?
  kill "$VPID" 2>/dev/null; wait "$VPID" 2>/dev/null
  grep -E '^(pswpin|pswpout|pgmajfault|pgfault) ' /proc/vmstat > "$OUT/procvmstat_after_r$r.txt"
  echo "$EC" > "$OUT/exitcode_r$r.txt"
  case $EC in 0) ST=OK;; 124) ST=TIMEOUT;; 137) ST="KILLED (likely out-of-memory)";; *) ST="ERROR $EC";; esac
  [ "$EC" = 137 ] && sudo dmesg 2>/dev/null | grep -i -E 'out of memory|killed process' | tail -3 > "$OUT/oom_r$r.txt"
  WALL=$(grep 'Elapsed (wall clock)' "$OUT/time_r$r.txt" | awk '{print $NF}')
  RSS=$(grep 'Maximum resident' "$OUT/time_r$r.txt" | awk '{print int($NF/1024)}')
  MAJ=$(grep 'Major (requiring I/O)' "$OUT/time_r$r.txt" | awk '{print $NF}')
  echo "   run $r/$REPS: $ST | wall $WALL | max RSS ${RSS} MiB | major faults $MAJ"
  sleep 5
done
echo "== $CFG finished. Results in $OUT  (take your screenshot now)"
