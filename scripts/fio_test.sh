#!/usr/bin/env bash
# Optional: real Linux I/O schedulers x access pattern, measured with fio inside the VM.
# Usage (from repo root):  bash scripts/fio_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/vm/fio; mkdir -p "$OUT"
DEV=$(lsblk -ndo NAME,TYPE | awk '$2=="disk"{print $1; exit}')
SCHED_FILE=/sys/block/$DEV/queue/scheduler
sudo -v
sudo modprobe bfq 2>/dev/null || true
echo "device: $DEV" | tee "$OUT/device.txt"
echo "available: $(cat $SCHED_FILE)" | tee -a "$OUT/device.txt"
ORIG=$(grep -o '\[[^]]*\]' $SCHED_FILE | tr -d '[]')
TESTFILE=$HOME/fio_testfile
for sched in mq-deadline bfq none; do
  if ! grep -qw "$sched" $SCHED_FILE; then echo "skip $sched (not available)" | tee -a "$OUT/device.txt"; continue; fi
  echo "$sched" | sudo tee $SCHED_FILE >/dev/null
  for pattern in read randread; do
    for rep in 1 2 3; do
      echo ">> $sched / $pattern / rep $rep"
      sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
      fio --name=$pattern --filename=$TESTFILE --size=512M --bs=4k \
          --rw=$pattern --direct=1 --ioengine=libaio --iodepth=16 \
          --runtime=20 --time_based --group_reporting \
          --output-format=json --output=$OUT/${sched}_${pattern}_r${rep}.json
    done
  done
done
echo "$ORIG" | sudo tee $SCHED_FILE >/dev/null
rm -f $TESTFILE
echo "DONE"
