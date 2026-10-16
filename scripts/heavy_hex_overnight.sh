#!/usr/bin/env bash
# Launch the unattended heavy-hex PPO run detached from this shell.
#
# Runs python -m autoqec_sim.experiments.heavy_hex_overnight drive under
# nohup with a global wall cap (default 10 h) and a thread cap (default 4),
# writes everything under results/heavy_hex_overnight/, and prints the pid.
# Rerunning the script resumes: every cell whose JSON exists is skipped.
#
#   scripts/heavy_hex_overnight.sh            # 10 h, 4 threads
#   HOURS=6 THREADS=2 scripts/heavy_hex_overnight.sh
#   scripts/heavy_hex_overnight.sh status     # heartbeat and last log lines
#   scripts/heavy_hex_overnight.sh stop       # SIGTERM to the driver (it kills its child)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/results/heavy_hex_overnight}"
HOURS="${HOURS:-10}"
THREADS="${THREADS:-4}"
cd "$ROOT"
mkdir -p "$OUT/logs"

case "${1:-start}" in
  status)
    if [ -f "$OUT/heartbeat.json" ]; then cat "$OUT/heartbeat.json"; echo; fi
    if [ -f "$OUT/driver.pid" ]; then
      pid=$(cat "$OUT/driver.pid")
      if kill -0 "$pid" 2>/dev/null; then echo "driver pid $pid is running"; else echo "driver pid $pid is not running (stale pid file)"; fi
    else
      echo "no driver.pid (not running)"
    fi
    [ -f "$OUT/log.txt" ] && tail -n 5 "$OUT/log.txt"
    exit 0
    ;;
  stop)
    if [ -f "$OUT/driver.pid" ]; then kill -TERM "$(cat "$OUT/driver.pid")" && echo "sent SIGTERM"; else echo "no driver.pid"; fi
    exit 0
    ;;
  start) ;;
  *) echo "usage: $0 [start|status|stop]" >&2; exit 2 ;;
esac

if [ -f "$OUT/driver.pid" ] && kill -0 "$(cat "$OUT/driver.pid")" 2>/dev/null; then
  echo "driver already running with pid $(cat "$OUT/driver.pid")" >&2
  exit 1
fi

# BLAS and torch thread caps for the children (autoqec_sim.agents.ppo.set_threads
# uses setdefault, so these take precedence over its own cap).
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1

nohup uv run --extra rl python -m autoqec_sim.experiments.heavy_hex_overnight drive \
  --hours "$HOURS" --threads "$THREADS" --out-dir "$OUT" \
  >> "$OUT/driver.out" 2>&1 < /dev/null &
DRIVER=$!
echo "$DRIVER" > "$OUT/launcher.pid"

# Backstop: if the driver is still alive 10 min past the cap, SIGTERM it
# (its handler terminates the running child). Detached as well. The python
# pid is read from driver.pid at fire time (uv run may or may not exec).
nohup bash -c "sleep $(( HOURS * 3600 + 600 )); for p in \$(cat '$OUT/driver.pid' 2>/dev/null) $DRIVER; do kill -0 \$p 2>/dev/null && kill -TERM \$p; done" >/dev/null 2>&1 < /dev/null &
echo "$!" > "$OUT/watchdog.pid"

echo "launched driver (uv wrapper pid $DRIVER); python pid appears in $OUT/driver.pid; hours=$HOURS threads=$THREADS"
echo "heartbeat: $OUT/heartbeat.json   log: $OUT/log.txt   resources: $OUT/resources.jsonl"
