#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec 9>arc_tacfold_queue.lock
flock -n 9 || exit 0

echo "$(date -Is) ARC-TAC campaign queued; waiting for existing training and monitor jobs."
while pgrep -f 'robomimic/scripts/train.py.*robocasa_mg3000_configs' >/dev/null \
   || pgrep -f '[c]ampaign_monitor_loop.py' >/dev/null \
   || pgrep -f '[h]arness.py.*--policy fm' >/dev/null; do
  sleep 300
  echo "$(date -Is) ARC-TAC queue still waiting."
done

if pgrep -f '[r]un_arc_tacfold_main_bench.sh' >/dev/null; then
  echo "ARC-TAC campaign is already running; queue exits."
  exit 0
fi

echo "$(date -Is) ARC-TAC prerequisites clear; starting campaign."
exec bash run_arc_tacfold_main_bench.sh
