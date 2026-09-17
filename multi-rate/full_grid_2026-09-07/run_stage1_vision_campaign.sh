#!/usr/bin/env bash
# Sequential Stage 1 only: own one vision bridge, then train/evaluate all four Table 2a tasks.
# Existing checkpoints resume at evaluation; completed structured results are skipped.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_PY="${POLICY_PY:-/home/user/anaconda3/envs/vla_smolvla_libero/bin/python}"
BRIDGE_PY="${BRIDGE_PY:-/home/user/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python}"
PORT="${PORT:-8766}"
DATA_DIR="${DATA_DIR:-$ROOT/robocasa_data_vision}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/stage1_vision_results}"
LOG_DIR="${LOG_DIR:-$ROOT/stage1_vision_logs}"
BRIDGE_NUMBA_CACHE_DIR="${BRIDGE_NUMBA_CACHE_DIR:-$LOG_DIR/numba_cache}"
STEPS="${STEPS:-15000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MICROBATCH="${MICROBATCH:-16}"
N_TRAIN="${N_TRAIN:-35}"
N_EVAL="${N_EVAL:-15}"
SEED="${SEED:-0}"
EVAL_SEED="${EVAL_SEED:-0}"
MAX_STEPS="${MAX_STEPS:-}"
FORCE="${FORCE:-0}"

usage() {
    echo "Usage: [POLICY_PY=...] [BRIDGE_PY=...] [PORT=8766] [MAX_STEPS=...] [FORCE=0|1] $0"
    echo "Runs only the four Stage 1 visual-policy tasks, sequentially. Other defaults may also be overridden by environment variable."
    echo "FORCE=1 re-evaluates a compatible completed checkpoint; incompatible artifacts require a distinct OUTPUT_DIR."
}

if (( $# > 0 )); then
    if (( $# == 1 )) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage
        exit 0
    fi
    usage >&2
    exit 2
fi

BRIDGE_PID=""
CURRENT_PID=""
CURRENT_LOG=""

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "$CURRENT_PID" ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
        kill -TERM "$CURRENT_PID" 2>/dev/null || true
        wait "$CURRENT_PID" 2>/dev/null || true
    fi
    if [[ -n "$BRIDGE_PID" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill -TERM "$BRIDGE_PID" 2>/dev/null || true
        wait "$BRIDGE_PID" 2>/dev/null || true
        echo "[bridge] stopped pid=$BRIDGE_PID"
    fi
    if (( status != 0 )) && [[ -n "$CURRENT_LOG" ]]; then
        echo "[campaign] stopped with status $status; task log: $CURRENT_LOG" >&2
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

for executable in "$POLICY_PY" "$BRIDGE_PY"; do
    if [[ ! -x "$executable" ]]; then
        echo "[campaign] Python executable is unavailable: $executable" >&2
        exit 2
    fi
done
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$BRIDGE_NUMBA_CACHE_DIR"

port_is_open() {
    "$POLICY_PY" - "$PORT" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.25):
    pass
PY
}

TASKS=(TurnOffSinkFaucet CoffeePressButton TurnOffMicrowave CloseSingleDoor)
MAX_STEPS_ARGS=()
if [[ -n "$MAX_STEPS" ]]; then
    MAX_STEPS_ARGS=(--max-steps "$MAX_STEPS")
fi
MODES=()
RUNNABLE=0
for task in "${TASKS[@]}"; do
    checkpoint="$OUTPUT_DIR/$task.pt"
    status=$("$POLICY_PY" "$ROOT/stage1_vision.py" "$task" --check-artifacts \
        --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" --checkpoint "$checkpoint" \
        --steps "$STEPS" --batch-size "$BATCH_SIZE" --microbatch "$MICROBATCH" \
        --n-train "$N_TRAIN" --n-eval "$N_EVAL" --seed "$SEED" --eval-seed "$EVAL_SEED" \
        "${MAX_STEPS_ARGS[@]}")
    case "$status" in
        train-eval|eval|skip) ;;
        *) echo "[campaign] invalid artifact status for $task: $status" >&2; exit 2 ;;
    esac
    if [[ "$FORCE" == 1 && "$status" == skip ]]; then
        status=eval
    fi
    MODES+=("$status")
    if [[ "$status" != skip ]]; then
        ((RUNNABLE += 1))
    fi
done
if (( RUNNABLE == 0 )); then
    echo "[campaign] all four matching Stage 1 results are already complete"
    exit 0
fi

if port_is_open; then
    echo "[campaign] refusing to replace an existing listener on 127.0.0.1:$PORT" >&2
    exit 2
fi

BRIDGE_LOG="$LOG_DIR/bridge.log"
echo "[bridge] starting on 127.0.0.1:$PORT; log=$BRIDGE_LOG"
NUMBA_CACHE_DIR="$BRIDGE_NUMBA_CACHE_DIR" \
    "$BRIDGE_PY" "$ROOT/robocasa_vision_bridge.py" --port "$PORT" --data-dir "$DATA_DIR" \
    >>"$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!
for _ in $(seq 1 60); do
    if port_is_open; then
        break
    fi
    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        echo "[bridge] exited during startup; see $BRIDGE_LOG" >&2
        exit 1
    fi
    sleep 1
done
if ! port_is_open; then
    echo "[bridge] did not become ready within 60 seconds; see $BRIDGE_LOG" >&2
    exit 1
fi

for index in "${!TASKS[@]}"; do
    task="${TASKS[$index]}"
    mode="${MODES[$index]}"
    checkpoint="$OUTPUT_DIR/$task.pt"
    result="$OUTPUT_DIR/${task}_native.json"
    CURRENT_LOG="$LOG_DIR/$task.log"
    if [[ "$mode" == skip ]]; then
        echo "[skip] $task has a complete result matching data, checkpoint, split, budgets, seeds, and max steps: $result"
        continue
    fi
    echo "[start] $task mode=$mode $(date --iso-8601=seconds); log=$CURRENT_LOG"
    "$POLICY_PY" "$ROOT/stage1_vision.py" "$task" \
        --mode "$mode" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" \
        --checkpoint "$checkpoint" --port "$PORT" --steps "$STEPS" \
        --batch-size "$BATCH_SIZE" --microbatch "$MICROBATCH" \
        --n-train "$N_TRAIN" --n-eval "$N_EVAL" --seed "$SEED" --eval-seed "$EVAL_SEED" \
        "${MAX_STEPS_ARGS[@]}" \
        >>"$CURRENT_LOG" 2>&1 &
    CURRENT_PID=$!
    wait "$CURRENT_PID"
    CURRENT_PID=""
    echo "[done] $task $(date --iso-8601=seconds); result=$result"
done

CURRENT_LOG=""
echo "[campaign] all four Stage 1 tasks completed; Stage 2 was not started"
