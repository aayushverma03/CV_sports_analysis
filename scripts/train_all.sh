#!/usr/bin/env bash
# Chain-run all four custom-detector trainings.
#
# Waits for medicine_ball_v1 (already running, PID 64911) to finish, then
# runs plyo_box_v1, hurdle_v1, cone_v1 sequentially. Each job logs to
# logs/train_<name>.log. A failure in one does NOT abort the chain.
#
#   nohup scripts/train_all.sh > logs/train_all.log 2>&1 &
#
# Check progress:
#   tail -f logs/train_all.log

set -u
cd "$(dirname "$0")/.."

MB_PID=64911
QUEUE=(
    "configs/training/plyo_box_v1.yaml"
    "configs/training/hurdle_v1.yaml"
    "configs/training/cone_v1.yaml"
)

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] chain start"

# Wait for the pre-existing medicine_ball_v1 training to finish, if alive.
if kill -0 "$MB_PID" 2>/dev/null; then
    echo "[$(ts)] waiting for medicine_ball_v1 (PID $MB_PID)..."
    while kill -0 "$MB_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[$(ts)] medicine_ball_v1 exited"
else
    echo "[$(ts)] medicine_ball_v1 already finished or PID not found, proceeding"
fi

# Run each config in order. Failure of one does not abort the chain.
for cfg in "${QUEUE[@]}"; do
    name=$(basename "$cfg" .yaml)
    log="logs/train_${name}.log"
    echo "[$(ts)] === starting $name ==="
    if uv run scripts/train_yolo.py --config "$cfg" > "$log" 2>&1; then
        echo "[$(ts)] $name done -> $log"
    else
        echo "[$(ts)] $name FAILED (exit $?) -> $log"
    fi
done

echo "[$(ts)] chain complete"
