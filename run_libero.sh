#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=${LOG_DIR:-"$ROOT_DIR/logs/libero"}
PYTHON_BIN=${PYTHON_BIN:-python}
NUM_TRIALS=${NUM_TRIALS:-1}
MUJOCO_GL=${MUJOCO_GL:-osmesa}
export MUJOCO_GL

mkdir -p "$LOG_DIR"

run_suite() {
	local suite_name="$1"
	local log_file="$LOG_DIR/${suite_name}.log"

	echo "[$(date '+%F %T')] Running suite: ${suite_name}" | tee "$log_file"
	"$PYTHON_BIN" "$ROOT_DIR/examples/libero/main.py" \
		--args.num-trials-per-task "$NUM_TRIALS" \
		--args.task-suite-name "$suite_name" 2>&1 | tee -a "$log_file"
}

if (($# > 0)); then
	suites=("$@")
else
	suites=(libero_spatial libero_object libero_goal libero_10)
fi

for suite in "${suites[@]}"; do
	run_suite "$suite"
done
