#!/usr/bin/env bash
# Start the Go server and N Python workers on this machine for one config, then print the result.
# usage: launch/local.sh [config.yaml] [seed]
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-tests/configs/tiny.yaml}
SEED=${2:-0}
make -s build
mkdir -p runs
python3 experiments/run.py --config "$CONFIG" --seed "$SEED" --out "runs/$(basename "$CONFIG" .yaml)_s$SEED.json"
