#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/newt-mpl"
conda run -n ms3 python eval_newt_tacfold.py \
  --task ms-pick-cube-eepose \
  --episodes 2 \
  --output smoke_results.json
