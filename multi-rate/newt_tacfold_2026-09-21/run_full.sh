#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/newt-mpl"
conda run -n ms3 python eval_newt_tacfold.py --episodes 100 --output results.json
conda run -n ms3 python analyze.py --input results.json --output REPORT.md
