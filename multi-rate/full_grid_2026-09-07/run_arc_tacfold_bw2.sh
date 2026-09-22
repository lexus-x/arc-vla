#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

pick_python() {
  local candidate
  for candidate in "$@"; do
    [[ -x $candidate ]] && { printf '%s\n' "$candidate"; return; }
  done
  return 1
}

export ARC_MS_PY=${ARC_MS_PY:-$(pick_python \
  "$HOME/envs/arc_ms/bin/python" \
  "$HOME/anaconda3/envs/ms3/bin/python" \
  "$HOME/miniconda3/envs/ms3/bin/python")}
export ARC_RM_PY=${ARC_RM_PY:-$(pick_python \
  "$HOME/envs/saptarshi/bin/python" \
  "$HOME/anaconda3/envs/vla_smolvla_libero/bin/python" \
  "$HOME/miniconda3/envs/vla_smolvla_libero/bin/python")}
export ARC_RC_PY=${ARC_RC_PY:-$(pick_python \
  "$HOME/envs/saptarshi/bin/python" \
  "$HOME/anaconda3/envs/gr00t/bin/python" \
  "$HOME/miniconda3/envs/gr00t/bin/python")}
export ARC_BRIDGE_PY=${ARC_BRIDGE_PY:-$(pick_python \
  "$HOME/envs/arc_robocasa/bin/python" \
  "$HOME/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python")}
NVIDIA_SMI=${NVIDIA_SMI:-$(pick_python /usr/bin/nvidia-smi /usr/lib/wsl/lib/nvidia-smi)}
export MANISKILL_DATA_DIR=${MANISKILL_DATA_DIR:-"$PWD/external_data/maniskill_data"}
export ROBOMIMIC_DATA_DIR=${ROBOMIMIC_DATA_DIR:-"$PWD/external_data/robomimic_data"}
export ROBOCASA_DATA_DIR=${ROBOCASA_DATA_DIR:-"$PWD/robocasa_data"}
export ARC_VISION_DATA=${ARC_VISION_DATA:-"$PWD/robocasa_data_vision"}
export ARC_VISION_OUT=${ARC_VISION_OUT:-"$PWD/stage1_arc_vision_results"}

for required in \
  "$MANISKILL_DATA_DIR/pusht_rl.h5" \
  "$MANISKILL_DATA_DIR/pusht_rl.json" \
  "$ROBOMIMIC_DATA_DIR/lift.hdf5" \
  "$ROBOMIMIC_DATA_DIR/can.hdf5" \
  "$ROBOMIMIC_DATA_DIR/square.hdf5" \
  "$ROBOCASA_DATA_DIR/TurnOffSinkFaucet_ld.hdf5" \
  "$ROBOCASA_DATA_DIR/CoffeePressButton_ld.hdf5" \
  "$ROBOCASA_DATA_DIR/TurnOffMicrowave_ld.hdf5" \
  "$ROBOCASA_DATA_DIR/CloseSingleDoor_ld.hdf5" \
  "$ARC_VISION_DATA/CloseSingleDoor_human_im.hdf5"; do
  [[ -f $required ]] || { echo "Missing campaign input: $required" >&2; exit 2; }
done

export NUMBA_CACHE_DIR=${ARC_NUMBA_CACHE_DIR:-"$PWD/.arc_numba_cache"}
mkdir -p "$NUMBA_CACHE_DIR"
"$ARC_MS_PY" -c 'import torch, scipy, mani_skill; assert torch.cuda.is_available()'
"$ARC_RM_PY" -c 'import torch, scipy, robosuite; assert torch.cuda.is_available()'
"$ARC_RC_PY" -c 'import torch, scipy; assert torch.cuda.is_available()'
"$ARC_BRIDGE_PY" -c 'import h5py, robocasa, robosuite'
"$ARC_RM_PY" -m pytest -q test_arc.py test_stage1_vision.py test_analyze_arc_tacfold.py
"$ARC_RM_PY" -m py_compile harness.py heads.py dp_min.py stage1_vision.py analyze_arc_tacfold.py

echo "$(date -Is) bw2 ARC-TAC campaign starting" | tee arc_tacfold_bw2.status
bash run_arc_tacfold_main_bench.sh 2>&1 | tee arc_tacfold_bw2.log
"$ARC_RM_PY" analyze_arc_tacfold.py | tee paper_arc_tac/ARC_TAC_RESULTS.md
{
  echo "completed_at=$(date -Is)"
  hostname
  "$NVIDIA_SMI" --query-gpu=index,name,driver_version --format=csv,noheader
  sha256sum result_*arcmain*.json dp_*arcmain*.pt "$ARC_VISION_OUT"/* 2>/dev/null || true
} > paper_arc_tac/ARC_TAC_PROVENANCE.sha256
echo "$(date -Is) bw2 ARC-TAC campaign complete" | tee arc_tacfold_bw2.status

tar -czf arc_tacfold_bw2_results.tgz \
  result_*arcmain*.json \
  paper_arc_tac/ARC_TAC_RESULTS.md \
  paper_arc_tac/ARC_TAC_PROVENANCE.sha256 \
  arc_tacfold_bw2.log arc_tacfold_bw2.status \
  "$ARC_VISION_OUT"

if command -v tailscale >/dev/null; then
  tailscale file cp arc_tacfold_bw2_results.tgz blackwell: || true
fi
