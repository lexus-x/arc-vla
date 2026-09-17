#!/usr/bin/env bash
# ==============================================================================
# Master Automated Flow Matching (8,300 steps) Training & Evaluation for 3 Suites:
# Suites: libero_object, libero_goal, libero_10 (long)
# Target Host: Blackwell GPU (RTX PRO 6000, 96GB VRAM)
# Recipe: Matched 8,300 steps (Stage 1: 1650, Stage 2: 6650, Batch: 48, seed: 0)
# Evaluation: Matched 20 Hz paired evaluation (asrc vs flow8300, n=50 per suite)
# ==============================================================================

set -u

TRAIN_DIR="/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH"
EVAL_DIR="/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
BASE_STORAGE="/media/user/C2FE578FFE577A9D/vla_matched"
PY="/home/user/anaconda3/envs/vla_smolvla_libero/bin/python"

mkdir -p "$BASE_STORAGE"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

MASTER_LOG="$BASE_STORAGE/master_suites_campaign.log"
echo "==============================================================================" | tee -a "$MASTER_LOG"
echo ">>> STARTING MASTER FLOW MATCHING SUITES CAMPAIGN at $(date)" | tee -a "$MASTER_LOG"
echo "==============================================================================" | tee -a "$MASTER_LOG"

SUITES=("object" "goal" "10")
DATASETS=("lerobot/libero_object_image" "lerobot/libero_goal_image" "lerobot/libero_10_image")
EVAL_SUITES=("libero_object" "libero_goal" "libero_10")
TAGS=("object" "goal" "long")

for i in "${!SUITES[@]}"; do
  S="${SUITES[$i]}"
  DS="${DATASETS[$i]}"
  ES="${EVAL_SUITES[$i]}"
  TAG="${TAGS[$i]}"
  OUT="$BASE_STORAGE/flow8300_${S}_s0"
  TRAIN_LOG="$BASE_STORAGE/flow8300_${S}_train.log"
  EVAL_LOG="$EVAL_DIR/matched8300_${TAG}_eval.log"
  EVAL_OUT="matched8300_${TAG}_out"

  echo "------------------------------------------------------------------------------" | tee -a "$MASTER_LOG"
  echo ">>> [SUITE $S] Phase 1: Training flow8300_${S}_s0 on $DS ($(date))" | tee -a "$MASTER_LOG"
  echo "------------------------------------------------------------------------------" | tee -a "$MASTER_LOG"

  cd "$TRAIN_DIR" || exit 1

  if [ -f "$OUT/checkpoints/8300/model.safetensors" ]; then
    echo ">>> Checkpoint already exists at $OUT/checkpoints/8300. Skipping training." | tee -a "$MASTER_LOG"
  else
    $PY train.py       --head flow       --variant baseline       --dataset "$DS"       --out "$OUT"       --seed 0       --stage1_steps 1650       --stage2_steps 6650       --stage1_batch 48       --stage2_batch 48       --warmup 500       --head_lr 0.0001       --backbone_lr 1e-05       --ema 0.999       --num_workers 8       --ckpt_every 10000       > "$TRAIN_LOG" 2>&1

    if [ -f "$OUT/checkpoints/8300/model.safetensors" ]; then
      echo ">>> Training completed successfully for $S at $(date)" | tee -a "$MASTER_LOG"
    else
      echo ">>> [ERROR] Training failed for $S! Check $TRAIN_LOG" | tee -a "$MASTER_LOG"
      exit 1
    fi
  fi

  echo "------------------------------------------------------------------------------" | tee -a "$MASTER_LOG"
  echo ">>> [SUITE $S] Phase 2: Running Matched Evaluation on $ES (asrc vs flow8300) ($(date))" | tee -a "$MASTER_LOG"
  echo "------------------------------------------------------------------------------" | tee -a "$MASTER_LOG"

  cd "$EVAL_DIR" || exit 1

  $PY -c "
import sys
sys.path.insert(0, '$EVAL_DIR')
import evaluate_multirate_honest as E
E.MODELS['flow8300_$TAG'] = ('flow', '$OUT/checkpoints/8300', None, None)
E.ARMS['flow8300_${TAG}_native_20env'] = ('flow8300_$TAG', 20, None, None)
E.MODEL_DATASET['flow8300_$TAG'] = '$DS'
sys.argv = ['evaluate_multirate_honest.py',
            '--arms', 'asrc_${TAG}_native_20env,flow8300_${TAG}_native_20env',
            '--trials_per_task', '5',
            '--suite', '$ES',
            '--out', '$EVAL_OUT']
E.main()
" > "$EVAL_LOG" 2>&1

  echo ">>> Evaluation complete for $S at $(date)" | tee -a "$MASTER_LOG"
done

echo "==============================================================================" | tee -a "$MASTER_LOG"
echo ">>> ALL 3 SUITES (OBJECT, GOAL, LONG) COMPLETE! Generating scorecard... ($(date))" | tee -a "$MASTER_LOG"
echo "==============================================================================" | tee -a "$MASTER_LOG"

cd "$EVAL_DIR" || exit 1
$PY matched8300_suites_report.py | tee -a "$MASTER_LOG"

echo ">>> ALL TASKS FINISHED at $(date)" | tee -a "$MASTER_LOG"
