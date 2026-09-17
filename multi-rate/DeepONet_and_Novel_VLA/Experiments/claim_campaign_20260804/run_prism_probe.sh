#!/usr/bin/env bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate vla_smolvla_libero

cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804
mkdir -p prism_probe_results

LOG="/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/prism_probe_results/probe_exec.log"
echo "=== PRISM PROBE LAUNCHED AT $(date) ===" > "$LOG"

PY="/home/user/anaconda3/envs/vla_smolvla_libero/bin/python /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/register_prism_arms.py"

# 1. Native 20 Hz Baseline (Replan 5 & 10)
echo "Running Native 20 Hz (replan 5)..." >> "$LOG"
$PY --arms v2_spatial_native_20env --replan_steps 5 --trials_per_task 5 --out prism_probe_results/v2_spatial_native_rp5.json >> "$LOG" 2>&1

echo "Running Native 20 Hz (replan 10)..." >> "$LOG"
$PY --arms v2_spatial_native_20env --replan_steps 10 --trials_per_task 5 --out prism_probe_results/v2_spatial_native_rp10.json >> "$LOG" 2>&1

# 2. 40 Hz Cubic Spline (Replan 10 & 20)
echo "Running 40 Hz Spline (replan 10)..." >> "$LOG"
$PY --arms v2_spatial_cadmag_spline_40env --replan_steps 10 --trials_per_task 5 --out prism_probe_results/v2_spatial_spline_rp10.json >> "$LOG" 2>&1

echo "Running 40 Hz Spline (replan 20)..." >> "$LOG"
$PY --arms v2_spatial_cadmag_spline_40env --replan_steps 20 --trials_per_task 5 --out prism_probe_results/v2_spatial_spline_rp20.json >> "$LOG" 2>&1

# 3. 40 Hz Zero-Order Hold (ZOH) (Replan 10 & 20)
echo "Running 40 Hz ZOH (replan 10)..." >> "$LOG"
$PY --arms v2_spatial_zoh_40env --replan_steps 10 --trials_per_task 5 --out prism_probe_results/v2_spatial_zoh_rp10.json >> "$LOG" 2>&1

echo "Running 40 Hz ZOH (replan 20)..." >> "$LOG"
$PY --arms v2_spatial_zoh_40env --replan_steps 20 --trials_per_task 5 --out prism_probe_results/v2_spatial_zoh_rp20.json >> "$LOG" 2>&1

# 4. 40 Hz PRISM (Dynamic Bandwidth-Gated PCHIP/ZOH) (Replan 10 & 20)
echo "Running 40 Hz PRISM (replan 10)..." >> "$LOG"
$PY --arms v2_spatial_prism_40env --replan_steps 10 --trials_per_task 5 --out prism_probe_results/v2_spatial_prism_rp10.json >> "$LOG" 2>&1

echo "Running 40 Hz PRISM (replan 20)..." >> "$LOG"
$PY --arms v2_spatial_prism_40env --replan_steps 20 --trials_per_task 5 --out prism_probe_results/v2_spatial_prism_rp20.json >> "$LOG" 2>&1

echo "=== PRISM PROBE COMPLETED AT $(date) ===" >> "$LOG"
