#!/usr/bin/env bash
cd /home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804 || exit 1
for K in 0p3 0p5 0p7 1p0; do
  ok=$(grep -c 'OK' "ksweep_k${K}.log" 2>/dev/null || true)
  fl=$(grep -cE 'x \(' "ksweep_k${K}.log" 2>/dev/null || true)
  last=$(tail -1 "ksweep_k${K}.log" 2>/dev/null | head -c 100)
  echo "k=${K}: ok=$ok fail=$fl | $last"
done
echo "native520: $(tail -1 r5_asrc_native_20env.log 2>/dev/null | head -c 100)"
echo "train: $(tail -1 train_asrc30k_s0.log 2>/dev/null | head -c 130)"
