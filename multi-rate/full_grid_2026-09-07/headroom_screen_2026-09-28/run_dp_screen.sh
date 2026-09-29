#!/bin/bash
# Official-DP headroom screen (SCREEN_RULE.md): usage  run_dp_screen.sh smoke|screen|conv|kitchen_smoke|kitchen
# Jobs run 4 at a time, CPU policy (torch 1.12.1 has no sm_120 kernels), 4 torch threads each.
cd "$(dirname "$0")"
D=/media/user/C2FE578FFE577A9D/dp_official_ckpts
declare -A CK=(
 [can_ph]="can_ph/diffusion_policy_transformer/train_0/checkpoints/epoch%3D0800-test_mean_score%3D1.000.ckpt 23301e7774109a0017eee1df8b292b0f2efbff386e2a447536665d7c72d78044"
 [square_ph]="square_ph/diffusion_policy_cnn/train_0/checkpoints/epoch%3D1750-test_mean_score%3D1.000.ckpt 3d2de56e9891b56c7beaabbb42c14a6fa1d5fc42acddc1ba46feac3eb8696816"
 [transport_ph]="transport_ph/diffusion_policy_transformer/train_0/checkpoints/epoch%3D3500-test_mean_score%3D1.000.ckpt 63107842e69f99cbd75d5bd294db4f84a01a48e82866bd936f5067c5ca68972c"
 [tool_hang_ph]="tool_hang_ph/diffusion_policy_transformer/train_0/checkpoints/epoch%3D4600-test_mean_score%3D1.000.ckpt a0ebb5aec0da31c5f2f18be326e73275460e17c855b9a6bc1ad264c929922fff"
 [kitchen]="kitchen/diffusion_policy_cnn/train_0/checkpoints/epoch%3D1700-test_mean_score%3D0.580.ckpt 40504002fde43a533b88536710578fe815f0231af09ae97b27422490f1d73269")
run() {  # task arm k n start_seed out_stem
  set -- $1 $2 $3 $4 $5 $6 ${CK[$1]}; DS=$D/data/robomimic/datasets; [ "$1" = kitchen ] && DS=$D/data/kitchen
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MUJOCO_GL=egl /home/user/anaconda3/envs/dp_official/bin/python -u ../eval_dp_official_rate.py \
    --ckpt "$D/$7" --ckpt_sha256 $8 --task $1 --arm $2 --k $3 --n $4 --start_seed $5 --n_envs 25 \
    --dataset_dir $DS --out $6.json > $6.log 2>&1
  echo "DONE $6 $(tail -c 300 $6.log | tr '\r' '\n' | grep -E '%$' | tail -1)"
}
export -f run; export D; export CK_DEF="$(declare -p CK)"
jobs() {
  if [ "$1" = smoke ]; then mkdir -p dp_smoke
    for t in square_ph transport_ph tool_hang_ph; do echo "$t native 1 20 900000 dp_smoke/dp_${t}_native_smoke"; done
  elif [ "$1" = kitchen_smoke ]; then mkdir -p dp_smoke  # amendment 4, burned seeds
    echo "kitchen native 1 10 900000 dp_smoke/dp_kitchen_native_smoke"; echo "kitchen zoh 4 10 900000 dp_smoke/dp_kitchen_zoh4_smoke"
  elif [ "$1" = kitchen ]; then  # amendment 4: native, zoh + converters at k=2,4
    echo "kitchen native 1 50 5000 dp_kitchen_native_k1"
    for k in 2 4; do for a in zoh spline_satfix tac_fold_satfix qp_anchor; do echo "kitchen $a $k 50 5000 dp_kitchen_${a}_k$k"; done; done
  elif [ "$1" = conv ]; then  # amendment 3 converter check, k=4
    for a in spline_satfix tac_fold_satfix qp_anchor; do for t in can_ph square_ph transport_ph tool_hang_ph; do echo "$t $a 4 50 5000 dp_${t}_${a}_k4"; done; done
  else
    for t in can_ph square_ph transport_ph tool_hang_ph; do echo "$t native 1 50 5000 dp_${t}_native_k1"; done
    for k in 2 4; do for t in can_ph square_ph transport_ph tool_hang_ph; do echo "$t zoh $k 50 5000 dp_${t}_zoh_k$k"; done; done
  fi
}
jobs $1 | xargs -P ${P:-4} -L 1 bash -c 'eval "$CK_DEF"; run "$@"' _
echo ALL_DONE $1
