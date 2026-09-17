#!/bin/bash
cd "$(dirname "$0")"
for i in $(seq 1 240); do [ -f demos_PushT-v1_200.npz ] && break; sleep 30; done
/home/user/anaconda3/envs/ms3/bin/python -c "import numpy as np; z=np.load('demos_PushT-v1_200.npz',allow_pickle=True); print('cache ok', len(z['O']))" || exit 1
for S in 1 2; do nohup ./run_heads_c2.sh PushT-v1 $S > stream_c2_PushT-v1_s${S}.log 2>&1 & done
echo "relaunched PushT s1 s2 $(date +%H:%M:%S)"
