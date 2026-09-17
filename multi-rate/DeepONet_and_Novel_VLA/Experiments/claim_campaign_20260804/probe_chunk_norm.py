"""Is the absorbing zero-action state in the CHUNK, or only in how it is consumed?

This is the decisive test for whether any decoder can escape it. The policy replans every 0.5 s
(REPLAN=10 at 20 Hz), so each chunk is freshly generated from the current observation. If the
freshly planned chunk is itself near-zero during a failure, the policy outputs nothing given that
observation -- and cubic spline, operator re-integration, ZOH and every other decoder are all
resampling the same near-zero numbers. That closes the ours-vs-spline axis with direct evidence
rather than inference.

Logs the RAW (unnormalized) pose-delta magnitude of every chunk at the moment it is produced.
"""
from __future__ import annotations
import os, sys
import numpy as np
os.environ.setdefault('DEEPONET_HEAD','asrc'); os.environ.setdefault('DEEPONET_FOURIER','6')
os.environ.setdefault('DEEPONET_STATE_HISTORY_STEPS','8'); os.environ.setdefault('DEEPONET_P','256')
sys.path.insert(0,'/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804')
import torch
import evaluate_multirate_honest as honest, evaluate_height_screen as screen
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from rate_integrated_deeponet import RateIntegratedDeepONetHead
honest.patch_control_freq(20)
screen.SUITE='libero_spatial'; screen.MAX_STEPS=220; screen.REPLAN=10; screen.CONTROL_FREQ=20
policy,(pre,post)=screen.load_policy('deeponet', honest.MODELS['asrc'][1],
                                     LeRobotDatasetMetadata(screen.DATASET).stats)
head=[m for m in policy.modules() if isinstance(m,RateIntegratedDeepONetHead)][0]
off=head.action_offset[:6].detach().float().cpu().numpy(); sc=head.action_scale[:6].detach().float().cpu().numpy()
chunks=[]
_orig=policy._get_action_chunk
def spy(*a,**k):
    ch=_orig(*a,**k)
    raw=ch[...,:6].detach().float().cpu().numpy()*sc+off      # RAW space; normalized magnitudes lie
    chunks.append(float(np.abs(raw[...,:3]).sum(-1).mean()))  # mean per-step translation magnitude
    return ch
policy._get_action_chunk=spy
for task,trials in ((8,range(5)),):
    for trial in trials:
        env=screen._make_env(task); env.num_steps_wait=honest.SETTLE_STEPS_20HZ
        env.init_state_id=trial; assert env.init_state_id==trial
        obs,_=env.reset(seed=1000+trial); policy.reset(); chunks.clear()
        ok=False
        for step in range(1,221):
            batch=pre(screen._policy_input(obs, env.task_description))
            with torch.no_grad(): act=policy.select_action(batch)
            act=post(act); an=np.asarray(act.squeeze(0).float().cpu() if torch.is_tensor(act) else act,float)
            obs,_,term,_,info=env.step(an)
            if info.get('is_success'): ok=True; break
            if term: break
        env.close()
        C=np.array(chunks)
        late=C[-4:] if len(C)>=4 else C
        print('[t%d trial %d] %s  n_chunks=%d  chunk_mag first4=%s  LAST4=%s  overall_mean=%.4f'
              %(task,trial,'OK  ' if ok else 'FAIL',len(C),
                np.round(C[:4],4).tolist(), np.round(late,4).tolist(), C.mean()), flush=True)
