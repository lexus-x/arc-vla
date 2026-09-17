"""Does asrc share flow's grasp-height mode collapse on task 5?

The known mechanism: the policy descends to a canonical grasp height instead of the target's
actual rim height, so an elevated target gets pushed rather than grasped. Measured directly:
eef_z minus target_z at the moment of closest horizontal approach. Negative => the gripper is
BELOW the target's centre, i.e. it drove through it. Task 2 is the success control.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
os.environ.setdefault('DEEPONET_HEAD','asrc'); os.environ.setdefault('DEEPONET_FOURIER','6')
os.environ.setdefault('DEEPONET_STATE_HISTORY_STEPS','8'); os.environ.setdefault('DEEPONET_P','256')
sys.path.insert(0,'/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804')
import torch
import evaluate_multirate_honest as honest, evaluate_height_screen as screen
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
ap=argparse.ArgumentParser(); ap.add_argument('--task_id',type=int,default=5)
ap.add_argument('--trials',type=int,default=3); a=ap.parse_args()
honest.patch_control_freq(20)
screen.SUITE='libero_spatial'; screen.MAX_STEPS=220; screen.REPLAN=10; screen.CONTROL_FREQ=20
policy,(pre,post)=screen.load_policy('deeponet', honest.MODELS['asrc'][1],
                                     LeRobotDatasetMetadata(screen.DATASET).stats)
for trial in range(a.trials):
    env=screen._make_env(a.task_id); env.num_steps_wait=honest.SETTLE_STEPS_20HZ
    env.init_state_id=trial; assert env.init_state_id==trial
    obs,_=env.reset(seed=1000+trial); policy.reset()
    raw=env._env.env._get_observations()
    # the manipulation target is whichever object the eef gets horizontally closest to
    objk=[k for k in raw if k.endswith('_pos') and not k.startswith('robot0') and 'to_robot0' not in k]
    tr=[]
    ok=False
    for step in range(1,221):
        batch=pre(screen._policy_input(obs, env.task_description))
        with torch.no_grad(): act=policy.select_action(batch)
        act=post(act); an=np.asarray(act.squeeze(0).float().cpu() if torch.is_tensor(act) else act,float)
        obs,_,term,_,info=env.step(an)
        r=env._env.env._get_observations()
        e=np.array(r['robot0_eef_pos'],float)
        tr.append((step,e,{k:np.array(r[k],float) for k in objk}))
        if info.get('is_success'): ok=True; break
        if term: break
    env.close()
    # closest horizontal approach to each object, and the vertical offset there
    best=None
    for k in objk:
        d=[(np.linalg.norm(e[:2]-o[k][:2]), s, e[2]-o[k][2], o[k][2]) for s,e,o in tr]
        xy,s,dz,oz=min(d)
        if best is None or xy<best[0]: best=(xy,s,dz,oz,k)
    xy,s,dz,oz,k=best
    print('[task %d trial %d] %s | target=%s  closest_xy=%.3f m @step %d  eef_z-target_z=%+.4f m  target_z=%.3f'
          %(a.task_id,trial,'SUCCESS' if ok else 'FAIL',k,xy,s,dz,oz), flush=True)
