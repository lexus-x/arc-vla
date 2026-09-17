"""Task-5 grasp mechanism, designed to survive the success control.

Three earlier metrics died to the task-2 control because they were ad hoc (gripper qpos sum,
object z-range, and min-horizontal-distance object over the episode -- which on successes picks
the PLATE, approached from above while already carrying the bowl, so it never measured a grasp).

This anchors on the event that actually defines a grasp attempt: the step where the gripper COMMAND
first closes. At that step it reports the vertical offset to EVERY candidate bowl, and separately
whether any bowl ever rises. No object is selected by proximity.
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
    bowls=[k for k in raw if 'bowl' in k and k.endswith('_pos') and 'to_robot0' not in k]
    z0={k:float(raw[k][2]) for k in bowls}
    if trial==0: print('[task %d] %r | bowls=%s'%(a.task_id, env.task_description, bowls), flush=True)
    close_at=None; snap=None; ok=False; zmax={k:z0[k] for k in bowls}
    for step in range(1,221):
        batch=pre(screen._policy_input(obs, env.task_description))
        with torch.no_grad(): act=policy.select_action(batch)
        act=post(act); an=np.asarray(act.squeeze(0).float().cpu() if torch.is_tensor(act) else act,float)
        obs,_,term,_,info=env.step(an)
        r=env._env.env._get_observations()
        e=np.array(r['robot0_eef_pos'],float)
        for k in bowls: zmax[k]=max(zmax[k], float(r[k][2]))
        if close_at is None and an[6] > 0.0:      # gripper COMMAND closes
            close_at=step
            snap={k:(float(np.linalg.norm(e[:2]-np.array(r[k],float)[:2])), float(e[2]-r[k][2])) for k in bowls}
        if info.get('is_success'): ok=True; break
        if term: break
    env.close()
    lift={k:zmax[k]-z0[k] for k in bowls}
    if snap is None:
        print('[task %d trial %d] %s | GRIPPER NEVER COMMANDED CLOSED in %d steps | lift=%s'
              %(a.task_id,trial,'SUCCESS' if ok else 'FAIL',step,{k:round(v,3) for k,v in lift.items()}), flush=True)
    else:
        d=' '.join('%s xy=%.3f dz=%+.4f'%(k.replace('_pos','').replace('akita_black_',''),v[0],v[1]) for k,v in snap.items())
        print('[task %d trial %d] %s | first_close@%d | %s | lift=%s'
              %(a.task_id,trial,'SUCCESS' if ok else 'FAIL',close_at,d,{k.replace('_pos','').replace('akita_black_',''):round(v,3) for k,v in lift.items()}), flush=True)
