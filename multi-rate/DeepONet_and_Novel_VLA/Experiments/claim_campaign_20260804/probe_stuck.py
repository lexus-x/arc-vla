"""Task-5 failure anatomy. Every arm scores 0-1/15 on libero_spatial task 5, and all failures sit
at the step cap with zero early terminations. Two very different causes look identical in a success
rate: the arm STUCK (commanded motion large, measured motion ~0 -> a state-feedback mechanism could
act on it) or a MIS-GRASP (motion fine, object never leaves the table -> feedback has nothing to
grab). This logs both so the mechanism is chosen from evidence, not theory.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

os.environ.setdefault('DEEPONET_HEAD','asrc'); os.environ.setdefault('DEEPONET_FOURIER','6')
os.environ.setdefault('DEEPONET_STATE_HISTORY_STEPS','8')
sys.path.insert(0,'/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804')
import torch
import evaluate_multirate_honest as honest
import evaluate_height_screen as screen
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ap=argparse.ArgumentParser()
ap.add_argument('--task_id',type=int,default=5)
ap.add_argument('--trials',type=int,default=3)
ap.add_argument('--out',default='probe_stuck.json')
a=ap.parse_args()

honest.patch_control_freq(20)
screen.SUITE='libero_spatial'; screen.MAX_STEPS=220; screen.REPLAN=10; screen.CONTROL_FREQ=20
policy,(pre,post)=screen.load_policy('deeponet', honest.MODELS['asrc'][1],
                                     LeRobotDatasetMetadata(screen.DATASET).stats)
out=[]
for trial in range(a.trials):
    env=screen._make_env(a.task_id)
    env.num_steps_wait=honest.SETTLE_STEPS_20HZ
    env.init_state_id=trial; assert env.init_state_id==trial
    obs,_=env.reset(seed=1000+trial)
    policy.reset()
    raw=env._env.env._get_observations()
    if trial==0:
        print('[keys] pos-like obs:', sorted(k for k in raw if k.endswith('_pos'))[:14], flush=True)
    objk=[k for k in raw if k.endswith('_pos') and not k.startswith('robot0')]
    eef0=np.array(raw['robot0_eef_pos'],float)
    rec={'trial':trial,'eef':[],'grip':[],'cmd':[],'obj':{k:[] for k in objk},'success_step':None}
    for step in range(1,221):
        batch=pre(screen._policy_input(obs, env.task_description))
        with torch.no_grad(): act=policy.select_action(batch)
        act=post(act); an=np.asarray(act.squeeze(0).float().cpu() if torch.is_tensor(act) else act,float)
        obs,_,term,_,info=env.step(an)
        r=env._env.env._get_observations()
        rec['eef'].append((np.array(r['robot0_eef_pos'],float)-eef0).tolist())
        rec['grip'].append(float(np.sum(r['robot0_gripper_qpos'])))
        rec['cmd'].append(float(np.abs(an[:3]).sum()))
        for k in objk: rec['obj'][k].append(np.array(r[k],float).tolist())
        if info.get('is_success') and rec['success_step'] is None: rec['success_step']=step
        if term or info.get('is_success'): break
    env.close()
    e=np.array(rec['eef']); n=len(e)
    # stuck test: measured motion in the LAST 2 s vs commanded over the same window
    w=min(40,n); late=np.linalg.norm(e[-1]-e[-w],axis=-1) if n>w else 0.0
    cmd_late=float(np.sum(rec['cmd'][-w:]))
    zs={k:(min(v[2] for v in vv), max(v[2] for v in vv)) for k,vv in rec['obj'].items()}
    lift=max((hi-lo) for lo,hi in zs.values()) if zs else 0.0
    rec['summary']={'steps':n,'success_step':rec['success_step'],
                    'net_disp_m':float(np.linalg.norm(e[-1])),
                    'late_2s_disp_m':float(late),'late_2s_cmd_sum':cmd_late,
                    'max_obj_lift_m':float(lift),
                    'grip_min':min(rec['grip']),'grip_max':max(rec['grip'])}
    print('[trial %d] %s' % (trial, rec['summary']), flush=True)
    out.append(rec)
json.dump([{'trial':r['trial'],'summary':r['summary']} for r in out], open(a.out,'w'), indent=1)
print('wrote', a.out)
