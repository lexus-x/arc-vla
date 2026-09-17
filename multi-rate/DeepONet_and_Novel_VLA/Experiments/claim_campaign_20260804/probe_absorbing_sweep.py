"""How prevalent is the absorbing zero-action state across LIBERO-Spatial?

Bounds the prize for an escape-from-fixed-point fix BEFORE any GPU is spent on one, which is the
counter-trap that turned rate conversion from an open loop into a closed result.

Classification uses a WITHIN-TASK, data-derived boundary -- no arbitrary constant: a failure is
'absorbing' if its commanded magnitude over the final 2 s falls below the minimum observed among
that task's own successes. Tasks with no successes report 'no in-task control' and are excluded
from the rate rather than guessed at.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
os.environ.setdefault('DEEPONET_HEAD','asrc'); os.environ.setdefault('DEEPONET_FOURIER','6')
os.environ.setdefault('DEEPONET_STATE_HISTORY_STEPS','8'); os.environ.setdefault('DEEPONET_P','256')
sys.path.insert(0,'/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804')
import torch
import evaluate_multirate_honest as honest, evaluate_height_screen as screen
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
honest.patch_control_freq(20)
screen.SUITE='libero_spatial'; screen.MAX_STEPS=220; screen.REPLAN=10; screen.CONTROL_FREQ=20
policy,(pre,post)=screen.load_policy('deeponet', honest.MODELS['asrc'][1],
                                     LeRobotDatasetMetadata(screen.DATASET).stats)
TRIALS=5; out={}
for task in range(10):
    rows=[]
    for trial in range(TRIALS):
        env=screen._make_env(task); env.num_steps_wait=honest.SETTLE_STEPS_20HZ
        env.init_state_id=trial; assert env.init_state_id==trial
        obs,_=env.reset(seed=1000+trial); policy.reset()
        eef0=np.array(env._env.env._get_observations()['robot0_eef_pos'],float)
        E=[]; C=[]; ok=False
        for step in range(1,221):
            batch=pre(screen._policy_input(obs, env.task_description))
            with torch.no_grad(): act=policy.select_action(batch)
            act=post(act); an=np.asarray(act.squeeze(0).float().cpu() if torch.is_tensor(act) else act,float)
            obs,_,term,_,info=env.step(an)
            E.append(np.array(env._env.env._get_observations()['robot0_eef_pos'],float))
            C.append(float(np.abs(an[:3]).sum()))
            if info.get('is_success'): ok=True; break
            if term: break
        env.close()
        E=np.array(E); n=len(E)
        rows.append(dict(ok=ok, steps=n, cmd_late=float(np.sum(C[-40:])),
                         late=float(np.linalg.norm(E[-1]-E[max(0,n-40)])),
                         net=float(np.linalg.norm(E[-1]-eef0))))
    S=[r for r in rows if r['ok']]; F=[r for r in rows if not r['ok']]
    thr = min(r['cmd_late'] for r in S) if S else None
    absorbing = sum(1 for r in F if thr is not None and r['cmd_late'] < thr)
    out[task]=dict(n=len(rows), succ=len(S), fail=len(F), thr=thr, absorbing=absorbing,
                   fail_cmd=[round(r['cmd_late'],2) for r in F],
                   succ_cmd=[round(r['cmd_late'],2) for r in S])
    print('task %d: %d/%d succ | fails=%d | in-task thr=%s | absorbing=%s | fail_cmd=%s succ_cmd=%s'
          %(task,len(S),len(rows),len(F),'n/a' if thr is None else round(thr,2),
            'n/a' if thr is None else absorbing, out[task]['fail_cmd'], out[task]['succ_cmd']), flush=True)
json.dump(out, open('probe_absorbing_sweep.json','w'), indent=1)
tf=sum(v['fail'] for v in out.values() if v['thr'] is not None)
ta=sum(v['absorbing'] for v in out.values() if v['thr'] is not None)
nocontrol=[k for k,v in out.items() if v['thr'] is None]
print('\n=== absorbing rate among CLASSIFIABLE failures: %d/%d = %.0f%% ==='%(ta,tf,100*ta/max(1,tf)))
print('tasks with no in-task control (excluded, not guessed):', nocontrol)
print('total failures incl. unclassifiable: %d of %d rollouts'%(sum(v['fail'] for v in out.values()), sum(v['n'] for v in out.values())))
