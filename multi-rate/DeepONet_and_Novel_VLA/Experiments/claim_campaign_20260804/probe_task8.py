"""Task-8 failure probe. Implements wiki/deeponet-vla/task8-probe-plan-2026-08-11.md.

Task 8 is the only non-saturated Spatial task (6-9/15), so successes and failures come from the
SAME task, same init-state pool, same run -- the success control is within-task. Task 5 needed a
cross-task control and three candidate metrics died to it.

Metrics are anchored on a defined EVENT (first gripper-close command) and reported for EVERY
object, never selected by proximity. Pre-registered kill rule: any metric whose failure-class
range overlaps its success-class range is discarded on the spot. No post-hoc metric selection.
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
ap=argparse.ArgumentParser(); ap.add_argument('--task_id',type=int,default=8)
ap.add_argument('--trials',type=int,default=15); a=ap.parse_args()
honest.patch_control_freq(20)
screen.SUITE='libero_spatial'; screen.MAX_STEPS=220; screen.REPLAN=10; screen.CONTROL_FREQ=20
policy,(pre,post)=screen.load_policy('deeponet', honest.MODELS['asrc'][1],
                                     LeRobotDatasetMetadata(screen.DATASET).stats)
recs=[]
for trial in range(a.trials):
    env=screen._make_env(a.task_id); env.num_steps_wait=honest.SETTLE_STEPS_20HZ
    env.init_state_id=trial; assert env.init_state_id==trial
    obs,_=env.reset(seed=1000+trial); policy.reset()
    r0=env._env.env._get_observations()
    objs=[k for k in r0 if k.endswith('_pos') and not k.startswith('robot0') and 'to_robot0' not in k]
    p0={k:np.array(r0[k],float) for k in objs}; eef0=np.array(r0['robot0_eef_pos'],float)
    close_at=None; snap=None; ok=False
    eefs=[]; cmds=[]; hist={k:[] for k in objs}
    for step in range(1,221):
        batch=pre(screen._policy_input(obs, env.task_description))
        with torch.no_grad(): act=policy.select_action(batch)
        act=post(act); an=np.asarray(act.squeeze(0).float().cpu() if torch.is_tensor(act) else act,float)
        obs,_,term,_,info=env.step(an)
        r=env._env.env._get_observations()
        e=np.array(r['robot0_eef_pos'],float); eefs.append(e); cmds.append(float(np.abs(an[:3]).sum()))
        for k in objs: hist[k].append(np.array(r[k],float))
        if close_at is None and an[6] > 0.0:
            close_at=step
            snap={k:(float(np.linalg.norm(e[:2]-np.array(r[k],float)[:2])), float(e[2]-r[k][2])) for k in objs}
        if info.get('is_success'): ok=True; break
        if term: break
    env.close()
    E=np.array(eefs); n=len(E)
    lift20={}; disp={}
    for k in objs:
        H=np.array(hist[k])
        lift20[k]=float(H[min(close_at+20,n)-1][2]-H[close_at-1][2]) if close_at else 0.0
        disp[k]=float(np.linalg.norm(H[-1]-p0[k]))
    recs.append(dict(trial=trial, ok=ok, steps=n, at_cap=(n>=220 and not ok), close_at=close_at,
                     net=float(np.linalg.norm(E[-1]-eef0)),
                     late=float(np.linalg.norm(E[-1]-E[max(0,n-40)])),
                     cmd_late=float(np.sum(cmds[-40:])),
                     # commanded vs delivered over the final 2 s: high command with ~zero motion is
                     # a SATURATED/stuck controller, which is the only case a state-feedback
                     # mechanism can act on. Low command means the policy itself stopped asking.
                     eff_late=float(np.linalg.norm(E[-1]-E[max(0,n-40)])/max(1e-6,np.sum(cmds[-40:]))),
                     snap=snap, lift20=lift20, disp=disp, objs=objs))
    print('[t%d trial %02d] %s steps=%d close@%s net=%.3f late2s=%.3f' %
          (a.task_id,trial,'OK  ' if ok else 'FAIL',n,close_at,recs[-1]['net'],recs[-1]['late']), flush=True)

S=[r for r in recs if r['ok']]; F=[r for r in recs if not r['ok']]
print('\n=== task %d: %d success / %d fail  (within-task control) ===' % (a.task_id,len(S),len(F)))
def rng(rs,f):
    v=[f(r) for r in rs if f(r) is not None]
    return (min(v),max(v),sum(v)/len(v)) if v else None
def show(name,f,fmt='%.3f'):
    s,fl=rng(S,f),rng(F,f)
    if not s or not fl: print('  %-26s insufficient data'%name); return
    overlap = not (s[1] < fl[0] or fl[1] < s[0])
    print(('  %-26s succ=['+fmt+','+fmt+'] mean '+fmt+'   fail=['+fmt+','+fmt+'] mean '+fmt+'   %s')
          %(name,s[0],s[1],s[2],fl[0],fl[1],fl[2],'OVERLAP -> KILLED' if overlap else '*** SEPARATES ***'))
show('steps to first close', lambda r: r['close_at'], '%.0f')
show('episode length',       lambda r: r['steps'], '%.0f')
show('net eef displacement', lambda r: r['net'])
show('eef disp in final 2s', lambda r: r['late'])
show('CMD magnitude final 2s',lambda r: r['cmd_late'], '%.2f')
show('delivered per cmd unit', lambda r: r['eff_late'], '%.5f')
allobj=recs[0]['objs']
for k in allobj:
    show('lift20 '+k.replace('_pos',''),  lambda r,k=k: r['lift20'].get(k))
for k in allobj:
    show('total disp '+k.replace('_pos',''), lambda r,k=k: r['disp'].get(k))
print('  fail_at_cap=%d of %d failures' % (sum(1 for r in F if r['at_cap']), len(F)))
