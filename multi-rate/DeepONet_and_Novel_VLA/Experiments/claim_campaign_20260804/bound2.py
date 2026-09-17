"""Per-suite spline-vs-exact bound, on REAL contexts.

The 0.75% commanded-path bound that closed the decoding route was measured on libero_spatial
only. If some suite's chunks are less smooth, spline's error is larger there and a win becomes
physically possible. Rate is set via head.runtime_rate_hz (not a forward kwarg).
"""
import os, sys
import numpy as np
os.environ.setdefault('DEEPONET_HEAD','asrc'); os.environ.setdefault('DEEPONET_FOURIER','6')
os.environ.setdefault('DEEPONET_STATE_HISTORY_STEPS','8'); os.environ.setdefault('DEEPONET_P','256')
os.environ.setdefault('DEEPONET_TRUNK_BANDLIMIT','1')
sys.path.insert(0,'/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804')
import torch
import evaluate_multirate_honest as honest, evaluate_height_screen as screen
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

SUITES=[("spatial","asrc","libero_spatial"),("object","asrc_object","libero_object"),
        ("goal","asrc_goal","libero_goal"),("long","asrc_long","libero_10")]
TASKS=3; STEPS=12
print(f"{'suite':<9}{'ctx':>5}{'mean path':>11}{'max':>8}{'endpoint':>10}   spline-vs-exact 20->40Hz")
print("-"*66)
for name,mkey,suite in SUITES:
    ck=honest.MODELS.get(mkey,[None,None])[1]
    if not ck or not os.path.isdir(ck): print(f"{name:<9} checkpoint missing"); continue
    honest.patch_control_freq(20)
    screen.SUITE=suite; screen.MAX_STEPS=220; screen.REPLAN=10; screen.CONTROL_FREQ=20
    screen.DATASET=honest.MODEL_DATASET.get(mkey, honest.DEFAULT_DATASET)
    policy,(pre,post)=screen.load_policy('deeponet',ck,LeRobotDatasetMetadata(screen.DATASET).stats)
    head=policy.model.deeponet
    E=[];M=[];D=[]
    for task in range(TASKS):
        env=screen._make_env(task); env.num_steps_wait=honest.SETTLE_STEPS_20HZ
        env.init_state_id=0; obs,_=env.reset(seed=1000); policy.reset()
        for step in range(STEPS):
            batch=pre(screen._policy_input(obs, env.task_description))
            with torch.no_grad():
                prefix,mask=policy.model.encode_prefix(batch) if hasattr(policy.model,'encode_prefix') else (None,None)
                if prefix is None: break
                head.runtime_rate_hz=20.0; A20=head(prefix,mask).squeeze(0).float().cpu().numpy()
                head.runtime_rate_hz=40.0; A40=head(prefix,mask).squeeze(0).float().cpu().numpy()
                head.runtime_rate_hz=20.0
                act=policy.select_action(batch)
            t20=np.linspace(0,1,len(A20)); t40=np.linspace(0,1,len(A40))
            SP=np.stack([np.interp(t40,t20,A20[:,d]) for d in range(A20.shape[1])],1)
            p_sp=np.cumsum(SP[:,:3],0); p_ex=np.cumsum(A40[:,:3],0)
            sc=np.linalg.norm(p_ex[-1])+1e-9
            e=np.linalg.norm(p_sp-p_ex,axis=1)/sc
            E.append(e.mean()); M.append(e.max()); D.append(np.linalg.norm(p_sp[-1]-p_ex[-1])/sc)
            an=np.asarray(post(act).squeeze(0).float().cpu(),float)
            obs,_,term,_,info=env.step(an)
            if term or info.get('is_success'): break
        env.close()
    if E: print(f"{name:<9}{len(E):>5}{np.mean(E)*100:>10.2f}%{np.mean(M)*100:>7.2f}%{np.mean(D)*100:>9.2f}%")
    else: print(f"{name:<9} no contexts captured")
