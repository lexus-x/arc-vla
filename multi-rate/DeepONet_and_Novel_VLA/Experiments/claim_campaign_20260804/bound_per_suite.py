"""Where is cubic spline WEAKEST? Measure the spline-vs-exact gap per suite.

The 0.75% commanded-path bound that closed the decoding route was measured on libero_spatial
ONLY. If a suite's action chunks are less smooth -- long-horizon multi-stage tasks are the
obvious candidate -- spline's interpolation error is larger there and a win becomes physically
possible. If the bound holds everywhere, no decoder can win and the route is closed for good.

Forward passes only. No rollouts. Compares, per context:
  spline : chunk emitted at 20 Hz, cubic-interpolated to 40 Hz
  exact  : same field re-integrated on the 40 Hz grid (what folding computes)
"""
import os, sys, json
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
N_CTX=16
print(f"{'suite':<9}{'n_ctx':>6}{'mean path err':>15}{'max':>9}{'endpoint':>10}   (spline vs exact, 20->40 Hz)")
print("-"*72)
for name,mkey,suite in SUITES:
    if mkey not in honest.MODELS: print(f"{name:<9} model {mkey} not registered"); continue
    ck=honest.MODELS[mkey][1]
    if not os.path.isdir(ck): print(f"{name:<9} checkpoint missing"); continue
    honest.patch_control_freq(20)
    screen.SUITE=suite; screen.MAX_STEPS=220; screen.REPLAN=10; screen.CONTROL_FREQ=20
    screen.DATASET=honest.MODEL_DATASET.get(mkey, honest.DEFAULT_DATASET)
    policy,(pre,post)=screen.load_policy('deeponet',ck,LeRobotDatasetMetadata(screen.DATASET).stats)
    head=policy.model.deeponet
    errs=[];maxs=[];ends=[]
    for i in range(N_CTX):
        ctx=torch.randn(1, 8, head.context_dim if hasattr(head,'context_dim') else 960,
                        device=next(head.parameters()).device,
                        dtype=next(head.parameters()).dtype)
        mask=torch.zeros(1,8,dtype=torch.bool,device=ctx.device)
        with torch.no_grad():
            a20=head(ctx,mask,rate=20.0) if 'rate' in head.forward.__code__.co_varnames else head(ctx,mask)
            a40=head(ctx,mask,rate=40.0) if 'rate' in head.forward.__code__.co_varnames else None
        if a40 is None: print(f"{name:<9} head takes no rate arg"); break
        A20=a20.squeeze(0).float().cpu().numpy(); A40=a40.squeeze(0).float().cpu().numpy()
        T20,T40=len(A20),len(A40)
        t20=np.linspace(0,1,T20); t40=np.linspace(0,1,T40)
        SP=np.stack([np.interp(t40,t20,A20[:,d]) for d in range(A20.shape[1])],1)
        p_sp=np.cumsum(SP[:,:3],0); p_ex=np.cumsum(A40[:,:3],0)
        scale=np.linalg.norm(p_ex[-1])+1e-9
        e=np.linalg.norm(p_sp-p_ex,axis=1)/scale
        errs.append(e.mean()); maxs.append(e.max()); ends.append(np.linalg.norm(p_sp[-1]-p_ex[-1])/scale)
    if errs:
        print(f"{name:<9}{len(errs):>6}{np.mean(errs)*100:>14.2f}%{np.mean(maxs)*100:>8.2f}%{np.mean(ends)*100:>9.2f}%")
