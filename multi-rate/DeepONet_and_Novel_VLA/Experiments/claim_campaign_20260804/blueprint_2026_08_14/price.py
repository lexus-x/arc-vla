"""Option 2 pricing: closed-loop trunk vs full-policy replanning. Measured, not assumed.

Counts real params per module and times them on the GPU. Key correction to the proposal:
_raw_rollout calls _feature+rhs FOUR times per step (RK4 k1..k4) plus one `direct` eval,
so the per-step operator cost is ~5 trunk passes, not 1.
"""
import json, time, torch, math
from safetensors import safe_open

CKPT = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/runs/asrc_s0/checkpoints/8300/model.safetensors"
groups = {}
with safe_open(CKPT, framework="pt") as h:
    for k in h.keys():
        n = math.prod(h.get_slice(k).get_shape())
        if   ".trunk."  in k: g = "trunk (per RK4 stage)"
        elif ".rhs."    in k: g = "rhs (per RK4 stage)"
        elif ".direct." in k: g = "direct (per step)"
        elif ".branch." in k: g = "branch (per chunk)"
        elif ".pool."   in k: g = "pool (per chunk)"
        elif "vlm" in k or "lm_head" in k: g = "BACKBONE (per replan)"
        elif "deeponet" in k: g = "head-other"
        else: g = "BACKBONE (per replan)"
        groups[g] = groups.get(g, 0) + n

tot = sum(groups.values())
print(f"{'module':26s} {'params':>14s}  {'% of total':>10s}")
for g, n in sorted(groups.items(), key=lambda x: -x[1]):
    print(f"{g:26s} {n:>14,}  {n/tot*100:>9.2f}%")
print(f"{'TOTAL':26s} {tot:>14,}\n")

per_stage = groups.get("trunk (per RK4 stage)", 0) + groups.get("rhs (per RK4 stage)", 0)
per_step  = 4*per_stage + groups.get("direct (per step)", 0)   # k1..k4 + direct
per_chunk = groups.get("branch (per chunk)", 0) + groups.get("pool (per chunk)", 0)
backbone  = groups.get("BACKBONE (per replan)", 0)
print(f"per-RK4-stage (trunk+rhs) : {per_stage:>12,}")
print(f"per-STEP (4 stages+direct): {per_step:>12,}   <-- 4x the naive 'one trunk' figure")
print(f"per-CHUNK (pool+branch)   : {per_chunk:>12,}")
print(f"BACKBONE (per replan)     : {backbone:>12,}\n")

print(f"{'regime':38s} {'params touched/sec':>20s} {'ratio vs replan':>17s}")
for exec_hz in (20, 50, 100):
    for replan_every in (1, 5):
        rp_hz = exec_hz / replan_every
        base = rp_hz * (backbone + per_chunk + per_step*replan_every)
        ours = (exec_hz/50) * (backbone + per_chunk) + exec_hz * per_step   # 1 chunk / 50 steps
        print(f"exec {exec_hz:>3} Hz, replan every {replan_every:>2} step(s) "
              f"{base:>18,.0f} {base/ours:>16.2f}x")
    print()
