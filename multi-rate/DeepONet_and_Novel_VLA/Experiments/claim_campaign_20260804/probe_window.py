"""Where is cubic spline actually weak? Four regimes, forward passes only, REAL contexts.

The 0.75% bound that closed the decoding route was measured over the WHOLE 50-step chunk on
libero_spatial at a 2x upsample. Three of those choices may be flattering the baseline:

  A. EXECUTED WINDOW. REPLAN_S=0.5 -> only the first 10 of 50 steps is ever executed. SciPy's
     default not-a-knot boundary makes the first two intervals share one cubic, and that is the
     least accurate region of the spline. Averaging over all 50 steps hides it.
  B. SUB-NATIVE RATE. At 10 Hz the chunk must be DOWNsampled. Spline point-samples its
     interpolant and aliases content above 5 Hz; folding integrates the cell (an average).
     Point-sampling a delta sequence is simply the wrong operator.
  C. KNOT COUNT. Error is O(h^4) in KNOT spacing. Truncating the chunk to K knots raises h to
     1/(K-1); K=8 should be ~2400x worse than K=50 if the bound is tight.

Ground truth for "exact" is RAI: integrate at the target dt with the trunk's rate token PINNED
to 20 Hz. That reproduces the native 20 Hz trajectory resolved on the target grid, which is
exactly what an ideal rate converter would emit. Both arms are compared in RAW action space on
cumulative displacement -- never on summed normalized actions, which accumulate the offset with
step count.

Contexts are captured from REAL observations via a forward hook on a short rollout, not
torch.randn: the field is only meaningful at branch codes the policy actually produces.

PRE-REGISTERED, fixed before the run:
  KILL   executed-window error < 2x the full-chunk error in every suite  -> axis A is dead.
  KILL   10 Hz error < 3%                                               -> axis B is dead.
  GO     any regime where the gap exceeds 5% of commanded path          -> build the arm.
A gap only counts if it appears on REAL contexts. Random contexts are reported separately and
are never the basis for a GO.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

os.environ.setdefault('DEEPONET_HEAD', 'asrc')
os.environ.setdefault('DEEPONET_FOURIER', '6')
os.environ.setdefault('DEEPONET_STATE_HISTORY_STEPS', '8')
os.environ.setdefault('DEEPONET_P', '256')
BASE = '/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804'
sys.path.insert(0, BASE)

import torch
import evaluate_multirate_honest as honest
import evaluate_height_screen as screen
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

N_CTX_PER_TASK = 2
TASKS = [0, 3, 5, 8]
REPLAN_20 = 10                      # 0.5 s at 20 Hz -- the only part ever executed


def capture_contexts(suite, model_key):
    """Run a few real env steps and hook the head to record its actual branch inputs."""
    honest.patch_control_freq(20)
    screen.SUITE = suite
    screen.MAX_STEPS = 40
    screen.REPLAN = REPLAN_20
    screen.CONTROL_FREQ = 20
    screen.DATASET = honest.MODEL_DATASET.get(model_key, honest.DEFAULT_DATASET)
    ck = honest.MODELS[model_key][1]
    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (pre, post) = screen.load_policy('deeponet', ck, stats)

    from rate_integrated_deeponet import RateIntegratedDeepONetHead
    head = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)][0]

    grabbed = []
    orig_forward = head.forward

    def hook(prefix, pad_mask):
        grabbed.append((prefix.detach().clone(), pad_mask.detach().clone()))
        return orig_forward(prefix, pad_mask)

    head.forward = hook
    for task in TASKS:
        env = screen._make_env(task)
        env.num_steps_wait = honest.SETTLE_STEPS_20HZ
        env.init_state_id = 0
        obs, _ = env.reset(seed=1000)
        policy.reset()
        want = len(grabbed) + N_CTX_PER_TASK
        for _ in range(40):
            if len(grabbed) >= want:
                break
            batch = pre(screen._policy_input(obs, env.task_description))
            with torch.no_grad():
                act = policy.select_action(batch)
            a = post(act)
            a = np.asarray(a.squeeze(0).float().cpu() if torch.is_tensor(a) else a, float)
            obs, _, term, _, _ = env.step(a)
            if term:
                break
        env.close()
    head.forward = orig_forward
    return head, grabbed


def rollout(head, prefix, mask, rate, anchor, horizon_s=None):
    """RAW (unnormalized) action rows. anchor=True pins the trunk rate token to 20 Hz."""
    saved = head._feature
    if anchor:
        def anchored(c, pose, t, rate_hz, _f=saved):
            return _f(c, pose, t, 20.0)
        head._feature = anchored
    try:
        with torch.no_grad():
            ctx = head.pool(prefix, mask)
            c = head.branch(ctx.flatten(1))
            raw = head._raw_rollout(c, float(rate), horizon_s)
    finally:
        head._feature = saved
    return raw.float().cpu().numpy()[0]          # (T, A) raw


def spline_like_production(head, raw20, target_len):
    """Exactly what the harness does: normalize, resample_delta_chunk, back to raw."""
    off = head.action_offset.detach().float().cpu().numpy()
    sc = head.action_scale.detach().float().cpu().numpy()
    norm = (raw20 - off) / sc
    t = torch.from_numpy(norm[None]).float()
    out = honest.resample_delta_chunk(t, target_len, raw20.shape[0],
                                      head.action_offset.detach().float().cpu(),
                                      head.action_scale.detach().float().cpu())
    return out.numpy()[0] * sc + off


def path_err(a, b, upto=None):
    """Relative L2 between cumulative displacement paths, pose dims only, RAW space."""
    n = upto or a.shape[0]
    pa = np.cumsum(a[:n, :6], axis=0)
    pb = np.cumsum(b[:n, :6], axis=0)
    denom = np.linalg.norm(pa) + 1e-12
    return 100.0 * np.linalg.norm(pa - pb) / denom


SUITES = [("spatial", "asrc", "libero_spatial"), ("object", "asrc_object", "libero_object"),
          ("goal", "asrc_goal", "libero_goal"), ("long", "asrc_long", "libero_10")]
report = {}

for name, mkey, suite in SUITES:
    if mkey not in honest.MODELS or not os.path.isdir(honest.MODELS[mkey][1]):
        print(f"{name}: checkpoint missing, skipped", flush=True)
        continue
    head, ctxs = capture_contexts(suite, mkey)
    if not ctxs:
        print(f"{name}: no contexts captured", flush=True)
        continue

    rows = {"full40": [], "win40": [], "down10": [], "k8": [], "k16": []}
    for prefix, mask in ctxs:
        raw20 = rollout(head, prefix, mask, 20, anchor=False)
        n20 = raw20.shape[0]

        # --- A. upsample 20 -> 40, full chunk vs executed window -------------------
        exact40 = rollout(head, prefix, mask, 40, anchor=True)
        spl40 = spline_like_production(head, raw20, exact40.shape[0])
        rows["full40"].append(path_err(spl40, exact40))
        rows["win40"].append(path_err(spl40, exact40, upto=REPLAN_20 * 2))

        # --- B. downsample 20 -> 10 ------------------------------------------------
        exact10 = rollout(head, prefix, mask, 10, anchor=True)
        spl10 = spline_like_production(head, raw20, exact10.shape[0])
        rows["down10"].append(path_err(spl10, exact10))

        # --- C. knot count: pretend the policy emitted a shorter chunk -------------
        for K, key in ((8, "k8"), (16, "k16")):
            hs = K / 20.0
            short20 = rollout(head, prefix, mask, 20, anchor=False, horizon_s=hs)
            shortE = rollout(head, prefix, mask, 40, anchor=True, horizon_s=hs)
            shortS = spline_like_production(head, short20, shortE.shape[0])
            rows[key].append(path_err(shortS, shortE))

    report[name] = {k: (float(np.mean(v)), float(np.max(v))) for k, v in rows.items() if v}
    m = report[name]
    print(f"\n=== {name} (n_ctx={len(ctxs)}, chunk={n20}) ===", flush=True)
    print(f"  A full chunk   20->40 : {m['full40'][0]:6.2f}%  (max {m['full40'][1]:.2f}%)")
    print(f"  A EXECUTED win 20->40 : {m['win40'][0]:6.2f}%  (max {m['win40'][1]:.2f}%)"
          f"   <-- ratio {m['win40'][0]/max(m['full40'][0],1e-9):.2f}x")
    print(f"  B downsample   20->10 : {m['down10'][0]:6.2f}%  (max {m['down10'][1]:.2f}%)")
    print(f"  C 16-step chunk       : {m['k16'][0]:6.2f}%")
    print(f"  C  8-step chunk       : {m['k8'][0]:6.2f}%")

json.dump(report, open(os.path.join(BASE, 'spline_weakness_probe.json'), 'w'), indent=2)
print("\nwrote spline_weakness_probe.json", flush=True)
print("\nGATES: axis A alive if executed/full >= 2x; axis B alive if 10 Hz >= 3%; "
      "GO on any regime >= 5%.", flush=True)
