"""Prove (or refute) two folding-vs-spline claims, offline, on real contexts.

CLAIM 1 -- EXACT INVARIANCE.
  Folding emits a_k = integral of v over [k/r, (k+1)/r], so sum_k a_k = s(H) for EVERY r, exactly.
  Spline resamples a delta sequence and multiplies by native_len/target_len, which conserves the
  sum only to quadrature accuracy. Test: total commanded displacement at rate r against the
  native 20 Hz total. Folding should be ~1e-6 (float noise). Spline should not.
  Reported for r in {10, 25, 30, 40, 50} -- one number per rate, not a single cherry-picked one.

CLAIM 2 -- SUB-NATIVE ALIASING.
  Going DOWN to 10 Hz, one 10 Hz action must cover the ground that two 20 Hz actions covered.
  The correct operator is the SUM of the two deltas it replaces (a cell integral). Spline instead
  POINT-SAMPLES its interpolant and scales by native_len/target_len -- that is a sample, not an
  average, so any content above r/2 folds back. Test: compare each method against the exact cell
  sum, and correlate the error with the trajectory's own high-frequency energy. If the claim is
  real, spline's error should grow with high-frequency content and folding's should not.

CLAIM 3 is NOT tested here because it is false against this baseline: cubic spline resamples to
any target length, so non-integer rates are not a folding advantage over spline (only over
integer zero-order hold).

Ground truth for "exact" is RAI: integrate at the target dt with the trunk rate token PINNED to
20 Hz, which reproduces the native trajectory on the target grid. All comparisons are in RAW
action space on pose dims 0:6 -- never on summed normalized actions.

HONEST NOTE ON WHAT THIS CAN SHOW: these are commanded-trajectory properties. LIBERO's OSC does
not deliver displacement proportional to the commanded integral (that is why magscale exists),
so an exact commanded invariant does NOT imply an exact delivered one. A win here is a property
claim, not a success-rate claim.
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

TASKS = [0, 2, 5, 7, 9]
N_PER_TASK = 3
RATES = [10, 25, 30, 40, 50]


def capture(suite, model_key):
    honest.patch_control_freq(20)
    screen.SUITE, screen.MAX_STEPS, screen.REPLAN, screen.CONTROL_FREQ = suite, 40, 10, 20
    screen.DATASET = honest.MODEL_DATASET.get(model_key, honest.DEFAULT_DATASET)
    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (pre, post) = screen.load_policy('deeponet', honest.MODELS[model_key][1], stats)
    from rate_integrated_deeponet import RateIntegratedDeepONetHead
    head = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)][0]

    got, orig = [], head.forward

    def hook(prefix, pad_mask):
        got.append((prefix.detach().clone(), pad_mask.detach().clone()))
        return orig(prefix, pad_mask)

    head.forward = hook
    for task in TASKS:
        env = screen._make_env(task)
        env.num_steps_wait = honest.SETTLE_STEPS_20HZ
        env.init_state_id = 0
        obs, _ = env.reset(seed=1000)
        policy.reset()
        want = len(got) + N_PER_TASK
        for _ in range(40):
            if len(got) >= want:
                break
            b = pre(screen._policy_input(obs, env.task_description))
            with torch.no_grad():
                a = policy.select_action(b)
            a = post(a)
            a = np.asarray(a.squeeze(0).float().cpu() if torch.is_tensor(a) else a, float)
            obs, _, term, _, _ = env.step(a)
            if term:
                break
        env.close()
    head.forward = orig
    return head, got


def roll(head, prefix, mask, rate, anchor):
    saved = head._feature
    if anchor:
        def anc(c, pose, t, r, _f=saved):
            return _f(c, pose, t, 20.0)
        head._feature = anc
    try:
        with torch.no_grad():
            c = head.branch(head.pool(prefix, mask).flatten(1))
            raw = head._raw_rollout(c, float(rate))
    finally:
        head._feature = saved
    return raw.float().cpu().numpy()[0]


def spline(head, raw20, target_len):
    off = head.action_offset.detach().float().cpu().numpy()[:raw20.shape[1]]
    sc = head.action_scale.detach().float().cpu().numpy()[:raw20.shape[1]]
    t = torch.from_numpy(((raw20 - off) / sc)[None]).float()
    out = honest.resample_delta_chunk(t, target_len, raw20.shape[0],
                                      head.action_offset.detach().float().cpu(),
                                      head.action_scale.detach().float().cpu())
    return out.numpy()[0] * sc + off


head, ctxs = capture('libero_spatial', 'asrc')
print(f"captured {len(ctxs)} real contexts\n", flush=True)

# =================== CLAIM 1 : exact invariance ===================
print("CLAIM 1  total commanded displacement vs the native 20 Hz total")
print("         (relative L2 over pose dims, %; lower = more invariant)\n")
print(f"{'rate':>6} | {'FOLDING':>22} | {'CUBIC SPLINE':>22}")
print(f"{'':>6} | {'mean':>10}{'max':>12} | {'mean':>10}{'max':>12}")
print("-" * 58)
claim1 = {}
for r in RATES:
    fo_e, sp_e = [], []
    for prefix, mask in ctxs:
        raw20 = roll(head, prefix, mask, 20, anchor=False)
        tot20 = raw20[:, :6].sum(axis=0)
        n = float(np.linalg.norm(tot20)) + 1e-12
        tgt = int(np.ceil(2.5 * r))
        fo = roll(head, prefix, mask, r, anchor=True)[:, :6].sum(axis=0)
        sp = spline(head, raw20, tgt)[:, :6].sum(axis=0)
        fo_e.append(100 * np.linalg.norm(fo - tot20) / n)
        sp_e.append(100 * np.linalg.norm(sp - tot20) / n)
    claim1[r] = dict(folding=[float(np.mean(fo_e)), float(np.max(fo_e))],
                     spline=[float(np.mean(sp_e)), float(np.max(sp_e))])
    print(f"{r:>6} | {np.mean(fo_e):>9.4f}%{np.max(fo_e):>11.4f}% | "
          f"{np.mean(sp_e):>9.4f}%{np.max(sp_e):>11.4f}%")

# =================== CLAIM 2 : sub-native aliasing ===================
# Exact 10 Hz action = SUM of the two 20 Hz deltas it replaces (a cell integral).
print("\n\nCLAIM 2  downsampling 20 -> 10 Hz, error against the exact cell sum")
print("         hf = fraction of the 20 Hz delta signal's energy above 5 Hz (the 10 Hz Nyquist)\n")
print(f"{'ctx':>4} {'hf_energy':>11} | {'folding err':>13} {'spline err':>13}")
print("-" * 48)
rows = []
for i, (prefix, mask) in enumerate(ctxs):
    raw20 = roll(head, prefix, mask, 20, anchor=False)
    n20 = raw20.shape[0] // 2 * 2
    cell = raw20[:n20, :6].reshape(-1, 2, 6).sum(axis=1)      # the exact 10 Hz action
    tgt = cell.shape[0]
    fo = roll(head, prefix, mask, 10, anchor=True)[:tgt, :6]
    sp = spline(head, raw20, int(np.ceil(2.5 * 10)))[:tgt, :6]
    d = float(np.linalg.norm(cell)) + 1e-12
    # spectral content of the 20 Hz delta signal above the 10 Hz Nyquist (5 Hz)
    F = np.fft.rfft(raw20[:, :6] - raw20[:, :6].mean(axis=0), axis=0)
    freqs = np.fft.rfftfreq(raw20.shape[0], d=1 / 20.0)
    P = (np.abs(F) ** 2).sum(axis=1)
    hf = float(P[freqs > 5.0].sum() / (P.sum() + 1e-12))
    fe = 100 * np.linalg.norm(fo - cell) / d
    se = 100 * np.linalg.norm(sp - cell) / d
    rows.append((hf, fe, se))
    print(f"{i:>4} {hf:>10.4f}  | {fe:>12.3f}% {se:>12.3f}%")

hf = np.array([r[0] for r in rows]); fe = np.array([r[1] for r in rows]); se = np.array([r[2] for r in rows])
print("-" * 48)
print(f"{'mean':>4} {hf.mean():>10.4f}  | {fe.mean():>12.3f}% {se.mean():>12.3f}%")
if len(rows) > 2 and hf.std() > 1e-9:
    print(f"\ncorrelation of error with high-frequency energy:")
    print(f"  folding : r = {np.corrcoef(hf, fe)[0,1]:+.3f}")
    print(f"  spline  : r = {np.corrcoef(hf, se)[0,1]:+.3f}   "
          f"(claim 2 predicts this is strongly POSITIVE and larger than folding's)")

json.dump({"claim1_invariance": claim1,
           "claim2_rows": [{"hf": a, "folding": b, "spline": c} for a, b, c in rows]},
          open(os.path.join(BASE, 'claims_proof.json'), 'w'), indent=2)
print("\nwrote claims_proof.json")
