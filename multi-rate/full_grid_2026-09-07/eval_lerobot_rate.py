"""Rate-conversion eval of official LeRobot hub checkpoints (Meta-World, ...) through lerobot_eval itself.

Usage (all normal lerobot_eval flags pass through; rate args come from env vars):
  RATE_K=4 RATE_ARM=zoh RATE_HOLD=1 MUJOCO_GL=egl python eval_lerobot_rate.py \
      --policy.path=lerobot/smolvla_metaworld --env.type=metaworld --env.task=push-v3 \
      --policy.empty_cameras=2 --rename_map='{"observation.image": "observation.images.camera1"}' \
      --eval.n_episodes=50 --eval.batch_size=10 --seed=5000 --output_dir=out

Each predicted chunk is unnormalized, clipped to [-1, 1] (the env's action box), passed through
harness.apply_arm (delta dims decimated at k and reconstructed, the RATE_HOLD trailing dims causal-held),
then renormalized. For delta-action envs only (Meta-World). Push-T/ALOHA use absolute targets: see
eval_pusht_hub_resamplers.py.
Pairing: flow/diffusion noise is drawn from a generator seeded by RATE_NOISE_SEED + call index and reset
with the policy, so every arm sees the same noise sequence on the same env seeds.
"""
import os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import harness  # pinned resample_math / resample_qp live next to it
sys.argv = _argv
import lerobot.scripts.lerobot_eval as le

K = int(os.environ.get("RATE_K", "1"))
ARM = os.environ.get("RATE_ARM", "native")
N_HOLD = int(os.environ.get("RATE_HOLD", "1"))
ABS = os.environ.get("RATE_ABS", "0") == "1"  # absolute targets (ALOHA): anchor = current observation.state
HOLD_IDX = [int(i) for i in os.environ.get("RATE_HOLD_IDX", "").split(",") if i]  # ABS mode: held dims (grippers)
ANGLE_IDX = [int(i) for i in os.environ.get("RATE_ANGLE_IDX", "").split(",") if i]  # ABS mode: radian dims to
# np.unwrap before diff (e.g. Euler orientation). Without this, a real target that crosses the +-pi seam
# (e.g. 3.1 -> -3.1, a small physical rotation) looks like a ~2pi jump: spline/TAC-Fold smooth straight
# through it (a fake wrist flip that masks any native-vs-converter gap), and QP-anchor's |v|<=1 box silently
# clips it. Unwrapping first makes the diff reflect the true small rotation, so no arm ever sees the jump.
NOISE_SEED = int(os.environ.get("RATE_NOISE_SEED", "20260928"))
harness.K = K


def convert(chunk, mean, std, anchor=None):
    """(B, T, A) normalized chunk -> same chunk after the rate arm, still normalized.

    Delta mode: clip to the [-1, 1] action box, harness.apply_arm (last N_HOLD dims causal-held).
    ABS mode (anchor = raw current state, (B, A)): continuous dims go through their displacement path
    (as eval_pusht_hub_resamplers.py), HOLD_IDX dims causal-hold the absolute target.
    """
    raw = chunk * std + mean
    if not ABS:
        out = np.stack([harness.apply_arm(c, ARM, N_HOLD) for c in np.clip(raw, -1.0, 1.0)])
        return (out - mean) / std
    cont = [i for i in range(raw.shape[2]) if i not in HOLD_IDX]
    ang_pos = [cont.index(i) for i in ANGLE_IDX if i in cont]  # position of angle dims within `cont`
    out = raw.copy()
    for b in range(len(raw)):
        seq = np.concatenate([anchor[b, cont][None], raw[b][:, cont]])
        if ang_pos:
            seq[:, ang_pos] = np.unwrap(seq[:, ang_pos], axis=0)
        d = np.diff(seq, axis=0)
        out[b][:, cont] = anchor[b, cont] + np.cumsum(harness.apply_arm(d, ARM, 0), axis=0)
        T = len(raw[b]); nb = T // harness.K
        held = np.repeat(raw[b][:nb * harness.K:harness.K], harness.K, 0)  # causal hold at block starts
        out[b][:nb * harness.K, HOLD_IDX] = held[:, HOLD_IDX]
    return (out - mean) / std


def wrap(policy):
    calls = [0]
    name = "_get_action_chunk" if hasattr(policy, "_get_action_chunk") else "predict_action_chunk"  # SmolVLA / ACT, DP
    orig_reset, orig_get = policy.reset, getattr(policy, name)

    def reset():
        calls[0] = 0
        return orig_reset()

    def sample_noise(shape, device):
        g = torch.Generator(device=device).manual_seed(NOISE_SEED + calls[0])
        calls[0] += 1
        return torch.randn(shape, generator=g, device=device, dtype=torch.float32)

    def get_chunk(batch, noise=None, **kw):
        a = orig_get(batch, noise, **kw) if name == "_get_action_chunk" else orig_get(batch, **kw)
        if ARM == "native" or K == 1:
            return a
        anchor = None
        if ABS:
            anchor = batch["observation.state"].float().cpu().numpy().reshape(len(a), -1) * wrap.s_std + wrap.s_mean
        out = convert(a.float().cpu().numpy(), wrap.mean, wrap.std, anchor)
        if calls[0] <= 1:
            print(f"[rate] chunk {tuple(a.shape)} max|converted-raw|={np.abs(out - a.float().cpu().numpy()).max():.4f}", flush=True)
        return torch.from_numpy(out.astype(np.float32)).to(a.device, a.dtype)

    policy.reset = reset
    setattr(policy, name, get_chunk)
    if hasattr(policy.model, "sample_noise"):  # flow/diffusion noise; ACT is deterministic at inference
        policy.model.sample_noise = sample_noise
    return policy


_make_policy, _make_pp = le.make_policy, le.make_pre_post_processors
le.make_policy = lambda *a, **kw: wrap(_make_policy(*a, **kw))


def make_pp(*a, **kw):
    pre, post = _make_pp(*a, **kw)
    stats = next(s for s in post.steps if hasattr(s, "stats") and s.stats and "action" in s.stats)
    wrap.mean = np.asarray(stats.stats["action"]["mean"], np.float32).reshape(-1)
    wrap.std = np.asarray(stats.stats["action"]["std"], np.float32).reshape(-1)
    if ABS:
        st = next(s for s in pre.steps if hasattr(s, "stats") and s.stats and "observation.state" in s.stats)
        wrap.s_mean = np.asarray(st.stats["observation.state"]["mean"], np.float32).reshape(-1)
        wrap.s_std = np.asarray(st.stats["observation.state"]["std"], np.float32).reshape(-1)
    return pre, post


le.make_pre_post_processors = make_pp

if "--env.type=robotwin" in sys.argv:
    # Official RoboTwin eval (script/eval_policy.py) skips seeds whose scene is unstable; lerobot crashes instead.
    # Skip to s + 100000*j: decided by (task, seed) before any action, so every arm skips the same seeds.
    import lerobot.envs.robotwin as rw
    _rw_reset = rw.RoboTwinEnv.reset

    def _rw_safe_reset(self, seed=None, **kw):
        s = self.episode_index if seed is None else seed
        for j in range(20):
            try:
                return _rw_reset(self, seed=s + 100000 * j, **kw)
            except Exception as e:
                if type(e).__name__ != "UnStableError":
                    raise
                print(f"[rate] unstable seed {s + 100000 * j}, skipping", flush=True)
                self._env.close_env()
        raise RuntimeError(f"20 unstable seeds from {s}")

    rw.RoboTwinEnv.reset = _rw_safe_reset

    # RoboTwin renders with the ray-tracing shader + OIDN denoiser, but sapien's bundled OIDN 2.0.1 has no sm_120
    # (Blackwell) kernels: it errors and leaves frames grainy (off the training distribution). OptiX denoises instead.
    import sapien.render as _sr
    _set_dn = _sr.set_ray_tracing_denoiser
    _sr.set_ray_tracing_denoiser = lambda name: _set_dn("optix" if name == "oidn" else name)

    # lerobot 0.6.2 hardcodes 10 video episodes per run; RoboTwinEnv.render() re-renders all 3 ray-traced cameras
    # every step, so videos cost ~17% of the GPU-bound screen. Off.
    _epa = le.eval_policy_all
    le.eval_policy_all = lambda *a, **kw: _epa(*a, **{**kw, "max_episodes_rendered": 0})


def self_check():
    """k=1 is the identity; zoh at k=2 keeps every block sum (delta) / block-end target (ABS)."""
    rng = np.random.default_rng(0)
    c = rng.uniform(-0.9, 0.9, (1, 8, 4)).astype(np.float32)
    mean, std = np.zeros(4, np.float32), np.ones(4, np.float32)
    global ARM, ABS, HOLD_IDX
    arm0, abs0, hold0, ARM, ABS = ARM, ABS, HOLD_IDX, "zoh", False
    harness.K = 1
    assert np.allclose(convert(c, mean, std), c)
    harness.K = 2
    z = convert(c, mean, std)
    assert np.allclose(z[0, :, :3].reshape(4, 2, 3).sum(1), c[0, :, :3].reshape(4, 2, 3).sum(1), atol=1e-5)
    assert np.allclose(z[0, 1::2, 3], c[0, 0::2, 3])  # gripper causal-held
    ABS, HOLD_IDX = True, [3]
    anchor = rng.uniform(-0.5, 0.5, (1, 4)).astype(np.float32)
    z = convert(c, mean, std, anchor)
    assert np.allclose(z[0, 1::2, :3], c[0, 1::2, :3], atol=1e-5)  # block-end targets kept
    assert np.allclose(z[0, 1::2, 3], c[0, 0::2, 3])

    global ANGLE_IDX
    ang0 = ANGLE_IDX
    ANGLE_IDX = [1]  # dim 1 is a radian angle that wraps the +-pi seam between anchor and the chunk
    true_phys = np.linspace(3.10, 3.10 + 0.4, 9)  # real continuous +0.4 rad rotation, anchor then 8-step chunk
    wrapped = (true_phys + np.pi) % (2 * np.pi) - np.pi  # what a wrapped policy output actually looks like
    anchor2 = np.array([[0.0, wrapped[0], 0.0, 0.0]], dtype=np.float32)
    c2 = c.copy(); c2[0, :, 1] = wrapped[1:]
    assert np.abs(np.diff(wrapped)).max() > 6  # sanity: the wrapped input really does jump ~2*pi somewhere
    z2 = convert(c2, mean, std, anchor2)
    # ZOH exactly reconstructs the true (unwrapped) physical block-end targets, not the wrapped ones
    assert np.allclose(z2[0, 1::2, 1], true_phys[2::2], atol=1e-4)
    # no fake ~2*pi jump anywhere in the reconstructed path
    assert np.abs(np.diff(np.concatenate([anchor2[0, 1:2], z2[0, :, 1]]))).max() < 1.0
    ANGLE_IDX = ang0
    harness.K, ARM, ABS, HOLD_IDX = K, arm0, abs0, hold0


if __name__ == "__main__":
    self_check()
    print(f"[rate] k={K} arm={ARM} hold={N_HOLD} noise_seed={NOISE_SEED}", flush=True)
    le.main()
