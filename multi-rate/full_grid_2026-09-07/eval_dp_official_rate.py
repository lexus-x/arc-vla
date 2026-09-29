"""Paired native/zoh eval of an official Diffusion Policy (Chi et al. 2023) low-dim checkpoint at slowdown k.

Env: conda `dp_official` (real-stanford/diffusion_policy pins: robomimic 0.2.0, robosuite @ cheng-chi 277ab95,
free-mujoco-py 2.1.6). The checkpoint is loaded exactly like the repo's eval.py (workspace cfg + payload, EMA
weights) and rolled out with the repo's own env_runner from the checkpoint cfg; only n_test / test_start_seed /
n_envs / dataset_path (env_meta source) are overridden, train-init and video episodes are switched off.

Per policy call:
  * torch noise is seeded from (noise_seed + call index); the index resets at every policy.reset(), i.e. per
    vectorized batch of episodes, so native and zoh see identical proposals while their obs agree.
  * the arm is applied to the executed n_action_steps chunk through harness.apply_arm (harness.K = k).
Action handling (robomimic OSC_POSE, per arm):
  * delta actions [dpos3, drot3 axis-angle, grip1]: chunk clipped to [-1,1] (robosuite clips anyway), the 6
    continuous dims are resampled, the gripper is causal-held.
  * absolute actions (abs_action=True, rotation_6d) [pos3, rot6d, grip1]: positions go through the absolute
    pattern (deltas = diff([anchor, chunk]); apply arm; anchor + cumsum) with anchor = the robot's measured
    eef position robot{i}_eef_pos from the newest obs frame (same world frame and metres as the action);
    rot6d + gripper (7 dims) are causal-held (value at each block start), since block sums of 6D rotations
    have no meaning. Dual-arm transport: both arms, same rule.
Success = the runner's own episode score: robomimic max sparse task reward (1.0 on task success) > 0.9.
Franka Kitchen: abs ckpt (joint positions) -> arm joints via the absolute pattern, anchor obs[0:7], fingers held;
delta ckpt -> all 9 resampled. Success = >= 4 subtasks completed (DP paper's p4).

Usage: python eval_dp_official_rate.py --ckpt X.ckpt --task can_ph --arm zoh --k 2 --n 50 --start_seed 5000 \
         --dataset_dir /media/.../data/robomimic/datasets --out dp_can_ph_zoh_k2.json
"""
import argparse, json, os, sys, tempfile
import numpy as np, torch, dill, hydra
from omegaconf import OmegaConf, open_dict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness

OmegaConf.register_new_resolver("eval", eval, replace=True)
KEY_DIMS = {"eef_pos": 3, "eef_quat": 4, "gripper_qpos": 2}


def action_layout(action_dim, abs_action):
    """-> (resampled dims, causal-held dims) for per-arm [pos3, rot, grip1] blocks."""
    per = 10 if abs_action else 7
    n_arms = action_dim // per; assert n_arms * per == action_dim, action_dim
    n_cont = 3 if abs_action else 6
    cont = [a * per + i for a in range(n_arms) for i in range(n_cont)]
    return cont, [d for d in range(action_dim) if d not in cont]


def eef_pos_idx(obs_keys, obs_dim):
    """Indices of robot{i}_eef_pos inside the concatenated low-dim obs vector."""
    dims = [next((v for s, v in KEY_DIMS.items() if k.endswith(s)), None) for k in obs_keys]
    dims = [obs_dim - sum(d for d in dims if d) if d is None else d for d in dims]  # 'object' = the rest
    off = np.cumsum([0] + dims)
    return [j for i, k in enumerate(obs_keys) if k.endswith("eef_pos") for j in range(off[i], off[i] + 3)]


def convert(chunk, arm, cont, hold, anchor=None):
    """Apply `arm` to one executed chunk (T, A). anchor=None -> delta actions, else absolute positions."""
    order = cont + hold; nc = len(cont)
    x = chunk[:, order].astype(np.float64)
    if anchor is None: x = np.clip(x, -1, 1)
    else: x[:, :nc] = np.diff(np.concatenate([anchor[None], x[:, :nc]]), axis=0)
    y = harness.apply_arm(x, arm, len(hold)).astype(np.float64)
    if anchor is not None: y[:, :nc] = anchor + np.cumsum(y[:, :nc], axis=0)
    out = np.empty(chunk.shape, np.float32); out[:, order] = y
    return out


def self_check():
    harness.K = 2
    rng = np.random.default_rng(0)
    cont, hold = action_layout(20, True); assert len(cont) == 6 and len(hold) == 14
    chunk = rng.normal(size=(8, 20)).astype(np.float32); anchor = rng.normal(size=6)
    assert np.allclose(convert(chunk, "native", cont, hold, anchor), chunk)
    z = convert(chunk, "zoh", cont, hold, anchor)
    assert np.allclose(z[1::2][:, cont], chunk[1::2][:, cont], atol=1e-5)     # block-end positions kept
    prev_end = np.vstack([anchor, chunk[1:-1:2][:, cont]])                    # anchor, c1, c3, c5
    assert np.allclose(z[0::2][:, cont], (prev_end + chunk[1::2][:, cont]) / 2, atol=1e-5)  # linear within block
    assert np.allclose(z[0::2][:, hold], chunk[0::2][:, hold]) and np.allclose(z[1::2][:, hold], chunk[0::2][:, hold])
    cont, hold = action_layout(7, False); assert hold == [6]
    d = rng.uniform(-1, 1, size=(8, 7)); zd = convert(d, "zoh", cont, hold)
    assert np.allclose(zd[0::2, :6], (d[0::2, :6] + d[1::2, :6]) / 2, atol=1e-6)
    assert eef_pos_idx(["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"], 23) == [14, 15, 16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--task", required=True)
    p.add_argument("--arm", choices=["native", "zoh", "spline_satfix", "tac_fold_satfix", "qp_anchor"], required=True); p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, default=50); p.add_argument("--start_seed", type=int, default=5000)
    p.add_argument("--n_envs", type=int, default=25); p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--device", default="cpu"); p.add_argument("--threads", type=int, default=4); p.add_argument("--dataset_dir", required=True)
    p.add_argument("--ckpt_sha256", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    self_check(); harness.K = a.k; torch.set_num_threads(a.threads)

    payload = torch.load(open(a.ckpt, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    ws = hydra.utils.get_class(cfg._target_)(cfg)  # transformer workspace takes no output_dir
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = ws.ema_model if cfg.training.use_ema else ws.model
    policy.to(torch.device(a.device)).eval()

    rc = cfg.task.env_runner
    kitchen = "kitchen" in rc._target_
    assert kitchen or "robomimic" in rc._target_, "only robomimic / kitchen runners are wired up"
    T = cfg.n_action_steps
    assert a.arm == "native" or T >= 2 * a.k, f"chunk rule: n_action_steps={T} < 2k={2 * a.k}"
    abs_action = bool(cfg.task.get("abs_action", False))
    # kitchen: 9 joint dims (7 arm + 2 finger). abs (Robot_PosAct): arm joints via the displacement path anchored at
    # the measured joint positions obs[0:7], fingers causal-held (as ALOHA). delta (velocities): all 9 resampled.
    if kitchen: cont, hold = (list(range(7)), [7, 8]) if abs_action else (list(range(cfg.action_dim)), [])
    else: cont, hold = action_layout(cfg.action_dim, abs_action)
    pos_idx = (list(range(7)) if kitchen else eef_pos_idx(list(rc.obs_keys), cfg.obs_dim)) if abs_action else None
    with open_dict(cfg):
        rc.n_train = 0; rc.n_train_vis = 0; rc.n_test = a.n; rc.n_test_vis = 0
        rc.test_start_seed = a.start_seed; rc.n_envs = min(a.n_envs, a.n)
        if kitchen: rc.dataset_dir = a.dataset_dir  # only all_init_qpos/qvel.npy are read (train inits, n_train=0)
        else: rc.dataset_path = os.path.join(a.dataset_dir, rc.dataset_path.split("datasets/", 1)[1])
    if kitchen:  # the runner logs only aggregate p_n: keep the per-episode maps it builds (2nd = n_completed)
        import collections, diffusion_policy.env_runner.kitchen_lowdim_runner as kr
        maps = []
        class _DD(collections.defaultdict):
            def __init__(self, f): super().__init__(f); maps.append(self)
        kr.collections = type("C", (), {"defaultdict": _DD})
    runner = hydra.utils.instantiate(rc, output_dir=tempfile.mkdtemp())

    calls = [0]; orig_predict, orig_reset = policy.predict_action, policy.reset

    def reset():
        calls[0] = 0; return orig_reset()

    def predict(obs_dict):
        torch.manual_seed(a.noise_seed + calls[0]); calls[0] += 1
        out = orig_predict(obs_dict)
        if a.arm == "native": return out
        act = out["action"].detach().cpu().numpy()
        anchors = obs_dict["obs"][:, -1, pos_idx].cpu().numpy().astype(np.float64) if abs_action else [None] * len(act)
        res = np.stack([convert(act[b], a.arm, cont, hold, anchors[b]) for b in range(len(act))])
        return {"action": torch.from_numpy(res).to(out["action"].device)}

    policy.reset, policy.predict_action = reset, predict
    log = runner.run(policy)
    if kitchen:  # success = DP paper's p4: >= 4 of the 7 kitchen subtasks completed; score = n completed
        scores = [float(n) for n in maps[1]["test/"]]; assert len(scores) == a.n and np.isclose(np.mean(np.array(scores) >= 4), log["test/p_4"])
        succ = [s >= 4 for s in scores]
    else:
        scores = [float(log[f"test/sim_max_reward_{a.start_seed + i}"]) for i in range(a.n)]
        succ = [s > 0.9 for s in scores]
    res = {"task": a.task, "k": a.k if a.arm != "native" else 1, "arm": a.arm, "start_seed": a.start_seed, "n": a.n,
           "successes": succ, "checkpoint": os.path.abspath(a.ckpt), "ckpt_sha256": a.ckpt_sha256,
           "max_rewards": scores, "abs_action": abs_action, "n_action_steps": T, "horizon": cfg.horizon,
           "noise_seed": a.noise_seed, "n_envs": rc.n_envs}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{a.task} {a.arm} k={res['k']}: {sum(res['successes'])}/{a.n} = {100 * np.mean(res['successes']):.1f}%")
    os._exit(0)  # AsyncVectorEnv workers can hang interpreter shutdown


if __name__ == "__main__":
    main()
