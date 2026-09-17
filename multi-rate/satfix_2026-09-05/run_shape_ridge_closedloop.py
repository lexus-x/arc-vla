"""
Closed-loop, real-sim success-rate test of a shape-aware ridge corrector vs ZOH and
cubic spline, on PickCube-v1 / pick_rl_joint.h5, k=2 -- the vault's own highest-headroom
cell. Reuses the sealed harness's helper functions (coarsen_actions, resamplers,
McNemar, bootstrap CI) UNMODIFIED via import; does not touch or overwrite the sealed
file. Ridge weights are FROZEN, fit only on a disjoint train split (see
fit_shape_ridge.py) -- every episode evaluated here was never used to fit the corrector.

This is exploratory, single-run, n<900. Reporting the true result either way.
"""
import sys, json, math
import numpy as np
import h5py
import gymnasium as gym
import mani_skill.envs  # noqa
import torch

sys.path.insert(0, "/home/user/Desktop/multi-rate/vla-vault/scratch")
from sweep_maniskill_decimation_ratios import (
    coarsen_actions, resample_exact_integral, resample_cubic_spline,
    resample_pchip, resample_tac_fold, exact_mcnemar, paired_bootstrap_ci,
)

SCRATCH_DIR = "/tmp/claude-1000/-home-user-Desktop/84eef76b-33e3-4b02-8c76-473993217a1e/scratchpad"
W = json.load(open(f"{SCRATCH_DIR}/shape_ridge_weights.json"))
WEIGHTS = np.array(W["weights_per_dim"])  # (7,4): [w_sim1, w_si, w_sip1, intercept]
TEST_IDS = set(W["test_episode_ids_heldout"])
K = W["k"]

def resample_shape_ridge(delta, gripper, target_len):
    """Frozen-ridge within-block split, conservation-exact (v2 = S_i - v1)."""
    n_blocks, D = delta.shape
    assert target_len == n_blocks * K
    out = np.empty((target_len, D), dtype=np.float32)
    for i in range(n_blocks):
        s_im1 = delta[i - 1] if i - 1 >= 0 else np.zeros(D, dtype=np.float32)
        s_i = delta[i]
        s_ip1 = delta[i + 1] if i + 1 < n_blocks else np.zeros(D, dtype=np.float32)
        feats = np.stack([s_im1, s_i, s_ip1, np.ones(D, dtype=np.float32)], axis=1)  # (D,4)
        v1 = np.einsum("df,df->d", feats, WEIGHTS)
        v2 = s_i - v1
        out[i * K] = v1
        out[i * K + 1] = v2
    if gripper is not None:
        grip = np.repeat(gripper, K, axis=0)
        return np.concatenate([out, grip], axis=1).astype(np.float32)
    return out.astype(np.float32)


H5_PATH = "/home/user/maniskill_data/pick_rl_joint.h5"
JSON_PATH = "/home/user/maniskill_data/pick_rl_joint.json"
N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 200

metadata = json.loads(open(JSON_PATH).read())
eligible = [e for e in metadata["episodes"]
            if int(e["episode_id"]) in TEST_IDS]
episodes = eligible[:N_EPISODES]
print(f"held-out eligible episodes available={len(eligible)}  using n={len(episodes)}")

env = gym.make("PickCube-v1", num_envs=1, obs_mode="state",
                control_mode="pd_joint_delta_pos", sim_backend="physx_cpu")

arms = ["original", "exact_integral", "cubic_spline", "shape_ridge"]
outcomes = {arm: [] for arm in arms}
episode_ids = []

with h5py.File(H5_PATH, "r") as handle:
    for idx, episode in enumerate(episodes):
        ep_id = int(episode["episode_id"])
        actions = np.asarray(handle[f"traj_{ep_id}"]["actions"], dtype=np.float32)
        act_len = (len(actions) // K) * K
        if act_len < K:
            continue
        actions = np.clip(actions[:act_len], -1.0, 1.0).astype(np.float32)
        delta, gripper = coarsen_actions(actions, K, has_gripper=True)
        target_len = len(actions)

        arm_actions = {
            "original": actions,
            "exact_integral": resample_exact_integral(delta, gripper, target_len),
            "cubic_spline": resample_cubic_spline(delta, gripper, target_len),
            "shape_ridge": resample_shape_ridge(delta, gripper, target_len),
        }

        state_group = handle[f"traj_{ep_id}"]["env_states"]
        state = {g: {n: torch.as_tensor(np.asarray(state_group[g][n])[0:1]) for n in state_group[g]}
                  for g in state_group}

        for arm in arms:
            acts = arm_actions[arm]
            if not np.isfinite(acts).all():
                raise RuntimeError(f"non-finite actions arm={arm} ep={ep_id}")
            env.reset(seed=episode["episode_seed"])
            try:
                env.unwrapped.set_state_dict(state)
            except Exception as exc:
                raise RuntimeError(f"set_state_dict failed ep={ep_id}: {exc!r}") from exc
            succeeded = False
            for act in acts:
                _, _, terminated, truncated, info = env.step(act[None])
                success = info.get("success")
                if success is not None and bool(np.asarray(success).reshape(-1)[0]):
                    succeeded = True
                if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                    break
            outcomes[arm].append(succeeded)
        episode_ids.append(ep_id)
        if (idx + 1) % 25 == 0 or (idx + 1) == len(episodes):
            counts = {a: int(sum(outcomes[a])) for a in arms}
            print(f"[{idx+1}/{len(episodes)}] {counts}", flush=True)

env.close()

print("\n=== FINAL SUCCESS RATES (k=2, PickCube-v1, held-out episodes) ===")
arrs = {a: np.asarray(outcomes[a], dtype=bool) for a in arms}
for a in arms:
    print(f"  {a:16s}: {arrs[a].mean()*100:6.2f}%  ({arrs[a].sum()}/{len(arrs[a])})")

print("\n=== PAIRED CONTRASTS (exact McNemar + 95% bootstrap CI on delta pp) ===")
for other in ["exact_integral", "cubic_spline"]:
    a_only, b_only, p = exact_mcnemar(arrs["shape_ridge"], arrs[other])
    lo, hi = paired_bootstrap_ci(arrs["shape_ridge"], arrs[other])
    delta_pp = 100.0 * (arrs["shape_ridge"].mean() - arrs[other].mean())
    print(f"shape_ridge vs {other}: delta={delta_pp:+.2f}pp  "
          f"shape_ridge_only_wins={a_only}  {other}_only_wins={b_only}  "
          f"exact_mcnemar_p={p:.4f}  bootstrap_95ci_pp=[{100*lo:.2f}, {100*hi:.2f}]")

result = {
    "n_episodes": len(episode_ids), "episode_ids": episode_ids,
    "success_rates": {a: float(arrs[a].mean()) for a in arms},
    "successes": {a: int(arrs[a].sum()) for a in arms},
    "episode_outcomes": {a: [bool(x) for x in outcomes[a]] for a in arms},
}
OUT = f"{SCRATCH_DIR}/pickcube_k2_shape_ridge_results.json"
json.dump(result, open(OUT, "w"), indent=2)
print(f"\nsaved -> {OUT}")
