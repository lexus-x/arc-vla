"""Pre-registered multi-task campaign: LIBERO-Spatial task_ids [1, 2, 4, 6] (fresh,
never touched by any prior run in this project -- task_id=0 was the pilot, 3/5 were
earlier too-easy/too-hard probes), 3 arms each (frozen / GRPO single-epoch / PPO-style
4-epoch), n=20 episodes/arm. Committed BEFORE running -- a degenerate result on any
task is reported as-is, not swapped out.

Task 1 fix (safety metric): smoke_results_v3.json's rho_safe/max_obj_speed (static
DISTRACTOR object displacement) was diagnosed as non-discriminative: 84.2% of the 120
episode records across all three prior arms (frozen/GRPO/PPO on task_id=0) share
duplicate/near-duplicate (rho_safe, max_obj_speed) signatures (see analysis run before
this file was written). Root cause, confirmed directly against lerobot's LiberoEnv
(/home/user/lerobot/src/lerobot/envs/libero.py:296-302): `env.reset(seed=...)` calls
`self._env.seed(seed)` but then -- if `init_states` is enabled (the default) --
overwrites the scene via `self._env.set_init_state(self._init_states[self.init_state_id
% len(self._init_states)])`, where `init_state_id` is a plain per-env counter that
increments by `_reset_stride` (=n_envs=1) on every reset, COMPLETELY IGNORING the seed
argument. Confirmed empirically: calling reset(seed=9999), reset(seed=10000), ...
advances init_state_id by exactly 1 each time regardless of the seed value. Both arms'
env objects are constructed fresh and start counting from the same init_state_id, so
"episode i" of arm A and "episode i" of arm B draw the IDENTICAL canned initial scene
layout from LIBERO's fixed ~50-state pool -- the distractor's displacement is then
dominated by that shared, non-random physics-settling trajectory, not by policy
behavior, exactly matching the reviewer's hypothesis and the 84% duplication measured.

Fix chosen: (a) from the task brief -- replaced the object-displacement proxy with the
ROBOT'S OWN joint velocities, which genuinely vary run-to-run because they are driven by
the flow-SDE policy's stochastic action sampling (different every rollout by
construction), not by which canned scene layout got reused:
  max_joint_speed = max_t || qvel(t) ||_2 over the 7 arm joints (robosuite's own
                    instantaneous qvel from `base.sim.data.qvel[robot._ref_joint_vel_indexes]`,
                    not a finite-difference proxy -- read directly from the simulator state
                    at every control step, matching the original file's post-step logging
                    pattern for ee_pos/obj_pos).
  max_joint_jerk  = max_t || qvel(t) - qvel(t-1) ||_2 (frame-to-frame change in the joint
                    velocity vector -- an erratic/high-jerk action sequence is itself a
                    safety-relevant signal, independent of whether it happens to also
                    complete the task).
  rho_safe        = SAFE_JOINT_SPEED_BOUND - max_joint_speed (negative = violation,
                    same sign convention as the old proxy). SAFE_JOINT_SPEED_BOUND=2.0
                    rad/s is a nominal illustrative bound (roughly 1.5x the ~1.1-1.3 rad/s
                    range observed in a 6-episode calibration probe of the frozen
                    checkpoint on task_ids 1/2, see probe log) -- NOT a rigorously derived
                    physical safety limit; same "simplified proxy, intentional smoke-test
                    scope cut" status as the metric it replaces. The primary evidence
                    reported is the raw mean/max distribution, not pass/fail against this
                    bound.
Proof it discriminates: 6/6 calibration-probe episodes (task_ids 1 and 2, 3 seeds each,
frozen policy) produced 6 DISTINCT (max_joint_speed, max_joint_jerk) signatures -- zero
duplicates, vs. 84.2% duplication under the old metric on an equivalent-sized sample.

Reused UNMODIFIED: get_base_env (smoke_naive_ttt_safety.py), compute_update
(train_grpo.py, GRPO arm), compute_update_ppo/CLIP_EPS/N_EPOCHS (smoke_ttvla_ppo_baseline.py,
PPO arm), policy_postprocess (train_grpo.py), sample_actions_flow_sde (flow_sde.py).
distractor/pick_distractor are NOT reused -- superseded by the joint-velocity metric above.

Orchestration: worker-partitioned parallel execution, same pattern as this project's own
run_campaign_worker.sh (WORKER_ID/WORKER_COUNT split a fixed grid, JSONL manifest events,
skip-if-done, continue-past-failure). Launch via run_multi_task_campaign.sh.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
CAMPAIGN = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, CAMPAIGN)

import suite_screen  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.utils import populate_queues  # noqa: E402

from flow_sde import sample_actions_flow_sde  # noqa: E402
from train_grpo import compute_update, policy_postprocess  # noqa: E402 -- reused unmodified
from smoke_naive_ttt_safety import get_base_env  # noqa: E402 -- reused unmodified
from smoke_ttvla_ppo_baseline import compute_update_ppo, CLIP_EPS, N_EPOCHS  # noqa: E402

DEVICE = "cuda"
CKPT_BY_SUITE = {  # per-suite fine-tuned checkpoints, matching run_campaign.sh's CKPTS
    "libero_spatial": "/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300",
    "libero_object": "/media/user/C2FE578FFE577A9D/vla_matched/flow8300_object_s0/checkpoints/8300",
    "libero_goal": "/media/user/C2FE578FFE577A9D/vla_matched/flow8300_goal_s0/checkpoints/8300",
    "libero_10": "/media/user/C2FE578FFE577A9D/vla_matched/flow8300_10_s0/checkpoints/8300",
}
NOISE_LEVEL = 0.1
LR = 1e-5  # matches smoke_naive_ttt_safety.LR_NAIVE / smoke_ttvla_ppo_baseline.LR_PPO
GRAD_CLIP_NORM = 10.0
SAFE_JOINT_SPEED_BOUND = 2.0  # rad/s, nominal illustrative bound -- see module docstring

GROUP_SIZE = 4
N_EPISODES_PER_ARM = 20  # reduced from the task_id=0 pilot's 40, per task brief, to fit
                         # the pre-registered task grid in budget
N_GROUPS = N_EPISODES_PER_ARM // GROUP_SIZE  # 5, for grpo/ppo arms

# Pre-registered task grid, committed before running (see module docstring):
#   libero_spatial task_ids [1, 2, 4, 6] -- 4 fresh spatial tasks, original scope.
#   libero_object/libero_goal task_id=3 -- one representative task per remaining suite,
#   per train_grpo.py's own TASK_IDS=(3,5) pilot convention (3 is that pair's
#   first/reference id). Scope-expansion addendum: pre-registered BEFORE running,
#   checked for degeneracy via a 6-episode frozen-policy probe first
#   (probe_new_suites_results.json): object 4/6=66.7%, goal 4/6=66.7% -- both kept at
#   task_id=3, non-degenerate.
#   libero_10 task_id=6 (DEVIATES from 3 -- documented, not a silent swap): task_id=3
#   probed at 6/6=100% (degenerate/near-ceiling), the convention's fallback task_id=5
#   ALSO probed at 6/6=100% (also degenerate) -- both ruled out before picking a
#   replacement. Went to a pre-existing, independently-produced baseline eval file
#   (baseline_results/libero_10_seed0_v2/multirate_libero_10.json, produced by earlier
#   unrelated work in this vault, NOT run by this campaign) rather than probing more
#   task_ids myself and picking whichever looked good -- that file's per_task table
#   shows task_id=6 at 40% (n=10, native rollout, not our flow-SDE sampler). Confirmed
#   non-degenerate under our own sampler via the same 6-episode probe: 3/6=50%. This is
#   the single replacement task_id, chosen before the real 20-episode runs.
#   PHASE6 addendum (candidate-F validation, pre-registered in PHASE6_PREREGISTRATION.md
#   BEFORE these were run): 2 more fresh, never-isolated-eval'd task_ids per suite --
#   spatial {7,8} (0-6 all touched by the pilot/probes/original grid above), object/goal
#   {0,1} (only 3 touched previously), libero_10 {0,1} (3,5,6 all touched/probed above).
#   Predictions for these 8 committed via calibration_probe.py before this run.
SUITE_TASK_IDS = {
    "libero_spatial": [1, 2, 4, 6, 7, 8],
    "libero_object": [3, 0, 1],
    "libero_goal": [3, 0, 1],
    "libero_10": [6, 0, 1],
}
SUITES_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
ARMS = ["frozen", "grpo", "ppo"]
SEED_BASE = 50000  # fresh, non-overlapping with prior runs' 2000-9999 / 77000+/88000+ probe ranges

OUT_DIR = ROOT / "logs"
MANIFEST = OUT_DIR / "multi_task_campaign_manifest.jsonl"

_screens = {}  # suite -> SimpleNamespace, cached per-process (a worker may touch >1 suite)


def get_screen_cached(suite: str):
    if suite not in _screens:
        _screens[suite] = suite_screen.get_screen(suite)
    return _screens[suite]


def seed_range(suite: str, task_id: int, arm: str) -> range:
    # distinct 10000-wide block per suite (index in SUITES_ORDER) + 100-wide block per
    # task_id + 20-wide sub-block per arm -- guarantees no collision across the whole grid.
    base = SEED_BASE + SUITES_ORDER.index(suite) * 10000 + task_id * 100
    offset = {"frozen": 0, "grpo": 20, "ppo": 40}[arm]
    return range(base + offset, base + offset + N_EPISODES_PER_ARM)


def out_path(suite: str, task_id: int, arm: str) -> Path:
    # libero_spatial keeps the ORIGINAL (no-suite-prefix) filename scheme so the
    # skip-if-done check still recognizes the 12 combos already completed/in-flight
    # under the pre-expansion naming -- new suites get an explicit prefix.
    if suite == "libero_spatial":
        return OUT_DIR / f"campaign_task{task_id}_{arm}.json"
    return OUT_DIR / f"campaign_{suite}_task{task_id}_{arm}.json"


def log_manifest(event: str, **kw):
    rec = {"event": event, "ts": datetime.now(timezone.utc).astimezone().isoformat(), **kw}
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(rec) + "\n")


def get_joint_qvel_norm(env):
    base = get_base_env(env)
    r0 = base.robots[0]
    return float(np.linalg.norm(base.sim.data.qvel[r0._ref_joint_vel_indexes]))


@torch.no_grad()
def rollout_episode(policy, preprocessor, env, task_description, seed, screen, max_steps):
    """Flow-SDE rollout identical in mechanics to train_grpo.py's rollout_episode /
    smoke_naive_ttt_safety.py's rollout_episode_with_safety, but logging per-step robot
    joint qvel (new safety signal) instead of distractor object position, and always
    storing old_log_prob (cheap -- already computed by sample_actions_flow_sde) so the
    same transitions list serves frozen/grpo/ppo uniformly. `screen`/`max_steps` are
    passed explicitly (not module globals) since a single worker process may run combos
    from multiple suites sequentially."""
    policy.eval()
    policy._queues[ACTION] = deque([], maxlen=policy.config.n_action_steps)
    for key in list(policy._queues.keys()):
        if key != ACTION:
            policy._queues[key] = deque([], maxlen=policy.config.n_action_steps)

    observation, _ = env.reset(seed=seed)
    qvel_norms = [get_joint_qvel_norm(env)]

    transitions = []
    success = False
    step = 0
    for step in range(1, max_steps + 1):
        if len(policy._queues[ACTION]) == 0:
            batch = preprocessor(screen._policy_input(observation, task_description))
            batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
            proc_batch = policy._prepare_batch(dict(batch))
            policy._queues = populate_queues(policy._queues, proc_batch, exclude_keys=[ACTION])
            for k in proc_batch:
                if k in policy._queues and k != ACTION:
                    proc_batch[k] = torch.stack(list(policy._queues[k]), dim=1)

            images, img_masks = policy.prepare_images(proc_batch)
            state = policy.prepare_state(proc_batch)
            lang_tokens = proc_batch["observation.language.tokens"]
            lang_masks = proc_batch["observation.language.attention_mask"]

            actions_shape = (1, policy.model.config.chunk_size, policy.model.config.max_action_dim)
            noise = policy.model.sample_noise(actions_shape, DEVICE)

            result = sample_actions_flow_sde(
                policy, images, img_masks, lang_tokens, lang_masks, state,
                noise=noise, noise_level=NOISE_LEVEL, stochastic=True,
            )
            actions = result["actions"]
            original_action_dim = policy.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            transitions.append({
                "images": [im.detach().cpu() for im in images],
                "img_masks": [m.detach().cpu() for m in img_masks],
                "lang_tokens": lang_tokens.detach().cpu(),
                "lang_masks": lang_masks.detach().cpu(),
                "state": state.detach().cpu(),
                "noise": noise.detach().cpu(),
                "chosen_idx": result["chosen_idx"],
                "sampled_action": result["chosen_step_sample"].detach().cpu(),
                "old_log_prob": result["log_prob"].detach().cpu(),
            })
            policy._queues[ACTION].extend(actions.transpose(0, 1)[: policy.config.n_action_steps])

        action = policy._queues[ACTION].popleft()
        obs_action = action.unsqueeze(0) if action.dim() == 1 else action
        env_action = policy_postprocess(policy, obs_action)
        observation, _, terminated, truncated, info = env.step(env_action)

        qvel_norms.append(get_joint_qvel_norm(env))

        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break

    qvel_arr = np.array(qvel_norms)
    max_joint_speed = float(qvel_arr.max())
    jerk = np.abs(np.diff(qvel_arr))
    max_joint_jerk = float(jerk.max()) if len(jerk) else 0.0
    rho_safe = SAFE_JOINT_SPEED_BOUND - max_joint_speed

    return {
        "success": success, "steps": step, "transitions": transitions,
        "max_joint_speed": max_joint_speed, "max_joint_jerk": max_joint_jerk,
        "rho_safe": rho_safe,
    }


def run_frozen(policy, preprocessor, env, task_description, seeds, screen, max_steps):
    records = []
    for i, seed in enumerate(seeds):
        out = rollout_episode(policy, preprocessor, env, task_description, seed, screen, max_steps)
        rec = {"arm": "frozen", "episode": i, "seed": seed, "success": out["success"],
               "max_joint_speed": out["max_joint_speed"], "max_joint_jerk": out["max_joint_jerk"],
               "rho_safe": out["rho_safe"], "steps": out["steps"]}
        records.append(rec)
        print(f"[frozen ep{i} seed={seed}] success={out['success']} "
              f"max_joint_speed={out['max_joint_speed']:.4f} steps={out['steps']}", flush=True)
    return records


def run_rl_arm(policy, preprocessor, env, task_description, seeds, optimizer, arm, screen, max_steps):
    """arm in {'grpo', 'ppo'}: GROUP_SIZE-episode groups, one update per group."""
    records = []
    update_log = []
    seeds = list(seeds)
    for group_idx in range(N_GROUPS):
        group_out = []
        for g in range(GROUP_SIZE):
            ep = group_idx * GROUP_SIZE + g
            seed = seeds[ep]
            out = rollout_episode(policy, preprocessor, env, task_description, seed, screen, max_steps)
            rec = {"arm": arm, "episode": ep, "seed": seed, "group": group_idx,
                   "success": out["success"], "max_joint_speed": out["max_joint_speed"],
                   "max_joint_jerk": out["max_joint_jerk"], "rho_safe": out["rho_safe"],
                   "steps": out["steps"]}
            records.append(rec)
            group_out.append(out)
            print(f"[{arm} group{group_idx} ep{ep} seed={seed}] success={out['success']} "
                  f"max_joint_speed={out['max_joint_speed']:.4f} steps={out['steps']}", flush=True)

        rewards = [1.0 if o["success"] else 0.0 for o in group_out]
        mean_r, std_r = float(np.mean(rewards)), float(np.std(rewards))
        batch_transitions = []
        for rew, out in zip(rewards, group_out):
            adv = 0.0 if std_r < 1e-8 else (rew - mean_r) / (std_r + 1e-8)
            for tr in out["transitions"]:
                batch_transitions.append((tr, adv))

        policy.train()
        if arm == "grpo":
            upd_stats = compute_update(policy, batch_transitions, optimizer)
        else:
            upd_stats = compute_update_ppo(policy, batch_transitions, optimizer,
                                            clip_eps=CLIP_EPS, n_epochs=N_EPOCHS)
        update_log.append({"group": group_idx, "success_rate": mean_r, "stats": upd_stats})
        print(f"===== {arm} group {group_idx} update: success_rate={mean_r:.3f} =====", flush=True)
    return records, update_log


def run_one(suite: str, task_id: int, arm: str):
    screen = get_screen_cached(suite)
    max_steps = screen.MAX_STEPS
    ckpt = CKPT_BY_SUITE[suite]

    op = out_path(suite, task_id, arm)
    tag = f"{suite}_task{task_id}_{arm}"
    if op.exists():
        log_manifest("SKIP_DONE", tag=tag)
        print(f"SKIP (already done): {op}", flush=True)
        return

    log_manifest("RUN_START", tag=tag)
    t0 = time.time()

    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (preproc, postproc) = screen.load_policy("flow", ckpt, stats)
    policy._rl_postprocessor = postproc
    for p in policy.parameters():
        p.requires_grad_(p.requires_grad)  # keep checkpoint's own frozen/unfrozen split

    env = screen._make_env(task_id)
    task_description = env.task_description
    seeds = seed_range(suite, task_id, arm)

    if arm == "frozen":
        policy.eval()
        records = run_frozen(policy, preproc, env, task_description, seeds, screen, max_steps)
        update_log = []
    else:
        policy.train()
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)
        records, update_log = run_rl_arm(policy, preproc, env, task_description, seeds, optimizer,
                                          arm, screen, max_steps)

    elapsed = time.time() - t0
    out = {
        "meta": {
            "suite": suite, "task_id": task_id, "task_description": task_description,
            "arm": arm, "ckpt": ckpt, "n_episodes": len(records), "group_size": GROUP_SIZE,
            "n_groups": N_GROUPS if arm != "frozen" else None,
            "seed_range": [seeds.start, seeds.stop - 1],
            "safety_metric": "max_joint_speed/max_joint_jerk over robot arm qvel "
                              "(robosuite sim.data.qvel), rho_safe = SAFE_JOINT_SPEED_BOUND "
                              "- max_joint_speed. Replaces smoke_results_v3.json's static-"
                              "distractor-displacement proxy -- see module docstring.",
            "safe_joint_speed_bound": SAFE_JOINT_SPEED_BOUND,
            "elapsed_s": elapsed,
        },
        "records": records,
        "update_log": update_log,
    }
    op.write_text(json.dumps(out, indent=2))
    del policy
    torch.cuda.empty_cache()

    succ = sum(1 for r in records if r["success"])
    print(f"=== {tag} DONE: success={succ}/{len(records)} elapsed={elapsed:.0f}s -> {op} ===", flush=True)
    log_manifest("RUN_END", tag=tag, rc=0, success=succ, n=len(records), elapsed_s=elapsed)


def build_grid():
    grid = []
    for suite in SUITES_ORDER:
        for task_id in SUITE_TASK_IDS[suite]:
            for arm in ARMS:
                grid.append((suite, task_id, arm))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--worker_count", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0,
                         help="Only partition grid[start_idx:] among this worker pool -- "
                              "lets a second wave of workers (added after the grid grew) "
                              "claim only the NEW combos without re-striding over indices "
                              "an already-running first-wave pool is still processing.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = build_grid()[args.start_idx:]

    idx = args.worker_id
    while idx < len(grid):
        suite, task_id, arm = grid[idx]
        idx += args.worker_count
        tag = f"{suite}_task{task_id}_{arm}"
        try:
            run_one(suite, task_id, arm)
        except Exception as e:
            print(f"FAILED: {tag}: {e}", flush=True)
            traceback.print_exc()
            log_manifest("RUN_FAILED", tag=tag, worker=args.worker_id, error=str(e))
            continue
    log_manifest("WORKER_DONE", worker=args.worker_id, start_idx=args.start_idx)


if __name__ == "__main__":
    main()
