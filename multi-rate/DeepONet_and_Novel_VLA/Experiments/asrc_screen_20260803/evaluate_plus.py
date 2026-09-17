#!/usr/bin/env python
"""
evaluate_plus.py
================
Closed-loop robustness evaluation of M1/M3/M4 on the LIBERO-Plus benchmark
(sylvestf/LIBERO-plus) over its 7 perturbation dimensions.

* Receding-horizon control: replan every --replan steps (config.n_action_steps),
  identical for all models.
* Per category, a fixed, difficulty-stratified subset of perturbed tasks is
  sampled (same task indices for every model) and run 1 trial each, per the
  LIBERO-Plus convention (num_trials_per_task = 1).
* Self-contained w.r.t. LIBERO: imports libero_plus_wrapper (which isolates the
  LIBERO-Plus package + config) and does NOT import the original-LIBERO eval path.

Models: repeatable  --model NAME=HEAD=CKPT  (HEAD in {flow,deeponet}).
Results stream to <out>/robustness_plus.json (resumable).
"""

from __future__ import annotations

import libero_plus_wrapper as LP  # MUST be imported first (sets sys.path + config)
from libero_plus_wrapper import LiberoPlusEnv, list_perturbed_tasks, CATEGORIES

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

logging.disable(logging.WARNING)
DEV = "cuda"
REPO_DATA = "lerobot/libero_spatial_image"
SUITE_DATASET = {
    "libero_spatial": "lerobot/libero_spatial_image",
    "libero_object": "lerobot/libero_object_image",
    "libero_goal": "lerobot/libero_goal_image",
    "libero_10": "lerobot/libero_10_image",
    "libero_90": "lerobot/libero_90_image",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_sha256(path: str) -> str:
    root = Path(path)
    files = sorted(root.glob("model*.safetensors"))
    if not files:
        raise SystemExit("[FATAL] no model*.safetensors under %s" % root)
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        digest.update(bytes.fromhex(_sha256_file(file)))
    return digest.hexdigest()


# --------------------------------------------------------------------------- model
def resolve_latest(ckpt_path: str) -> str:
    p = Path(ckpt_path)
    if p.name in ("LATEST", "BEST"):
        ptr = p.parent / f"{p.name}.txt"
        if not ptr.exists() and p.name == "BEST":
            ptr = p.parent / "LATEST.txt"
        if ptr.exists():
            return str(p.parent / ptr.read_text().strip())
    return str(p)


def load_policy(head: str, ckpt: str, dataset_stats, replan: int):
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    ckpt = resolve_latest(ckpt)
    # GUARD: trunk band SPACING and pool channel-norm are invisible to the weights-vs-weights
    # guard below -- freqs is a non-persistent buffer, and geometric vs linear spacing yields
    # byte-identical tensor SHAPES. A mismatch would silently evaluate a different architecture
    # and return a plausible number. Cross-check the arm own run_config.json, which train.py
    # writes and is authoritative.
    import os as _osc, json as _jsonc
    _rc = _osc.path.join(ckpt, "..", "..", "run_config.json")
    if head == "deeponet":
        if not _osc.path.exists(_rc):
            raise SystemExit("[FATAL] missing training provenance: %s" % _rc)
        try:
            _a = _jsonc.load(open(_rc))["args"]
        except Exception as exc:
            raise SystemExit("[FATAL] invalid training provenance %s: %s" % (_rc, exc))
        _trained_head = _a.get("deeponet_head", "deeponet")
        _eval_head = _osc.environ.get("DEEPONET_HEAD", "deeponet")
        if _trained_head != _eval_head:
            raise SystemExit("[FATAL] eval/train head mismatch: trained %s, eval %s" %
                             (_trained_head, _eval_head))
        for _var, _key in (("DEEPONET_TRUNK_BANDLIMIT", "trunk_bandlimit"),
                           ("DEEPONET_POOL_CHANNEL_NORM", "pool_channel_norm")):
            # Older checkpoints predate these flags; their training default was False.
            _trained = bool(_a.get(_key, False))
            _got = bool(int(_osc.environ.get(_var, "0")))
            if _trained != _got:
                raise SystemExit(
                    "[FATAL] eval/train mismatch: ckpt trained with %s=%s but eval env has "
                    "%s=%s.\n  run_config: %s\n  The weights-vs-weights guard CANNOT catch "
                    "this (identical shapes)." % (_key, _trained, _var, _got, _rc))
    if head == "csa":
        from contextual_operator_arbitration import load_stability_policy
        policy = load_stability_policy(ckpt)
    elif head == "coa":
        from contextual_operator_arbitration import load_policy as load_coa_policy
        policy = load_coa_policy(ckpt)
    elif head == "flow":
        from modeling_smolvla_ph import SmolVLAPHPolicy
        policy = SmolVLAPHPolicy.from_pretrained(ckpt, ph_enabled=False)
    elif head == "deeponet":
        import os as _os
        from modeling_smolvla_deeponet_v2 import SmolVLADeepONetPolicy
        policy = SmolVLADeepONetPolicy.from_pretrained(
            ckpt, ph_enabled=False,
            deeponet_p=int(_os.environ.get("DEEPONET_P", 256)),
            deeponet_blocks=int(_os.environ.get("DEEPONET_BLOCKS", 3)),
            deeponet_queries=int(_os.environ.get("DEEPONET_QUERIES", 8)),
            deeponet_fourier=int(_os.environ.get("DEEPONET_FOURIER", 16)),
            deeponet_head=_os.environ.get("DEEPONET_HEAD", "deeponet"),
            deeponet_pool_norm=bool(int(_os.environ.get("DEEPONET_POOL_CHANNEL_NORM", "0"))),
            deeponet_trunk_bandlimit=bool(int(_os.environ.get("DEEPONET_TRUNK_BANDLIMIT", "0"))))
    else:
        raise ValueError(head)

    if head == "deeponet" and _osc.environ.get("DEEPONET_HEAD") == "tempo":
        tempo_speed = float(_osc.environ.get("TEMPO_SPEED", "1.0"))
        policy.set_execution_speed(tempo_speed)
        print("[tempo] execution speed=%.3f" % tempo_speed, flush=True)

    _f64 = bool(int(__import__("os").environ.get("DEEPONET_TRUNK_FP64", "0")))
    if _f64 and head == "deeponet":
        _fh = [m for m in policy.modules() if type(m).__name__ == "DeepONetHeadV2"]
        if len(_fh) != 1:
            raise SystemExit("[FATAL] fp64: found %d DeepONetHeadV2" % len(_fh))
        _fh[0].set_trunk_fp64(True)
        print("[smooth] trunk carriers computed in float64", flush=True)
    _ss = int(__import__("os").environ.get("DEEPONET_SUPERSAMPLE", "1"))
    _lp = bool(int(__import__("os").environ.get("DEEPONET_CHUNK_LOWPASS", "0")))
    if (_ss > 1 or _lp) and head == "deeponet":
        _sh = [m for m in policy.modules() if type(m).__name__ == "DeepONetHeadV2"]
        if len(_sh) != 1:
            raise SystemExit("[FATAL] smooth: found %d DeepONetHeadV2" % len(_sh))
        if _ss > 1:
            _sh[0].set_supersample(_ss)
            print("[smooth] trunk cell-integration supersample=%d" % _ss, flush=True)
        if _lp:
            _sh[0].set_chunk_lowpass(True)
            print("[smooth] 3-tap binomial chunk lowpass ON", flush=True)

    # guard: the built head must match the checkpoint's stored head, weights vs weights.
    # A non-strict load silently random-inits missing tensors and yields a valid-looking
    # number instead of crashing. head=flow has no deeponet keys on either side -> {}=={}.
    import glob as _glob, os as _o
    import safetensors.torch as _st
    _have = {}
    for _f in _glob.glob(_o.path.join(ckpt, "model*.safetensors")):
        with _st.safe_open(_f, framework="pt") as _h:
            _have.update({k: tuple(_h.get_slice(k).get_shape())
                          for k in _h.keys() if "deeponet" in k})
    _want = {k: tuple(v.shape) for k, v in policy.state_dict().items() if "deeponet" in k}
    if _want != _have:
        raise SystemExit(
            "[FATAL] head/ckpt mismatch: DEEPONET_HEAD=%s built %d head tensors, ckpt has %d\n"
            "  ckpt: %s\n"
            "  randomly-initialized (in model, absent from ckpt): %s\n"
            "  trained-but-dropped  (in ckpt, absent from model): %s"
            % (_o.environ.get("DEEPONET_HEAD", "<unset>"), len(_want), len(_have), ckpt,
               sorted(set(_want) - set(_have)), sorted(set(_have) - set(_want))))

    _rate_head = _o.environ.get("DEEPONET_HEAD", "deeponet") in ("ti", "asrc")
    if head == "deeponet" and _rate_head:
        from rate_integrated_deeponet import RateIntegratedDeepONetHead
        _rh = [m for m in policy.modules() if isinstance(m, RateIntegratedDeepONetHead)]
        if len(_rh) != 1:
            raise SystemExit("[FATAL] expected exactly one rate-integrated head, found %d" % len(_rh))
        _rh[0].configure_action_stats(dataset_stats["action"]["mean"],
                                      dataset_stats["action"]["std"])
        _rate = float(_o.environ.get("DEEPONET_RATE_HZ", "0"))
        _rh[0].set_rate(_rate)
        print("[rate-integrated] head=%s rate=%.3fHz raw deltas already include dt" %
              (_o.environ.get("DEEPONET_HEAD"), _rate), flush=True)

    # ---- off-grid tau query + alias folding (all three default to OFF) --------
    # Applied AFTER the run_config guard and AFTER the weights-vs-weights shape guard,
    # so neither can be perturbed. The head sign vectors are non-persistent buffers and
    # never enter state_dict(), so the shape guard above is unaffected either way.
    _qf = int(_o.environ.get("DEEPONET_QUERY_FACTOR", "1"))
    _qp = int(_o.environ.get("DEEPONET_QUERY_POINTS", "0"))
    _fold = bool(int(_o.environ.get("DEEPONET_FOLD_ALIASES", "0")))
    _gaar = bool(int(_o.environ.get("DEEPONET_GAAR", "0")))
    _return_grid = bool(int(_o.environ.get("DEEPONET_RETURN_QUERY_GRID", "0")))
    _action_interp = int(_o.environ.get("DEEPONET_ACTION_INTERP_FACTOR", "1"))
    if _fold and _gaar:
        raise SystemExit("[FATAL] DEEPONET_FOLD_ALIASES and DEEPONET_GAAR are mutually exclusive")
    if _action_interp != 1 and (_qf != 1 or _qp > 0 or _fold or _gaar or _return_grid):
        raise SystemExit("[FATAL] action interpolation is an unmodified-native control")
    if _return_grid and _qf == 1 and _qp == 0:
        raise SystemExit("[FATAL] DEEPONET_RETURN_QUERY_GRID requires a denser query grid")
    if head == "deeponet" and (_qf != 1 or _qp > 0 or _fold or _gaar or _return_grid
                               or _action_interp != 1):
        from deeponet_head_v2 import DeepONetHeadV2 as _DHV2
        # The head may be WRAPPED (spectral/timewarp/hybrid/multioperator head_types all
        # nest a DeepONetHeadV2, and three of them build TWO). Resolve by type, and
        # refuse to guess when the answer is ambiguous.
        _hits = [(n, m) for n, m in policy.named_modules() if isinstance(m, _DHV2)]
        if len(_hits) != 1:
            raise SystemExit(
                "[FATAL] DEEPONET_QUERY_FACTOR/POINTS/FOLD_ALIASES/GAAR set, but found %d "
                "DeepONetHeadV2 instances under DEEPONET_HEAD=%s (need exactly 1).\n"
                "  found: %s" % (len(_hits), _o.environ.get("DEEPONET_HEAD", "<unset>"),
                                 [n for n, _ in _hits] or "none"))
        _hname, _dh = _hits[0]
        _msg = []
        if _action_interp != 1:
            _dh.set_action_interpolation(_action_interp)
            _msg.append("linear action interpolation x%d [NON-operator control]" % _action_interp)
        elif _fold:
            _d = _dh.fold_aliases()
            _msg.append("fold_aliases: cycles %s signs %s (on-grid err f64 %.2e / f32 %.2e)"
                        % ([round(c, 3) for c in _d["alias_cycles"]],
                           [int(s) for s in _d["sin_sign"]],
                           _d["max_ongrid_err_f64"], _d["max_ongrid_err_f32"]))
        elif _gaar:
            _dh.eval()
            _d = _dh.anchor_alias_repair()
            _msg.append("GAAR: folded carriers + linearly interpolated native-feature residual "
                        "(anchors use original path; fold err f64 %.2e / f32 %.2e)"
                        % (_d["max_ongrid_err_f64"], _d["max_ongrid_err_f32"]))
        if _qp > 0:
            _dh.set_query_grid(_qp)
            _msg.append("set_query_grid(%d) [NON-nested: genuinely interpolates]" % _qp)
        elif _qf != 1:
            _dh.set_query_resolution(_qf)
            _msg.append("set_query_resolution(%d) [NESTED]" % _qf)
        if _return_grid:
            _dh.return_query_grid()
            _msg.append("return every queried action")
        returned = ((max(_dh.tau.shape[0], _dh.chunk_size) - 1) * _action_interp + 1
                    if _action_interp != 1 else
                    (_dh.tau.shape[0] if _return_grid else _dh.chunk_size))
        print("[retarget] %s -> %s ; tau grid %d pts -> returns %d steps"
              % (_hname, " ; ".join(_msg), _dh.tau.shape[0], returned), flush=True)

    policy = policy.to(DEV).eval()
    policy.config.n_action_steps = replan
    pre, post = make_smolvla_pre_post_processors(policy.config, dataset_stats=dataset_stats)
    return policy, pre, post


# --------------------------------------------------------------------------- obs conv
def _quat2axisangle(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.to(torch.float32)
    w = quat[:, 3].clamp(-1.0, 1.0)
    den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    out = torch.zeros((quat.shape[0], 3))
    mask = den > 1e-10
    if mask.any():
        angle = 2.0 * torch.acos(w[mask])
        axis = quat[mask, :3] / den[mask].unsqueeze(1)
        out[mask] = axis * angle.unsqueeze(1)
    return out


def plus_obs_to_policy_input(obs, task_description):
    """Raw LIBERO-Plus robosuite obs -> policy batch, matching training convention
    (180-deg image flip; state = [eef_pos(3), quat2axisangle(3), gripper_qpos(2)])."""
    def img(a):
        t = torch.as_tensor(np.asarray(a)).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        return torch.flip(t, dims=[2, 3])

    eef_pos = torch.as_tensor(np.asarray(obs["robot0_eef_pos"])).float().reshape(1, 3)
    eef_quat = torch.as_tensor(np.asarray(obs["robot0_eef_quat"])).float().reshape(1, 4)
    grip = torch.as_tensor(np.asarray(obs["robot0_gripper_qpos"])).float().reshape(1, 2)
    state = torch.cat([eef_pos, _quat2axisangle(eef_quat), grip], dim=-1)
    return {
        "observation.images.image": img(obs["agentview_image"]),
        "observation.images.wrist_image": img(obs["robot0_eye_in_hand_image"]),
        "observation.state": state,
        "task": [task_description],
    }


@torch.no_grad()
def rollout(policy, pre, post, env, task_description, max_steps):
    policy.reset()
    obs = env.reset(seed=0)
    for _ in range(max_steps):
        pin = pre(plus_obs_to_policy_input(obs, task_description))
        pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(pin)
        action = post(action)
        a = action.to("cpu").float().numpy().reshape(-1)
        obs, reward, done, info = env.step(a)
        if env.check_success():
            return True
        if done:
            break
    return False


# --------------------------------------------------------------------------- sampling
def stratified_sample(tasks, n, seed=0):
    """Pick n tasks spread across difficulty levels (deterministic)."""
    import random
    rng = random.Random(seed)
    by_diff = {}
    for t in tasks:
        by_diff.setdefault(t["difficulty_level"], []).append(t)
    for v in by_diff.values():
        v.sort(key=lambda t: t["id"])
        rng.shuffle(v)
    out, levels = [], sorted(by_diff, key=lambda d: (d is None, d))  # None-safe (Goal has None difficulties)
    i = 0
    while len(out) < min(n, len(tasks)):
        lv = levels[i % len(levels)]
        if by_diff[lv]:
            out.append(by_diff[lv].pop())
        i += 1
        if all(len(by_diff[lv]) == 0 for lv in levels):
            break
    return out


def _save(results, out_path):
    Path(out_path).write_text(json.dumps(results, indent=2))


def run_plus_for_policy(policy, pre, post, suite, n_per_cat=12, max_steps=300,
                        out_dir=None, replan=5, img_size=256, categories=None,
                        control_freq=20, scale_pose_deltas=True):
    """Run the LIBERO-Plus robustness sweep for ONE already-loaded policy and
    return its robustness_average (mean over category averages). Reuses the same
    stratified sampling + per-category rollout as main(). Used by eval_pi05.py so
    the pi0.5 variants share this exact harness."""
    cats = CATEGORIES if not categories else [c.strip() for c in categories.split(",")]
    bench, tasks = list_perturbed_tasks(suite)
    by_cat = {c: [t for t in tasks if t["category"] == c] for c in cats}
    sampled = {c: stratified_sample(by_cat[c], n_per_cat, seed=42) for c in cats}
    cat_avgs = []
    per_cat = {}
    for c in cats:
        vals = []
        for t in sampled[c]:
            env = LiberoPlusEnv(bench, t["index"], img_size=img_size,
                                control_freq=control_freq,
                                scale_pose_deltas=scale_pose_deltas)
            succ = rollout(policy, pre, post, env, env.task_description, max_steps)
            env.close()
            vals.append(bool(succ))
        avg = float(np.mean(vals)) if vals else None
        per_cat[c] = avg
        if avg is not None:
            cat_avgs.append(avg)
        print(f"[plus] {c}: {avg}", flush=True)
    rob = float(np.mean(cat_avgs)) if cat_avgs else None
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        _save({"per_category": per_cat, "robustness_average": rob},
              Path(out_dir) / "robustness_plus.json")
    return rob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="repeatable NAME=HEAD=CKPT (HEAD in flow|deeponet)")
    ap.add_argument("--out", default="runs/eval_plus")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--stats_path", default=None)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--n_per_cat", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--categories", default=None, help="comma list to restrict categories")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--control_freq", type=int, default=20)
    args = ap.parse_args()

    return_grid = bool(int(os.environ.get("DEEPONET_RETURN_QUERY_GRID", "0")))
    query_factor = int(os.environ.get("DEEPONET_QUERY_FACTOR", "1"))
    action_interp = int(os.environ.get("DEEPONET_ACTION_INTERP_FACTOR", "1"))
    retarget_factor = query_factor if return_grid else action_interp
    rate_head = os.environ.get("DEEPONET_HEAD", "deeponet") in ("ti", "asrc")
    if rate_head:
        if (args.control_freq % 5 or return_grid or query_factor != 1 or action_interp != 1 or
                int(os.environ.get("DEEPONET_QUERY_POINTS", "0")) or
                int(os.environ.get("DEEPONET_FOLD_ALIASES", "0")) or
                int(os.environ.get("DEEPONET_GAAR", "0"))):
            raise SystemExit("[FATAL] rate-integrated heads require direct queries and rates divisible by 5")
        expected = (args.control_freq, args.control_freq // 5, 15 * args.control_freq)
        got = (args.control_freq, args.replan, args.max_steps)
        if float(os.environ.get("DEEPONET_RATE_HZ", "0")) != args.control_freq or got != expected:
            raise SystemExit("[FATAL] rate-integrated physical-time mismatch: expected %s, got %s" %
                             (expected, got))
    elif return_grid or action_interp != 1:
        expected = (20 * retarget_factor, 5 * retarget_factor, 300 * retarget_factor)
        got = (args.control_freq, args.replan, args.max_steps)
        if return_grid and (query_factor <= 1 or
                            int(os.environ.get("DEEPONET_QUERY_POINTS", "0")) > 0):
            raise SystemExit("[FATAL] executable rate retargeting requires a nested query factor")
        if got != expected:
            raise SystemExit(
                "[FATAL] physical-time mismatch: (control_freq,replan,max_steps) "
                f"must be {expected}, got {got}"
            )
    elif args.control_freq != 20:
        raise SystemExit("[FATAL] non-retargeted evaluation must use the native 20Hz control rate")

    cats = CATEGORIES if not args.categories else \
        [c.strip() for c in args.categories.split(",")]

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    # ponytail: norm stats MUST match the suite action distribution; a fixed Spatial
    # default silently zeroed Object/Long Plus (wrong unnorm -> garbage actions).
    dataset = args.dataset or SUITE_DATASET.get(args.suite, REPO_DATA)
    print(f"[plus] norm dataset = {dataset} (suite={args.suite})", flush=True)
    meta = LeRobotDatasetMetadata(dataset)
    norm_stats = torch.load(args.stats_path) if args.stats_path else meta.stats

    # enumerate + sample tasks per category (same for all models)
    bench, tasks = list_perturbed_tasks(args.suite)
    by_cat = {c: [t for t in tasks if t["category"] == c] for c in cats}
    sampled = {c: stratified_sample(by_cat[c], args.n_per_cat, seed=42) for c in cats}
    for c in cats:
        print(f"[plus] {c}: sampled {len(sampled[c])} / {len(by_cat[c])} tasks", flush=True)

    models = {}
    for spec in args.model:
        name, head, ckpt = spec.split("=", 2)
        models[name] = (head, resolve_latest(ckpt))

    runtime_vars = {
        "query_factor": int(os.environ.get("DEEPONET_QUERY_FACTOR", "1")),
        "query_points": int(os.environ.get("DEEPONET_QUERY_POINTS", "0")),
        "fold_aliases": bool(int(os.environ.get("DEEPONET_FOLD_ALIASES", "0"))),
        "gaar": bool(int(os.environ.get("DEEPONET_GAAR", "0"))),
        "return_query_grid": bool(int(os.environ.get("DEEPONET_RETURN_QUERY_GRID", "0"))),
        "action_interp_factor": int(os.environ.get("DEEPONET_ACTION_INTERP_FACTOR", "1")),
        "trunk_bandlimit": bool(int(os.environ.get("DEEPONET_TRUNK_BANDLIMIT", "0"))),
        "pool_channel_norm": bool(int(os.environ.get("DEEPONET_POOL_CHANNEL_NORM", "0"))),
        "rate_hz": float(os.environ.get("DEEPONET_RATE_HZ", "0")),
    }
    run_config = {
        "suite": args.suite, "dataset": dataset, "replan": args.replan,
        "n_per_cat": args.n_per_cat, "max_steps": args.max_steps,
        "img_size": args.img_size, "control_freq": args.control_freq, "categories": cats,
        "relative_pose_delta_scale": 1.0 if rate_head else 20.0 / args.control_freq,
        "task_sample_seed": 42, "rollout_reset_seed": 0,
        "protocol": ("raw 6D action-velocity proxy integrated component-wise" if rate_head else
                     "nested query grid executed at matched physical rate" if return_grid else
                     "linearly interpolated final actions at matched physical rate"
                     if action_interp != 1 else
                     "off-grid queries resampled to 50 outputs at native 20Hz"),
        "physical_replan_seconds": args.replan / args.control_freq,
        "physical_horizon_seconds": args.max_steps / args.control_freq,
        "runtime": runtime_vars,
        "models": {name: {"head": head, "checkpoint": str(Path(ckpt).resolve()),
                           "checkpoint_sha256": _checkpoint_sha256(ckpt)}
                   for name, (head, ckpt) in models.items()},
        "code_sha256": {name: _sha256_file(Path(__file__).with_name(name))
                         for name in ("evaluate_plus.py", "deeponet_head_v2.py",
                                      "rate_integrated_deeponet.py", "libero_plus_wrapper.py")},
        "sampled_indices": {c: [t["index"] for t in sampled[c]] for c in cats},
    }

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) / "robustness_plus.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    if "_config" in results and results["_config"] != run_config:
        raise SystemExit("[FATAL] refusing mixed resume: saved _config differs from this run")
    results["_config"] = run_config
    _save(results, out_path)

    for name, (head, ckpt) in models.items():
        mres = results.setdefault(name, {})
        policy, pre, post = load_policy(head, ckpt, norm_stats, args.replan)
        print(f"[plus] loaded {name} (head={head}) <- {ckpt}", flush=True)
        for c in cats:
            cres = mres.setdefault(c, {"per_task": {}, "average": None})
            for t in sampled[c]:
                idx = t["index"]
                key = str(idx)
                if key in cres["per_task"]:
                    continue
                env = LiberoPlusEnv(bench, idx, img_size=args.img_size,
                                    control_freq=args.control_freq,
                                    scale_pose_deltas=not rate_head)
                succ = rollout(policy, pre, post, env, env.task_description, args.max_steps)
                env.close()
                cres["per_task"][key] = {"name": t["name"], "difficulty": t["difficulty_level"],
                                         "success": bool(succ)}
                print(f"[plus/{name}/{c}] idx{idx} d{t['difficulty_level']}: "
                      f"{'OK' if succ else 'x'}", flush=True)
                _save(results, out_path)
            vals = [v["success"] for v in cres["per_task"].values()]
            cres["average"] = float(np.mean(vals)) if vals else None
            _save(results, out_path)
        cat_avgs = [mres[c]["average"] for c in cats if mres.get(c, {}).get("average") is not None]
        mres["robustness_average"] = float(np.mean(cat_avgs)) if cat_avgs else None
        _save(results, out_path)
        print(f"[plus] {name} robustness avg = {mres['robustness_average']}", flush=True)
        del policy
        torch.cuda.empty_cache()

    print("[plus] DONE", flush=True)
    _save(results, out_path)


if __name__ == "__main__":
    main()
