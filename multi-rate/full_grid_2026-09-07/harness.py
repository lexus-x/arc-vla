"""Closed-loop DP/FM + resampler eval. One script, three sim backends:
  ManiSkill  (env `ms3`):               python harness.py PickCube-v1 | PushT-v1
  RoboMimic  (env `vla_smolvla_libero`): python harness.py lift | can | square
  RoboCasa   (env `gr00t`, via socket bridge -- robosuite/robocasa need numpy<1.24, which can't
             share a process with Blackwell-capable torch): python harness.py RC-OpenDrawer |
             RC-PnPCounterToStove  (start robocasa_bridge.py in robocasa_uv first)
  --policy dp|fm selects DiffusionPolicy (dp_min) or FlowMatchingPolicy (fm_min); same U-Net,
  same train()/EMA, differ only in objective+sampler. Default dp (matches the 2026-09-06 run).

Protocol (pre-registered, do not tune after seeing results):
  * DP = dp_min.DiffusionPolicy, state-only, n_obs=2, horizon=16, execute 8, EMA weights, DDIM-10.
  * Train on the first N_TRAIN demos; eval from the initial states of N_EVAL *held-out* demos.
  * Arms, all applied to the SAME executed 8-step chunk after clipping to [-1,1]:
      native (k=1, chunk as-is) + {zoh, spline, spline_satfix, pchip, pchip_satfix,
      tac_fold, tac_fold_satfix, gripper_sync} at k=2 on the delta dims. gripper_sync
      uses spline_satfix on continuous dims and reconstructs gripper timing from one
      transition offset/value summary per block;
      other arms causal-hold the gripper dim.
  * Paired: same init state per episode across arms, and the DP's sampling noise is re-seeded
    per (episode, replan) so arms see identical policy proposals whenever their obs agree.
  * Stats: exact McNemar vs zoh AND vs native, per arm. Loss column always reported.
Usage: python harness.py TASK [--steps 30000] [--n_train 200] [--n_eval 100] [--smoke]
"""
import argparse, hashlib, json, math, os, socket, struct, pickle, sys, time
import multiprocessing as mp
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MANISKILL_DATA_DIR = os.environ.get("MANISKILL_DATA_DIR", "/home/user/maniskill_data")
ROBOMIMIC_DATA_DIR = os.environ.get("ROBOMIMIC_DATA_DIR", "/home/user/robomimic_data")
from dp_min import DiffusionPolicy, MinMax, make_chunks, train
from fm_min import FlowMatchingPolicy
from resample_math import coarsen_gripper_transitions, decimate_and_resample, gripper_sync
import resample_math
from resample_qp import resample_qp, resample_qp_anchor
resample_math.RESAMPLERS["qp"] = resample_qp
resample_math.RESAMPLERS["qp_anchor"] = resample_qp_anchor
from resample_bspline2 import resample_bspline_eps
import heads  # PLAN_BLOCKSUM_HEAD: blocksum / bspline action heads
resample_math.RESAMPLERS["bspline_eps_satfix"] = resample_math._satfix_wrap(lambda b, k: resample_bspline_eps(b, k, eps=0.005))  # PREREG_BSPLINE eps
resample_math.RESAMPLERS["bspline_eps05_satfix"] = resample_math._satfix_wrap(lambda b, k: resample_bspline_eps(b, k, eps=0.05))  # 1X verify: looser eps bracket
resample_math.RESAMPLERS["bspline_eps_raw"] = lambda b, k: resample_bspline_eps(b, k, eps=0.005)  # 1X verify: paper +BSP, no satfix (satfix at k=1 is identity)
resample_math.RESAMPLERS["bspline_eps05_raw"] = lambda b, k: resample_bspline_eps(b, k, eps=0.05)
resample_math.RESAMPLERS["qp_eps05"] = lambda b, k: resample_qp(b, k, eps=0.05)  # PREREG_1X: bounded-position-error QP
resample_math.RESAMPLERS["qp_eps10"] = lambda b, k: resample_qp(b, k, eps=0.10)

POLICIES = {"dp": DiffusionPolicy, "fm": FlowMatchingPolicy}

K = 2  # overridden by --k
FOLD = -1  # RoboCasa held-out fold (--fold); -1 = legacy split
ARMS = ["native", "zoh", "spline", "spline_satfix", "pchip", "pchip_satfix", "tac_fold", "tac_fold_satfix", "bspline", "bspline_satfix", "gripper_sync"]
MANISKILL = {"PickCube-v1": dict(h5="pick_rl_joint", ctrl="pd_joint_delta_pos", gripper=True, max_steps=100),
             "PushT-v1":    dict(h5="pusht_rl",       ctrl="pd_ee_delta_pose",    gripper=False, max_steps=150)}
# 2026-09-23 saturation-screened expansion: official ManiSkill RL demos, joint-delta controller,
# gripper convention as PickCube, max_steps = each env's own default (from the demo json).
MANISKILL.update({t: dict(h5=f"raw/{t}/rl/trajectory.none.pd_joint_delta_pos.physx_cuda", ctrl="pd_joint_delta_pos",
                          gripper=t != "AnymalC-Reach-v1", max_steps=m)
                  for t, m in [("RollBall-v1", 80), ("PullCube-v1", 50), ("LiftPegUpright-v1", 50), ("PushCube-v1", 50),
                               ("AnymalC-Reach-v1", 200), ("PokeCube-v1", 50), ("StackCube-v1", 50)]})
ROBOMIMIC = {"lift": 200, "can": 400, "square": 400}  # max steps (DP paper)
RM_OBS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]  # DP low-dim obs
ROBOCASA = {"RC-OpenDrawer": dict(env_task="OpenDrawer", max_steps=500, n_hold=6),
            "RC-PnPCounterToStove": dict(env_task="PnPCounterToStove", max_steps=500, n_hold=6),
            # B-spline Policy (arXiv 2607.09648) Table 2a tasks; horizons from robocasa dataset_registry
            "RC-TurnOffSinkFaucet": dict(env_task="TurnOffSinkFaucet", max_steps=500, n_hold=6),
            "RC-CoffeePressButton": dict(env_task="CoffeePressButton", max_steps=300, n_hold=6),
            "RC-TurnOffMicrowave": dict(env_task="TurnOffMicrowave", max_steps=500, n_hold=6),
            "RC-CloseSingleDoor": dict(env_task="CloseSingleDoor", max_steps=500, n_hold=6)}
BRIDGE_HOST, BRIDGE_PORT = "127.0.0.1", 8765


def to_np(x): return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    ao, bo = int((a & ~b).sum()), int((~a & b).sum()); d = ao + bo
    if d == 0: return ao, bo, 1.0
    return ao, bo, float(min(1.0, 2 * sum(math.comb(d, i) for i in range(min(ao, bo) + 1)) / 2 ** d))


def apply_arm(chunk, arm, n_hold):
    """Apply one arm to a clipped ``(T, A)`` policy chunk.

    Continuous delta dimensions are decimated and reconstructed at the configured K.
    Trailing dimensions are causal-held except that ``gripper_sync`` reconstructs the
    first trailing dimension (the gripper) from one transition offset/value summary per
    block, on top of its default ``spline_satfix`` continuous reconstruction.
    """
    if arm == "native": return chunk
    nd = chunk.shape[1] - n_hold
    out = decimate_and_resample(chunk[:, :nd], K, arm)
    if n_hold:
        T = len(chunk); nb = T // K
        g = np.repeat(chunk[:nb * K:K, nd:], K, 0)
        if nb * K < T: g = np.concatenate([g, chunk[nb * K:, nd:]], 0)
        out = np.concatenate([out, g], 1)
        if arm == "gripper_sync":
            transition_summary = coarsen_gripper_transitions(chunk[:, nd], K)
            out = gripper_sync(out, transition_summary, n_hold, K)
    return out.astype(np.float32)


def apply_arm_ctx(pred, arm, n_hold, prev_bs):
    """``<base>_ctx`` arm: resample the executed 8 steps with context instead of in isolation.

    Block sums = [previously executed blocks | all blocks of the clipped predicted horizon
    (executed 8 steps + the unexecuted tail the step head already predicts)]; run the base
    resampler over that whole sequence and keep the executed window. Block-sum conservation
    and the box are per-block, so qp_anchor_ctx keeps both on the executed blocks.
    Gripper/trailing dims are causal-held exactly as in apply_arm.
    """
    assert 8 % K == 0, "ctx arms need whole blocks in the 8-step executed chunk"
    nd = pred.shape[1] - n_hold
    bs = resample_math.coarsen_delta(pred[:, :nd], K)
    p = 0 if prev_bs is None else len(prev_bs)
    if p: bs = np.concatenate([prev_bs, bs])
    out = apply_arm(pred[:8], "zoh", n_hold)
    out[:, :nd] = resample_math.RESAMPLERS[arm](bs, K)[p * K:p * K + 8]
    return out.astype(np.float32)


# ------------------------------------------------------------------ sim adapters
class ManiSkillSim:
    def __init__(self, task):
        import gymnasium as gym, mani_skill.envs, h5py  # noqa
        cfg = MANISKILL[task]; self.cfg = cfg
        self.env = gym.make(task, num_envs=1, obs_mode="state", control_mode=cfg["ctrl"],
                            sim_backend="physx_cuda", reconfiguration_freq=1)
        self.u = self.env.unwrapped; self.has_gripper = cfg["gripper"]; self.n_hold = 1 if cfg["gripper"] else 0
        self.max_steps = cfg["max_steps"]
        self.meta = json.load(open(f"{MANISKILL_DATA_DIR}/{cfg['h5']}.json"))["episodes"]
        self.h5 = h5py.File(f"{MANISKILL_DATA_DIR}/{cfg['h5']}.h5", "r")
        self.eligible = [e for e in self.meta if len(self.h5[f"traj_{int(e['episode_id'])}"]["actions"]) >= 16]

    def _state(self, ep):
        sg = self.h5[f"traj_{int(ep['episode_id'])}"]["env_states"]
        return {g: {n: torch.as_tensor(np.asarray(sg[g][n])[0:1]) for n in sg[g]} for g in sg}

    def split(self, n_train, n_eval, offset=0):  # plenty of demos -> eval from held-out demo init states
        a = n_train + offset; eps = self.eligible[:n_train], self.eligible[a:a + n_eval]
        assert len(eps[1]) == n_eval, f"not enough held-out demos: {len(eps[1])}"
        return eps

    def reset_to(self, ep):
        self.env.reset(seed=ep["episode_seed"]); self.u.set_state_dict(self._state(ep))
        return to_np(self.u.get_obs()).reshape(-1).astype(np.float32)

    def step(self, a):
        obs, _, term, trunc, info = self.env.step(a[None])
        s = bool(to_np(info.get("success")).reshape(-1)[0])
        done = bool(to_np(term).reshape(-1)[0]) or bool(to_np(trunc).reshape(-1)[0])
        return to_np(obs).reshape(-1).astype(np.float32), s, done

    def harvest(self, eps):
        """No obs stored in these h5s -> re-simulate each demo with its recorded (pre-clip) actions."""
        O, A = [], []
        for i, ep in enumerate(eps):
            acts = np.asarray(self.h5[f"traj_{int(ep['episode_id'])}"]["actions"], np.float32)
            obs = self.reset_to(ep); ob = []
            for t in range(len(acts)):
                ob.append(obs); obs, _, done = self.step(np.clip(acts[t], -1, 1))
                if done: break
            n = len(ob); O.append(np.stack(ob)); A.append(acts[:n])  # train on the RL policy's raw output
            if (i + 1) % 50 == 0: print(f"[harvest] {i+1}/{len(eps)}", flush=True)
        return O, A

    def close(self): self.env.close(); self.h5.close()


class RoboMimicSim:
    def __init__(self, task):
        import robosuite, h5py
        self.f = h5py.File(f"{ROBOMIMIC_DATA_DIR}/{task}.hdf5", "r")
        env_meta = json.loads(self.f["data"].attrs["env_args"])
        self.env = robosuite.make(env_meta["env_name"], **env_meta["env_kwargs"])
        self.has_gripper = True; self.n_hold = 1; self.max_steps = ROBOMIMIC[task]
        self.eligible = [d for d in self.f["data"].keys() if self.f["data"][d]["actions"].shape[0] >= 16]

    _LIVE = {"object": "object-state"}  # robomimic hdf5 key -> live robosuite obs key
    def _obs_from_dict(self, od): return np.concatenate([np.asarray(od[self._LIVE.get(k, k)], np.float32).reshape(-1) for k in RM_OBS])

    def split(self, n_train, n_eval):  # eval "episodes" are reset seeds, not demos
        return self.eligible[:n_train], list(range(n_eval))

    def reset_to(self, seed):
        np.random.seed(10_000 + seed)  # robosuite placement samplers use the global np RNG
        od = self.env.reset()
        return self._obs_from_dict(od)

    def step(self, a):
        od, _, _, _ = self.env.step(a)
        return self._obs_from_dict(od), bool(self.env._check_success()), False

    def harvest(self, eps):
        O, A = [], []
        for d in eps:
            g = self.f["data"][d]
            O.append(np.concatenate([np.asarray(g["obs"][k], np.float32).reshape(len(g["actions"]), -1) for k in RM_OBS], 1))
            A.append(np.asarray(g["actions"], np.float32))
        return O, A

    def close(self): self.env.close(); self.f.close()


class RoboCasaSim:
    """Socket client to robocasa_bridge.py (runs in the robocasa_uv venv -- robosuite/robocasa
    hard-assert numpy<1.24, which can't share a process with the Blackwell-capable torch this
    policy needs). Wire format: 4-byte big-endian length prefix + pickle, localhost only."""
    def __init__(self, task, host=BRIDGE_HOST, port=BRIDGE_PORT, random_eval=False, eval_seed=0):
        cfg = ROBOCASA[task]; self.task = task; self.env_task = cfg["env_task"]
        self.random_eval = random_eval; self.eval_seed = eval_seed
        self.n_hold = cfg["n_hold"]; self.has_gripper = True; self.max_steps = cfg["max_steps"]
        self.sock = socket.create_connection((host, port), timeout=60)
        self._send({"cmd": "ping"}); assert self._recv()["ok"]
        n = self._send_recv({"cmd": "n_demos", "task": self.env_task})["n"]
        self.n_demos = n

    def _send(self, obj):
        data = pickle.dumps(obj, protocol=4)
        self.sock.sendall(struct.pack(">I", len(data)) + data)

    def _recv(self):
        hdr = b""
        while len(hdr) < 4:
            chunk = self.sock.recv(4 - len(hdr))
            if not chunk: raise ConnectionError("bridge closed connection")
            hdr += chunk
        (n,) = struct.unpack(">I", hdr)
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(min(65536, n - len(buf)))
            if not chunk: raise ConnectionError("bridge closed mid-message")
            buf += chunk
        return pickle.loads(buf)

    def _send_recv(self, obj):
        self._send(obj); return self._recv()

    def split(self, n_train, n_eval):  # demo indices, or fresh deterministic reset seeds for eval
        if self.random_eval:
            if FOLD < 0:
                train = list(range(n_train))
            else:
                held = set(range(FOLD * 15, min((FOLD + 1) * 15, self.n_demos)))
                train = [i for i in range(self.n_demos) if i not in held][:n_train]
            assert len(train) == n_train, f"only {self.n_demos} RoboCasa demos, need {n_train} for training"
            return train, [("random", 1_000_000 + self.eval_seed * n_eval + i) for i in range(n_eval)]
        assert n_train + n_eval <= self.n_demos, f"only {self.n_demos} RoboCasa demos, need {n_train+n_eval}"
        if FOLD < 0: return list(range(n_train)), list(range(n_train, n_train + n_eval))
        ev = list(range(FOLD * n_eval, (FOLD + 1) * n_eval)); assert ev[-1] < self.n_demos, "fold beyond demo count"
        return [i for i in range(self.n_demos) if i not in ev][:n_train], ev

    def reset_to(self, idx):
        if isinstance(idx, tuple) and idx[0] == "random":
            return self._send_recv({"cmd": "reset_random", "task": self.env_task, "seed": idx[1]})["obs"]
        return self._send_recv({"cmd": "reset_to", "task": self.env_task, "idx": idx})["obs"]

    def step(self, a):
        r = self._send_recv({"cmd": "step", "task": self.env_task, "action": a})
        return r["obs"], r["success"], r["done"]

    def harvest(self, idx_list):
        r = self._send_recv({"cmd": "harvest", "task": self.env_task, "idx": idx_list})
        return r["O"], r["A"]

    def close(self):
        try: self._send_recv({"cmd": "close", "task": self.env_task})
        except Exception: pass
        self.sock.close()


# ------------------------------------------------------------------ eval (shared by sequential + parallel paths)
def make_sim(task, fold, port):
    if task in MANISKILL: return ManiSkillSim(task)
    if task in ROBOMIMIC: return RoboMimicSim(task)
    if task in ROBOCASA: return RoboCasaSim(task, port=port)
    raise ValueError(f"unknown task {task}")


def run_episode(sim, policy, onorm, anorm, dev, ei, ep, head, K, ND, nblocks):
    """One episode, every arm in ARMS. Identical whether called sequentially or from a
    worker process -- the only thing that can legitimately change eval numbers is which
    process executes this function, never its logic. Seed is a pure function of (ei, replan),
    independent of arm and of execution order/worker assignment -- parallel-safe by construction
    (see harness.py module docstring: "paired noise per (episode, replan)")."""
    out = {}
    for arm in ARMS:
        obs = sim.reset_to(ep); hist = [obs, obs]; s = False; steps = 0; sat_e = []
        replan = 0; prev_bs = None
        while steps < sim.max_steps and not s:
            o = torch.as_tensor(onorm.norm(np.stack(hist[-2:]))[None], device=dev)
            torch.manual_seed(1_000_003 * ei + replan)  # paired noise across arms
            if head == "arc":
                rate_condition = torch.tensor([[math.log2(K) / 2.0]], device=dev)
                sampled = policy.sample(o, rate_condition=rate_condition)
            else:
                sampled = policy.sample(o)
            pred = anorm.denorm(sampled.cpu().numpy()[0])
            if head == "step":
                chunk = pred[:8]; sat_e.append(float((np.abs(chunk) > 1).mean()))
                if arm == "mlp_bc":  # small network alone as the policy (DP proposal ignored)
                    import shape_governor
                    exec_chunk = shape_governor.decode_bc(os.environ["SHAPE_BC_PATH"], np.stack(hist[-2:]))
                elif arm in ("qp_learned", "learned_raw", "learned_tanh"):  # shape_governor.py: learned anchor (+ governor) / tanh net
                    import shape_governor
                    nd = chunk.shape[1] - sim.n_hold; exec_chunk = apply_arm(np.clip(chunk, -1, 1).astype(np.float32), "zoh", sim.n_hold)
                    exec_chunk[:, :nd] = shape_governor.decode(os.environ["SHAPE_TANH_PATH" if arm == "learned_tanh" else "SHAPE_GOV_PATH"], np.stack(hist[-2:]),
                                                               np.clip(chunk[:, :nd], -1, 1).astype(np.float32), arm == "qp_learned")
                elif arm.endswith("_ctx"):
                    clipped = np.clip(pred, -1, 1).astype(np.float32)
                    exec_chunk = apply_arm_ctx(clipped, arm[:-4], sim.n_hold, prev_bs)
                    prev_bs = resample_math.coarsen_delta(clipped[:8, :clipped.shape[1] - sim.n_hold], K)
                else:
                    exec_chunk = apply_arm(np.clip(chunk, -1, 1).astype(np.float32), arm, sim.n_hold)
            else:
                exec_chunk = heads.decode(pred, head, ND, K, arm, nblocks)
                sat_e.append(float((np.abs(exec_chunk[:, :ND]) > 1).mean())); exec_chunk = np.clip(exec_chunk, -1, 1)
            for a in exec_chunk:
                obs, s_t, done = sim.step(a); hist.append(obs); steps += 1; s |= s_t
                if s or done or steps >= sim.max_steps: break
            replan += 1
        out[arm] = (s, float(np.mean(sat_e)))
    return out


def _eval_sequential(sim, policy, onorm, anorm, dev, eval_eps, args, ND):
    succ = {a: [] for a in ARMS}; sat = {a: [] for a in ARMS}
    for ei, ep in enumerate(eval_eps):
        res = run_episode(sim, policy, onorm, anorm, dev, ei, ep, args.head, K, ND, args.nblocks)
        for a in ARMS: succ[a].append(res[a][0]); sat[a].append(res[a][1])
        if (ei + 1) % 10 == 0 or ei + 1 == len(eval_eps):
            print(f"[eval] [{ei+1}/{len(eval_eps)}] " + " ".join(f"{a}={sum(succ[a])}" for a in ARMS), flush=True)
    return succ, sat


_WORKER = {}  # per-process globals, set once by _worker_init


def _worker_init(task, fold, port, policy_name, ckpt, obs_dim, act_dim, head_kwargs,
                  onorm_state, anorm_state, dev, k, arms, fold_global):
    global K, ARMS, FOLD
    K, ARMS, FOLD = k, arms, fold_global
    _WORKER["sim"] = make_sim(task, fold, port)
    policy = POLICIES[policy_name](obs_dim, act_dim, **head_kwargs)
    policy.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    policy.to(dev).eval()
    _WORKER["policy"] = policy
    _WORKER["onorm"] = MinMax.from_state(onorm_state)
    _WORKER["anorm"] = MinMax.from_state(anorm_state)
    _WORKER["dev"] = dev


def _worker_run(job):
    ei, ep, head, nd, nblocks = job
    w = _WORKER
    res = run_episode(w["sim"], w["policy"], w["onorm"], w["anorm"], w["dev"], ei, ep, head, K, nd, nblocks)
    return ei, res


def _eval_parallel(n_workers, args, ND, eval_eps, ckpt, obs_dim, act_dim, head_kwargs, onorm, anorm):
    """Same math as _eval_sequential (both call run_episode, unmodified) -- only the
    distribution across processes differs. Correctness rests entirely on run_episode's
    seed being a pure function of (ei, replan): which worker/process computes a given
    episode cannot change its result. Validate with --workers 1 vs N on a small --n_eval
    before trusting a real campaign to this path (see VALIDATE_WORKERS.md)."""
    onorm_state, anorm_state = onorm.state(), anorm.state()
    ctx = mp.get_context("spawn")
    jobs = [(ei, ep, args.head, ND, args.nblocks) for ei, ep in enumerate(eval_eps)]
    results = {}
    with ctx.Pool(
        n_workers, initializer=_worker_init,
        initargs=(args.task, args.fold, args.port, args.policy, ckpt, obs_dim, act_dim, head_kwargs,
                  onorm_state, anorm_state, "cuda", K, ARMS, FOLD),
    ) as pool:
        done = 0
        for ei, res in pool.imap_unordered(_worker_run, jobs):
            results[ei] = res; done += 1
            if done % 10 == 0 or done == len(jobs):
                succ_so_far = {a: sum(1 for e in range(done) if e in results and results[e][a][0]) for a in ARMS}
                print(f"[eval] [{done}/{len(jobs)}] " + " ".join(f"{a}={succ_so_far[a]}" for a in ARMS), flush=True)
    succ = {a: [] for a in ARMS}; sat = {a: [] for a in ARMS}
    for ei in range(len(eval_eps)):  # reassemble in strict episode order -- output identical to sequential
        for a in ARMS:
            succ[a].append(results[ei][a][0]); sat[a].append(results[ei][a][1])
    return succ, sat


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--n_train", type=int, default=200); ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--policy", choices=list(POLICIES), default="dp")
    ap.add_argument("--k", type=int, default=2); ap.add_argument("--arms", default=None, help="comma list; default all")
    ap.add_argument("--suffix", default="", help="appended to the result tag (never to the checkpoint)")
    ap.add_argument("--checkpoint-suffix", default="", help="isolates checkpoint provenance")
    ap.add_argument("--head", choices=["step", "blocksum", "bspline", "arc"], default="step"); ap.add_argument("--nblocks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="training seed; eval episodes and paired noise are seed-independent")
    ap.add_argument("--fold", type=int, default=-1, help="RoboCasa: held-out fold index (n_eval demos per fold)"); ap.add_argument("--port", type=int, default=BRIDGE_PORT)
    ap.add_argument("--rc-random-eval", action="store_true", help="RoboCasa: evaluate deterministic fresh resets instead of held-out demo states")
    ap.add_argument("--eval-seed", type=int, default=0, help="RoboCasa random-reset seed block")
    ap.add_argument("--workers", type=int, default=1, help="parallel eval processes across episodes; 1=sequential (default, unchanged behavior). >1 must produce identical results (see VALIDATE_WORKERS.md) -- pairing depends only on (episode, replan), never on which worker runs it.")
    ap.add_argument("--eval-offset", type=int, default=0, help="ManiSkill: skip this many held-out demos before the eval window (dev vs confirmation windows)")
    ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    if args.smoke: args.steps, args.n_train, args.n_eval = 50, 3, 2
    dev = "cuda"; torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.task in MANISKILL: sim = ManiSkillSim(args.task)
    elif args.task in ROBOMIMIC: sim = RoboMimicSim(args.task)
    elif args.task in ROBOCASA: sim = RoboCasaSim(args.task, port=args.port, random_eval=args.rc_random_eval, eval_seed=args.eval_seed)
    else: raise ValueError(f"unknown task {args.task}")
    global K, ARMS, FOLD
    K = args.k; FOLD = args.fold
    if args.arms: ARMS = [a.strip() for a in args.arms.split(",")]
    elif args.head == "arc": ARMS = ["zoh", "tac_fold", "tac_fold_satfix"]
    hsfx = ("" if args.head == "step" else f"_{args.head}" + (f"_b{args.nblocks}" if args.head == "blocksum" and args.nblocks != 4 else "")) + (f"_s{args.seed}" if args.seed else "") + (f"_f{args.fold}" if args.fold >= 0 else "")
    if args.head != "step": assert args.policy == "dp", "heads implemented for dp only"
    tag = args.policy + "_" + args.task + hsfx + (f"_k{K}" if K != 2 else "") + args.suffix + ("_smoke" if args.smoke else "")
    out_target = f"{HERE}/result_{tag}.json"
    if os.path.exists(out_target) and not args.smoke:
        try:
            with open(out_target) as _fp: _d = json.load(_fp)
            _s = _d.get("success", {})
            req_proto = "random_reset" if args.rc_random_eval else "heldout_demo_state"
            if (len(_s) > 0 and all(a in _s and len(_s[a]) == args.n_eval for a in ARMS)
                    and _d.get("eval_protocol") == req_proto
                    and _d.get("eval_seed") == args.eval_seed
                    and _d.get("n_train") == args.n_train
                    and _d.get("train_steps") == args.steps
                    and _d.get("eval_offset", 0) == args.eval_offset):
                print(f"[skip] {out_target} already complete ({args.n_eval} eps, all arms present). Skipping.")
                sim.close()
                return
        except Exception:
            pass
    train_eps, eval_eps = sim.split(args.n_train, args.n_eval, **({"offset": args.eval_offset} if args.eval_offset else {}))

    cache = f"{HERE}/demos_{args.task}_{args.n_train}" + (f"_f{args.fold}" if args.fold >= 0 else "") + ".npz"  # harvested demos are deterministic per (task, n_train)
    t0 = time.time()
    if os.path.exists(cache) and not args.smoke:
        z = np.load(cache, allow_pickle=True); O, A = list(z["O"]), list(z["A"]); print(f"[data] loaded {cache}")
    else:
        O, A = sim.harvest(train_eps)
        if not args.smoke:  # atomic: parallel streams of the same task write identical content
            tmp = f"{cache[:-4]}.{os.getpid()}.tmp.npz"; np.savez(tmp, O=np.array(O, dtype=object), A=np.array(A, dtype=object)); os.replace(tmp, cache)
    print(f"[data] {len(O)} demos harvested in {time.time()-t0:.0f}s", flush=True)
    onorm, anorm = MinMax(np.concatenate(O)), MinMax(np.concatenate(A))
    ND = A[0].shape[-1] - sim.n_hold  # continuous (resampled) dims; trailing dims are causal-hold
    rate_conditions = None
    if args.head == "step":
        chunks = [make_chunks(onorm.norm(o), anorm.norm(a), 2, 16) for o, a in zip(O, A)]
        OC, AC = np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])
    elif args.head == "arc":
        chunks = [make_chunks(onorm.norm(o), a, 2, heads.ARC_BLOCKS * max(heads.ARC_RATES))
                  for o, a in zip(O, A)]
        base_oc = np.concatenate([c[0] for c in chunks])
        raw = np.concatenate([c[1] for c in chunks])
        targets = [heads.arc_targets(raw, ND, rate) for rate in heads.ARC_RATES]
        OC = np.concatenate([base_oc for _ in heads.ARC_RATES])
        AC = np.concatenate(targets)
        rate_conditions = np.concatenate([
            np.full((len(base_oc), 1), math.log2(rate) / 2.0, np.float32)
            for rate in heads.ARC_RATES
        ])
        anorm = MinMax(AC.reshape(-1, AC.shape[-1])); AC = anorm.norm(AC)
    else:  # head targets from RAW 16-step windows; normaliser fit on the targets themselves
        chunks = [make_chunks(onorm.norm(o), a, 2, 16) for o, a in zip(O, A)]
        OC = np.concatenate([c[0] for c in chunks])
        AC = heads.targets(np.concatenate([c[1] for c in chunks]), args.head, ND, args.nblocks)
        anorm = MinMax(AC.reshape(-1, AC.shape[-1])); AC = anorm.norm(AC)
    print(f"[data] {len(OC)} windows, obs_dim={OC.shape[-1]}, act_dim={AC.shape[-1]}, "
          f"raw |a|>1 frac={float((np.abs(np.concatenate(A))>1).mean()):.3f}", flush=True)

    ckpt = (f"{HERE}/{args.policy}_{args.task}{hsfx}{args.checkpoint_suffix}" +
            ("_smoke" if args.smoke else "") + ".pt")  # k-agnostic
    head_kwargs = ({"horizon": AC.shape[1]} if args.head != "step" else {})
    if args.head == "arc": head_kwargs["rate_condition_dim"] = 1
    policy = POLICIES[args.policy](OC.shape[-1], AC.shape[-1], **head_kwargs)
    if os.path.exists(ckpt) and not args.smoke:
        policy.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True)); policy.to(dev).eval(); print(f"[train] loaded {ckpt}")
    else:
        policy = train(policy, OC, AC, steps=args.steps, dev=dev,
                       rate_conditions=rate_conditions)
        checkpoint_tmp = f"{ckpt}.{os.getpid()}.partial"
        torch.save(policy.state_dict(), checkpoint_tmp)
        os.replace(checkpoint_tmp, ckpt)
        print(f"[train] saved {ckpt}")
    policy.eval()
    if any(a in ("qp_learned", "learned_raw", "learned_tanh") for a in ARMS):  # fit on the training demos only; env var reaches spawned workers
        import shape_governor
        for env, tanh in (("SHAPE_GOV_PATH", False), ("SHAPE_TANH_PATH", True)):
            sp = f"{HERE}/shape{'_tanh' if tanh else ''}_{args.task}_k{K}_n{args.n_train}.pt"
            if not os.path.exists(sp): torch.save(shape_governor.fit(O, A, K, ND, tanh=tanh), sp); print(f"[shape] saved {sp}")
            os.environ[env] = sp
    if "mlp_bc" in ARMS:
        import shape_governor
        sp = f"{HERE}/shape_bc_{args.task}_n{args.n_train}.pt"
        if not os.path.exists(sp): torch.save(shape_governor.fit_bc(O, A, ND), sp); print(f"[shape] saved {sp}")
        os.environ["SHAPE_BC_PATH"] = sp

    # ---- eval
    t_eval = time.time()
    n_workers = max(1, args.workers)
    if n_workers == 1:
        succ, sat = _eval_sequential(sim, policy, onorm, anorm, dev, eval_eps, args, ND)
        sim.close()
    else:
        sim.close()  # not reused for parallel eval -- each worker builds its own
        succ, sat = _eval_parallel(n_workers, args, ND, eval_eps, ckpt, OC.shape[-1], AC.shape[-1], head_kwargs, onorm, anorm)
    print(f"[eval] wall {time.time()-t_eval:.0f}s for {len(eval_eps)} eps x {len(ARMS)} arms"
          f"{f' ({n_workers} workers)' if n_workers > 1 else ''}", flush=True)

    S = {a: np.asarray(succ[a], bool) for a in ARMS}
    print(f"\n=== {args.task} closed-loop {args.policy.upper()}, k={K}, n={len(eval_eps)} ===")
    res = {"task": args.task, "policy": args.policy, "k": K, "head": args.head, "seed": args.seed, "fold": args.fold,
           "n": len(eval_eps), "eval_offset": args.eval_offset, "n_train": args.n_train, "train_steps": args.steps, "arms": ARMS,
           "checkpoint_path": os.path.abspath(ckpt), "checkpoint_sha256": sha256_file(ckpt),
           "eval_protocol": "random_reset" if args.rc_random_eval else "heldout_demo_state", "eval_seed": args.eval_seed,
           "success": {a: succ[a] for a in ARMS}, "policy_raw_sat_frac": {a: float(np.mean(sat[a])) for a in ARMS}, "contrasts": {}}
    if args.head == "arc": res["arc_training_rates"] = list(heads.ARC_RATES)
    for a in ARMS:
        line = f"  {a:16s} {100*S[a].mean():6.1f}% ({S[a].sum()}/{len(S[a])})"
        res["contrasts"][a] = {}
        refs = ["zoh", "native"]
        if a == "gripper_sync":
            refs.extend(["spline_satfix", "bspline_eps_satfix"])
        for ref in refs:
            if a == ref or ref not in S: continue  # arms list may omit zoh
            ao, bo, p = exact_mcnemar(S[a], S[ref]); d = 100 * (S[a].mean() - S[ref].mean())
            res["contrasts"][a][f"vs_{ref}"] = {"delta_pp": d, "p": p, "a_only": ao, "ref_only": bo}
            line += f"  | vs {ref}: {d:+6.1f}pp p={p:.3g}"
        print(line)
    print(f"  policy raw |a|>1 frac (pre-clip): {res['policy_raw_sat_frac'][ARMS[0]]:.3f}")
    out = f"{HERE}/result_{tag}.json"
    result_tmp = f"{out}.{os.getpid()}.partial"
    with open(result_tmp, "w") as stream:
        json.dump(res, stream, indent=2)
    os.replace(result_tmp, out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
