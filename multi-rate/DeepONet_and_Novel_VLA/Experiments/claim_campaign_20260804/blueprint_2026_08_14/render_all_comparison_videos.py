import os, sys, json, numpy as np, h5py, torch
import gymnasium as gym
from scipy.interpolate import CubicSpline
import imageio
import mani_skill.envs

OUT_DIR = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/blueprint_2026_08_14/comparison_videos"
os.makedirs(OUT_DIR, exist_ok=True)
D = os.path.expanduser("~/maniskill_data")

meta = json.load(open(f"{D}/pick_rl_joint.json"))
eps = meta["episodes"]

def fold(coarse, n):
    k = n // coarse.shape[0]
    return np.repeat(coarse / k, k, axis=0)

def spline(coarse, n):
    x = np.linspace(0, 1, coarse.shape[0])
    return CubicSpline(x, coarse, axis=0)(np.linspace(0, 1, n)) * coarse.shape[0] / n

print("Initializing PickCube-v1 environment with rgb_array render mode...")
env = gym.make("PickCube-v1", num_envs=1, obs_mode="state",
               render_mode="rgb_array", control_mode="pd_joint_delta_pos",
               sim_backend="physx_cpu")

N_VIDEOS = 5
rendered_count = 0

with h5py.File(f"{D}/pick_rl_joint.h5", "r") as h:
    for ep_idx, e in enumerate(eps):
        if rendered_count >= N_VIDEOS:
            break
        a = np.array(h[f"traj_{e['episode_id']}"]["actions"]).astype(np.float32)
        n = (len(a)//2)*2; a = a[:n]
        coarse = a.reshape(n//2, 2, a.shape[1]).sum(1)
        
        arms = {
            "spline": spline(coarse, n).astype(np.float32),
            "folding": fold(coarse, n).astype(np.float32)
        }
        g = h[f"traj_{e['episode_id']}"]["env_states"]
        st = {grp: {nm: torch.as_tensor(np.array(g[grp][nm])[0:1]) for nm in g[grp]} for grp in g}
        
        arm_frames = {}
        arm_success = {}
        
        for arm_name, acts in arms.items():
            env.reset(seed=e["episode_seed"])
            env.unwrapped.set_state_dict(st)
            frames = []
            ok = False
            
            # Initial frame
            frame = env.render()
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]
            frames.append(frame)
            
            for t in range(len(acts)):
                _, _, term, trunc, info = env.step(acts[t][None])
                s = info.get("success")
                if s is not None and bool(np.asarray(s).reshape(-1)[0]):
                    ok = True
                f = env.render()
                if isinstance(f, torch.Tensor):
                    f = f.cpu().numpy()
                if f.ndim == 4:
                    f = f[0]
                frames.append(f)
                if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                    break
            arm_frames[arm_name] = frames
            arm_success[arm_name] = ok

        # Look for discordant cases where folding succeeded and spline failed
        is_discordant = arm_success["folding"] and not arm_success["spline"]
        if is_discordant or rendered_count < 2:
            max_len = max(len(arm_frames["spline"]), len(arm_frames["folding"]))
            
            def pad_frames(fr_list, target_len):
                padded = list(fr_list)
                while len(padded) < target_len:
                    padded.append(fr_list[-1])
                return padded
            
            f_spline = pad_frames(arm_frames["spline"], max_len)
            f_folding = pad_frames(arm_frames["folding"], max_len)
            
            side_by_side = [np.concatenate([s_fr, fold_fr], axis=1) for s_fr, fold_fr in zip(f_spline, f_folding)]
            
            out_file = f"{OUT_DIR}/pickcube_ep{e['episode_id']}_fold_{arm_success['folding']}_spline_{arm_success['spline']}.mp4"
            imageio.mimsave(out_file, side_by_side, fps=20)
            print(f"[Saved Video {rendered_count+1}/{N_VIDEOS}] {out_file} (Folding: {arm_success['folding']}, Spline: {arm_success['spline']})")
            rendered_count += 1

env.close()
print(f"All {rendered_count} side-by-side comparison videos generated in {OUT_DIR}")
