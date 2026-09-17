"""Closed-Loop Parallel Evaluation of DeepONet vs. Flow Matching on ManiSkill PickCube RL.

Evaluates under closed-loop replanning (replan=4) across n=300 episodes.
"""
import os, sys, json, time, math
import numpy as np, h5py, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import CubicSpline
import gymnasium as gym
import mani_skill.envs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_PATH = os.path.expanduser("~/maniskill_data")
OUT_DIR = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/blueprint_2026_08_14"

# -------------------------------------------------------------
# 1. Dataset Extraction: All (s_t -> Action Chunk (H, 8))
# -------------------------------------------------------------
def get_flat_s(o):
    if isinstance(o, dict):
        return np.concatenate([v.cpu().numpy().reshape(-1) if isinstance(v, torch.Tensor) else np.array(v).reshape(-1) for v in o.values()])
    elif isinstance(o, torch.Tensor):
        return o.cpu().numpy().reshape(-1)
    return np.array(o).reshape(-1)

def build_closed_loop_dataset(chunk_size=16):
    print(f"Extracting transition dataset (chunk_size={chunk_size})...")
    meta = json.load(open(f"{D_PATH}/pick_rl_joint.json"))
    eps = meta["episodes"]
    
    env = gym.make("PickCube-v1", num_envs=1, obs_mode="state",
                   control_mode="pd_joint_delta_pos", sim_backend="physx_cpu")
    
    all_states = []
    all_chunks = []
    
    with h5py.File(f"{D_PATH}/pick_rl_joint.h5", "r") as h:
        for idx, e in enumerate(eps):
            a = np.array(h[f"traj_{e['episode_id']}"]["actions"]).astype(np.float32)
            n = len(a)
            if n < chunk_size:
                continue
            
            g = h[f"traj_{e['episode_id']}"]["env_states"]
            st = {grp: {nm: torch.as_tensor(np.array(g[grp][nm])[0:1]) for nm in g[grp]} for grp in g}
            
            obs, _ = env.reset(seed=e["episode_seed"])
            env.unwrapped.set_state_dict(st)
            
            for t in range(0, n - chunk_size + 1, 4): # stride 4
                s_t = get_flat_s(obs)
                chunk = a[t:t+chunk_size]
                all_states.append(s_t)
                all_chunks.append(chunk)
                
                # step env forward
                for step_i in range(t, min(t+4, n)):
                    obs, _, term, trunc, _ = env.step(a[step_i][None])
                    if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                        break
            if (idx + 1) % 250 == 0:
                print(f"  Processed {idx+1}/{len(eps)} trajectories...")
                
    env.close()
    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_chunks, dtype=np.float32)
    print(f"Closed-loop dataset ready: States {states.shape}, Chunks {actions.shape}")
    return states, actions, eps

# -------------------------------------------------------------
# 2. Architectures
# -------------------------------------------------------------
class FourierTrunk(nn.Module):
    def __init__(self, n_fourier=16, hidden_dim=256, p_dim=128, act_dim=8):
        super().__init__()
        self.n_fourier = n_fourier
        in_dim = 1 + 2 * n_fourier
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, p_dim * act_dim)
        )
        self.p_dim = p_dim
        self.act_dim = act_dim

    def forward(self, tau):
        if tau.ndim == 1:
            tau = tau.unsqueeze(-1)
        freqs = torch.arange(1, self.n_fourier + 1, device=tau.device, dtype=tau.dtype) * 2 * math.pi
        feat = torch.cat([tau, torch.sin(tau * freqs), torch.cos(tau * freqs)], dim=-1)
        out = self.mlp(feat)
        return out.view(*out.shape[:-1], self.p_dim, self.act_dim)

class DeepONetActionHead(nn.Module):
    def __init__(self, state_dim, p_dim=128, act_dim=8, hidden_dim=256):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, p_dim)
        )
        self.trunk = FourierTrunk(n_fourier=16, hidden_dim=hidden_dim, p_dim=p_dim, act_dim=act_dim)
        self.bias = nn.Parameter(torch.zeros(act_dim))

    def forward(self, state, tau_grid):
        b = self.branch(state)
        t = self.trunk(tau_grid)
        return torch.einsum("bp, tpa -> bta", b, t) + self.bias

class FlowMatchingActionHead(nn.Module):
    def __init__(self, state_dim, chunk_len=16, act_dim=8, hidden_dim=256):
        super().__init__()
        self.chunk_len = chunk_len
        self.act_dim = act_dim
        in_dim = (chunk_len * act_dim) + state_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, chunk_len * act_dim)
        )

    def forward(self, x_t, t, state):
        feat = torch.cat([x_t, state, t], dim=-1)
        return self.mlp(feat)

    def sample(self, state, steps=10):
        B = state.shape[0]
        x = torch.randn(B, self.chunk_len * self.act_dim, device=state.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B, 1), i * dt, device=state.device, dtype=torch.float32)
            v = self.forward(x, t, state)
            x = x + v * dt
        return x.view(B, self.chunk_len, self.act_dim)

# -------------------------------------------------------------
# 3. Training & Evaluation
# -------------------------------------------------------------
def train_and_eval_closed_loop(n_eval=300, replan=4):
    chunk_size = 16
    states, actions, eps = build_closed_loop_dataset(chunk_size=chunk_size)
    
    state_dim = states.shape[1]
    act_dim = actions.shape[2]
    
    class DS(Dataset):
        def __init__(self, s, a):
            self.s = torch.tensor(s, dtype=torch.float32)
            self.a = torch.tensor(a, dtype=torch.float32)
        def __len__(self): return len(self.s)
        def __getitem__(self, i): return self.s[i], self.a[i]
        
    loader = DataLoader(DS(states, actions), batch_size=64, shuffle=True)
    
    print("\n--- Training DeepONet (Closed-Loop) ---")
    don = DeepONetActionHead(state_dim=state_dim, p_dim=128, act_dim=act_dim).to(DEVICE)
    opt_don = optim.AdamW(don.parameters(), lr=1e-3, weight_decay=1e-4)
    tau_grid = torch.linspace(0, 1, chunk_size, device=DEVICE).unsqueeze(-1)
    
    for ep in range(30):
        don.train()
        l_sum = 0
        for s_b, a_b in loader:
            s_b, a_b = s_b.to(DEVICE), a_b.to(DEVICE)
            pred = don(s_b, tau_grid)
            loss = nn.functional.mse_loss(pred, a_b)
            opt_don.zero_grad(); loss.backward(); opt_don.step()
            l_sum += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"  [DeepONet Epoch {ep+1:02d}/30] Loss: {l_sum/len(loader):.6f}")
            
    print("\n--- Training Flow Matching (Closed-Loop) ---")
    flow = FlowMatchingActionHead(state_dim=state_dim, chunk_len=chunk_size, act_dim=act_dim).to(DEVICE)
    opt_flow = optim.AdamW(flow.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for ep in range(30):
        flow.train()
        l_sum = 0
        for s_b, a_b in loader:
            s_b = s_b.to(DEVICE)
            a_flat = a_b.to(DEVICE).view(a_b.shape[0], -1)
            x_0 = torch.randn_like(a_flat)
            x_1 = a_flat
            t = torch.rand(a_b.shape[0], 1, device=DEVICE)
            x_t = (1 - t) * x_0 + t * x_1
            pred_v = flow(x_t, t, s_b)
            loss = nn.functional.mse_loss(pred_v, x_1 - x_0)
            opt_flow.zero_grad(); loss.backward(); opt_flow.step()
            l_sum += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"  [Flow Matching Epoch {ep+1:02d}/30] Loss: {l_sum/len(loader):.6f}")

    # Evaluation
    print(f"\n=======================================================")
    print(f"HIGH-CONFIDENCE CLOSED-LOOP EVALUATION (n={n_eval}, replan={replan})")
    print(f"=======================================================")
    
    env = gym.make("PickCube-v1", num_envs=1, obs_mode="state",
                   control_mode="pd_joint_delta_pos", sim_backend="physx_cpu")
                   
    don.eval()
    flow.eval()
    
    arms = ["deeponet_closed_loop", "flow_closed_loop", "deeponet_spline", "deeponet_folding"]
    results = {k: [] for k in arms}
    
    with h5py.File(f"{D_PATH}/pick_rl_joint.h5", "r") as h:
        for idx in range(min(n_eval, len(eps))):
            e = eps[idx]
            g = h[f"traj_{e['episode_id']}"]["env_states"]
            st = {grp: {nm: torch.as_tensor(np.array(g[grp][nm])[0:1]) for nm in g[grp]} for grp in g}
            
            # Evaluate DeepONet closed loop
            obs, _ = env.reset(seed=e["episode_seed"])
            env.unwrapped.set_state_dict(st)
            ok_don = False
            for step in range(0, 50, replan):
                s_t = torch.tensor(get_flat_s(obs)[None], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    acts = don(s_t, tau_grid)[0].cpu().numpy()
                for k in range(min(replan, 50 - step)):
                    obs, _, term, trunc, info = env.step(acts[k][None])
                    s = info.get("success")
                    if s is not None and bool(np.asarray(s).reshape(-1)[0]): ok_don = True
                    if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]): break
                if ok_don: break
            results["deeponet_closed_loop"].append(ok_don)
            
            # Evaluate Flow closed loop
            obs, _ = env.reset(seed=e["episode_seed"])
            env.unwrapped.set_state_dict(st)
            ok_flow = False
            for step in range(0, 50, replan):
                s_t = torch.tensor(get_flat_s(obs)[None], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    acts = flow.sample(s_t, steps=10)[0].cpu().numpy()
                for k in range(min(replan, 50 - step)):
                    obs, _, term, trunc, info = env.step(acts[k][None])
                    s = info.get("success")
                    if s is not None and bool(np.asarray(s).reshape(-1)[0]): ok_flow = True
                    if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]): break
                if ok_flow: break
            results["flow_closed_loop"].append(ok_flow)
            
            # DeepONet with Spline vs Folding
            obs, _ = env.reset(seed=e["episode_seed"])
            env.unwrapped.set_state_dict(st)
            ok_spline = False
            for step in range(0, 50, replan):
                s_t = torch.tensor(get_flat_s(obs)[None], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    acts_raw = don(s_t, tau_grid)[0].cpu().numpy()
                coarse = acts_raw.reshape(chunk_size//2, 2, act_dim).sum(1)
                x_orig = np.linspace(0, 1, len(coarse))
                acts_spline = CubicSpline(x_orig, coarse, axis=0)(np.linspace(0, 1, chunk_size)) * 0.5
                for k in range(min(replan, 50 - step)):
                    obs, _, term, trunc, info = env.step(acts_spline[k][None])
                    s = info.get("success")
                    if s is not None and bool(np.asarray(s).reshape(-1)[0]): ok_spline = True
                    if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]): break
                if ok_spline: break
            results["deeponet_spline"].append(ok_spline)

            obs, _ = env.reset(seed=e["episode_seed"])
            env.unwrapped.set_state_dict(st)
            ok_fold = False
            for step in range(0, 50, replan):
                s_t = torch.tensor(get_flat_s(obs)[None], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    acts_raw = don(s_t, tau_grid)[0].cpu().numpy()
                coarse = acts_raw.reshape(chunk_size//2, 2, act_dim).sum(1)
                acts_fold = np.repeat(coarse / 2, 2, axis=0)
                for k in range(min(replan, 50 - step)):
                    obs, _, term, trunc, info = env.step(acts_fold[k][None])
                    s = info.get("success")
                    if s is not None and bool(np.asarray(s).reshape(-1)[0]): ok_fold = True
                    if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]): break
                if ok_fold: break
            results["deeponet_folding"].append(ok_fold)

            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"Evaluated {idx+1}/{n_eval} episodes...")

    env.close()
    print("\n" + "="*50)
    print("FINAL HIGH-CONFIDENCE CLOSED-LOOP RESULTS (n=300)")
    print("="*50)
    for k, v in results.items():
        print(f"  {k:25s}: {sum(v):3d}/{len(v):3d} = {100*np.mean(v):5.1f}%")
        
    out_file = f"{OUT_DIR}/closed_loop_deeponet_flow_results_n300.json"
    with open(out_file, "w") as fp:
        json.dump({k: {"successes": sum(v), "total": len(v), "rate": float(np.mean(v))} for k, v in results.items()}, fp, indent=2)
    print(f"\nSaved results to {out_file}")

if __name__ == "__main__":
    train_and_eval_closed_loop(n_eval=300, replan=4)
