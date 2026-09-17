"""Train & evaluate DeepONet vs. Flow Matching on ManiSkill PickCube RL high-frequency trajectories.

Arms evaluated:
  1. DeepONet (Trained) -> Native, Spline @ 40Hz, Folding @ 40Hz
  2. Flow Matching (Trained) -> Native, Spline @ 40Hz, ZOH @ 40Hz
  3. Replay Baselines (Untrained) -> ZOH vs. Spline controls
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
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Dataset Extraction: Paired (s_0 -> Action Chunk (50, 8))
# -------------------------------------------------------------
def build_dataset():
    print("Extracting paired (state_0 -> action_chunk) dataset from PickCube RL...")
    meta = json.load(open(f"{D_PATH}/pick_rl_joint.json"))
    eps = meta["episodes"]
    
    env = gym.make("PickCube-v1", num_envs=1, obs_mode="state",
                   control_mode="pd_joint_delta_pos", sim_backend="physx_cpu")
    
    states = []
    actions_list = []
    
    with h5py.File(f"{D_PATH}/pick_rl_joint.h5", "r") as h:
        for idx, e in enumerate(eps):
            a = np.array(h[f"traj_{e['episode_id']}"]["actions"]).astype(np.float32)
            n = (len(a) // 2) * 2
            a = a[:n]
            if len(a) != 50:
                continue
            
            g = h[f"traj_{e['episode_id']}"]["env_states"]
            st = {grp: {nm: torch.as_tensor(np.array(g[grp][nm])[0:1]) for nm in g[grp]} for grp in g}
            
            obs, _ = env.reset(seed=e["episode_seed"])
            env.unwrapped.set_state_dict(st)
            
            # Extract state vector
            if isinstance(obs, dict):
                # Flatten dict state
                flat_s = np.concatenate([v.cpu().numpy().reshape(-1) if isinstance(v, torch.Tensor) else np.array(v).reshape(-1) for v in obs.values()])
            elif isinstance(obs, torch.Tensor):
                flat_s = obs.cpu().numpy().reshape(-1)
            else:
                flat_s = np.array(obs).reshape(-1)
                
            states.append(flat_s)
            actions_list.append(a)
            
            if (idx + 1) % 200 == 0:
                print(f"  Extracted {idx+1}/{len(eps)} trajectories...")
                
    env.close()
    
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions_list, dtype=np.float32)
    print(f"Dataset ready: States {states.shape}, Actions {actions.shape}")
    return states, actions, eps

class ActionChunkDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
    def __len__(self):
        return len(self.states)
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# -------------------------------------------------------------
# 2. Architectures: DeepONet vs Flow Matching
# -------------------------------------------------------------
class FourierTrunk(nn.Module):
    def __init__(self, n_fourier=16, hidden_dim=256, p_dim=128, act_dim=8):
        super().__init__()
        self.n_fourier = n_fourier
        # tau in [0, 1] -> Fourier basis (1 + 2*n_fourier)
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
        # tau: (B, T, 1) or (T, 1)
        if tau.ndim == 1:
            tau = tau.unsqueeze(-1)
        freqs = torch.arange(1, self.n_fourier + 1, device=tau.device, dtype=tau.dtype) * 2 * math.pi
        sin_f = torch.sin(tau * freqs)
        cos_f = torch.cos(tau * freqs)
        feat = torch.cat([tau, sin_f, cos_f], dim=-1)
        out = self.mlp(feat) # (..., p * act_dim)
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
        # state: (B, S), tau_grid: (T, 1)
        b = self.branch(state) # (B, p)
        t = self.trunk(tau_grid) # (T, p, act_dim)
        # einsum: b(B, p), t(T, p, A) -> out(B, T, A)
        out = torch.einsum("bp, tpa -> bta", b, t) + self.bias
        return out

class FlowMatchingActionHead(nn.Module):
    def __init__(self, state_dim, chunk_len=50, act_dim=8, hidden_dim=256):
        super().__init__()
        self.chunk_len = chunk_len
        self.act_dim = act_dim
        in_dim = (chunk_len * act_dim) + state_dim + 1 # x_t + state + t
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
        # x_t: (B, H*D), t: (B, 1), state: (B, S)
        feat = torch.cat([x_t, state, t], dim=-1)
        v = self.mlp(feat)
        return v

    def sample(self, state, steps=10):
        # Euler ODE solver from noise x_0 ~ N(0, I) to x_1
        B = state.shape[0]
        x = torch.randn(B, self.chunk_len * self.act_dim, device=state.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B, 1), i * dt, device=state.device, dtype=torch.float32)
            v = self.forward(x, t, state)
            x = x + v * dt
        return x.view(B, self.chunk_len, self.act_dim)

# -------------------------------------------------------------
# 3. Training Function
# -------------------------------------------------------------
def train_models(states, actions, epochs=50):
    dataset = ActionChunkDataset(states, actions)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    state_dim = states.shape[1]
    act_dim = actions.shape[2]
    T_steps = actions.shape[1]
    
    print(f"\n--- Training DeepONet on PickCube RL ({epochs} epochs) ---")
    don = DeepONetActionHead(state_dim=state_dim, p_dim=128, act_dim=act_dim).to(DEVICE)
    opt_don = optim.AdamW(don.parameters(), lr=1e-3, weight_decay=1e-4)
    tau_grid = torch.linspace(0, 1, T_steps, device=DEVICE).unsqueeze(-1)
    
    t0 = time.time()
    for ep in range(epochs):
        don.train()
        total_loss = 0
        for s_b, a_b in loader:
            s_b, a_b = s_b.to(DEVICE), a_b.to(DEVICE)
            pred = don(s_b, tau_grid)
            loss = nn.functional.mse_loss(pred, a_b)
            opt_don.zero_grad()
            loss.backward()
            opt_don.step()
            total_loss += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [DeepONet Epoch {ep+1:02d}/{epochs}] MSE Loss: {total_loss/len(loader):.6f}")
    print(f"DeepONet trained in {time.time()-t0:.1f}s.")
    
    print(f"\n--- Training Flow Matching on PickCube RL ({epochs} epochs) ---")
    flow = FlowMatchingActionHead(state_dim=state_dim, chunk_len=T_steps, act_dim=act_dim).to(DEVICE)
    opt_flow = optim.AdamW(flow.parameters(), lr=1e-3, weight_decay=1e-4)
    
    t0 = time.time()
    for ep in range(epochs):
        flow.train()
        total_loss = 0
        for s_b, a_b in loader:
            s_b = s_b.to(DEVICE)
            a_flat = a_b.to(DEVICE).view(a_b.shape[0], -1)
            x_0 = torch.randn_like(a_flat)
            x_1 = a_flat
            t = torch.rand(a_b.shape[0], 1, device=DEVICE)
            x_t = (1 - t) * x_0 + t * x_1
            target_v = x_1 - x_0
            
            pred_v = flow(x_t, t, s_b)
            loss = nn.functional.mse_loss(pred_v, target_v)
            opt_flow.zero_grad()
            loss.backward()
            opt_flow.step()
            total_loss += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [Flow Matching Epoch {ep+1:02d}/{epochs}] MSE Loss: {total_loss/len(loader):.6f}")
    print(f"Flow Matching trained in {time.time()-t0:.1f}s.")
    
    return don, flow

# -------------------------------------------------------------
# 4. Evaluation Function
# -------------------------------------------------------------
def resample_spline(action_chunk, target_len=50):
    orig_len = action_chunk.shape[0]
    x_orig = np.linspace(0, 1, orig_len)
    x_target = np.linspace(0, 1, target_len)
    cs = CubicSpline(x_orig, action_chunk, axis=0)
    return cs(x_target) * (orig_len / target_len)

def resample_zoh(action_chunk, target_len=50):
    k = target_len // action_chunk.shape[0]
    return np.repeat(action_chunk / k, k, axis=0)

def evaluate_models(don, flow, states, eps, n_eval=150):
    print(f"\n=======================================================")
    print(f"EVALUATING TRAINED DEEPONET & FLOW ON PICKCUBE ({n_eval} EPISODES)")
    print(f"=======================================================")
    
    env = gym.make("PickCube-v1", num_envs=1, obs_mode="state",
                   control_mode="pd_joint_delta_pos", sim_backend="physx_cpu")
    
    arms = [
        "deeponet_native", "deeponet_spline_40hz", "deeponet_folding_40hz",
        "flow_native", "flow_spline_40hz", "flow_zoh_40hz"
    ]
    results = {k: [] for k in arms}
    
    don.eval()
    flow.eval()
    tau_native = torch.linspace(0, 1, 50, device=DEVICE).unsqueeze(-1)
    tau_fine = torch.linspace(0, 1, 100, device=DEVICE).unsqueeze(-1)
    
    with h5py.File(f"{D_PATH}/pick_rl_joint.h5", "r") as h:
        for idx in range(min(n_eval, len(eps))):
            e = eps[idx]
            g = h[f"traj_{e['episode_id']}"]["env_states"]
            st = {grp: {nm: torch.as_tensor(np.array(g[grp][nm])[0:1]) for nm in g[grp]} for grp in g}
            
            # Predict from initial state
            s_tensor = torch.tensor(states[idx:idx+1], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                # DeepONet predictions
                a_don_native = don(s_tensor, tau_native)[0].cpu().numpy()
                # DeepONet fine query (continuous operator evaluation)
                a_don_fine = don(s_tensor, tau_fine)[0].cpu().numpy() * 0.5 # scale for 2x rate
                # Flow prediction
                a_flow_native = flow.sample(s_tensor, steps=10)[0].cpu().numpy()
                
            # Decimate coarse for spline/folding comparison
            coarse_don = a_don_native.reshape(25, 2, 8).sum(1)
            a_don_spline = resample_spline(coarse_don, 50)
            a_don_fold = resample_zoh(coarse_don, 50)
            
            coarse_flow = a_flow_native.reshape(25, 2, 8).sum(1)
            a_flow_spline = resample_spline(coarse_flow, 50)
            a_flow_zoh = resample_zoh(coarse_flow, 50)
            
            eval_actions = {
                "deeponet_native": a_don_native,
                "deeponet_spline_40hz": a_don_spline,
                "deeponet_folding_40hz": a_don_fold,
                "flow_native": a_flow_native,
                "flow_spline_40hz": a_flow_spline,
                "flow_zoh_40hz": a_flow_zoh
            }
            
            for arm_name in arms:
                acts = eval_actions[arm_name]
                env.reset(seed=e["episode_seed"])
                env.unwrapped.set_state_dict(st)
                ok = False
                for t in range(len(acts)):
                    _, _, term, trunc, info = env.step(acts[t][None])
                    s = info.get("success")
                    if s is not None and bool(np.asarray(s).reshape(-1)[0]):
                        ok = True
                    if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                        break
                results[arm_name].append(ok)
                
            if (idx + 1) % 25 == 0 or idx == 0:
                print(f"Evaluated {idx+1}/{n_eval} episodes...")
                
    env.close()
    
    print("\n" + "="*50)
    print("FINAL HEAD-TO-HEAD RESULTS ON PICKCUBE RL")
    print("="*50)
    for k, v in results.items():
        print(f"  {k:25s}: {sum(v):3d}/{len(v):3d} = {100*np.mean(v):5.1f}%")
        
    out_file = f"{OUT_DIR}/trained_deeponet_flow_pickcube_results.json"
    summary = {k: {"successes": sum(v), "total": len(v), "rate": float(np.mean(v))} for k, v in results.items()}
    with open(out_file, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSaved results to {out_file}")

def main():
    states, actions, eps = build_dataset()
    don, flow = train_models(states, actions, epochs=50)
    evaluate_models(don, flow, states, eps, n_eval=150)

if __name__ == "__main__":
    main()
