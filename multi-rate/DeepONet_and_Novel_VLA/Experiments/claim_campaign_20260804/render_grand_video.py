import os, sys, json
from pathlib import Path
import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw

C = Path.home() / "DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
sys.path.insert(0, str(C))
os.chdir(C)

import evaluate_multirate_grand as G
import evaluate_height_screen as screen
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

RATE = 40
TASK_ID = 0
SEED = 0
ARMS = ["v2_spatial_dec2_tacfold_40env", "v2_spatial_dec2_bspline_40env", "v2_spatial_dec2_spline_40env"]
LABEL = {ARMS[0]: "ARC-VLA", ARMS[1]: "B-SPLINE", ARMS[2]: "CUBIC SPLINE"}
COL = {ARMS[0]: (16, 122, 87), ARMS[1]: (180, 60, 40), ARMS[2]: (40, 80, 180)}

def setup_arm(arm):
    model_key, env_freq, head_rate, transform = G.ARMS[arm]
    backbone, checkpoint, deeponet_head, fourier = G.MODELS[model_key]
    steps = int(round(G.SUITE_WALLCLOCK.get("libero_spatial", G.WALLCLOCK_S) * env_freq))
    replan = int(round(G.REPLAN_S * env_freq))
    G.patch_control_freq(env_freq, horizon=steps + 200)
    screen.SUITE = "libero_spatial"
    screen.MAX_STEPS = steps
    screen.REPLAN = replan
    screen.CONTROL_FREQ = int(G.NATIVE_HZ)
    screen.DATASET = G.MODEL_DATASET.get(model_key, G.DEFAULT_DATASET)
    stats = LeRobotDatasetMetadata(screen.DATASET).stats
    policy, (pre, post) = screen.load_policy(backbone, checkpoint, stats)
    dev = next(policy.parameters()).device
    A_OFF = torch.as_tensor(stats["action"]["mean"], dtype=torch.float32).flatten().to(dev)
    A_SC = torch.as_tensor(stats["action"]["std"], dtype=torch.float32).flatten().to(dev)
    steps_ = transform.split("+")
    _v2_dec = [t for t in steps_ if t.startswith("decimate")]
    _v2_res = [t for t in steps_ if t in G._V2_RESAMPLERS]
    assert len(_v2_res) == 1
    k = int(_v2_dec[0][len("decimate"):])
    target_len = int(np.ceil(G.HORIZON_S * env_freq))
    prev = policy._get_action_chunk
    stat = {"n": 0, "coarsened": 0}
    def hooked(*a, **kw):
        chunk = prev(*a, **kw)
        out, in_len, dec_len, out_len = G.v2_transform_chunk(chunk, k, G._V2_RESAMPLERS[_v2_res[0]], target_len, A_OFF, A_SC)
        stat["n"] += 1
        if dec_len < in_len:
            stat["coarsened"] += 1
        return out
    policy._get_action_chunk = hooked
    policy._v2_stat = stat
    return policy, pre, post, steps

def rollout_frames(policy, pre, post, task_desc, seed, max_steps):
    policy.reset()
    env = screen._make_env(TASK_ID)
    obs, _ = env.reset(seed=seed)
    desc = env.task_description
    frames, success, step = [], False, 0
    for step in range(1, max_steps + 1):
        frames.append(np.asarray(obs["pixels"]["image"]).copy())
        batch = pre(screen._policy_input(obs, desc))
        batch = {kk: v.to("cuda") if torch.is_tensor(v) else v for kk, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(batch)
        obs, _, terminated, truncated, info = env.step(post(action).to("cpu").float().numpy().reshape(-1))
        if info.get("is_success", False):
            frames.append(np.asarray(obs["pixels"]["image"]).copy())
            success = True
            break
        if terminated or truncated:
            break
    env.close()
    return success, frames, step, desc

def annotate(frame, text, color):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 256, 22], fill=color)
    d.text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(img)

def main():
    results = {}
    runs = {}
    for arm in ARMS:
        policy, pre, post, steps = setup_arm(arm)
        success, frames, steps_used, desc = rollout_frames(policy, pre, post, None, SEED, steps)
        runs[arm] = (success, frames)
        results[arm] = {"success": success, "steps": steps_used, "task": TASK_ID, "seed": SEED,
                        "queries": policy._v2_stat["n"], "coarsened": policy._v2_stat["coarsened"],
                        "desc": str(desc)}
        del policy
        torch.cuda.empty_cache()
        print(f"[{arm}] success={success} steps={steps_used} queries={results[arm]['queries']} coarsened={results[arm]['coarsened']}", flush=True)
    max_len = max(len(f) for _, f in runs.values())
    padded = {}
    for arm, (s, f) in runs.items():
        while len(f) < max_len:
            f.append(f[-1])
        padded[arm] = f
    a, b, c3 = ARMS
    banner_h = 26
    top_banner = np.full((banner_h, 512, 3), 255, dtype=np.uint8)
    bot_banner = np.full((banner_h, 512, 3), 255, dtype=np.uint8)
    im = Image.fromarray(top_banner); d = ImageDraw.Draw(im)
    d.text((4, 6), f"LIBERO-Spatial | {RATE} Hz serving | native 20 Hz | task {TASK_ID} seed {SEED} | DeepONet-v2 backbone", fill=(20, 20, 20))
    im2 = Image.fromarray(bot_banner); d2 = ImageDraw.Draw(im2)
    d2.text((4, 6), f"same comparison vs cubic spline | identical task+seed", fill=(20, 20, 20))
    top_banner, bot_banner = np.asarray(im), np.asarray(im2)
    rows = []
    for pair, bn in (((a, b), top_banner), ((a, c3), bot_banner)):
        left = annotate(padded[pair[0]][0], f"{LABEL[pair[0]]} {'OK' if runs[pair[0]][0] else 'FAIL'}", COL[pair[0]])
        right = annotate(padded[pair[1]][0], f"{LABEL[pair[1]]} {'OK' if runs[pair[1]][0] else 'FAIL'}", COL[pair[1]])
        l_imgs, r_imgs = [], []
        for i in range(max_len):
            l_imgs.append(annotate(padded[pair[0]][i], f"{LABEL[pair[0]]} {'OK' if runs[pair[0]][0] else 'FAIL'}", COL[pair[0]]))
            r_imgs.append(annotate(padded[pair[1]][i], f"{LABEL[pair[1]]} {'OK' if runs[pair[1]][0] else 'FAIL'}", COL[pair[1]]))
        row = [np.concatenate([np.concatenate([bn, np.concatenate([l_imgs[0], r_imgs[0]], axis=1)], axis=0)])]
        del l_imgs, r_imgs
        # memory-light: rebuild per frame without storing all annotations twice
        row_frames = []
        for i in range(max_len):
            li = annotate(padded[pair[0]][i], f"{LABEL[pair[0]]} {'OK' if runs[pair[0]][0] else 'FAIL'}", COL[pair[0]])
            ri = annotate(padded[pair[1]][i], f"{LABEL[pair[1]]} {'OK' if runs[pair[1]][0] else 'FAIL'}", COL[pair[1]])
            row_frames.append(np.concatenate([bn, np.concatenate([li, ri], axis=1)], axis=0))
        rows.append(row_frames)
        del row_frames
    n = min(len(rows[0]), len(rows[1]))
    full = [np.concatenate([rows[0][i], rows[1][i]], axis=0) for i in range(n)]
    out_dir = C / "grand_videos"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "arc_vs_bspline_vs_cubic_v2_spatial_40hz.mp4"
    imageio.mimsave(out, full, fps=RATE)
    (out_dir / "arc_vs_bspline_vs_cubic_results.json").write_text(json.dumps(results, indent=1))
    print("SAVED", out, flush=True)

if __name__ == "__main__":
    main()