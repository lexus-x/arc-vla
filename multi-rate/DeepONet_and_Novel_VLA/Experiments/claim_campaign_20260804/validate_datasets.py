"""Validate candidate LIBERO dataset IDs before spending GPU on training.

Checks, per dataset: metadata loads, fps, episode/frame counts, action+state dims,
and that action stats are finite. Compares everything against the Spatial dataset
that asrc_s0 was actually trained on, since a mismatch there silently changes the
head's dynamics (rate_integrated_deeponet uses action_scale inside RK4).
"""
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

REF = "lerobot/libero_spatial_image"
CANDS = [REF,
         "lerobot/libero_object_image",
         "lerobot/libero_goal_image",
         "lerobot/libero_10_image"]

ref = None
for ds in CANDS:
    try:
        m = LeRobotDatasetMetadata(ds)
    except Exception as e:
        print(f"FAIL  {ds}\n        {type(e).__name__}: {str(e)[:160]}")
        continue

    stats = m.stats or {}
    ak = next((k for k in stats if k.endswith("action")), None)
    sk = next((k for k in stats if "state" in k), None)
    adim = len(stats[ak]["mean"]) if ak else "?"
    sdim = len(stats[sk]["mean"]) if sk else "?"
    finite = "ok"
    if ak:
        import math
        vals = list(stats[ak]["mean"]) + list(stats[ak]["std"])
        finite = "ok" if all(math.isfinite(float(v)) for v in vals) else "NON-FINITE"

    try:
        ntasks = len(m.tasks)
    except Exception:
        ntasks = "?"
    info = dict(fps=m.fps, eps=m.total_episodes, frames=m.total_frames,
                adim=adim, sdim=sdim, finite=finite, tasks=ntasks)
    if ds == REF:
        ref = info
        print(f"REF   {ds}\n        {info}")
    else:
        flags = []
        if ref:
            if info["fps"] != ref["fps"]:
                flags.append(f"FPS MISMATCH {info['fps']} vs ref {ref['fps']}")
            if info["adim"] != ref["adim"]:
                flags.append(f"ACTION DIM {info['adim']} vs {ref['adim']}")
            if info["sdim"] != ref["sdim"]:
                flags.append(f"STATE DIM {info['sdim']} vs {ref['sdim']}")
        print(f"{'WARN ' if flags else 'OK   '} {ds}\n        {info}")
        for f in flags:
            print(f"        !! {f}")
