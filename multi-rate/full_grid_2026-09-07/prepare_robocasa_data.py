"""One-time conversion: RoboCasa human_raw demo hdf5 (states+actions only) -> low-dim-obs hdf5
(obs/* populated by replaying the recorded states through the real sim), via robocasa's own
dataset_states_to_obs pipeline -- called programmatically (camera_names=[]) rather than through
its CLI, since argparse's nargs="+" can't express an empty camera list. Must run inside
robocasa_uv (numpy 1.23-pinned).

Usage: robocasa_uv/.venv/bin/python prepare_robocasa_data.py
"""
import argparse, os
import robocasa.scripts.dataset_states_to_obs as sto

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f"{HERE}/robocasa_data"
TASKS = ["OpenDrawer", "PnPCounterToStove"]


def convert(task):
    raw = f"{DATA_DIR}/{task}_raw.hdf5"
    out_name = f"{task}_ld.hdf5"
    out_path = f"{DATA_DIR}/{out_name}"
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists"); return
    args = argparse.Namespace(
        dataset=raw, output_name=out_name, filter_key=None, n=None, shaped=False,
        camera_names=[], camera_height=84, camera_width=84, done_mode=0,
        copy_rewards=False, copy_dones=False, include_next_obs=False, no_compress=False,
        num_procs=4, add_datagen_info=False, generative_textures=False, randomize_cameras=False,
    )
    print(f"[convert] {task}: {raw} -> {out_path}", flush=True)
    sto.dataset_states_to_obs_multiprocessing(args)


if __name__ == "__main__":
    import sys
    for t in (sys.argv[1:] or TASKS):
        convert(t)
    print("[done] all conversions complete")
