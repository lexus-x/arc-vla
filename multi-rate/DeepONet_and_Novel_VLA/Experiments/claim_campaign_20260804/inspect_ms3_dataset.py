import h5py, os, glob

def inspect_all():
    base = os.path.expanduser("~/maniskill_data")
    for h5file in sorted(glob.glob(f"{base}/*.h5")):
        print("="*50)
        print("FILE:", h5file)
        with h5py.File(h5file, "r") as f:
            trajs = list(f.keys())
            print("  Total trajs:", len(trajs))
            if len(trajs) > 0:
                t0 = f[trajs[0]]
                print("  Keys:", list(t0.keys()))
                if "obs" in t0:
                    if hasattr(t0["obs"], "shape"):
                        print(f"  obs: shape={t0['obs'].shape}, dtype={t0['obs'].dtype}")
                    else:
                        print("  obs group:", list(t0["obs"].keys()))
                if "actions" in t0:
                    print(f"  actions: shape={t0['actions'].shape}, dtype={t0['actions'].dtype}")

if __name__ == "__main__":
    inspect_all()
