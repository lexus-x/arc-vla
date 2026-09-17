# -*- coding: utf-8 -*-
"""Unit-validate the harness v2 chunk transform WITHOUT loading a policy.

Extracts the v2 block from the live harness and execs it against numpy/torch/CubicSpline,
so what is tested is the code that will actually run.
"""
import io, re, sys
import numpy as np
import torch
from scipy.interpolate import CubicSpline

P = "evaluate_multirate_honest.py"
s = io.open(P, encoding="utf-8").read()

POSE_DIMS = int(re.search(r"^POSE_DIMS\s*=\s*(\d+)", s, re.M).group(1))
HORIZON_S = float(re.search(r"^HORIZON_S\s*=\s*([0-9.]+)", s, re.M).group(1))
start = s.index("def _v2_akima_slopes")
end = s.index("def build_arm")
ns = {"np": np, "torch": torch, "CubicSpline": CubicSpline, "POSE_DIMS": POSE_DIMS}
exec(compile(s[start:end], "v2block", "exec"), ns)
v2 = ns["v2_transform_chunk"]
R = ns["_V2_RESAMPLERS"]
print(f"POSE_DIMS={POSE_DIMS} HORIZON_S={HORIZON_S} resamplers={sorted(R)}")

A = POSE_DIMS + 1
T = int(np.ceil(HORIZON_S * 20))
rng = np.random.default_rng(0)
off = torch.arange(A, dtype=torch.float32) * 0.1 + 0.3     # non-zero offset: the whole point
sc = torch.arange(A, dtype=torch.float32) * 0.05 + 1.7
norm = torch.from_numpy(rng.normal(0, 1, (1, T, A)).astype(np.float32))

o = off[:POSE_DIMS].numpy()
c = sc[:POSE_DIMS].numpy()
raw_in = norm[0, :, :POSE_DIMS].numpy() * c + o
tgt_in = raw_in.sum(axis=0)

fail = 0
print(f"\n{'arm':<26} {'in':>4} {'dec':>4} {'out':>4}  integral |err|max   verdict")
for k, name in [(2, "tac_fold"), (2, "spline_cum"), (2, "zoh_fold"), (1, "zoh_fold")]:
    out, il, dl, ol = v2(norm, k, R[name], T, off, sc)
    raw_out = out[0, :, :POSE_DIMS].numpy() * c + o
    err = np.abs(raw_out.sum(axis=0) - tgt_in).max()
    coarsened = (dl < il and ol == T)
    ok = (ol == T) and (dl == T // k) and (err < 1e-3)
    label = f"decimate{k}+{name}"
    print(f"{label:<26} {il:>4} {dl:>4} {ol:>4}  {err:>14.2e}   {'ok' if ok else 'FAIL'}"
          f"{'' if coarsened else '   <- gate: INERT (would FATAL)'}")
    if not ok:
        fail += 1
    if k == 1 and name == "zoh_fold":
        ident = np.abs(out.numpy() - norm.numpy()).max()
        print(f"{'  identity check':<26} max|out-in| = {ident:.2e} "
              f"({'identity confirmed' if ident < 1e-5 else 'NOT identity'})")
        if coarsened:
            print("  *** GATE BROKEN: inert arm reported as coarsened ***")
            fail += 1

# normalized-space trap: summing normalized deltas must NOT equal the raw sum
naive = norm[0, :, :POSE_DIMS].numpy().sum(axis=0)
print(f"\nRAW-space discipline: |raw_sum - normalized_sum| = {np.abs(tgt_in - naive).max():.3f} "
      "(large => summing in normalized space would have been wrong, as the spec warns)")

print("\nRESULT:", "ALL PASS" if fail == 0 else f"{fail} FAILURE(S)")
sys.exit(1 if fail else 0)
