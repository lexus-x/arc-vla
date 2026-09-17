import sys, os, time
import numpy as np, torch

HERE = "/home/user/Desktop/multi-rate/full_grid_2026-09-07"
sys.path.insert(0, HERE)

from resample_math import resample_tac_fold, resample_spline
from resample_bspline2 import resample_bspline_eps

def test_interpolation():
    print("Testing mathematical interpolation functions:")
    # 8 displacement steps
    deltas = np.array([
        [0.1, -0.05], [0.2, 0.1], [0.05, 0.2], [-0.1, 0.15],
        [-0.2, -0.1], [-0.05, -0.2], [0.1, -0.1], [0.15, 0.0]
    ], dtype=np.float32)
    
    # 2x upsampling (10 Hz -> 20 Hz, 16 steps)
    tac_2x = resample_tac_fold(deltas, 2)
    spline_2x = resample_spline(deltas, 2)
    bspline_2x = resample_bspline_eps(deltas, 2, eps=0.005)
    
    print(f"Original 10 Hz deltas shape: {deltas.shape} | sum: {deltas.sum(axis=0)}")
    print(f"20 Hz (2x) TAC-Fold shape: {tac_2x.shape} | sum: {tac_2x.sum(axis=0)}")
    print(f"20 Hz (2x) Spline shape: {spline_2x.shape} | sum: {spline_2x.sum(axis=0)}")
    print(f"20 Hz (2x) B-Spline shape: {bspline_2x.shape} | sum: {bspline_2x.sum(axis=0)}")
    
    # 4x upsampling (10 Hz -> 40 Hz, 32 steps)
    tac_4x = resample_tac_fold(deltas, 4)
    spline_4x = resample_spline(deltas, 4)
    bspline_4x = resample_bspline_eps(deltas, 4, eps=0.005)
    print(f"40 Hz (4x) TAC-Fold shape: {tac_4x.shape} | sum: {tac_4x.sum(axis=0)}")
    print(f"40 Hz (4x) Spline shape: {spline_4x.shape} | sum: {spline_4x.sum(axis=0)}")
    print(f"40 Hz (4x) B-Spline shape: {bspline_4x.shape} | sum: {bspline_4x.sum(axis=0)}")
    assert tac_2x.shape == (16, 2)
    assert spline_2x.shape == (16, 2)
    assert bspline_2x.shape == (16, 2)
    assert tac_4x.shape == (32, 2)
    assert spline_4x.shape == (32, 2)
    assert bspline_4x.shape == (32, 2)
    print("All interpolation shapes verified!\n")

if __name__ == "__main__":
    test_interpolation()
