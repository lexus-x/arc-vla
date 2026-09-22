import numpy as np
from resample_arc import resample_arc
from resample_math import resample_tac_fold, resample_spline

# 1. Test Decimation (k=4)
deltas = np.array([
    [0.8, -0.6], [0.9, -0.7], [0.7, -0.5], [0.8, -0.6],
    [-0.5, 0.4], [-0.6, 0.5], [-0.4, 0.3], [-0.5, 0.4]
])
# Coarsen to k=4
k = 4
coarse = deltas.reshape(2, 4, 2).sum(axis=1) # 2 blocks of sum
arc_dec = resample_arc(coarse, k, is_up=False)
tf_dec = resample_tac_fold(coarse, k)
sp_dec = resample_spline(coarse, k)

print('=== Decimation (k=4) ===')
print('Max abs delta ARC:     ', np.max(np.abs(arc_dec)))
print('Max abs delta TAC-Fold:', np.max(np.abs(tf_dec)))
print('Max abs delta Spline:  ', np.max(np.abs(sp_dec)))
print('ARC sum error:        ', np.max(np.abs(arc_dec.reshape(2, 4, 2).sum(axis=1) - coarse)))

# 2. Test Upsampling (50 Hz, mult=5)
acts = np.array([
    [0.2, 0.05], [0.4, 0.12], [0.6, 0.25], [0.7, 0.45],
    [0.75, 0.70], [0.76, 0.90], [0.75, 1.00], [0.72, 1.05],
])
d = np.diff(np.concatenate([np.zeros((1, 2)), acts], axis=0), axis=0)
arc_up = resample_arc(d, 5, is_up=True)
tf_up = resample_tac_fold(d, 5)
sp_up = resample_spline(d, 5)

print('\n=== Upsampling (5x, 50 Hz) ===')
arc_jerk = np.max(np.abs(np.diff(np.diff(arc_up, axis=0), axis=0)))
tf_jerk  = np.max(np.abs(np.diff(np.diff(tf_up, axis=0), axis=0)))
sp_jerk  = np.max(np.abs(np.diff(np.diff(sp_up, axis=0), axis=0)))
print(f'Max jerk jump ARC:      {arc_jerk:.6f}')
print(f'Max jerk jump TAC-Fold: {tf_jerk:.6f}')
print(f'Max jerk jump Spline:   {sp_jerk:.6f}')
print('ARC fine delta sum error:', np.max(np.abs(arc_up.reshape(8, 5, 2).sum(axis=1) - d)))
