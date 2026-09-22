import numpy as np
from resample_math import resample_tac_fold, resample_spline
import matplotlib.pyplot as plt

# Generate a sample 8-step action chunk representing pushing around a corner
np.random.seed(42)
# Waypoints representing moving in x then turning in y
anchor = np.array([0.0, 0.0])
chunk = np.array([
    [0.2, 0.05],
    [0.4, 0.12],
    [0.6, 0.25],
    [0.7, 0.45],
    [0.75, 0.70],
    [0.76, 0.90],
    [0.75, 1.00],
    [0.72, 1.05],
])
deltas = np.diff(np.concatenate([anchor[None], chunk], axis=0), axis=0)

mult = 5
tf_deltas = resample_tac_fold(deltas, mult)
sp_deltas = resample_spline(deltas, mult)

tf_pos = anchor[None] + np.cumsum(tf_deltas, axis=0)
sp_pos = anchor[None] + np.cumsum(sp_deltas, axis=0)

print("Number of fine points:", len(tf_pos))
print("Final position TAC-Fold:", tf_pos[-1])
print("Final position Spline:  ", sp_pos[-1])
print("Target position (chunk[-1]):", chunk[-1])

# Check acceleration / smoothness (diff of deltas)
tf_acc = np.diff(tf_deltas, axis=0)
sp_acc = np.diff(sp_deltas, axis=0)
print(f"TAC-Fold max jerk/acc-jump: {np.max(np.abs(np.diff(tf_acc, axis=0))):.5f}")
print(f"Spline max jerk/acc-jump:   {np.max(np.abs(np.diff(sp_acc, axis=0))):.5f}")
print(f"TAC-Fold mean squared jerk: {np.mean(np.diff(tf_acc, axis=0)**2):.5f}")
print(f"Spline mean squared jerk:   {np.mean(np.diff(sp_acc, axis=0)**2):.5f}")
