"""Official B-Spline Policy (arXiv 2607.09648) action head, called verbatim from
github.com/B-spline-policy/bspline-policy @61ed5f4 (MIT), vendored at ../external/bspline-policy.

targets: BSplineChunkSampler (per-episode reduced-knot cubic fit, 10-span chunks, stride 1, knots relative to
         the current step) -> (16, 1+A) parameter matrix per timestep; obs = 2 frames, edge-padded like make_chunks.
decode : exactly PolicyLocalBSpline._flush_predictor (safer_knots, BSpline default extrapolate) sampled at native
         speed, t = 0..7 (speed_up_times=1, time-align off: sim inference is synchronous, a new plan starts at t=0).
Rate-free by design: no k enters either function."""
import os, sys, types
import numpy as np
from scipy.interpolate import BSpline

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "bspline-policy")
for p in ("bspline_policy", "diffusion_policy"): sys.path.insert(0, os.path.join(_ROOT, p))
# ponytail: ReplayBuffer is only a type in bspline_action.py; stub it instead of installing zarr into this env
sys.modules.setdefault("diffusion_policy.common.replay_buffer", types.SimpleNamespace(ReplayBuffer=object))
from bspline_policy.common.bspline_action import BSplineChunkSampler  # noqa: E402

DEGREE, CHUNK, MAX_ERROR, N_EXEC = 3, 16 - 2 * 3, 0.01, 8  # official config: horizon 16, n_obs 2, n_action_steps 8; class default max_error


class _Buf(dict):  # the two things BSplineChunkSampler reads from a ReplayBuffer
    def __init__(self, A):
        super().__init__(action=np.concatenate(A).astype(np.float64)); self.episode_ends = np.cumsum([len(a) for a in A])


def targets(O, A, n_obs=2):
    """O, A: lists of per-episode (T, D) normalised obs and (T, A) raw actions -> obs (N, n_obs, D), params (N, 16, 1+A)."""
    ad = A[0].shape[1]
    s = BSplineChunkSampler(_Buf(A), chunk_size=CHUNK, degree=DEGREE, max_error=MAX_ERROR, stride=1, keys=["action"],
                            n_action_steps=CHUNK + 2 * DEGREE, n_action_channels=1 + ad)
    obs = []
    for o in O:
        P = np.concatenate([np.repeat(o[:1], n_obs - 1, 0), o], 0); obs += [P[t:t + n_obs] for t in range(len(o))]
    obs = np.stack(obs)
    return obs[s.valid_timesteps], s.all_actions[s.timestep_to_chunk[s.valid_timesteps]].astype(np.float32)


def safer_knots(knots):  # verbatim, bspline_policy/scripts/policy_local_bspline.py:37 (that module pulls real-robot deps)
    knots = np.asarray(knots, dtype=np.float64).copy()
    for idx in range(1, len(knots)):
        if knots[idx] < knots[idx - 1]:
            knots[idx] = knots[idx - 1] + 1e-6
    return knots


def _spline(params):  # PolicyLocalBSpline._flush_predictor
    params = np.asarray(params, np.float64)
    spl = BSpline(t=safer_knots(params[:, 0]), c=params[: -(DEGREE + 1), 1:], k=DEGREE)
    return spl, spl.t[DEGREE], spl.t[-DEGREE - 1]


def decode(params, n=N_EXEC):
    """params: denormalised (16, 1+A) -> (n, A) actions at native speed from t=0 (conformance check only)."""
    return _spline(params)[0](np.arange(n, dtype=np.float64)).astype(np.float32)


# PolicyLocalBSpline defaults (policy_local_bspline.py:474-482); speed_up_times = 1
SPEED, PBE, OTS, THRESH, LARGER_T = 1.0, 0.06, 10.0, 0.1, 0.2


def _closest(spl, target, lo, hi):  # _find_closest_t_to_target; compare arm dims only (consider_gripper_during_align=False)
    from scipy.optimize import minimize_scalar
    d = len(target)
    res = minimize_scalar(lambda t: np.sqrt((spl(t)[:d] - target) ** 2).sum(), bounds=(lo, hi), method="bounded")
    return res.x, float(np.abs(spl(res.x)[:d] - target).max())


def _align(spl, old, lo, hi):
    """_align_new_plan with obs_time == arrival time (synchronous sim inference: elapsed 0)."""
    new_max_t = float(np.clip(0.0 * SPEED * OTS, lo, hi))
    allowed = hi - PBE * OTS - 0.1
    allowed = min(allowed, hi * LARGER_T + lo * (1.0 - LARGER_T)); allowed = max(allowed, lo + 1e-3)
    lam, best_t, best_err = 1.0, lo, np.inf
    while best_err > THRESH:
        this = min(new_max_t * lam, allowed)
        if this <= lo: break
        best_t, best_err = _closest(spl, old, lo, this)
        if lam * new_max_t > allowed or lam > 20: break
        lam *= 1.5
    return best_t


def plan(params, prev=None, nd=None):
    """One official plan executed at native speed: start at t=0 (first plan) or the aligned t, then step t += 1 while
    the replan trigger (max_t - t) / OTS < PBE * SPEED is not yet hit. prev = last executed raw action of the old plan."""
    spl, lo, hi = _spline(params)
    t0 = 0.0 if prev is None else _align(spl, np.asarray(prev, np.float64)[:nd], lo, hi)
    n = max(1, int(np.floor(hi - PBE * SPEED * OTS - t0)) + 1)  # ponytail: >=1 step so a spent plan cannot stall the sync loop
    return spl(t0 + SPEED * np.arange(n)).astype(np.float32)


if __name__ == "__main__":  # conformance: official targets -> our decode reproduces the demo actions it was fitted to
    rng = np.random.default_rng(0)
    A = [np.cumsum(rng.normal(0, .05, (T, 4)), 0) for T in (40, 73, 120)]; O = [rng.normal(size=(len(a), 5)) for a in A]
    ob, par = targets(O, A)
    assert ob.shape == (sum(map(len, A)), 2, 5) and par.shape == (len(ob), 16, 5), (ob.shape, par.shape)
    errs, g = [], 0
    for a in A:
        for t in range(len(a) - N_EXEC):
            errs.append(np.abs(decode(par[g + t]) - a[t:t + N_EXEC]).max())
        g += len(a)
    errs = np.asarray(errs); print(f"round-trip max|err| median={np.median(errs):.4f} p95={np.quantile(errs, .95):.4f} max={errs.max():.4f}")
    assert np.median(errs) < 2 * MAX_ERROR, "decode at t=0..7 does not reproduce the fitted demo actions"
    # replay with the official plan/align loop: every executed action must match the demo at the spline's own time
    fit, jump, g = [], [], 0
    for a in A:
        tau, prev = 0, None
        while tau < len(a) - 1:
            spl, lo, hi = _spline(par[g + tau]); acts = plan(par[g + tau], prev, nd=3)
            t0 = 0.0 if prev is None else _align(spl, prev[:3], lo, hi)
            idx = np.clip(np.rint(tau + t0 + np.arange(len(acts))).astype(int), 0, len(a) - 1)
            fit.append(np.abs(acts - a[idx]).max())
            if prev is not None: jump.append(np.abs(acts[0, :3] - prev[:3]).max())
            prev, tau = acts[-1], tau + len(acts)
        g += len(a)
    fit, jump = np.asarray(fit), np.asarray(jump)
    print(f"plan/align replay: plans={len(fit)} steps/plan={sum(map(len, A)) / len(fit):.1f} fit median={np.median(fit):.4f} "
          f"p95={np.quantile(fit, .95):.4f} | first-step jump median={np.median(jump):.4f} p95={np.quantile(jump, .95):.4f}")
    assert np.median(fit) < 2 * MAX_ERROR
    print("bsp_official conformance ok")
