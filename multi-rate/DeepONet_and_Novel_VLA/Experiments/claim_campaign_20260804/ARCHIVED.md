# ARCHIVED 2026-08-12 — asrc head and operator folding (multi-rate line)

CLOSED BY DECISION, ON EVIDENCE. Nothing here is deleted. Do not restart without a benchmark
that contains more than one control rate.

Reason:
  asrc vs til        +6.3 pp, p=0.056   (n=300)      -- never beat don_v2
  don_v2 vs flow    +20.9 pp, p<1e-5    (8.3K, 5 seeds)
  folding vs spline  4 of 4 POWERED TIES:
      Spatial n=300         -3.0 pp, p=0.23
      Goal    n=300         -1.3 pp, p=0.5034
      same-ckpt 40Hz n=300  +0.3 pp, p=1.0000
      LIBERO-Plus n=315     +0.6 pp, 16/14, p=0.8555
  LIBERO-Plus n=315 @ real 40 Hz:
      flow @20 native                29.5%
      flow @40 + plain cubic spline  27.6%   (-1.9 pp vs its own 20 Hz, p=0.53)
      asrc_s0@8300 @40 + folding     14.6%
      asrc_s0@8300 @40 + spline      14.0%
      ours+folding vs flow+spline   -13.0 pp, 22/63, p<1e-4

CONSEQUENCE: don_v2 is DeepONetHeadV2 -- forward(prefix, pad_mask), NO rate input.
Folding lives only in RateIntegratedDeepONetHead. Archiving asrc removes multi-rate from the
project rather than relocating it. Never describe a don_v2 result as rate-invariant.

RETIRED: RateIntegratedDeepONetHead variants ti/til/asrc, the 0.2 s rate-consistency loss, the
log2(r/20) trunk token, the Fourier bandlimit guard, every folding-vs-spline comparison, and the
unfixed 25 Hz ceil(H*r) partial-step bug.

SURVIVES (method-independent, keep using):
  - raw-space magnitude x env_freq/20 : +54.0 pp, exact McNemar p=7.4e-6 (cadence +8.0 pp, n.s.,
    interaction exactly zero). A CONTROLLER property every decoder needs equally.
  - the unfixed 40 Hz floor is 12%, not 0/50.
  - assert n_tasks_verified == 2402 in any LIBERO-Plus harness (guards import shadowing).
  - eval noise floor: ~10.5 pp per-category drift on identical reruns; n=50 per-category deltas
    under ~20 pp are noise. 3 of 3 n=50 estimates reversed under power.

RETAINED ON DISK (do not delete):
  runs/asrc_s0  runs/asrc_goal_s0  runs/asrc_object_s0  runs/asrc_long_s0  runs/til_s0
  rate_integrated_deeponet.py  evaluate_multirate_honest.py  evaluate_plus_multirate.py
  pow_plus_*_out/

IF RE-OPENED: needs held-out control rates at test, a classical rate-transfer reference
(DMP/ProMP time-scaling) and a learned one (CrossFormer-style per-embodiment heads). On a
single-rate suite the question is not answerable by anyone.

Full record: vault wiki/deeponet-vla/asrc-multirate-archived-2026-08-12.md
