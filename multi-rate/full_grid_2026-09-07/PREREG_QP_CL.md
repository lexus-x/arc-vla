# Pre-registration — closed-loop power extension + second policy family for the QP resampler
Written 2026-09-08 03:50 KST, before any of these results exist.

## A. Diffusion Policy, PickCube-v1, k=4, n=400 held-out (episodes 200-599), same checkpoint
Prior cell: n=100 (episodes 200-299): native 90, ZOH 40, spline_satfix 48, tac_fold_satfix 48, qp 52;
qp vs satfix arms +4pp, 6/2 discordant, p=0.29 — underpowered. This run extends the SAME protocol
(dp_PickCube-v1.pt, md5 9f2255fe787c, k=4, exec chunk 8 -> 2 blocks of 4, paired noise per
(episode, replan)) to n=400. Arms: native, zoh, spline_satfix, tac_fold_satfix, bspline_eps_satfix, qp.
Primary: qp vs spline_satfix, paired exact McNemar; secondary: qp vs tac_fold_satfix, qp vs
bspline_eps_satfix, qp vs zoh (Holm m=4). Prediction: qp > spline_satfix by 2-5pp; whether it reaches
p<0.05 at n=400 is exactly what this run decides — either outcome is reported as the answer.
Not allowed: changing n after seeing results, dropping arms, swapping reference.

## B. Flow Matching, PickCube-v1, k=4, n=100 (fm_PickCube-v1.pt, eval-only)
Same arms. Prediction: the k=4 ordering (qp >= satfix arms > ZOH) reproduces in direction under FM;
significance not expected at n=100. Reported as measured.
