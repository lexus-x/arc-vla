"""Add --replan_s so the replan interval can be varied without touching anything else.

Why this matters: the original authors' eval of these same 30K checkpoints used replan=5, and got
v2 85.0 / 87.0 / 58.5. The q6 grid used replan=10 (REPLAN_S=0.5 at 20 Hz) and got 27.3 / 28.7 /
20.3, while FLOW barely moved (79.5->80.7, 87.5->89.7, 66.5->65.0). On Long the horizon was
identical (520 steps both), so replan is the only remaining difference. This flag isolates it.
"""
P = "evaluate_multirate_honest.py"
s = open(P).read()

a1 = "REPLAN_S = 0.5\n"
assert s.count(a1) == 1, f"REPLAN_S anchor count={s.count(a1)}"
s = s.replace(a1, "REPLAN_S = 0.5\nREPLAN_OVERRIDE_STEPS = None  # set by --replan_steps; None => REPLAN_S * env_freq\n", 1)

a2 = "    replan = int(round(REPLAN_S * env_freq))\n"
assert s.count(a2) == 1, f"replan calc count={s.count(a2)}"
s = s.replace(a2, "    # A fixed STEP count (not a fixed wall-clock) is what the original protocol used, so the\n"
                  "    # override is in steps -- matching replan=5 exactly rather than approximating it.\n"
                  "    replan = REPLAN_OVERRIDE_STEPS or int(round(REPLAN_S * env_freq))\n", 1)

a3 = '    parser.add_argument("--smoke", action="store_true", help="run verification gate only")\n'
assert s.count(a3) == 1, f"argparse anchor count={s.count(a3)}"
s = s.replace(a3, a3 + '    parser.add_argument("--replan_steps", type=int, default=None,\n'
                       '                        help="fixed replan interval in env steps; overrides REPLAN_S")\n', 1)

a4 = '    print(f"[suite] {args.suite} -> {results_path}", flush=True)\n'
assert s.count(a4) == 1, f"suite print anchor count={s.count(a4)}"
s = s.replace(a4, '    if args.replan_steps is not None:\n'
                  '        if args.replan_steps < 1:\n'
                  '            raise SystemExit("[FATAL] --replan_steps must be >= 1")\n'
                  '        global REPLAN_OVERRIDE_STEPS\n'
                  '        REPLAN_OVERRIDE_STEPS = int(args.replan_steps)\n'
                  '        print(f"[replan] OVERRIDE: {REPLAN_OVERRIDE_STEPS} env steps "\n'
                  '              f"(default would be {int(round(REPLAN_S * 20))} at 20 Hz)", flush=True)\n'
                  + a4, 1)

open(P, "w").write(s)
import ast; ast.parse(s)
print("PATCH OK + SYNTAX OK")
