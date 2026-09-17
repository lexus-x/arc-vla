# Pre-registration — RoboCasa native reproduction, paper's own training codebase

Written 2026-09-14 21:20 KST, before any RoboCasa training run. Goal: reproduce B-spline
Policy (arXiv 2607.09648) Table 2(a) RoboCasa base cells on the paper's actual scale/codebase,
replacing our prior 55-demo/task reproduction that landed 59pp below paper on SinkFaucet
(20% vs 79%, `PROGRESS_UPDATE_DECK.md:16`).

## Target cells (paper's Table 2a, verified against this repo's own prior citations where
## available — PushT 72%/SinkFaucet 79% independently corroborated pre-existing repo docs;
## the other 4 numbers single-sourced from a paper fetch, not yet PDF-cross-checked)

| Task | Diff. Base | Reg. Base |
|---|---|---|
| TurnOffSinkFaucet | 79% | 84% |
| CoffeePressButton | 93% | 89% |
| TurnOffMicrowave | 77% | 93% |
| CloseSingleDoor | 27% | 40% |

No paper cell exists for OpenDrawer/PnPCounterToStove (our existing 55-demo preps for those
tasks are outside this reproduction's scope) or for Flow Matching (paper reports only Diff. and
Reg.; FM dropped from this effort per 2026-09-14 decision).

## Root cause of the prior gap (already diagnosed, now fully confirmed)

Not a metric/env mismatch (unlike PushT) — a **demo-scale** mismatch. RoboCasa's official
dataset registry (`robocasa/utils/dataset_registry.py`) has three tiers per task:
`human_raw`/`human_im` (~55 demos, what we had) and **`mg_im`: 3000 MimicGen-generated demos
with images**. The latter is what robomimic's own RoboCasa training configs
(`config_gen/diffusion_gen.py`, `config_gen/bc_xfmr_gen.py`) train on by default
(`filter_key="3000_demos"`). This is almost certainly the paper's actual training scale.

## Training codebase — reuse, don't reimplement

RoboCasa's own docs (`docs/use_cases/policy_learning.md`) name the canonical training repo:
`ARISE-Initiative/robomimic` branch `robocasa`. Installed 2026-09-14 into `vla_smolvla_libero`
(`pip install -e . --no-deps`, plus `tianshou` for `train_utils.py`) — confirmed NOT to require
numpy<1.24 or an old torch at import time despite `setup.py`'s hard pins
(`numpy==1.23.2`, `torch==2.0.1`): those pins are for robosuite/mujoco-C-extension live
simulation, which training-from-hdf5 never touches. Verified: `torch.cuda.is_available()`
still `True` on 2.10.0+cu128 after install, `lerobot.scripts.lerobot_train` still imports
cleanly (the env the concurrent PushT job depends on was not broken).

- Diff. → `robomimic/algo/diffusion_policy.py`, config template
  `robomimic/exps/templates/diffusion_policy.json` (num_epochs=2000, batch_size=256,
  seq_length=15 — robomimic's own shipped default, used as-is per ladder rung 3, not invented)
- Reg. → BC-Transformer, `robomimic/algo/bc.py`, config template
  `robomimic/exps/templates/bc_transformer.json` (num_epochs=2000, batch_size=100,
  seq_length=10)
- Dataset resolution: `config_gen_utils.get_robocasa_ds(ds_names=[<task>], src="mg",
  filter_key="3000_demos")` — same function robomimic's own generators use, not hand-rolled.
- Both templates ship `rollout: {enabled: true, n: 50, rate: 50 epochs, horizon: 400,
  terminate_on_success: true}` — this closed-loop eval already matches paper convention
  (n=50, terminate on success) without extra harness work, though it runs inside robomimic's
  own env wrapper, not our socket bridge — decide at launch time whether to use this built-in
  eval or route through `RoboCasaSim` for consistency with our other result tables.

## Data

`mg_im`, 4 tasks (TurnOffSinkFaucet, CoffeePressButton, TurnOffMicrowave, CloseSingleDoor),
downloading now via robocasa's own `scripts/download_datasets.py` (not a custom downloader),
targeting `/media/user/C2FE578FFE577A9D/desktop_datasets/robocasa_mg/` (475GB free; the
84GB-free main disk cannot hold this). Confirmed size: 24.2GB (SinkFaucet) / 32.4GB
(CloseSingleDoor) — total for 4 tasks likely ~110-130GB. ETA ~65-70 min at measured ~24MB/s.

## Not allowed

Training on anything less than the full `mg_im`/3000_demos set for these 4 tasks — that was
the exact prior mistake. Changing robomimic's shipped epoch/batch defaults without a stated
reason. Reporting a RoboCasa number without stating it came from `mg_im`/3000-demo scale (a
54-demo-trained number is not comparable and must not be placed in the same table column).

## Open items before training fires

1. Confirm all 4 downloads complete + pass robocasa's own validation.
2. Wall-clock per run is unknown — robomimic's 2000-epoch/3000-demo budget has no prior timing
   data in this repo. Smoke-test (small epoch count) before committing to a full run, same
   discipline as the PushT smoke test.
3. Decide rollout-eval path (robomimic's built-in vs. our `RoboCasaSim` bridge) before the
   first full run, not after — for result-table consistency with every other cell.
