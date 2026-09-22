# ARC--TAC positive Q2 method paper: execution checklist

## Definition of done

The project is complete only when a fresh, provenance-valid campaign supports a scoped positive
ARC--TAC claim and closes the five current reviewer objections. Completing a run is not sufficient.

## 0. Authorize and stage on BW2

- [x] Configure passwordless SSH alias `bw2` for `islab@100.114.151.66`.
- [x] Validate the RTX PRO 6000 Blackwell GPU and available disk space.
- [x] Create and validate isolated BW2 runtimes:
  - `~/envs/arc_ms/bin/python` for ManiSkill/Push-T.
  - `~/envs/saptarshi/bin/python` for RoboMimic, policy training, and analysis.
  - `~/envs/arc_robocasa/bin/python` for the CPU-only RoboCasa bridge.
- [x] Freeze bundle `arc_tacfold_bw2_bundle_20260918_v5.tar.gz`.
- [x] Verify local SHA-256:
  `3bddb93322e1b8aeb1cc448a6ed60f0079bb5ea19823e41474aac9b464b1c293`.
- [ ] Obtain explicit authorization to transfer the private source and benchmark datasets.
- [ ] Copy v5 to BW2, verify the remote SHA-256, and extract under
  `~/arc_tacfold_campaign`.

BW2 staging sequence after authorization:

```bash
scp /home/user/Desktop/arc_tacfold_bw2_bundle_20260918_v5.tar.gz bw2:~/
ssh bw2 'mkdir -p ~/arc_tacfold_campaign && tar -xzf ~/arc_tacfold_bw2_bundle_20260918_v5.tar.gz -C ~/arc_tacfold_campaign'
ssh bw2 'sha256sum ~/arc_tacfold_bw2_bundle_20260918_v5.tar.gz'
```

Acceptance: remote checksum exactly matches the local checksum; no source or dataset is silently
replaced.

## 1. BW2 preflight and smoke evaluation

- [ ] Run the 54 unit/integration tests on BW2.
- [ ] Compile all campaign Python files and parse every shell runner.
- [ ] Validate dataset paths and hashes.
- [ ] Start one RoboCasa bridge and complete a reset/step smoke test.
- [ ] Complete one Push-T, one RoboMimic, and one RoboCasa smoke rollout.
- [ ] Confirm that each ARC checkpoint is shared across `k = 1, 2, 4`.
- [ ] Confirm that no stale result or checkpoint causes an unintended skip.

Acceptance: all tests pass, each simulator produces a real closed-loop rollout, and the GPU is
visible from both training runtimes.

## 2. Run the existing locked signal campaign on BW2

- [ ] Launch `run_arc_tacfold_bw2.sh` in a persistent `tmux` session with a timestamped log.
- [ ] Run Push-T: 200 demonstrations, 30k steps, 400 paired evaluations.
- [ ] Run RoboMimic Lift/Can/Square: 200 demonstrations, 30k steps, 100 paired evaluations each.
- [ ] Run four RoboCasa tasks: 39 demonstrations, 15k steps, 100 random-reset paired evaluations
  each.
- [ ] Run RGB-plus-proprioception CloseSingleDoor: 35 demonstrations, 15k steps, 15 paired
  evaluations per rate.
- [ ] Preserve all checkpoint hashes, result JSONs, full logs, and failures.
- [ ] Monitor BW2 without restarting successful cells or editing the preregistration.

Recommended launch/monitor process:

```bash
ssh bw2
cd ~/arc_tacfold_campaign
tmux new -s arc_tac
bash run_arc_tacfold_bw2.sh
# Detach with Ctrl-b d
```

Monitor from Blackwell 1:

```bash
ssh bw2 'tail -n 80 ~/arc_tacfold_campaign/arc_tacfold_bw2.log'
ssh bw2 '/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader'
ssh bw2 'cat ~/arc_tacfold_campaign/arc_tacfold_bw2.status'
```

Acceptance: all 48 state-policy cells and all three visual rates exist and pass artifact validation.

## 3. Decide whether ARC--TAC has a real positive signal

- [ ] Run `analyze_arc_tacfold.py` without manual table editing.
- [ ] Require six positive, non-void primary comparisons with Holm-adjusted `p < 0.05`.
- [ ] Require at least one corrected ARC--TAC versus ARC+ZOH mechanism win.
- [ ] Require the pooled native-rate 90% lower bound to exceed `-5` percentage points.
- [ ] Inspect task-level negative cells, saturation incidence, and displacement residuals.
- [ ] Sync the complete result archive back to Blackwell 1 and verify hashes.

Decision:

- All gates pass: freeze ARC--TAC and proceed to named-policy baselines.
- Gates fail: do not write a positive claim. Diagnose on declared development splits, record an
  amendment, change the method once, and rerun a fresh holdout.

## 4. Add true named-policy competitors

### B-Spline Policy

- [ ] Pin the official repository commit and license.
- [ ] Port its learned control-point/knot head into the matched diffusion backbone.
- [ ] Reproduce its target fitting, continuity alignment, and temporal scaling.
- [ ] Add unit tests against the official implementation on identical synthetic trajectories.
- [ ] Match demonstrations, optimizer steps, observation inputs, execution horizon, and evaluation
  episodes to ARC--TAC.

Acceptance: this arm can truthfully be called a B-Spline Policy reproduction, rather than a
B-spline reconstruction control.

### CAT

- [ ] Implement the paper's action encoder/decoder, fixed continuous latent tokens,
  frequency-aware temporal coordinates, and trajectory regularization.
- [ ] Verify tensor shapes, reconstruction loss, shared-frequency behavior, and parameter count.
- [ ] Document every ambiguity caused by unavailable official code.
- [ ] Match the ARC--TAC backbone, data, budgets, and paired evaluation protocol.

Acceptance: an independent reviewer can map every implemented component to the CAT specification.

### Spline Policy

- [ ] Implement the learned quadratic Bernstein-spline output representation.
- [ ] Reproduce target fitting, spline decoding, and declared continuity constraints.
- [ ] Separate basic spline execution from optional vector-field control.
- [ ] Add analytic curve/derivative and continuity tests.
- [ ] Match the common backbone, data, budgets, and paired evaluation protocol.

Acceptance: the learned head is paper-faithful and is not confused with the existing post-hoc
cubic interpolation arm.

## 5. Development without cherry-picking

- [ ] Reserve explicit development episodes or training-only diagnostics before examining the
  final holdout.
- [ ] Compare target reconstruction error, saturation, smoothness, and conditioned-rate usage.
- [ ] Change ARC--TAC only for a documented mechanism supported across multiple development tasks.
- [ ] Do not remove a task, rate, seed, or competitor because it is unfavorable.
- [ ] Freeze the final method and named-baseline implementations before the confirmatory run.

Acceptance: one immutable code/config hash for every method before final evaluation.

## 6. Fresh confirmatory head-to-head campaign

- [ ] Evaluate ARC--TAC, CAT, Spline Policy, and B-Spline Policy on identical paired episodes.
- [ ] Cover at least two benchmark families with sufficient non-void discordant pairs.
- [ ] Correct the complete declared comparison family for multiplicity.
- [ ] Require ARC--TAC to beat B-Spline Policy and at least one of CAT or Spline Policy with a
  positive corrected result.
- [ ] Require native-rate no-harm and the ARC/TAC component mechanism gate.
- [ ] Report all complete negative and null cells.

Acceptance: the positive conclusion survives the locked statistical analysis and provenance audit.

## 7. Reviewer re-evaluation

- [ ] Re-review originality against CAT, Spline Policy, and B-Spline Policy.
- [ ] Re-review methodology, multiplicity, paired design, and effect sizes.
- [ ] Re-review evidence breadth, failure cases, and claim boundaries.
- [ ] Ask the binary Q2 question again: publishable positive method paper, yes or no.

Acceptance: no critical evidence or comparator objection remains. Manuscript work resumes only
after this point.

## ETA from transfer authorization

| Stage | Expected elapsed time |
|---|---:|
| Transfer, checksum, BW2 preflight | 2--4 hours |
| Existing full ARC--TAC signal campaign | 18--36 hours |
| Analysis and one method diagnosis | 4--8 hours |
| Official B-Spline Policy integration and conformance tests | 1--2 days |
| CAT reproduction and conformance tests | 2--4 days |
| Spline Policy reproduction and conformance tests | 2--3 days |
| Named-baseline development runs | 2--4 days |
| Fresh confirmatory head-to-head campaign | 2--4 days |
| Final provenance audit and reviewer decision | 0.5--1 day |

Best case, with a strong first ARC--TAC signal and no reproduction surprises: **8--12 days**.

Realistic case, including one documented ARC--TAC revision and one failed baseline-integration
attempt: **14--21 days**.

If the method needs multiple substantive revisions: **3--5 weeks**. A positive result cannot be
guaranteed by a date; the defensible commitment is to continue the method/evidence loop without
fabricating, relabeling, or selectively excluding results.
