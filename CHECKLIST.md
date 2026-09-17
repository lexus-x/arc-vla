# Workspace & Harness Audit — Results

Run 2026-09-10. Findings + fixes applied to `/home/user/Desktop` and `~/.claude`.

## Score: 1 PASS / 1 PARTIAL / 17 FAIL (before fixes)

| # | Item | Verdict | Evidence |
|---|------|---------|----------|
| 1 | project context | FAIL → fixed | No `CLAUDE.md` existed. Written: env names, eval entry point, stats house rules. |
| 2 | project check | FAIL (open) | `/home/user/Desktop/.git/` has only `info/` — not a real repo. Not initialized (278GB of data/checkpoints under this dir makes root-level git the wrong unit — flagged, not acted on). |
| 3 | useless tools | FAIL → fixed | 13 MCP servers configured, 0 working. Removed `filesystem`/`browser-server`/`fetch`/`memory`/`executeautomation-playwright-server` (placeholders, dead, or superseded); fixed `huggingface` (missing `"type":"http"`) — now connects. |
| 4 | rules.md | FAIL → fixed | `.clinerules/` + `.agents/rules/` (19 files) were Cline/Antigravity formats Claude Code never reads. Moved out; house rules now live in `CLAUDE.md`. |
| 5 | github | FAIL (open) | No repo, no remote. Not something to auto-create — needs your call on repo boundaries. |
| 6 | session pre-loading | PARTIAL (open) | `remember` hook works; `openviking` still 401s (needs your credential fix). |
| 7 | model settings / cheap model | FAIL → fixed | Top-level `effortLevel: xhigh` removed (per-model override on `claude-opus-5` kept). Added `model: sonnet` to 4 custom agents that had none (`context-manager`, `devops-engineer`, `python-pro`, `test-engineer`). Built-in agent types (Explore/general-purpose/Plan) aren't overridable this way — no global "subagent model" setting exists. |
| 8 | plan-mode-first | FAIL → fixed | `permissions.defaultMode`: `auto` → `plan`. |
| 9 | claude.md optimize | FAIL → fixed | Created, ~50 lines. |
| 10 | no skills in claude.md | FAIL → fixed | `AGENTS.md` was entirely an agent/skill/command roster. Rewritten to 8 lines, no catalog. |
| 11 | proper file structure | FAIL → partially fixed | Moved 2 stray root `.mp4`s into `multi-rate/`, deleted a duplicate `.xlsx` and a stray file literally named `camera 10`. Did **not** touch `Ayush/`, `trajmdn_vla/`, `vla-gap-work/`, `omni_vla_research/` — bigger restructuring needs a separate call from you. |
| 12 | interview me first | DONE → now durable | Was done ad hoc; now written into `CLAUDE.md`'s working conventions. |
| 13 | agents' output check | FAIL → now durable | No verification convention existed. `CLAUDE.md` now states: check a subagent's claim against the source file before repeating it. |
| 14 | skill building | FAIL → fixed | 646 skill dirs, 0 enabled anywhere; 8.2GB plugin cache for 3 active plugins. Pruned (see below). |
| 15 | computer use or scripts | PASS | Research is already script-driven. No change needed. |
| 16 | mcp / api / cli connections | FAIL → fixed | Same as #3. `filesystem` scope conflict (user vs project) still exists — that's the *user-level* entry, out of this project's scope to fix. |
| 17 | example reference — good only | FAIL (open) | `.agents/skills/*/examples/` shipped built outputs, not curated references — moot now, that tree is pruned. No replacement curated set exists; flagged, not built (would need you to name what's worth keeping as a reference). |
| 18 | feedback loop set | FAIL → fixed | Added a `Stop` hook (`'.claude/hooks/stop-pycompile.sh`) that `py_compile`s every `.py` file touched this session and blocks silently succeeding on a syntax error. Self-tested against a broken file, a clean file, and empty input. |
| 19 | only .md, not docs/pdf | FAIL → fixed | Converted the live scoreboard `.xlsx` → `BENCHMARK_TRACKER.md` (verified greppable, per-sheet tables intact). Archived `.pdf`/`.pptx`/`.docx` left alone — historical records, not working documents. |

## What actually changed on disk

- **8.2GB → 16MB**: uninstalled 290 disabled plugins via `claude plugin uninstall` (manifest), then deleted their now-orphaned cache clones (CLI has no command for this — confirmed with you before deleting).
- **Moved to `.trash_pruned_2026-09-10/`** (not deleted — `rm -rf` is blocked at the permission layer regardless of plan approval, so this is the reversible equivalent): `.agents/` (47MB, Antigravity, never read by Claude Code), `.clinerules/` (Cline, never read), `.claude/skills/` (188KB, all 173 duplicated elsewhere, all disabled), `skills-lock.json`, both `.xlsx` copies of the tracker, the stray `camera 10` file. Empty it yourself once you've confirmed nothing's needed: `rm -rf /home/user/Desktop/.trash_pruned_2026-09-10`.
- **`.mcp.json`**: 8 servers → 3 (`context7`, `github`, `huggingface`); `huggingface` now actually connects.
- **`~/.claude/settings.json`**: `defaultMode: plan`, top-level `effortLevel` removed, dead 227-entry `skillOverrides` block removed.
- **`.claude/settings.local.json`**: dead 172-entry `skillOverrides` removed, dead `additionalDirectories` entries removed, new `Stop` hook registered.
- **New**: `CLAUDE.md`, `BENCHMARK_TRACKER.md`, `.claude/hooks/stop-pycompile.sh`.
- **Rewritten**: `AGENTS.md` (roster → 8-line pointer).
- **4 agent defs** gained `model: sonnet` (were inheriting the expensive default).

## Deliberately not touched (your call, not mine)

- **Secrets**: `.mcp.json` still has a live GitHub PAT, HF token, and Context7 key in plaintext, file mode 644. Flagged only, per your instruction.
- **`disabledMcpjsonServers`**: still lists `context7`/`github` as disabled even though `.mcp.json` now configures them cleanly — editing that specific field was blocked by the permission classifier (MCP-enablement is guarded against agent self-edits, reasonably). Flip it yourself via `/mcp` if you want them live.
- **Git init at repo root**: not done — 278GB of datasets/checkpoints makes this the wrong unit for a single repo.
- **#2, #5, #6, #17**: need a decision from you, not a mechanical fix — see table above.

## Parked from earlier this session (not part of this checklist)

Three research agents already came back on the FOLD-vs-DeepONet results question your advisor raised (protocol audit, results-table audit, eval-harness trace). Key thread to pick up: `assemble_paper_table.py`'s merge key is wrong and currently prints PickCube 1X Base as 1% when the true value is 90% — the published table is actively broken, independent of any DeepONet-comparison question. Say the word when you want to move back to that.
