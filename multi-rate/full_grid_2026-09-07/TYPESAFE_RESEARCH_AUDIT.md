# TypeSafe research audit

This adds semantic review around the existing deterministic evaluation pipeline. It does not
change the harness, success vectors, statistical tests, or preregistered decision rules.

## Workflows

`typesafe_research_audit.py` covers five review tasks:

1. **Claim verification:** candidate JSON evidence is paired with each numerical or comparative
   Markdown claim; TypeSafe labels it supported, contradicted, unsupported, or misattributed.
2. **Preregistration compliance:** four independent judgments check endpoint coverage, sample
   size, correction family, and whether confirmatory wording is justified.
3. **Claim strength:** wording is classified as calibrated, overstated, understated, or unclear.
4. **Experiment triage:** code marks obvious floor/ceiling candidates and TypeSafe supplies a
   bounded semantic interpretation. The deterministic tag remains visible.
5. **Related-work screening:** annotated bibliography entries receive separate relevance,
   methodological-overlap, provenance-quality, and primary-source judgments.

All raw percentages, deltas, discordant pairs, and exact McNemar p-values are recomputed from the
episode boolean vectors. TypeSafe never calculates or replaces scientific results.

## Prepare offline

```bash
cd /home/user/Desktop/multi-rate/full_grid_2026-09-07
python3 typesafe_research_audit.py self-test
python3 typesafe_research_audit.py prepare
```

The second command writes `typesafe_audit_bundle.json`. Review this file before any API call: it
contains every state object, question, evidence candidate, and preregistration excerpt that will
be sent. Defaults cap triage at 40 result files; change `--max-triage` deliberately after checking
request cost.

Audit the paper draft instead of the QP results draft with:

```bash
python3 typesafe_research_audit.py prepare \
  --document paper_arc_tac/phase4_draft/MANUSCRIPT.md \
  --output paper_arc_tac/typesafe_audit_bundle.json
```

## Run TypeSafe

Install the current SDK in the intended environment and supply the API key without committing it:

```bash
pip install typesafe-sdk
export TYPESAFE_API_KEY=...
python3 typesafe_research_audit.py run typesafe_audit_bundle.json
```

The default review threshold is `0.8`. Choice and Score use the returned confidence. For Noul,
the runner derives certainty from distance to 0.5; probabilities near 0.5 are reviewed. Thresholds
must be calibrated on labeled project examples before CI enforcement.

## Boundaries

- A raw p-value is never treated as Holm-adjusted.
- Open-loop replay and closed-loop rollout evidence are not interchangeable.
- Confirmatory language requires the preregistered endpoint, threshold, and sample size.
- Semantic outputs are review signals, not permission to edit results or launch experiments.
- Keep `typesafe_audit_report.json` as an audit artifact only after checking API-input privacy.

The implementation follows the live TypeSafe Python SDK, Choice/Noul/Score, confidence-routing,
and citation-check guidance current on 2026-09-22.
