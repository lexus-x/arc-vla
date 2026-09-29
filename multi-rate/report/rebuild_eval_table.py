#!/usr/bin/env python3
"""Rebuild report/eval_table.html from raw per-episode result JSONs.

Tier A: paired closed-loop booleans in full_grid_2026-09-07/ -> exact McNemar at build time;
        Holm only over pre-registered primary families.
Tier C: unpaired point estimates archived in eval_table.outdated.html (RoboMimic, MetaWorld,
        ManiSkill k=2 Spline/B-Spline). No raw JSON backs them, so no tests, not citable.

ARC = the `qp_anchor` arm. Means are unweighted task means: descriptive, not tests.
No cross-suite mean (Push-T is 1 task, MetaWorld is N=6/env).

Run:  python3 rebuild_eval_table.py      (idempotent; writes eval_table.html)
"""
import hashlib
import json
import math
import os
import re
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = "/home/user/Desktop/multi-rate/full_grid_2026-09-07"
OUTDATED = os.path.join(HERE, "eval_table.outdated.html")
OUT = os.path.join(HERE, "eval_table.html")

MS_TASKS = ["PickCube-v1", "LiftPegUpright-v1", "PushCube-v1", "PullCube-v1",
            "RollBall-v1", "StackCube-v1", "AnymalC-Reach-v1", "PokeCube-v1"]
LABEL = {"native": "Native", "zoh": "ZOH", "spline": "Spline", "spline_satfix": "Spline + satfix",
         "bspline_eps_raw": "B-Spline", "bspline_eps_satfix": "B-Spline + satfix",
         "tac_fold_satfix": "TAC-Fold", "qp": "QP", "qp_anchor": "ARC",
         "qp_learned": "Learned governor"}
# Archived row layout; only the k=2 block is shown (the 2x-faster cells have no raw JSON).
ARCHIVE_COLS = ["native", "spline", "bspline", "tacfold", "qpa", "spline_f", "bspline_f", "qpa_f"]
ARCHIVE_SHOW = ["native", "spline", "bspline", "tacfold", "qpa"]
SUITE_LABEL = {"maniskill": "ManiSkill 3", "robomimic": "RoboMimic", "metaworld": "MetaWorld"}
SOURCES = []  # (file, sha256, n) for every raw JSON read


def load(fname):
    with open(os.path.join(GRID, fname), "rb") as f:
        raw = f.read()
    d = json.loads(raw)
    SOURCES.append((fname, hashlib.sha256(raw).hexdigest(), d["n"]))
    return d, {a: [bool(x) for x in v] for a, v in d["success"].items()}


def pct(v):
    return 100.0 * sum(v) / len(v)


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans. Returns (delta_pp, p, a_only, b_only)."""
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    if n == 0:
        return 0.0, 1.0, 0, 0
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n)
    return (a_only - b_only) / len(a) * 100.0, p, a_only, b_only


def holm(ps):
    """Holm step-down adjusted p-values, returned in input order."""
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(sorted(range(len(ps)), key=ps.__getitem__)):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def fmt_p(p):
    # House rule: tiny p-values are formatting artifacts, not extra evidence.
    return "&lt;0.001" if p < 0.001 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def achievable(v, n):
    """Can a 1-decimal percentage v be s/n for some integer s?"""
    return any(abs(100.0 * s / n - v) <= 0.05 + 1e-6 for s in range(n + 1))


def wtl(diffs):
    return f"{sum(d > 0 for d in diffs)}/{sum(d == 0 for d in diffs)}/{sum(d < 0 for d in diffs)}"


def self_check():
    assert all(abs(x - y) < 1e-12 for x, y in zip(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
    assert achievable(55.7, 400) and not achievable(51.1, 400) and achievable(83.6, 293)
    assert mcnemar([True, True, False], [False, True, True])[1:] == (1.0, 1, 1)


def maniskill(k):
    """Per-task paired rows from the k-specific confirm JSONs, plus a summary."""
    rows = []
    for task in MS_TASKS:
        d, s = load(f"result_dp_{task}_k{k}_confirm.json")
        rows.append({"task": task, "n": d["n"], "pct": {a: pct(v) for a, v in s.items()},
                     "vs_zoh": mcnemar(s["qp_anchor"], s["zoh"]),
                     "vs_tac": mcnemar(s["qp_anchor"], s["tac_fold_satfix"])})
    diff = lambda a, b: [r["pct"][a] - r["pct"][b] for r in rows]
    gaps = [(r["pct"]["qp_anchor"] - r["pct"]["zoh"]) / (r["pct"]["native"] - r["pct"]["zoh"])
            for r in rows if r["pct"]["native"] - r["pct"]["zoh"] > 5]
    summary = {"k": k,
               "mean": {a: sum(r["pct"][a] for r in rows) / len(rows)
                        for a in ("native", "zoh", "tac_fold_satfix", "qp_anchor", "qp_learned")},
               "wtl_zoh": wtl(diff("qp_anchor", "zoh")),
               "wtl_tac": wtl(diff("qp_anchor", "tac_fold_satfix")),
               "wtl_learned": wtl(diff("qp_learned", "qp_anchor")),
               "gap": 100.0 * sum(gaps) / len(gaps), "n_gap": len(gaps)}
    return rows, summary


def contrasts(files, family):
    """ARC vs every other arm, per file; Holm over the pre-registered `family` arms across files."""
    recs = []
    for fname in files:
        d, s = load(fname)
        for arm in s:
            if arm == "qp_anchor":
                continue
            delta, p, a_only, b_only = mcnemar(s["qp_anchor"], s[arm])
            recs.append({"k": d["k"], "n": d["n"], "arm": arm, "arm_pct": pct(s[arm]),
                         "arc_pct": pct(s["qp_anchor"]), "delta": delta, "p": p,
                         "a_only": a_only, "b_only": b_only, "adj": None,
                         "same": [LABEL.get(o, o) for o in s
                                  if o not in (arm, "qp_anchor") and s[o] == s[arm]]})
    fam = [r for r in recs if r["arm"] in family]
    for r, adj in zip(fam, holm([r["p"] for r in fam])):
        r["adj"] = adj
    return recs


def archive_rows():
    """Per-task rows scraped from the archived report (ManiSkill, RoboMimic, MetaWorld only)."""
    rows = []
    page = open(OUTDATED).read()
    for suite, body in re.findall(r'<tr data-suite="([^"]+)">(.*?)</tr>', page, re.S):
        if suite not in SUITE_LABEL:
            continue
        vals = [float(x) for x in re.findall(r'class="num[^"]*">\s*(\d+\.\d+)%\s*</td>', body)]
        if len(vals) != len(ARCHIVE_COLS):
            continue  # footer/summary rows
        task = re.search(r'task-title">([^<]+)<', body).group(1)
        meta = re.search(r'task-meta">([^<]+)<', body).group(1)
        rows.append({"suite": suite, "task": task, "n": int(re.search(r"N=(\d+)", meta).group(1)),
                     "v": dict(zip(ARCHIVE_COLS, vals))})
    counts = {s: sum(r["suite"] == s for r in rows) for s in SUITE_LABEL}
    assert counts == {"maniskill": 8, "robomimic": 3, "metaworld": 1}, counts
    return rows


CSS = """
:root{--ink:#1e293b;--mut:#64748b;--line:#e2e8f0;--bg:#f8fafc;--pos:#047857;--neg:#b91c1c}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 system-ui,sans-serif;color:var(--ink);background:var(--bg)}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:26px;margin:0 0 4px}.sub{color:var(--mut);font-size:13px}
.verdict{margin:20px 0 8px;padding:16px 18px;border-left:5px solid #0f766e;background:#fff;border-radius:8px}
h2{font-size:19px;margin:36px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:15px;margin:22px 0 8px}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;background:#fff;font-size:12.5px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;
font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{background:#0f172a;color:#e2e8f0;font-size:11px}
tr.foot td{background:#f1f5f9;font-weight:600}
.arc{color:#1d4ed8;font-weight:600}.pos{color:var(--pos);font-weight:600}.neg{color:var(--neg);font-weight:600}
.p{color:var(--mut);font-weight:400}.warn{background:#fef3c7}
.note{color:var(--mut);font-size:12.5px;margin:8px 0 0}
footer{margin-top:40px;color:var(--mut);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}
"""


def table(head, rows, foot=()):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = lambda cells, cls="": f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    body = "".join(tr(r) for r in rows) + "".join(tr(r, ' class="foot"') for r in foot)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def delta(d, p=None):
    cls = "pos" if d > 0 else "neg" if d < 0 else ""
    tail = f' <span class="p">(p={fmt_p(p)})</span>' if p is not None else ""
    return f'<span class="{cls}">{d:+.1f}</span>{tail}'


def averages_table(sums):
    head = ["k", "Native", "ZOH", "TAC-Fold", "ARC", "ARC W/T/L vs ZOH",
            "ARC W/T/L vs TAC-Fold", "Gap to native recovered"]
    rows = [[f"k={s['k']}"] + [f"{s['mean'][a]:.2f}%" for a in ("native", "zoh", "tac_fold_satfix")]
            + [f'<span class="arc">{s["mean"]["qp_anchor"]:.2f}%</span>', s["wtl_zoh"], s["wtl_tac"],
               f"{s['gap']:.1f}% <span class='p'>({s['n_gap']} tasks)</span>"] for s in sums]
    return table(head, rows)


def maniskill_table(rows, s):
    head = ["Task", "n", "Native", "ZOH", "TAC-Fold", "ARC", "ARC &minus; ZOH, pp", "ARC &minus; TAC-Fold, pp"]
    body = [[r["task"], r["n"]] + [f"{r['pct'][a]:.2f}%" for a in ("native", "zoh", "tac_fold_satfix")]
            + [f'<span class="arc">{r["pct"]["qp_anchor"]:.2f}%</span>',
               delta(r["vs_zoh"][0], r["vs_zoh"][1]), delta(r["vs_tac"][0], r["vs_tac"][1])] for r in rows]
    m = s["mean"]
    foot = [["Mean (8 tasks)", ""] + [f"{m[a]:.2f}%" for a in ("native", "zoh", "tac_fold_satfix")]
            + [f"{m['qp_anchor']:.2f}%", delta(m["qp_anchor"] - m["zoh"]),
               delta(m["qp_anchor"] - m["tac_fold_satfix"])],
            ["ARC win/tie/loss", "", "", "", "", "", s["wtl_zoh"], s["wtl_tac"]]]
    return table(head, body, foot)


def contrast_table(recs):
    head = ["k", "vs arm", "Arm", "ARC", "ARC &minus; arm, pp", "ARC-only / arm-only",
            "p (exact McNemar)", "Holm p (pre-reg)", "Note"]
    rows = [[f"k={r['k']}", LABEL.get(r["arm"], r["arm"]), f"{r['arm_pct']:.2f}%",
             f"{r['arc_pct']:.2f}%", delta(r["delta"]), f"{r['a_only']} / {r['b_only']}", fmt_p(r["p"]),
             "&mdash; secondary" if r["adj"] is None
             else f"<b>{fmt_p(r['adj'])}</b> {'sig' if r['adj'] < 0.05 else 'n.s.'}",
             ("&equiv; " + ", ".join(r["same"]) + " (identical per episode)") if r["same"] else ""]
            for r in recs]
    return table(head, rows)


def learned_table(sums):
    head = ["k", "ARC mean", "Learned governor mean", "Learned &minus; ARC, pp", "Learned W/T/L vs ARC"]
    rows = [[f"k={s['k']}", f"{s['mean']['qp_anchor']:.2f}%", f"{s['mean']['qp_learned']:.2f}%",
             delta(s["mean"]["qp_learned"] - s["mean"]["qp_anchor"]), s["wtl_learned"]] for s in sums]
    return table(head, rows)


def archive_table(rows):
    head = ["Task", "Suite", "N", "Native", "Spline", "B-Spline", "TAC-Fold", "ARC"]

    def cell(r, c):
        v = r["v"][c]
        ok = r["suite"] == "metaworld" or achievable(v, r["n"])  # MetaWorld N is per env
        txt = f'<span class="arc">{v:.1f}%</span>' if c == "qpa" else f"{v:.1f}%"
        return txt if ok else f'<span class="warn" title="not k/{r["n"]} for any integer k">&#9888; {txt}</span>'

    out = []
    for suite, label in SUITE_LABEL.items():
        sub = [r for r in rows if r["suite"] == suite]
        body = [[r["task"], label, f"{r['n']}/env" if suite == "metaworld" else r["n"]]
                + [cell(r, c) for c in ARCHIVE_SHOW] for r in sub]
        foot = ([[f"Mean ({len(sub)} tasks)", label, ""]
                 + [f"{sum(r['v'][c] for r in sub) / len(sub):.2f}%" for c in ARCHIVE_SHOW]]
                if len(sub) > 1 else [])
        out.append(table(head, body, foot))
    return "".join(out)


def main():
    self_check()
    (ms2, s2), (ms4, s4) = maniskill(2), maniskill(4)
    pick = contrasts(["result_dp_PickCube-v1_k4_n400_anchor.json"],
                     {"spline_satfix", "tac_fold_satfix", "bspline_eps_satfix"})
    pusht = contrasts(["result_dp_PushT-v1_fair_n400.json", "result_dp_PushT-v1_k3_fair_n400.json"],
                      {"spline", "tac_fold_satfix"})
    archive = archive_rows()

    pk = next(r for r in pick if r["arm"] == "tac_fold_satfix")
    pt_fam = [r for r in pusht if r["adj"] is not None]
    pt_sp2 = next(r for r in pusht if r["k"] == 2 and r["arm"] == "spline")
    gain = lambda s: s["mean"]["qp_anchor"] - s["mean"]["tac_fold_satfix"]
    verdict = (
        f"<b>ManiSkill 3</b> (8 tasks): ARC beats ZOH on {s2['wtl_zoh'].split('/')[0]}/8 tasks at k=2 "
        f"and {s4['wtl_zoh'].split('/')[0]}/8 at k=4, but recovers only {s2['gap']:.0f}% / {s4['gap']:.0f}% "
        f"of the gap to native. Its mean margin over TAC-Fold is {gain(s2):+.1f} / {gain(s4):+.1f} pp. "
        f"<b>PickCube k=4</b> (pre-registered, n={pk['n']}): ARC {pk['delta']:+.1f} pp over TAC-Fold "
        f"&equiv; Spline + satfix (Holm p={fmt_p(pk['adj'])}). "
        f"<b>Push-T</b> (pre-registered, n={pt_sp2['n']}): {sum(r['adj'] < 0.05 for r in pt_fam)}/"
        f"{len(pt_fam)} primary contrasts Holm-significant; ARC &minus; Spline at k=2 is "
        f"{pt_sp2['delta']:+.1f} pp (p={fmt_p(pt_sp2['p'])}). "
        f"The learned governor, a separate method, averages {s2['mean']['qp_learned']:.1f}% / "
        f"{s4['mean']['qp_learned']:.1f}% vs ARC's {s2['mean']['qp_anchor']:.1f}% / "
        f"{s4['mean']['qp_anchor']:.1f}%.")

    sources = "".join(f"<li><code>{f}</code> &middot; n={n} &middot; sha256 {h[:12]}</li>"
                      for f, h, n in dict.fromkeys(SOURCES))
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-Rate Resampling Results</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>Multi-Rate Action Resampling &mdash; Results</h1>
<div class="sub">Generated {date.today().isoformat()} &middot; all results closed-loop (live rollout)
&middot; ARC = <code>qp_anchor</code> arm &middot; Tier A numbers recomputed from per-episode booleans</div>
<div class="verdict">{verdict}</div>

<h2>Averages &mdash; ManiSkill 3, 8 tasks (Tier A, paired)</h2>
{averages_table([s2, s4])}
<p class="note">Unweighted task means: descriptive, not a test. W/T/L counts tasks by success rate.
Gap recovered = mean of (ARC &minus; ZOH) / (Native &minus; ZOH) over tasks where ZOH is &gt;5 pp below native.
No cross-suite mean: Push-T is one task and MetaWorld has N=6 per env.</p>

<h2>ManiSkill 3 &mdash; 2&times; slower (k=2)</h2>
{maniskill_table(ms2, s2)}
<p class="note">p = exact McNemar, unadjusted. PickCube n=293, others n=400.</p>

<h2>ManiSkill 3 &mdash; 4&times; slower (k=4)</h2>
{maniskill_table(ms4, s4)}

<h2>Pre-registered contrasts (Tier A)</h2>
<h3>PickCube-v1, k=4, n=400 &mdash; <code>PREREG_QP_ANCHOR_CL.md</code>, Holm m=3</h3>
{contrast_table(pick)}
<p class="note">Spline + satfix and TAC-Fold are identical episode by episode here, as are B-Spline + satfix
and ZOH, so the m=3 family holds two distinct comparisons.</p>
<h3>Push-T, k=2 and k=3, n=400 &mdash; <code>PREREG_PUSHT_CL_FAIR_N400.md</code>, Holm m=4</h3>
{contrast_table(pusht)}
<p class="note">Spline and B-Spline are raw (satfix withheld, per the pre-registration). Rows marked
&ldquo;secondary&rdquo; are outside the primary family and unadjusted.</p>

<h2>Learned governor (separate pre-registered family, not ARC)</h2>
{learned_table([s2, s4])}
<p class="note">Same confirm JSONs (<code>qp_learned</code> arm). Hypothesis checks for this family:
<code>PREREG_LEARNED_GOVERNOR.md</code>, <code>Q2_PUBLICATION_EVIDENCE_DOSSIER.md</code>.</p>

<h2>Appendix &mdash; archived point estimates, k=2 (Tier C: unpaired, not citable)</h2>
{archive_table(archive)}
<p class="note">Scraped from <code>eval_table.outdated.html</code>; no raw JSON backs these cells, so no tests.
&#9888; = value cannot be s/N for any integer s at the stated N. MetaWorld: N=6 per env, weak evidence.
The archived 2&times;-faster and native-rate resampler cells are omitted: nothing backs them.</p>

<footer><b>Tier A sources</b> (full_grid_2026-09-07/)<ul>{sources}</ul></footer>
</div></body></html>
"""
    with open(OUT, "w") as f:
        f.write(doc)
    print(f"wrote {OUT} ({len(doc)} bytes, {len(dict.fromkeys(SOURCES))} source JSONs)")


if __name__ == "__main__":
    main()
