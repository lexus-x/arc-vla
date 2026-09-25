"""PREREG_LEARNED_GOVERNOR.md analysis, k=4 confirmation. Reads result_dp_<task>_k4_confirm.json only.
Exact two-sided McNemar per comparison, Holm step-down within each family, void if both arms <10% or >90%."""
import json, math
from harness import exact_mcnemar

TASKS = ['PickCube-v1', 'RollBall-v1', 'PullCube-v1', 'LiftPegUpright-v1', 'PushCube-v1', 'AnymalC-Reach-v1', 'PokeCube-v1', 'StackCube-v1']
WIN = {'PickCube-v1': (500, 293)}  # (eval_offset, n); all others (100, 400)
FAMILIES = {'H1 vs tac_fold_satfix': ('tac_fold_satfix', TASKS), 'H2 vs qp_anchor': ('qp_anchor', TASKS),
            'H4 vs learned_raw': ('learned_raw', ['RollBall-v1', 'LiftPegUpright-v1', 'PullCube-v1', 'AnymalC-Reach-v1']),
            'H6 vs learned_tanh': ('learned_tanh', TASKS), 'H3 vs mlp_bc (no direction)': ('mlp_bc', TASKS)}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


S = {}
for t in TASKS:
    d = json.load(open(f'result_dp_{t}_k4_confirm.json')); off, n = WIN.get(t, (100, 400))
    assert d['eval_offset'] == off and d['n'] == n and d['k'] == 4, (t, d['eval_offset'], d['n'])
    S[t] = d['success']

print('Success rates (%), k=4, live, DP:')
arms = list(S[TASKS[0]])
print(f"{'task':18s}" + ''.join(f'{a[:12]:>13s}' for a in arms))
for t in TASKS: print(f'{t:18s}' + ''.join(f'{100 * sum(S[t][a]) / len(S[t][a]):13.1f}' for a in arms))

for fam, (ref, tasks) in FAMILIES.items():
    rows = []
    for t in tasks:
        a, b = S[t]['qp_learned'], S[t][ref]; ao, bo, p = exact_mcnemar(a, b)
        ra, rb = sum(a) / len(a), sum(b) / len(b); void = (ra < .1 and rb < .1) or (ra > .9 and rb > .9)
        rows.append([t, 100 * (ra - rb), ao, bo, p, void])
    adj = holm([r[4] for r in rows])
    print(f'\n{fam} (Holm m={len(rows)})')
    for r, q in zip(rows, adj):
        verdict = 'VOID' if r[5] else ('WIN' if q < .05 and r[1] > 0 else 'LOSS' if q < .05 else 'n.s.')
        print(f'  {r[0]:18s} {r[1]:+6.1f}pp  {r[2]:3d}/{r[3]:<3d} p={r[4]:.2e}  holm={q:.2e}  {verdict}')
    if fam.startswith('H1'):
        wins = sum(q < .05 and r[1] > 0 and not r[5] for r, q in zip(rows, adj)); loss = sum(q < .05 and r[1] < 0 and not r[5] for r, q in zip(rows, adj))
        print(f'  H1 claim rule (>=4/8 Holm wins, 0 Holm losses): wins={wins} losses={loss} -> {"PASS" if wins >= 4 and loss == 0 else "FAIL"}')
