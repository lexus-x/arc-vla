import json, glob, itertools
from math import comb

ARMS = {}
for p in glob.glob('pow_plus_*_out/plus_multirate.json'):
    for k, v in json.load(open(p)).items():
        ARMS[k] = v

# find the per-episode record list
def episodes(v):
    for key in ('episodes', 'records', 'results', 'trials'):
        if isinstance(v, dict) and key in v and isinstance(v[key], list):
            return v[key]
    if isinstance(v, list):
        return v
    return None

for k, v in ARMS.items():
    e = episodes(v)
    print(k, type(v).__name__, list(v)[:8] if isinstance(v, dict) else '', '| eps:', (len(e) if e else None))
    if e:
        print('   sample:', {kk: e[0][kk] for kk in list(e[0])[:8]})
