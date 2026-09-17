import glob
import re

files = sorted(glob.glob('/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804/results_matched8300_parallel_eval/arm_*/eval.log'))
print(f"{'Evaluation Arm':32s} | {'Success/Trials':14s} | {'Success Rate':12s}")
print("-" * 65)

for f in files:
    arm = f.split('/')[-2].replace('arm_', '')
    content = open(f, encoding='utf-8').read()
    m = re.search(r'-->\s*([a-zA-Z0-9_]+):\s*([0-9\.]+)%', content)
    diag = re.search(r'diagnostics:\s*([0-9]+)/([0-9]+)\s*ok', content)
    if m and diag:
        succ = int(diag.group(1))
        tot = int(diag.group(2))
        pct = float(m.group(2))
        print(f"{arm:32s} | {succ:2d} / {tot:2d}        | {pct:5.1f}%")
    elif diag:
        succ = int(diag.group(1))
        tot = int(diag.group(2))
        pct = succ / tot * 100 if tot > 0 else 0
        print(f"{arm:32s} | {succ:2d} / {tot:2d}        | {pct:5.1f}%")
    else:
        oks = len(re.findall(r': OK', content))
        trials = len(re.findall(r'trial [0-9]+:', content))
        pct = oks / trials * 100 if trials > 0 else 0
        print(f"{arm:32s} | {oks:2d} / {trials:2d} (raw)  | {pct:5.1f}%")
