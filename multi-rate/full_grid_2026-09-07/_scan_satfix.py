import glob, re
d = "/home/user/Desktop/multi-rate/satfix_2026-09-05/"
print("files:", sorted(glob.glob(d + "*.md")) + sorted(glob.glob(d + "*.txt"))[:10])
for f in glob.glob(d + "*.md") + glob.glob(d + "*.txt"):
    txt = open(f, errors="replace").read()
    for m in re.finditer(r".*(0/36|Holm|0 of 36|holm).*", txt):
        line = m.group(0).strip()
        if "0/36" in line or "0 of 36" in line:
            print(f.split("/")[-1], "::", line[:300])
