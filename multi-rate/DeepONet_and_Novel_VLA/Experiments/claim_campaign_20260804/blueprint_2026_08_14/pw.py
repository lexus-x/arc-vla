import glob, math
from safetensors import safe_open
base="/home/user/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH/v2/runs"
def head_params(run):
    f=f"{base}/{run}/checkpoints/8300/model.safetensors"
    with safe_open(f, framework="pt") as h:
        return sum(math.prod(h.get_slice(k).get_shape()) for k in h.keys()
                   if any(t in k for t in ("deeponet","head","trunk","branch","pool")))
for r in ["reg_s0","bandlimit_s0","abl_noF_s0","abl_1blk_s0"]:
    print(f"{r:14s} head params = {head_params(r):,}")

# seeds needed to detect the v2-vs-reg robustness gap
m1,s1,n1 = 38.5,6.9,5      # full v2
m2,s2,n2 = 32.7,0.9,3      # reg (no operator)
sp = math.sqrt(((n1-1)*s1**2+(n2-1)*s2**2)/(n1+n2-2))
d  = (m1-m2)/sp
n  = 2*((1.96+0.84)/d)**2   # two-sample, alpha=.05, power=.80
print(f"\npooled sd={sp:.2f}  Cohen d={d:.2f}  -> n={math.ceil(n)} SEEDS PER ARM")
