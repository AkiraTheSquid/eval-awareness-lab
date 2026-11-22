"""Pilots X1 and X2 (research/ideas.md) on the Qwen3-32B reference
rollouts. No model needed.

X1: is "asking" beating "testing" already at k = 0 (from the prompt),
    or only once reasoning is sampled? Paired bootstrap of
    AUROC(asking) - AUROC(testing) at each k.
X2: where in the reasoning does the winning boundary sit? Relative
    position (argmax boundary / reasoning length) per rollout, median
    per prompt, eval vs deploy.

usage: python lab/x1_x2.py [rollouts.jsonl] [R]   -> runs/pilots/x1_x2*.json
"""
import json
import os
import sys

import torch as t

import analysis as A

path = sys.argv[1] if len(sys.argv) > 1 else \
    "reference/qwen3-32b-vast/rollouts_Qwen3-32B_R4_T384_seed0.jsonl"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 4
tag = os.path.basename(os.path.dirname(path))
ev, de = A.load_pairs(path)
g = t.Generator().manual_seed(7)
out = {"source": path, "pairs": len(ev)}

tp, tn = A.cube(ev, "testing", R), A.cube(de, "testing", R)
ap, an = A.cube(ev, "asking", R), A.cube(de, "asking", R)
x1 = {}
for k in range(R + 1):
    d, lo, hi = A.paired_diff_ci(
        A.monitor_scores(ap, k), A.monitor_scores(an, k),
        A.monitor_scores(tp, k), A.monitor_scores(tn, k), g)
    x1[k] = dict(asking_minus_testing=d, lo=lo, hi=hi)
    print(f"X1 k={k}: AUROC(asking) - AUROC(testing) = {d:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]")
out["x1"] = x1


def rel_argmax(rows):
    """Per prompt: median over rollouts of (argmax boundary position /
    reasoning length), boundary 0 excluded so k = 0 does not dominate.
    Rollouts with fewer than 3 boundaries are skipped."""
    per = []
    for r in rows:
        vals = []
        for x in r["rollouts"][:R]:
            s, b = x["testing"][1:], x["bounds"][1:]
            if len(s) < 2:
                continue
            i = max(range(len(s)), key=s.__getitem__)
            vals.append(b[i] / x["n_cot"])
        if vals:
            per.append(t.tensor(vals).median().item())
    return t.tensor(per, dtype=t.float64)


pe, pd = rel_argmax(ev), rel_argmax(de)
diff = (pe.median() - pd.median()).item()
boots = []
for _ in range(2000):
    a = pe[t.randint(0, len(pe), (len(pe),), generator=g)]
    b = pd[t.randint(0, len(pd), (len(pd),), generator=g)]
    boots.append((a.median() - b.median()).item())
boots = t.tensor(boots)
lo, hi = t.quantile(boots, 0.025).item(), t.quantile(boots, 0.975).item()
# also: how often is boundary 0 the overall max of a rollout?
b0 = [float(x["testing"][0] >= max(x["testing"]))
      for r in ev + de for x in r["rollouts"][:R]]
out["x2"] = dict(eval_median=pe.median().item(),
                 deploy_median=pd.median().item(), diff=diff, lo=lo,
                 hi=hi, auroc_eval_earlier=A.auroc(-pe, -pd),
                 share_rollouts_max_at_boundary0=sum(b0) / len(b0))
print(f"X2: median relative argmax eval {pe.median():.2f} vs deploy "
      f"{pd.median():.2f}; diff {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]; "
      f"boundary 0 is the rollout max in {sum(b0) / len(b0):.0%}")
os.makedirs("runs/pilots", exist_ok=True)
json.dump(out, open(f"runs/pilots/x1_x2_{tag}.json", "w"), indent=1)
