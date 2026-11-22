"""E0: re-analyse the Qwen3-32B reference rollouts with lab/analysis.py.

usage: python lab/e0_reference.py
Writes runs/e0/{table.md,results.json,auroc_by_k.png,variance.json}.
Fails loudly if any point estimate differs from the reference summary
by more than 0.0005 (docs/02-experiments.md E0).
"""
import json
import os

import matplotlib
import torch as t
from einops import reduce

import analysis as A

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REF = "reference/qwen3-32b-vast"
OUT = "runs/e0"
os.makedirs(OUT, exist_ok=True)
assert A.self_test()
ev, de = A.load_pairs(f"{REF}/rollouts_Qwen3-32B_R4_T384_seed0.jsonl")
table = A.score_table(ev, de, ks=[0, 1, 2, 3, 4])
rows = A.report(table)

# the reference summary's labels -> ours
ref = {r["label"]: r["auroc"] for r in
       json.load(open(f"{REF}/summary_Qwen3-32B.json"))["rows"]}
MAP = {"baseline: prompt length (want ~0.5)": "prompt length",
       "baseline: number of boundaries scored": "boundaries scored, k=4",
       "start of reasoning only (k=0)": "monitor k=0",
       "logit monitor minus control phrase, k=4":
           "max primary - max control k=4",
       "control phrase alone, k=4": "control alone k=4"}
MAP.update({f"logit monitor, k={k} rollouts": f"monitor k={k}"
            for k in range(1, 5)})
MAP.update({f"keyword verbalisation rate, k={k}":
            f"keyword verbalisation k={k}" for k in range(1, 5)})
ours = {lab: a for lab, a, _, _ in rows}
match = []
for theirs, mine in MAP.items():
    gap = abs(ref[theirs] - ours[mine])
    match.append((mine, ref[theirs], ours[mine], gap))
    assert gap < 5e-4, (theirs, ref[theirs], ours[mine])
print(f"E0: all {len(match)} reference AUROCs reproduced to 3 decimals")

# AUROC by k: primary, control, difference
g = t.Generator().manual_seed(1)
ks = [0, 1, 2, 3, 4]
fig, ax = plt.subplots(figsize=(6, 4))
curves = {}
for name, fmt in (("testing", "monitor k={}"),
                  ("asking (control)", "control alone k={}"),
                  ("testing - asking", "same-boundary primary - control k={}")):
    pts = [(A.auroc(*table[fmt.format(k)]),
            *A.bootstrap_ci(*table[fmt.format(k)], g)) for k in ks]
    curves[name] = pts
    ax.plot(ks, [p[0] for p in pts], marker="o", label=name)
    ax.fill_between(ks, [p[1] for p in pts], [p[2] for p in pts],
                    alpha=0.15)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
ax.set(xlabel="rollouts used (k); k = 0 is the start of reasoning",
       ylabel="AUROC, Fortress vs WildChat", ylim=(0.2, 0.9),
       title=f"Qwen3-32B reference, {len(ev)} matched pairs")
ax.legend()
fig.savefig(f"{OUT}/auroc_by_k.png", dpi=130, bbox_inches="tight")

# Q2 first look: spread of rollout maxima within a prompt vs spread of
# prompt means between prompts
tp = A.cube(ev + de, "testing", 4)
per_roll = reduce(tp, "p r b -> p r", "max")
within = reduce(per_roll.var(dim=1), "p ->", "mean").item()
between = per_roll.mean(dim=1).var().item()
icc = between / (between + within)
var = {"within_prompt_var_of_rollout_max": within,
       "between_prompt_var_of_prompt_mean": between,
       "share_between": icc,
       "note": "naive ICC-style share; R6 fits the proper mixed model"}
json.dump(var, open(f"{OUT}/variance.json", "w"), indent=1)
print(f"variance of per-rollout max: within {within:.2f}, between "
      f"{between:.2f}, share between {icc:.2f}")

json.dump({"pairs": len(ev), "rows": rows, "curves": curves,
           "reference_match": match}, open(f"{OUT}/results.json", "w"),
          indent=1)
open(f"{OUT}/table.md", "w").write(
    f"Qwen3-32B reference, {len(ev)} pairs, R=4, T=384 "
    f"(command: python lab/e0_reference.py)\n\n" + A.markdown(rows) + "\n")
print(A.markdown(rows))
