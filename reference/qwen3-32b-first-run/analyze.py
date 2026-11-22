"""Read k0.jsonl and rollouts.jsonl; print AUROCs with bootstrap CIs using my reference functions."""
import json
import sys

import torch as t

import logit_monitor as L

g = t.Generator().manual_seed(20260928)
POS, NEG = "fortress_adv", "wildchat_benign"


def report(label, pos, neg, n_boot=1000):
    pos, neg = t.tensor(pos, dtype=t.float64), t.tensor(neg, dtype=t.float64)
    a = L.auroc(pos, neg)
    lo, hi = L.bootstrap_ci(pos, neg, n_boot, g)
    print(f"  {label:44} AUROC {a:.3f}  [{lo:.3f}, {hi:.3f}]  n={len(pos)}/{len(neg)}")


def k0(path):
    rows = [json.loads(l) for l in open(path)]
    skipped = sum(1 for r in rows if r.get("skipped"))
    rows = [r for r in rows if not r.get("skipped")]
    print(f"== start of reasoning (k=0), {len(rows)} prompts scored, {skipped} skipped for length")
    diff = max(abs(r["j_primary"] - r["mine"]) for r in rows)
    print(f"  max |johntrob construction - my function| = {diff:.2e} nats")
    by = lambda f: ([f(r) for r in rows if r["set"] == POS], [f(r) for r in rows if r["set"] == NEG])
    report("johntrob: primary phrase", *by(lambda r: r["j_primary"]))
    report("johntrob: primary - control", *by(lambda r: r["j_primary"] - r["j_control"]))
    report("frankyaoxiao: primary, no think tag", *by(lambda r: r["f_primary"]))
    report("mine (same sequences as johntrob)", *by(lambda r: r["mine"]))
    report("baseline: prompt length in tokens", *by(lambda r: float(r["n_prompt"])))
    gap = [r["j_primary"] - r["f_primary"] for r in rows]
    print(f"  johntrob - frankyaoxiao score: mean {sum(gap) / len(gap):.1f} nats (min {min(gap):.1f}, max {max(gap):.1f})")
    x = t.tensor([r["j_primary"] for r in rows]); y = t.tensor([r["f_primary"] for r in rows])
    print(f"  correlation johntrob vs frankyaoxiao scores: {L.pearson(x, y):.3f}")


def rollouts(path):
    rows = [json.loads(l) for l in open(path)]
    R = len(rows[0]["rollouts"])
    print(f"== main monitor (max over boundaries and rollouts), {len(rows)} prompts, {R} rollouts each")
    nb = [len(x["bounds"]) for r in rows for x in r["rollouts"]]
    fin = sum(x["finished"] for r in rows for x in r["rollouts"])
    print(f"  boundaries per rollout: mean {sum(nb) / len(nb):.1f}; reasoning finished inside the token budget: {fin}/{len(nb)}")

    def table(field, sets):
        sel = [r for r in rows if r["set"] == sets]
        return L.pad_traces([L.pad_traces([t.tensor(x[field]) for x in r["rollouts"]]).flatten() for r in sel]), sel

    def cube(field, name):
        sel = [r for r in rows if r["set"] == name]
        longest = max(len(x[field]) for r in sel for x in r["rollouts"])
        c = t.full((len(sel), R, longest), float("-inf"))
        for i, r in enumerate(sel):
            for j, x in enumerate(r["rollouts"]):
                c[i, j, :len(x[field])] = t.tensor(x[field])
        return c, sel

    (tp, sp), (tn, sn) = cube("testing", POS), cube("testing", NEG)
    (ap, _), (an, _) = cube("asking", POS), cube("asking", NEG)
    report("k=0 start score (this subset)", L.start_scores(tp).tolist(), L.start_scores(tn).tolist())
    for k in range(1, R + 1):
        report(f"logit monitor, k={k} rollouts", L.monitor_scores(tp, k).tolist(), L.monitor_scores(tn, k).tolist())
    report(f"logit monitor - control phrase, k={R}", L.control_difference(tp, ap, R).tolist(), L.control_difference(tn, an, R).tolist())
    fp = t.tensor([[x["verbal"] for x in r["rollouts"]] for r in sp])
    fn = t.tensor([[x["verbal"] for x in r["rollouts"]] for r in sn])
    print(f"  keyword-flagged rollouts: eval {int(fp.sum())}/{fp.numel()}, deploy {int(fn.sum())}/{fn.numel()}")
    for k in (1, R):
        report(f"keyword verbalisation rate, k={k}", L.verbalization_scores(fp, k).tolist(), L.verbalization_scores(fn, k).tolist())
    report("baseline: prompt length", [float(r["n_prompt"]) for r in sp], [float(r["n_prompt"]) for r in sn])
    nbp = [float(sum(len(x["bounds"]) for x in r["rollouts"])) for r in sp]
    nbn = [float(sum(len(x["bounds"]) for x in r["rollouts"])) for r in sn]
    report("baseline: number of boundaries scored", nbp, nbn)


if __name__ == "__main__":
    k0(sys.argv[1])
    if len(sys.argv) > 2:
        rollouts(sys.argv[2])
