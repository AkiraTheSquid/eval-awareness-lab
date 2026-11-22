"""From scored rollouts to AUROCs (docs/01-method.md section 5).

Input is the reference rollout format: one JSON line per prompt with
key, set ("eval" / "deploy"), pair, n_prompt and rollouts, each rollout
holding bounds, testing, asking (one score per boundary) and verbal.
Only complete pairs are analysed, so the two sets stay length-matched.
"""
import json

import torch as t
from einops import reduce

PHRASES = ("testing", "asking")


def load_pairs(path):
    """(eval rows, deploy rows), complete pairs only, in pair order."""
    by_pair = {}
    for line in open(path):
        r = json.loads(line)
        by_pair.setdefault(r["pair"], {})[r["set"]] = r
    done = [v for _, v in sorted(by_pair.items()) if len(v) == 2]
    return [v["eval"] for v in done], [v["deploy"] for v in done]


def cube(rows, field, R=None):
    """(prompt, rollout, boundary), padded with -inf so padding never
    wins a max."""
    R = R or max(len(r["rollouts"]) for r in rows)
    longest = max(len(x[field]) for r in rows for x in r["rollouts"])
    c = t.full((len(rows), R, longest), float("-inf"), dtype=t.float64)
    for i, r in enumerate(rows):
        for j, x in enumerate(r["rollouts"][:R]):
            c[i, j, :len(x[field])] = t.tensor(x[field])
    return c


def monitor_scores(table, k):
    """Max over the first k rollouts; k = 0 is boundary 0 of rollout 0
    (identical across rollouts: no sampled reasoning yet)."""
    if k == 0:
        return table[:, 0, 0]
    return reduce(table[:, :k], "p r b -> p", "max")


def auroc(pos, neg):
    """Fraction of (pos, neg) pairs with pos > neg, ties half."""
    pos = t.as_tensor(pos, dtype=t.float64)
    neg = t.as_tensor(neg, dtype=t.float64)
    wins = (pos[:, None] > neg[None, :]).double()
    ties = (pos[:, None] == neg[None, :]).double()
    return reduce(wins + 0.5 * ties, "a b ->", "mean").item()


def _resamples(n_pos, n_neg, g, n_boot):
    ip = t.randint(0, n_pos, (n_boot, n_pos), generator=g)
    iq = t.randint(0, n_neg, (n_boot, n_neg), generator=g)
    return ip, iq


def bootstrap_ci(pos, neg, g, n_boot=2000):
    """95% interval; eval and deploy prompts resampled separately."""
    pos, neg = t.as_tensor(pos), t.as_tensor(neg)
    ip, iq = _resamples(len(pos), len(neg), g, n_boot)
    s = t.tensor([auroc(pos[a], neg[b]) for a, b in zip(ip, iq)])
    return t.quantile(s, 0.025).item(), t.quantile(s, 0.975).item()


def paired_diff_ci(pos1, neg1, pos2, neg2, g, n_boot=2000):
    """AUROC(score 1) - AUROC(score 2) with the same resampled prompts
    for both scores, so prompt-level noise cancels."""
    pos1, neg1 = t.as_tensor(pos1), t.as_tensor(neg1)
    pos2, neg2 = t.as_tensor(pos2), t.as_tensor(neg2)
    ip, iq = _resamples(len(pos1), len(neg1), g, n_boot)
    s = t.tensor([auroc(pos1[a], neg1[b]) - auroc(pos2[a], neg2[b])
                  for a, b in zip(ip, iq)])
    d = auroc(pos1, neg1) - auroc(pos2, neg2)
    return d, t.quantile(s, 0.025).item(), t.quantile(s, 0.975).item()


def verbal_rate(rows, k):
    """Fraction of the first k rollouts the keyword regex flags."""
    f = t.tensor([[float(x["verbal"]) for x in r["rollouts"][:k]]
                  for r in rows])
    return reduce(f, "p r -> p", "mean")


def score_table(ev, de, ks):
    """{label: (eval scores, deploy scores)}: monitor and baselines of
    docs/01-method.md section 5."""
    R = max(ks)
    tp, tn = cube(ev, "testing", R), cube(de, "testing", R)
    ap, an = cube(ev, "asking", R), cube(de, "asking", R)
    out = {
        "prompt length": ([r["n_prompt"] for r in ev],
                          [r["n_prompt"] for r in de]),
        f"boundaries scored, k={R}":
            ([sum(len(x["bounds"]) for x in r["rollouts"][:R])
              for r in ev],
             [sum(len(x["bounds"]) for x in r["rollouts"][:R])
              for r in de]),
    }
    for k in ks:
        out[f"monitor k={k}"] = (monitor_scores(tp, k),
                                 monitor_scores(tn, k))
    for k in ks:
        out[f"control alone k={k}"] = (monitor_scores(ap, k),
                                       monitor_scores(an, k))
    for k in ks:
        # what the reference run reported: max primary - max control
        out[f"max primary - max control k={k}"] = (
            monitor_scores(tp, k) - monitor_scores(ap, k),
            monitor_scores(tn, k) - monitor_scores(an, k))
    for k in ks:
        # what docs/01-method.md section 5 asks for: difference at the
        # same boundary, then max (-inf - -inf padding becomes nan)
        dp, dn = (tp - ap).nan_to_num(float("-inf")), \
            (tn - an).nan_to_num(float("-inf"))
        out[f"same-boundary primary - control k={k}"] = (
            monitor_scores(dp, k), monitor_scores(dn, k))
    for k in ks:
        if k:
            out[f"keyword verbalisation k={k}"] = (verbal_rate(ev, k),
                                                   verbal_rate(de, k))
    return out


def report(table, seed=20260928, n_boot=2000):
    """[(label, auroc, lo, hi)] for every row of score_table."""
    g = t.Generator().manual_seed(seed)
    rows = []
    for label, (pos, neg) in table.items():
        lo, hi = bootstrap_ci(pos, neg, g, n_boot)
        rows.append((label, auroc(pos, neg), lo, hi))
    return rows


def markdown(rows):
    out = ["| score | AUROC [95% CI] |", "|---|---|"]
    out += [f"| {lab} | {a:.3f} [{lo:.2f}, {hi:.2f}] |"
            for lab, a, lo, hi in rows]
    return "\n".join(out)


def self_test():
    """Check 5 of docs/01-method.md section 6."""
    T = lambda *x: t.tensor(x, dtype=t.float64)
    assert auroc(T(3, 4), T(1, 2)) == 1.0
    assert auroc(T(1, 2), T(3, 4)) == 0.0
    assert auroc(T(2, 2), T(2, 2)) == 0.5
    assert auroc(T(0.1, 0.4, 0.35, 0.8), T(0.2, 0.3)) == 0.75
    inf = float("-inf")
    c = t.tensor([[[1.0, 5.0], [9.0, inf]], [[2.0, inf], [3.0, 4.0]]])
    assert monitor_scores(c, 1).tolist() == [5.0, 2.0]
    assert monitor_scores(c, 2).tolist() == [9.0, 4.0]
    assert monitor_scores(c, 0).tolist() == [1.0, 2.0]
    return True
