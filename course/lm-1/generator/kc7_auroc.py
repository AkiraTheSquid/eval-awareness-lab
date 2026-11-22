"""monitor.auroc: how well the scores separate evaluation prompts from deployment prompts."""
from framework import maker

KC = "monitor.auroc"
A = [("lm-auroc", 1.0)]
AB = [("lm-bootstrap-ci", 1.0), ("lm-auroc", 0.5)]
AP = [("lm-auroc", 1.0), ("lm-prompt-score", 0.5)]
SPEC = dict(
    id=KC, title="AUROC with a bootstrap interval",
    concepts=["auroc-as-pair-wins", "bootstrap-interval"],
    supporting=["monitor.prompt-score", "torch.broadcasting-rules", "torch.boolean-masking", "torch.random-seeding"],
    new_syntax=["torch.quantile"],
)

AUC = """    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return ((wins + 0.5 * ties) / (len(pos) * len(neg))).item()"""
BOOT = """    P = pos[idx_pos]
    N = {n}
    wins = (P[:, :, None] > N[:, None, :]).sum(dim=2).sum(dim=1)
    ties = (P[:, :, None] == N[:, None, :]).sum(dim=2).sum(dim=1)
    aucs = (wins + 0.5 * ties) / (len(pos) * len(neg))
"""

SEGMENTS = [
    dict(title="AUROC is the share of pairs the monitor ranks correctly", concept="""
The monitor gives every prompt a score. Some prompts come from evaluations (tests written to probe the model) and some from deployment (real conversations). The monitor is useful if evaluation prompts tend to score higher.

A threshold would turn scores into yes/no answers, but every threshold gives a different accuracy. The *AUROC* (area under the receiver operating characteristic curve) avoids choosing one. It has a direct meaning: pick one evaluation prompt and one deployment prompt at random, and the AUROC is the probability that the evaluation prompt has the higher score. A tie counts as half a win.

$$\\text{AUROC} = \\frac{\\#\\{(e, d) : s_e > s_d\\} + \\tfrac{1}{2}\\,\\#\\{(e, d) : s_e = s_d\\}}{n_e \\cdot n_d}$$

A value of 1 means every evaluation prompt outscores every deployment prompt. A value of 0.5 means the scores carry no information, the same as guessing. A value below 0.5 means the monitor ranks them backwards.

Every evaluation score has to be compared with every deployment score. Giving the first a trailing axis and the second a leading axis makes the comparison a table with one entry per pair.
""", worked="""
Three evaluation prompts scored 3, 1 and 2, and two deployment prompts scored 0 and 2. The plan: build the table of comparisons, count the wins and the ties, and divide by the number of pairs. There are six pairs, so the counts can be checked by hand.

```python
import torch as t
pos = t.tensor([3.0, 1.0, 2.0])
neg = t.tensor([0.0, 2.0])
greater = pos[:, None] > neg[None, :]
print(greater)
# Hidden checks
assert greater.tolist() == [[True, True], [True, False], [True, False]]
```

Row 0 is the score 3, which beats both deployment scores. Row 1 is the score 1, which beats 0 only. Row 2 is the score 2, which beats 0 and ties with the other 2, and a tie is not a win. That makes four wins and one tie:

```python
wins = greater.sum()
ties = (pos[:, None] == neg[None, :]).sum()
auroc = (wins + 0.5 * ties) / (len(pos) * len(neg))
print(wins, ties, auroc)
# Hidden checks
assert wins.item() == 4 and ties.item() == 1
assert abs(auroc.item() - 0.75) < 1e-6
```

The AUROC is 4.5 out of 6, which is 0.75. Leaving out the tie would give 0.67 and treat a tied pair as a loss.
"""),
    dict(title="A bootstrap interval says how far the AUROC could move", concept="""
An AUROC from a few hundred prompts is an estimate. A different sample of prompts would have given a somewhat different number, and a reported difference between two monitors means little unless it is larger than that wobble.

The *bootstrap* measures the wobble using only the data in hand. Draw a new set of evaluation scores by picking from the real ones at random WITH replacement, the same number as before, so some scores appear twice and some not at all. Do the same for the deployment scores. Compute the AUROC of this resampled data. Repeat many times, for example 1000.

The spread of those 1000 values stands in for the spread across fresh samples. The *95% confidence interval* runs from the value below which 2.5% of them fall to the value below which 97.5% fall. Those cut points are *quantiles*, and `t.quantile(values, q)` returns them: `q` is a fraction between 0 and 1, or a tensor of several fractions.

Random index tables do the resampling for all repetitions at once. `t.randint(0, n, (B, n), generator=g)` gives B rows of n indices, each between 0 and n − 1, and indexing the scores with that table gives B resampled sets.
""", worked="""
Use the same five scores as before and 200 resamples, with a seeded generator so the run can be repeated. The plan: draw index tables for both groups, index the scores with them, and compute one AUROC per row.

```python
import torch as t
pos = t.tensor([3.0, 1.0, 2.0])
neg = t.tensor([0.0, 2.0])
g = t.Generator().manual_seed(0)
idx_pos = t.randint(0, 3, (200, 3), generator=g)
idx_neg = t.randint(0, 2, (200, 2), generator=g)
P = pos[idx_pos]
N = neg[idx_neg]
print(P.shape, N.shape)
# Hidden checks
assert P.shape == (200, 3) and N.shape == (200, 2)
```

Each row of `P` is one resampled set of evaluation scores. The pair comparison gains a leading axis for the resample, and the counts are taken over the two pair axes:

```python
greater = P[:, :, None] > N[:, None, :]
equal = P[:, :, None] == N[:, None, :]
wins = greater.sum(dim=2).sum(dim=1)
ties = equal.sum(dim=2).sum(dim=1)
aucs = (wins + 0.5 * ties) / 6
low = t.quantile(aucs, 0.025)
high = t.quantile(aucs, 0.975)
print(low, high)
# Hidden checks
assert aucs.shape == (200,)
assert 0.0 <= low.item() <= high.item() <= 1.0
assert high.item() - low.item() > 0.3
```

The interval is very wide. With three and two prompts the AUROC of 0.75 is compatible with almost anything, including 0.5. That is the point of computing it: the estimate alone looked like a result.
"""),
]

MISCONCEPTIONS = [
    ("AUROC is the accuracy at the best threshold.", "It uses no threshold. It is the share of evaluation and deployment pairs ranked the right way."),
    ("A tie is a loss.", "A tie counts as half. Scores that tie often would otherwise be marked far too low."),
    ("An AUROC of 0.5 is half right.", "0.5 is what random scores give. It means no information."),
    ("Resample without replacement.", "That returns the same set every time. Drawing with replacement is what makes the resamples differ."),
]

D = maker("au", KC)
PN = "pos = t.tensor([3., 1., 2.])\nneg = t.tensor([0., 2.])"
PN2 = "pos = t.tensor([5., 5., 1.])\nneg = t.tensor([5., 0., 2., 6.])"
PN3 = "pos = t.tensor([1., 2.])\nneg = t.tensor([3., 4.])"
PN4 = "pos = t.tensor([0., 0., 1., 1.])\nneg = t.tensor([0., 0., 0., 1.])"
IX = "\nidx_pos = t.tensor([[0, 0, 1], [2, 1, 0], [1, 1, 1], [0, 2, 2]])\nidx_neg = t.tensor([[0, 1], [1, 1], [0, 0], [1, 0]])"
IX2 = "\nidx_pos = t.tensor([[0, 1, 2], [2, 2, 2], [0, 0, 1]])\nidx_neg = t.tensor([[0, 1, 2, 3], [3, 3, 0, 0], [1, 2, 1, 2]])"
AK = "aucs = t.tensor([.5, .6, .7, .8, .9, .55, .65, .75, .85, .95, .7])"
AK2 = "aucs = t.tensor([.2, .9, .4, .4, .6, .5, .5, .3])"
DRILLS = [
    D("f1", "faded", 1,
      "`pos` holds the scores of the evaluation prompts and `neg` the scores of the deployment prompts. Return the AUROC as a Python float: the share of pairs in which the evaluation prompt scores higher, with a tie counted as half.",
      "pos, neg", "AUROC of evaluation scores against deployment scores.",
      AUC,
      [(PN, "solve(pos, neg)"), (PN2, "solve(pos, neg)"), (PN3, "solve(pos, neg)"), (PN4, "solve(pos, neg)")],
      (AUC.replace("0.5 * ties", "0 * ties"), PN,
       "The tied pair was dropped from the count, which treats it as a loss. A tie is worth half a win."),
      35, A, scaffold=AUC.replace("0.5", "_____")),
    D("f2", "faded", 2,
      "`aucs` holds the AUROC of each of many resamples. Return the two ends of the 95% confidence interval, as a tensor of shape `(2,)`.",
      "aucs", "95% interval from resampled AUROCs.",
      "    return t.quantile(aucs, t.tensor([0.025, 0.975]))",
      [(AK, "solve(aucs).tolist()"), (AK2, "solve(aucs).tolist()"), ("aucs = t.tensor([.5, .5, .5])", "solve(aucs).tolist()"), ("aucs = t.arange(101.) / 100", "solve(aucs).tolist()")],
      ("    return t.quantile(aucs, t.tensor([0.05, 0.95]))", AK,
       "These cut points leave 5% outside on each side, which makes a 90% interval. A 95% interval leaves 2.5% on each side."),
      25, AB, scaffold="    return t._____(aucs, t.tensor([_____, 0.975]))"),
    # ---- solo ------------------------------------------------------------------------------------------------
    D("s1", "solo", 0,
      "`pos` holds the scores of the evaluation prompts and `neg` the scores of the deployment prompts. Return the number of pairs (one of each) in which the evaluation prompt scores strictly higher, as a Python int.",
      "pos, neg", "Number of pairs the evaluation prompt wins.",
      "    return (pos[:, None] > neg[None, :]).sum().item()",
      [(PN, "solve(pos, neg)"), (PN2, "solve(pos, neg)"), (PN3, "solve(pos, neg)"), (PN4, "solve(pos, neg)")],
      ("    return (pos[:, None] >= neg[None, :]).sum().item()", PN,
       "Tied pairs were counted as wins. A tie says the monitor could not tell the two prompts apart."),
      35, A),
    D("s2", "solo", 0,
      "`pos` holds the scores of the evaluation prompts and `neg` the scores of the deployment prompts. For each evaluation prompt, return the fraction of deployment prompts it scores strictly higher than. Shape like `pos`.",
      "pos, neg", "Share of deployment prompts each evaluation prompt beats.",
      "    return (pos[:, None] > neg[None, :]).sum(dim=1) / len(neg)",
      [(PN, "solve(pos, neg).tolist()"), (PN2, "solve(pos, neg).tolist()"), (PN3, "solve(pos, neg).tolist()"), (PN4, "solve(pos, neg).tolist()")],
      ("    return (pos[:, None] > neg[None, :]).sum(dim=0) / len(neg)", PN,
       "The count ran down the columns, which belong to the deployment prompts. This gives one number per deployment prompt: how many evaluation prompts beat it."),
      40, A),
    D("s3", "solo", 0,
      "`scores` holds one score per prompt and `is_eval` is a boolean tensor of the same shape, True for evaluation prompts and False for deployment prompts. Return the AUROC as a Python float, with a tie counted as half.",
      "scores, is_eval", "AUROC from scores and labels.",
      "    pos = scores[is_eval]\n    neg = scores[~is_eval]\n" + AUC,
      [("scores = t.tensor([3., 0., 1., 2., 2.])\nis_eval = t.tensor([True, False, True, False, True])", "solve(scores, is_eval)"),
       ("scores = t.tensor([1., 2., 3., 4.])\nis_eval = t.tensor([False, False, True, True])", "solve(scores, is_eval)"),
       ("scores = t.tensor([1., 2., 3., 4.])\nis_eval = t.tensor([True, False, True, False])", "solve(scores, is_eval)"),
       ("scores = t.tensor([5., 5., 5.])\nis_eval = t.tensor([True, False, False])", "solve(scores, is_eval)")],
      ("    pos = scores[is_eval]\n    neg = scores\n" + AUC, "scores = t.tensor([3., 0., 1., 2., 2.])\nis_eval = t.tensor([True, False, True, False, True])",
       "The evaluation prompts were compared against every prompt, themselves included. Each one ties with itself, which drags the result toward 0.5."),
      50, A),
    D("s4", "solo", 0,
      "Return a table of resampling indices of shape `(B, n)`: every entry is a whole number from 0 to n − 1 inclusive, drawn uniformly with the generator `g`.",
      "n, B, g", "Bootstrap index table.",
      "    return t.randint(0, n, (B, n), generator=g)",
      [("n = 3\nB = 4\ng = t.Generator().manual_seed(0)", "solve(n, B, g).tolist()"), ("n = 2\nB = 6\ng = t.Generator().manual_seed(1)", "solve(n, B, g).tolist()"),
       ("n = 5\nB = 2\ng = t.Generator().manual_seed(2)", "solve(n, B, g).tolist()"), ("n = 4\nB = 3\ng = t.Generator().manual_seed(3)", "solve(n, B, g).tolist()")],
      ("    return t.randint(0, n - 1, (B, n), generator=g)", "n = 3\nB = 4\ng = t.Generator().manual_seed(0)",
       "The upper end of the range is never drawn, so this can never pick the last prompt. That prompt is silently dropped from every resample."),
      35, AB),
    D("s5", "solo", 0,
      "`pos` and `neg` hold the evaluation and deployment scores. `idx_pos`, shape `(B, n)`, and `idx_neg`, shape `(B, m)`, hold resampling indices into them. Return the AUROC of each of the B resamples, shape `(B,)`, with a tie counted as half.",
      "pos, neg, idx_pos, idx_neg", "AUROC of every bootstrap resample.",
      BOOT.format(n="neg[idx_neg]") + "    return aucs",
      [(PN + IX, "solve(pos, neg, idx_pos, idx_neg).tolist()"), (PN2 + IX2, "solve(pos, neg, idx_pos, idx_neg).tolist()"),
       (PN + "\nidx_pos = t.tensor([[0, 1, 2]])\nidx_neg = t.tensor([[0, 1]])", "solve(pos, neg, idx_pos, idx_neg).tolist()"),
       (PN + "\nidx_pos = t.tensor([[1, 1, 1], [0, 0, 0]])\nidx_neg = t.tensor([[1, 1], [1, 1]])", "solve(pos, neg, idx_pos, idx_neg).tolist()")],
      (BOOT.format(n="neg[None, :]") + "    return aucs", PN + IX,
       "Only the evaluation prompts were resampled. The deployment prompts are a sample too, and leaving them fixed hides half of the uncertainty."),
      60, AB),
    D("s6", "solo", 0,
      "`aucs` holds the AUROC of each of many resamples. Return True if the 95% confidence interval lies entirely above 0.5, else False.",
      "aucs", "Whether the monitor is reliably better than chance.",
      "    return (t.quantile(aucs, 0.025) > 0.5).item()",
      [(AK, "solve(aucs)"), (AK2, "solve(aucs)"), ("aucs = t.tensor([.6, .7, .8, .9])", "solve(aucs)"), ("aucs = t.tensor([.3, .6, .6, .6, .6, .6, .6, .6])", "solve(aucs)")],
      ("    return (aucs.mean() > 0.5).item()", "aucs = t.tensor([.3, .6, .6, .6, .6, .6, .6, .6])",
       "The average resample is above 0.5, but a noticeable share of resamples is not. The claim needs the low end of the interval to clear 0.5."),
      40, AB, show=3),
    # ---- integrated ------------------------------------------------------------------------------------------
    D("i1", "integrated", 0,
      "`ev` and `dep` have shape `(prompt, rollout, pos)` and hold the phrase log-probabilities for evaluation prompts and deployment prompts. `ks` is a list of rollout counts. For each count k, score every prompt with its first k rollouts and compute the AUROC, with a tie counted as half. Return the AUROCs as a 1-D tensor with one entry per count, in the order given.",
      "ev, dep, ks", "AUROC at each rollout count.",
      "    out = t.zeros(len(ks))\n    i = 0\n    for k in ks:\n        pos = einops.reduce(ev[:, :k], \"prompt rollout pos -> prompt\", \"max\")\n        neg = einops.reduce(dep[:, :k], \"prompt rollout pos -> prompt\", \"max\")\n        wins = (pos[:, None] > neg[None, :]).sum()\n        ties = (pos[:, None] == neg[None, :]).sum()\n        out[i] = (wins + 0.5 * ties) / (len(pos) * len(neg))\n        i += 1\n    return out",
      [("ev = t.tensor([[[-9.], [-3.]], [[-5.], [-9.]]])\ndep = t.tensor([[[-6.], [-2.]], [[-8.], [-8.]], [[-7.], [-4.]]])\nks = [1, 2]", "solve(ev, dep, ks).tolist()"),
       ("ev = t.tensor([[[-9.], [-3.]], [[-5.], [-9.]]])\ndep = t.tensor([[[-6.], [-2.]], [[-8.], [-8.]], [[-7.], [-4.]]])\nks = [2]", "solve(ev, dep, ks).tolist()"),
       ("ev = t.tensor([[[-1., -5.], [-5., -5.]]])\ndep = t.tensor([[[-2., -2.], [-1., -3.]], [[-4., -4.], [-4., -4.]]])\nks = [1, 2]", "solve(ev, dep, ks).tolist()"),
       ("ev = t.tensor([[[-3.], [-3.], [-1.]], [[-2.], [-6.], [-6.]]])\ndep = t.tensor([[[-4.], [-2.], [-2.]]])\nks = [3, 1, 2]", "solve(ev, dep, ks).tolist()")],
      ("    out = t.zeros(len(ks))\n    i = 0\n    for k in ks:\n        pos = einops.reduce(ev[:, :k], \"prompt rollout pos -> prompt\", \"max\")\n        neg = einops.reduce(dep, \"prompt rollout pos -> prompt\", \"max\")\n        wins = (pos[:, None] > neg[None, :]).sum()\n        ties = (pos[:, None] == neg[None, :]).sum()\n        out[i] = (wins + 0.5 * ties) / (len(pos) * len(neg))\n        i += 1\n    return out",
       "ev = t.tensor([[[-9.], [-3.]], [[-5.], [-9.]]])\ndep = t.tensor([[[-6.], [-2.]], [[-8.], [-8.]], [[-7.], [-4.]]])\nks = [1, 2]",
       "The deployment prompts were scored with all their rollouts while the evaluation prompts got only k. More rollouts raise a maximum, so the deployment side was given an unfair lift."),
      85, AP),
    D("i2", "integrated", 0,
      "`pos` and `neg` hold the evaluation and deployment scores. `idx_pos`, shape `(B, n)`, and `idx_neg`, shape `(B, m)`, hold resampling indices into them. Return the two ends of the 95% confidence interval of the AUROC (a tie counts as half), as a tensor of shape `(2,)`.",
      "pos, neg, idx_pos, idx_neg", "Bootstrap 95% interval of the AUROC.",
      BOOT.format(n="neg[idx_neg]") + "    return t.quantile(aucs, t.tensor([0.025, 0.975]))",
      [(PN + IX, "solve(pos, neg, idx_pos, idx_neg).tolist()"), (PN2 + IX2, "solve(pos, neg, idx_pos, idx_neg).tolist()"),
       (PN4 + "\nidx_pos = t.tensor([[0, 1, 2, 3], [2, 2, 3, 3], [0, 0, 0, 1], [3, 1, 3, 1], [2, 0, 2, 0]])\nidx_neg = t.tensor([[0, 1, 2, 3], [3, 3, 3, 0], [1, 2, 0, 0], [3, 2, 1, 0], [3, 3, 1, 1]])", "solve(pos, neg, idx_pos, idx_neg).tolist()"),
       (PN3 + "\nidx_pos = t.tensor([[0, 1], [1, 1]])\nidx_neg = t.tensor([[0, 1], [0, 0]])", "solve(pos, neg, idx_pos, idx_neg).tolist()")],
      (BOOT.format(n="neg[idx_neg]") + "    return t.stack([aucs.min(), aucs.max()])", PN + IX,
       "This is the full range of the resamples, which keeps growing as more resamples are drawn. A 95% interval cuts off the most extreme 2.5% at each end."),
      85, AB),
    D("i3", "integrated", 0,
      "Two monitors scored the same prompts. `pos_a` and `neg_a` hold the first monitor's scores for the evaluation and deployment prompts, `pos_b` and `neg_b` the second monitor's. Return the first monitor's AUROC minus the second's, as a Python float. A tie counts as half.",
      "pos_a, neg_a, pos_b, neg_b", "AUROC difference between two monitors.",
      "    wa = (pos_a[:, None] > neg_a[None, :]).sum() + 0.5 * (pos_a[:, None] == neg_a[None, :]).sum()\n    wb = (pos_b[:, None] > neg_b[None, :]).sum() + 0.5 * (pos_b[:, None] == neg_b[None, :]).sum()\n    return ((wa - wb) / (len(pos_a) * len(neg_a))).item()",
      [("pos_a = t.tensor([3., 1., 2.])\nneg_a = t.tensor([0., 2.])\npos_b = t.tensor([1., 0., 0.])\nneg_b = t.tensor([0., 0.])", "solve(pos_a, neg_a, pos_b, neg_b)"),
       ("pos_a = t.tensor([1., 2.])\nneg_a = t.tensor([3., 4.])\npos_b = t.tensor([1., 1.])\nneg_b = t.tensor([1., 0.])", "solve(pos_a, neg_a, pos_b, neg_b)"),
       ("pos_a = t.tensor([5., 6.])\nneg_a = t.tensor([1., 2.])\npos_b = t.tensor([0., 0.])\nneg_b = t.tensor([0., 0.])", "solve(pos_a, neg_a, pos_b, neg_b)"),
       ("pos_a = t.tensor([2., 2., 2.])\nneg_a = t.tensor([2.])\npos_b = t.tensor([1., 0., 1.])\nneg_b = t.tensor([0.])", "solve(pos_a, neg_a, pos_b, neg_b)")],
      ("    wa = (pos_a[:, None] > neg_a[None, :]).sum()\n    wb = (pos_b[:, None] > neg_b[None, :]).sum()\n    return ((wa - wb) / (len(pos_a) * len(neg_a))).item()",
       "pos_a = t.tensor([3., 1., 2.])\nneg_a = t.tensor([0., 2.])\npos_b = t.tensor([1., 0., 0.])\nneg_b = t.tensor([0., 0.])",
       "Ties were counted as losses. The second monitor's scores are mostly identical, so it ties in most pairs and was marked far below what it deserves."),
      80, A),
]
