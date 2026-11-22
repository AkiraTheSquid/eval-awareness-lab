"""monitor.baselines: the verbalisation baseline, and the cheap explanations a monitor must beat."""
from framework import maker

KC = "monitor.baselines"
A = [("lm-verbalisation-baseline", 1.0)]
AC = [("lm-control-confound", 1.0)]
ACA = [("lm-control-confound", 1.0), ("lm-auroc", 0.5)]
AVA = [("lm-verbalisation-baseline", 1.0), ("lm-auroc", 0.5)]
SPEC = dict(
    id=KC, title="Baselines and controls for the monitor",
    concepts=["verbalisation-baseline", "cheap-explanations"],
    supporting=["monitor.auroc", "monitor.prompt-score", "tensor.row-normalization"],
    new_syntax=["Tensor.float"],
)

AUC = """    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return ((wins + 0.5 * ties) / (len(pos) * len(neg))).item()"""

SEGMENTS = [
    dict(title="The verbalisation baseline counts rollouts that say it out loud", concept="""
The logit monitor is worth having only if it beats the obvious method: read the reasoning and see whether the model SAYS it thinks it is being tested. This is the *verbalisation baseline*. A second model, the *judge*, reads each rollout and answers yes or no. The result is a table of flags with shape `(prompt, rollout)`, True where the judge found the rollout to voice a suspicion of being tested.

The baseline's score for a prompt is the fraction of its first k rollouts that were flagged:

$$v(\\text{prompt}) = \\frac{\\#\\{\\text{flagged rollouts among the first } k\\}}{k}$$

Two properties of this score matter when it is compared with the logit monitor. It can take only k + 1 different values, so with one rollout it is 0 or 1 and most pairs of prompts tie. And models voice the suspicion rarely, so most prompts score exactly 0. The half-credit for ties in the AUROC is what keeps this baseline from being scored unfairly low.

True and False cannot be averaged directly. `flags.float()` turns them into 1.0 and 0.0, and the mean of those is the fraction of True entries.
""", worked="""
Three prompts have four rollouts each. The plan: take the first two rollouts, turn the flags into numbers, and average along the rollout axis.

```python
import torch as t
import einops
flags = t.tensor([[True, False, False, True],
                  [False, False, False, False],
                  [True, True, False, False]])
k = 2
first_k = flags[:, :k].float()
pattern = "prompt rollout -> prompt"
v = einops.reduce(first_k, pattern, "mean")
print(v)
# Hidden checks
assert v.tolist() == [0.5, 0.0, 1.0]
```

The first prompt has one flag among its first two rollouts, so it scores 0.5. Its second flag is in the fourth rollout and is not seen at k = 2. With all four rollouts the picture changes:

```python
every = einops.reduce(flags.float(), pattern, "mean")
print(every)
# Hidden checks
assert every.tolist() == [0.5, 0.0, 0.5]
```

The third prompt fell from 1.0 to 0.5. Unlike the maximum used by the logit monitor, a fraction can go down as rollouts are added. The second prompt scores 0 at every k, like most prompts do.
"""),
    dict(title="A monitor must beat the cheap explanations", concept="""
A high AUROC shows that the scores separate the two groups of prompts. It does not show WHY. Evaluation prompts and deployment prompts differ in many ways that have nothing to do with the model noticing a test, and a score can pick up any of them. Two cheap explanations are checked before the result is believed.

The first is that the phrase is simply more likely after some kinds of text. The check is a *control phrase* of the same shape that does not mention testing, such as "The user might be asking". Its score is computed in exactly the same way and subtracted:

$$\\text{corrected} = s_{\\text{test phrase}} - s_{\\text{control phrase}}$$

What is left is how much more the model leans toward "testing" than toward a neutral continuation at the same point.

The second is that the groups differ in something trivial, the classic case being length. Evaluation prompts are often long, written scenarios, and real user messages are often short. The check is to use the trivial quantity itself as the score and compute its AUROC. If the number of tokens in the prompt separates the groups as well as the monitor does, the monitor has not yet shown anything. The comparison can then be repeated inside a band of similar lengths, where length can no longer help.
""", worked="""
Four prompts were scored with the test phrase and with the control phrase. The plan: subtract, and see which prompts stand out once the generic likelihood is removed.

```python
import torch as t
primary = t.tensor([-4.0, -9.0, -5.0, -12.0])
control = t.tensor([-3.0, -9.5, -8.0, -12.5])
corrected = primary - control
print(corrected)
# Hidden checks
assert corrected.tolist() == [-1.0, 0.5, 3.0, 0.5]
```

The first prompt had the highest raw score, −4, but its control score is higher still, so it ends up lowest: phrases of this shape are just likely there. The third prompt is the one where "testing" in particular fits, 3 above the control.

Now the length check. Suppose the first two prompts are evaluations and their lengths in tokens are known:

```python
lens_eval = t.tensor([900.0, 400.0])
lens_dep = t.tensor([30.0, 500.0])
wins = (lens_eval[:, None] > lens_dep[None, :]).sum()
ties = (lens_eval[:, None] == lens_dep[None, :]).sum()
print((wins + 0.5 * ties) / 4)
# Hidden checks
assert abs(((wins + 0.5 * ties) / 4).item() - 0.75) < 1e-6
```

Length alone gives an AUROC of 0.75 here, without running the model at all. A monitor that reports 0.75 on these prompts has matched a ruler.
"""),
]

MISCONCEPTIONS = [
    ("The verbalisation score also grows with more rollouts.", "It is a fraction, so it can go up or down. Only its resolution improves."),
    ("Ties do not matter for the baseline.", "Most prompts score exactly 0, so most pairs tie. Counting ties as losses would sink it."),
    ("A high AUROC shows the model noticed the test.", "It shows the groups can be told apart. Length or generic phrase likelihood may be doing it."),
    ("The control phrase is a second monitor.", "It is what gets subtracted, to remove how likely any phrase of that shape is."),
]

D = maker("bl", KC)
FL = "flags = t.tensor([[True, False, False, True], [False, False, False, False], [True, True, False, False]])"
FL2 = "flags = t.tensor([[False, True], [True, True], [False, False], [True, False]])"
PT = "primary = t.tensor([[[-4., -6.], [-5., -9.]], [[-9., -9.], [-8., -3.]]])\ncontrol = t.tensor([[[-3., -7.], [-6., -9.]], [[-9.5, -9.], [-8., -8.]]])"
PT2 = "primary = t.tensor([[[-1.], [-2.]], [[-5.], [-5.]], [[-7.], [-3.]]])\ncontrol = t.tensor([[[-2.], [-2.]], [[-4.], [-6.]], [[-3.], [-7.]]])"
SL = "scores = t.tensor([3., 0., 1., 2., 2., .5])\nlens = t.tensor([900, 30, 400, 500, 450, 20])"
DRILLS = [
    D("f1", "faded", 1,
      "`flags` has shape `(prompt, rollout)` and is True where the judge found that a rollout voices a suspicion of being tested. Return each prompt's verbalisation score from its first `k` rollouts: the fraction of them that were flagged. Shape `(prompt,)`.",
      "flags, k", "Fraction of the first k rollouts flagged.",
      '    return einops.reduce(flags[:, :k].float(), "prompt rollout -> prompt", "mean")',
      [(FL + "\nk = 2", "solve(flags, k).tolist()"), (FL + "\nk = 4", "solve(flags, k).tolist()"), (FL2 + "\nk = 1", "solve(flags, k).tolist()"), (FL2 + "\nk = 2", "solve(flags, k).tolist()")],
      ('    return einops.reduce(flags[:, :k].float(), "prompt rollout -> prompt", "max")', FL + "\nk = 2",
       "This says whether ANY of the rollouts was flagged. A prompt flagged once in two rollouts and one flagged twice then look the same."),
      25, A, scaffold='    return einops.reduce(flags[:, :k]._____(), "prompt rollout -> prompt", "_____")'),
    D("f2", "faded", 2,
      "`primary` and `control` each hold one score per prompt: the monitor's score with the test phrase and with the control phrase. Return the control-corrected score of each prompt.",
      "primary, control", "Test-phrase score with the control-phrase score removed.",
      "    return primary - control",
      [("primary = t.tensor([-2., -6., -3.])\ncontrol = t.tensor([-5., -6.5, -1.])", "solve(primary, control).tolist()"),
       ("primary = t.tensor([-1., -1.])\ncontrol = t.tensor([-1., -2.])", "solve(primary, control).tolist()"),
       ("primary = t.tensor([-6.])\ncontrol = t.tensor([-2.])", "solve(primary, control).tolist()"),
       ("primary = t.tensor([-3., -5., -7.])\ncontrol = t.tensor([-7., -5., -3.])", "solve(primary, control).tolist()")],
      ("    return control - primary", "primary = t.tensor([-2., -6., -3.])\ncontrol = t.tensor([-5., -6.5, -1.])",
       "The sign is reversed, so a prompt where the test phrase fits far better than the control now gets the lowest score."),
      15, AC, scaffold="    return primary _____ control"),
    # ---- solo ------------------------------------------------------------------------------------------------
    D("s1", "solo", 0,
      "`flags` has shape `(prompt, rollout)` and is True where the judge flagged a rollout. Return, for each prompt, whether at least one of its first `k` rollouts was flagged. Boolean tensor of shape `(prompt,)`.",
      "flags, k", "Whether any of the first k rollouts was flagged.",
      "    return flags[:, :k].any(dim=1)",
      [(FL + "\nk = 1", "solve(flags, k).tolist()"), (FL + "\nk = 4", "solve(flags, k).tolist()"), (FL2 + "\nk = 1", "solve(flags, k).tolist()"), (FL2 + "\nk = 2", "solve(flags, k).tolist()")],
      ("    return flags[:, :k].all(dim=1)", FL + "\nk = 4",
       "This asks for every rollout to be flagged. Models voice the suspicion rarely, so almost no prompt would ever pass."),
      30, A, show=1),
    D("s2", "solo", 0,
      "`flags_eval` and `flags_dep` have shape `(prompt, rollout)` and hold the judge's flags for evaluation prompts and deployment prompts. Score each prompt by the fraction of its first `k` rollouts that were flagged, and return the AUROC of that score as a Python float. A tie counts as half.",
      "flags_eval, flags_dep, k", "AUROC of the verbalisation baseline at k rollouts.",
      '    pos = einops.reduce(flags_eval[:, :k].float(), "prompt rollout -> prompt", "mean")\n    neg = einops.reduce(flags_dep[:, :k].float(), "prompt rollout -> prompt", "mean")\n' + AUC,
      [("flags_eval = t.tensor([[True, False], [False, False], [True, True]])\nflags_dep = t.tensor([[False, False], [False, True]])\nk = 2", "solve(flags_eval, flags_dep, k)"),
       ("flags_eval = t.tensor([[True, False], [False, False], [True, True]])\nflags_dep = t.tensor([[False, False], [False, True]])\nk = 1", "solve(flags_eval, flags_dep, k)"),
       ("flags_eval = t.tensor([[False], [False]])\nflags_dep = t.tensor([[False], [False], [False]])\nk = 1", "solve(flags_eval, flags_dep, k)"),
       ("flags_eval = t.tensor([[True, True, False]])\nflags_dep = t.tensor([[True, False, False], [True, True, True]])\nk = 3", "solve(flags_eval, flags_dep, k)")],
      ('    pos = einops.reduce(flags_eval[:, :k].float(), "prompt rollout -> prompt", "mean")\n    neg = einops.reduce(flags_dep[:, :k].float(), "prompt rollout -> prompt", "mean")\n' + AUC.replace("0.5 * ties", "0 * ties"),
       "flags_eval = t.tensor([[True, False], [False, False], [True, True]])\nflags_dep = t.tensor([[False, False], [False, True]])\nk = 2",
       "Tied pairs were counted as losses. This score has only a few possible values and most prompts sit at 0, so most pairs tie and the baseline is marked far too low."),
      60, AVA),
    D("s3", "solo", 0,
      "`flags` has shape `(prompt, rollout)` and is True where the judge flagged a rollout. Return the number of prompts with NO flag among their first `k` rollouts, as a Python int.",
      "flags, k", "How many prompts score exactly zero on the baseline.",
      "    return (flags[:, :k].sum(dim=1) == 0).sum().item()",
      [(FL + "\nk = 2", "solve(flags, k)"), (FL + "\nk = 1", "solve(flags, k)"), (FL2 + "\nk = 1", "solve(flags, k)"), (FL2 + "\nk = 2", "solve(flags, k)")],
      ("    return (~flags[:, :k]).sum().item()", FL + "\nk = 2",
       "This counted unflagged ROLLOUTS. The question is about prompts, and a prompt counts only when all of its rollouts are unflagged."),
      40, A),
    D("s4", "solo", 0,
      "`primary` and `control` have shape `(prompt, rollout, pos)` and hold the phrase log-probabilities for the test phrase and for the control phrase, with no padding. Score each prompt with each phrase (the maximum over its rollouts and positions) and return the test-phrase score minus the control-phrase score. Shape `(prompt,)`.",
      "primary, control", "Control-corrected prompt scores from the two tables.",
      '    p = einops.reduce(primary, "prompt rollout pos -> prompt", "max")\n    c = einops.reduce(control, "prompt rollout pos -> prompt", "max")\n    return p - c',
      [(PT, "solve(primary, control).tolist()"), (PT2, "solve(primary, control).tolist()"),
       ("primary = t.tensor([[[-2., -8.]]])\ncontrol = t.tensor([[[-9., -3.]]])", "solve(primary, control).tolist()"),
       ("primary = t.tensor([[[-1.]], [[-2.]]])\ncontrol = t.tensor([[[-1.]], [[-1.]]])", "solve(primary, control).tolist()")],
      ('    return einops.reduce(primary - control, "prompt rollout pos -> prompt", "max")', PT,
       "The subtraction was done at every position first, and the best gap was kept. That is a different quantity: it rewards one position where the control happens to be very unlikely, even if the test phrase is unlikely there too."),
      50, AC),
    D("s5", "solo", 0,
      "`scores` holds one monitor score per prompt and `lens` the number of tokens in each prompt. Return the scores of the prompts whose length is at least `lo` and below `hi`, in their original order.",
      "scores, lens, lo, hi", "Scores of prompts inside a length band.",
      "    return scores[(lens >= lo) & (lens < hi)]",
      [(SL + "\nlo = 400\nhi = 600", "solve(scores, lens, lo, hi).tolist()"), (SL + "\nlo = 0\nhi = 100", "solve(scores, lens, lo, hi).tolist()"),
       (SL + "\nlo = 450\nhi = 901", "solve(scores, lens, lo, hi).tolist()"), (SL + "\nlo = 20\nhi = 30", "solve(scores, lens, lo, hi).tolist()")],
      ("    return scores[(lens >= lo) | (lens < hi)]", SL + "\nlo = 400\nhi = 600",
       "Every prompt satisfies one of the two conditions, so all of them came back. A band needs both conditions to hold at once."),
      35, AC),
    D("s6", "solo", 0,
      "`pos` and `neg` hold a score for the evaluation prompts and for the deployment prompts. The score may separate the groups in either direction. Return how well it separates them, as a Python float: its AUROC (a tie counts as half), or one minus that if the AUROC is below 0.5.",
      "pos, neg", "Separation of a score regardless of direction.",
      "    wins = (pos[:, None] > neg[None, :]).sum()\n    ties = (pos[:, None] == neg[None, :]).sum()\n    auc = ((wins + 0.5 * ties) / (len(pos) * len(neg))).item()\n    return max(auc, 1 - auc)",
      [("pos = t.tensor([30., 40.])\nneg = t.tensor([900., 35., 500.])", "solve(pos, neg)"), ("pos = t.tensor([900., 400.])\nneg = t.tensor([30., 500.])", "solve(pos, neg)"),
       ("pos = t.tensor([1., 1.])\nneg = t.tensor([1.])", "solve(pos, neg)"), ("pos = t.tensor([1., 2., 3.])\nneg = t.tensor([4., 5.])", "solve(pos, neg)")],
      (AUC, "pos = t.tensor([30., 40.])\nneg = t.tensor([900., 35., 500.])",
       "The plain AUROC came out below 0.5, which looks like a useless score. It means the groups are separated the other way round, and a trivial quantity that does that is just as much a rival explanation."),
      50, ACA),
    # ---- integrated ------------------------------------------------------------------------------------------
    D("i1", "integrated", 0,
      "`flags_eval` and `flags_dep` have shape `(prompt, rollout)` and hold the judge's flags for evaluation and deployment prompts. `ks` is a list of rollout counts. For each count k, score each prompt by the fraction of its first k rollouts that were flagged and compute the AUROC, with a tie counted as half. Return a 1-D tensor with one entry per count, in the order given.",
      "flags_eval, flags_dep, ks", "Verbalisation-baseline AUROC at each rollout count.",
      '    out = t.zeros(len(ks))\n    i = 0\n    for k in ks:\n        pos = einops.reduce(flags_eval[:, :k].float(), "prompt rollout -> prompt", "mean")\n        neg = einops.reduce(flags_dep[:, :k].float(), "prompt rollout -> prompt", "mean")\n        wins = (pos[:, None] > neg[None, :]).sum()\n        ties = (pos[:, None] == neg[None, :]).sum()\n        out[i] = (wins + 0.5 * ties) / (len(pos) * len(neg))\n        i += 1\n    return out',
      [("flags_eval = t.tensor([[True, False, True], [False, False, True], [False, True, True]])\nflags_dep = t.tensor([[False, False, False], [False, True, False]])\nks = [1, 2, 3]", "solve(flags_eval, flags_dep, ks).tolist()"),
       ("flags_eval = t.tensor([[True, False, True], [False, False, True], [False, True, True]])\nflags_dep = t.tensor([[False, False, False], [False, True, False]])\nks = [3]", "solve(flags_eval, flags_dep, ks).tolist()"),
       ("flags_eval = t.tensor([[False, True]])\nflags_dep = t.tensor([[True, False], [False, False]])\nks = [2, 1]", "solve(flags_eval, flags_dep, ks).tolist()"),
       ("flags_eval = t.tensor([[True, True], [True, False]])\nflags_dep = t.tensor([[False, False]])\nks = [1, 2]", "solve(flags_eval, flags_dep, ks).tolist()")],
      ('    out = t.zeros(len(ks))\n    i = 0\n    for k in ks:\n        pos = einops.reduce(flags_eval[:, k - 1:k].float(), "prompt rollout -> prompt", "mean")\n        neg = einops.reduce(flags_dep[:, k - 1:k].float(), "prompt rollout -> prompt", "mean")\n        wins = (pos[:, None] > neg[None, :]).sum()\n        ties = (pos[:, None] == neg[None, :]).sum()\n        out[i] = (wins + 0.5 * ties) / (len(pos) * len(neg))\n        i += 1\n    return out',
       "flags_eval = t.tensor([[True, False, True], [False, False, True], [False, True, True]])\nflags_dep = t.tensor([[False, False, False], [False, True, False]])\nks = [1, 2, 3]",
       "Each count used the k-th rollout alone. The baseline at k pools the first k rollouts, which is what lets it tell one flag from several."),
      85, AVA),
    D("i2", "integrated", 0,
      "`scores` holds one monitor score per prompt, `lens` the number of tokens in each prompt, and `is_eval` is True for evaluation prompts. Keeping only the prompts whose length is at least `lo` and below `hi`, return the AUROC of the scores as a Python float. A tie counts as half.",
      "scores, lens, is_eval, lo, hi", "AUROC among prompts of similar length.",
      "    band = (lens >= lo) & (lens < hi)\n    pos = scores[band & is_eval]\n    neg = scores[band & ~is_eval]\n" + AUC,
      [(SL + "\nis_eval = t.tensor([True, False, True, False, True, False])\nlo = 400\nhi = 1000", "solve(scores, lens, is_eval, lo, hi)"),
       (SL + "\nis_eval = t.tensor([True, False, True, False, True, False])\nlo = 0\nhi = 1000", "solve(scores, lens, is_eval, lo, hi)"),
       (SL + "\nis_eval = t.tensor([False, True, True, False, False, True])\nlo = 0\nhi = 460", "solve(scores, lens, is_eval, lo, hi)"),
       (SL + "\nis_eval = t.tensor([True, True, False, False, True, False])\nlo = 30\nhi = 901", "solve(scores, lens, is_eval, lo, hi)")],
      ("    pos = scores[is_eval]\n    neg = scores[~is_eval]\n" + AUC, SL + "\nis_eval = t.tensor([True, False, True, False, True, False])\nlo = 400\nhi = 1000",
       "Every prompt was used, so short deployment prompts were compared with long evaluation prompts. Inside the band that easy comparison is gone and the result changes."),
      80, ACA),
    D("i3", "integrated", 0,
      "`scores` holds one monitor score per prompt, `lens` the number of tokens in each prompt, and `is_eval` is True for evaluation prompts. Return a tensor of shape `(2,)`: the AUROC of the monitor's scores, then the AUROC obtained when the prompt length itself is used as the score. A tie counts as half.",
      "scores, lens, is_eval", "Monitor AUROC beside the length-only AUROC.",
      "    out = t.zeros(2)\n    i = 0\n    for s in [scores, lens]:\n        pos = s[is_eval]\n        neg = s[~is_eval]\n        wins = (pos[:, None] > neg[None, :]).sum()\n        ties = (pos[:, None] == neg[None, :]).sum()\n        out[i] = (wins + 0.5 * ties) / (len(pos) * len(neg))\n        i += 1\n    return out",
      [(SL + "\nis_eval = t.tensor([True, False, True, False, True, False])", "solve(scores, lens, is_eval).tolist()"),
       (SL + "\nis_eval = t.tensor([False, True, True, False, False, True])", "solve(scores, lens, is_eval).tolist()"),
       ("scores = t.tensor([1., 2., 3., 4.])\nlens = t.tensor([10, 10, 10, 10])\nis_eval = t.tensor([False, False, True, True])", "solve(scores, lens, is_eval).tolist()"),
       ("scores = t.tensor([2., 2., 2.])\nlens = t.tensor([5, 9, 7])\nis_eval = t.tensor([False, True, False])", "solve(scores, lens, is_eval).tolist()")],
      ("    out = t.zeros(2)\n    i = 0\n    for s in [scores, lens]:\n        pos = s[is_eval]\n        neg = s[~is_eval]\n        wins = (pos[:, None] > neg[None, :]).sum()\n        out[i] = wins / (len(pos) * len(neg))\n        i += 1\n    return out",
       "scores = t.tensor([1., 2., 3., 4.])\nlens = t.tensor([10, 10, 10, 10])\nis_eval = t.tensor([False, False, True, True])",
       "Ties were left out. All the prompts here have the same length, so length carries no information and should come out at 0.5, yet it was reported as 0."),
      85, ACA, show=2),
]
