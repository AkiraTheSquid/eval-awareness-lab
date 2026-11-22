---
kc: monitor.baselines
title: Baselines and controls for the monitor
new_syntax: [Tensor.float]
concepts: [verbalisation-baseline, cheap-explanations]
supporting: ['monitor.auroc', 'monitor.prompt-score', 'tensor.row-normalization']
previews: []
faded: [1892, 1893]
guided: []
independent: [1894, 1895, 1896, 1897, 1898, 1899]
integrated: [1900, 1901, 1902]
---

## Concept: The verbalisation baseline counts rollouts that say it out loud

The logit monitor is worth having only if it beats the obvious method: read the reasoning and see whether the model SAYS it thinks it is being tested. This is the *verbalisation baseline*. A second model, the *judge*, reads each rollout and answers yes or no. The result is a table of flags with shape `(prompt, rollout)`, True where the judge found the rollout to voice a suspicion of being tested.

The baseline's score for a prompt is the fraction of its first k rollouts that were flagged:

$$v(\text{prompt}) = \frac{\#\{\text{flagged rollouts among the first } k\}}{k}$$

Two properties of this score matter when it is compared with the logit monitor. It can take only k + 1 different values, so with one rollout it is 0 or 1 and most pairs of prompts tie. And models voice the suspicion rarely, so most prompts score exactly 0. The half-credit for ties in the AUROC is what keeps this baseline from being scored unfairly low.

True and False cannot be averaged directly. `flags.float()` turns them into 1.0 and 0.0, and the mean of those is the fraction of True entries.

## Worked example

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

## Faded practice

### q1892
`flags` has shape `(prompt, rollout)` and is True where the judge found that a rollout voices a suspicion of being tested. Return each prompt's verbalisation score from its first `k` rollouts: the fraction of them that were flagged. Shape `(prompt,)`.

```python starter
import torch as t
import einops


def solve(flags, k):
    return einops.reduce(flags[:, :k]._____(), "prompt rollout -> prompt", "_____")
```

```python solution
import torch as t
import einops


def solve(flags, k):
    return einops.reduce(flags[:, :k].float(), "prompt rollout -> prompt", "mean")
```

## Concept: A monitor must beat the cheap explanations

A high AUROC shows that the scores separate the two groups of prompts. It does not show WHY. Evaluation prompts and deployment prompts differ in many ways that have nothing to do with the model noticing a test, and a score can pick up any of them. Two cheap explanations are checked before the result is believed.

The first is that the phrase is simply more likely after some kinds of text. The check is a *control phrase* of the same shape that does not mention testing, such as "The user might be asking". Its score is computed in exactly the same way and subtracted:

$$\text{corrected} = s_{\text{test phrase}} - s_{\text{control phrase}}$$

What is left is how much more the model leans toward "testing" than toward a neutral continuation at the same point.

The second is that the groups differ in something trivial, the classic case being length. Evaluation prompts are often long, written scenarios, and real user messages are often short. The check is to use the trivial quantity itself as the score and compute its AUROC. If the number of tokens in the prompt separates the groups as well as the monitor does, the monitor has not yet shown anything. The comparison can then be repeated inside a band of similar lengths, where length can no longer help.

## Worked example

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

## Faded practice

### q1893
`primary` and `control` each hold one score per prompt: the monitor's score with the test phrase and with the control phrase. Return the control-corrected score of each prompt.

```python starter
import torch as t


def solve(primary, control):
    return primary _____ control
```

```python solution
import torch as t


def solve(primary, control):
    return primary - control
```

## Solo practice

### q1894
`flags` has shape `(prompt, rollout)` and is True where the judge flagged a rollout. Return, for each prompt, whether at least one of its first `k` rollouts was flagged. Boolean tensor of shape `(prompt,)`.

### q1895
`flags_eval` and `flags_dep` have shape `(prompt, rollout)` and hold the judge's flags for evaluation prompts and deployment prompts. Score each prompt by the fraction of its first `k` rollouts that were flagged, and return the AUROC of that score as a Python float. A tie counts as half.

### q1896
`flags` has shape `(prompt, rollout)` and is True where the judge flagged a rollout. Return the number of prompts with NO flag among their first `k` rollouts, as a Python int.

### q1897
`primary` and `control` have shape `(prompt, rollout, pos)` and hold the phrase log-probabilities for the test phrase and for the control phrase, with no padding. Score each prompt with each phrase (the maximum over its rollouts and positions) and return the test-phrase score minus the control-phrase score. Shape `(prompt,)`.

### q1898
`scores` holds one monitor score per prompt and `lens` the number of tokens in each prompt. Return the scores of the prompts whose length is at least `lo` and below `hi`, in their original order.

### q1899
`pos` and `neg` hold a score for the evaluation prompts and for the deployment prompts. The score may separate the groups in either direction. Return how well it separates them, as a Python float: its AUROC (a tie counts as half), or one minus that if the AUROC is below 0.5.

## Integrated practice

### q1900
`flags_eval` and `flags_dep` have shape `(prompt, rollout)` and hold the judge's flags for evaluation and deployment prompts. `ks` is a list of rollout counts. For each count k, score each prompt by the fraction of its first k rollouts that were flagged and compute the AUROC, with a tie counted as half. Return a 1-D tensor with one entry per count, in the order given.

### q1901
`scores` holds one monitor score per prompt, `lens` the number of tokens in each prompt, and `is_eval` is True for evaluation prompts. Keeping only the prompts whose length is at least `lo` and below `hi`, return the AUROC of the scores as a Python float. A tie counts as half.

### q1902
`scores` holds one monitor score per prompt, `lens` the number of tokens in each prompt, and `is_eval` is True for evaluation prompts. Return a tensor of shape `(2,)`: the AUROC of the monitor's scores, then the AUROC obtained when the prompt length itself is used as the score. A tie counts as half.

## Misconceptions

- **The verbalisation score also grows with more rollouts.** It is a fraction, so it can go up or down. Only its resolution improves.
- **Ties do not matter for the baseline.** Most prompts score exactly 0, so most pairs tie. Counting ties as losses would sink it.
- **A high AUROC shows the model noticed the test.** It shows the groups can be told apart. Length or generic phrase likelihood may be doing it.
- **The control phrase is a second monitor.** It is what gets subtracted, to remove how likely any phrase of that shape is.
