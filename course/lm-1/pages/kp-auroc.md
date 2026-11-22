---
kc: monitor.auroc
title: AUROC with a bootstrap interval
new_syntax: [torch.quantile]
concepts: [auroc-as-pair-wins, bootstrap-interval]
supporting: ['monitor.prompt-score', 'torch.broadcasting-rules', 'torch.boolean-masking', 'torch.random-seeding']
previews: []
faded: [1881, 1882]
guided: []
independent: [1883, 1884, 1885, 1886, 1887, 1888]
integrated: [1889, 1890, 1891]
---

## Concept: AUROC is the share of pairs the monitor ranks correctly

The monitor gives every prompt a score. Some prompts come from evaluations (tests written to probe the model) and some from deployment (real conversations). The monitor is useful if evaluation prompts tend to score higher.

A threshold would turn scores into yes/no answers, but every threshold gives a different accuracy. The *AUROC* (area under the receiver operating characteristic curve) avoids choosing one. It has a direct meaning: pick one evaluation prompt and one deployment prompt at random, and the AUROC is the probability that the evaluation prompt has the higher score. A tie counts as half a win.

$$\text{AUROC} = \frac{\#\{(e, d) : s_e > s_d\} + \tfrac{1}{2}\,\#\{(e, d) : s_e = s_d\}}{n_e \cdot n_d}$$

A value of 1 means every evaluation prompt outscores every deployment prompt. A value of 0.5 means the scores carry no information, the same as guessing. A value below 0.5 means the monitor ranks them backwards.

Every evaluation score has to be compared with every deployment score. Giving the first a trailing axis and the second a leading axis makes the comparison a table with one entry per pair.

## Worked example

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

## Faded practice

### q1881
`pos` holds the scores of the evaluation prompts and `neg` the scores of the deployment prompts. Return the AUROC as a Python float: the share of pairs in which the evaluation prompt scores higher, with a tie counted as half.

```python starter
import torch as t


def solve(pos, neg):
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return ((wins + _____ * ties) / (len(pos) * len(neg))).item()
```

```python solution
import torch as t


def solve(pos, neg):
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return ((wins + 0.5 * ties) / (len(pos) * len(neg))).item()
```

## Concept: A bootstrap interval says how far the AUROC could move

An AUROC from a few hundred prompts is an estimate. A different sample of prompts would have given a somewhat different number, and a reported difference between two monitors means little unless it is larger than that wobble.

The *bootstrap* measures the wobble using only the data in hand. Draw a new set of evaluation scores by picking from the real ones at random WITH replacement, the same number as before, so some scores appear twice and some not at all. Do the same for the deployment scores. Compute the AUROC of this resampled data. Repeat many times, for example 1000.

The spread of those 1000 values stands in for the spread across fresh samples. The *95% confidence interval* runs from the value below which 2.5% of them fall to the value below which 97.5% fall. Those cut points are *quantiles*, and `t.quantile(values, q)` returns them: `q` is a fraction between 0 and 1, or a tensor of several fractions.

Random index tables do the resampling for all repetitions at once. `t.randint(0, n, (B, n), generator=g)` gives B rows of n indices, each between 0 and n − 1, and indexing the scores with that table gives B resampled sets.

## Worked example

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

## Faded practice

### q1882
`aucs` holds the AUROC of each of many resamples. Return the two ends of the 95% confidence interval, as a tensor of shape `(2,)`.

```python starter
import torch as t


def solve(aucs):
    return t._____(aucs, t.tensor([_____, 0.975]))
```

```python solution
import torch as t


def solve(aucs):
    return t.quantile(aucs, t.tensor([0.025, 0.975]))
```

## Solo practice

### q1883
`pos` holds the scores of the evaluation prompts and `neg` the scores of the deployment prompts. Return the number of pairs (one of each) in which the evaluation prompt scores strictly higher, as a Python int.

### q1884
`pos` holds the scores of the evaluation prompts and `neg` the scores of the deployment prompts. For each evaluation prompt, return the fraction of deployment prompts it scores strictly higher than. Shape like `pos`.

### q1885
`scores` holds one score per prompt and `is_eval` is a boolean tensor of the same shape, True for evaluation prompts and False for deployment prompts. Return the AUROC as a Python float, with a tie counted as half.

### q1886
Return a table of resampling indices of shape `(B, n)`: every entry is a whole number from 0 to n − 1 inclusive, drawn uniformly with the generator `g`.

### q1887
`pos` and `neg` hold the evaluation and deployment scores. `idx_pos`, shape `(B, n)`, and `idx_neg`, shape `(B, m)`, hold resampling indices into them. Return the AUROC of each of the B resamples, shape `(B,)`, with a tie counted as half.

### q1888
`aucs` holds the AUROC of each of many resamples. Return True if the 95% confidence interval lies entirely above 0.5, else False.

## Integrated practice

### q1889
`ev` and `dep` have shape `(prompt, rollout, pos)` and hold the phrase log-probabilities for evaluation prompts and deployment prompts. `ks` is a list of rollout counts. For each count k, score every prompt with its first k rollouts and compute the AUROC, with a tie counted as half. Return the AUROCs as a 1-D tensor with one entry per count, in the order given.

### q1890
`pos` and `neg` hold the evaluation and deployment scores. `idx_pos`, shape `(B, n)`, and `idx_neg`, shape `(B, m)`, hold resampling indices into them. Return the two ends of the 95% confidence interval of the AUROC (a tie counts as half), as a tensor of shape `(2,)`.

### q1891
Two monitors scored the same prompts. `pos_a` and `neg_a` hold the first monitor's scores for the evaluation and deployment prompts, `pos_b` and `neg_b` the second monitor's. Return the first monitor's AUROC minus the second's, as a Python float. A tie counts as half.

## Misconceptions

- **AUROC is the accuracy at the best threshold.** It uses no threshold. It is the share of evaluation and deployment pairs ranked the right way.
- **A tie is a loss.** A tie counts as half. Scores that tie often would otherwise be marked far too low.
- **An AUROC of 0.5 is half right.** 0.5 is what random scores give. It means no information.
- **Resample without replacement.** That returns the same set every time. Drawing with replacement is what makes the resamples differ.
