---
kc: monitor.prompt-score
title: A prompt's score is the maximum over its rollouts and boundaries
new_syntax: []
concepts: [ragged-traces-to-a-table, max-over-the-first-k-rollouts]
supporting: ['monitor.forced-logprob', 'monitor.reasoning-boundaries', 'einops.reduce-model', 'torch.constructors', 'torch.stack-concat-interleave', 'torch.slice-assignment', 'python.control-flow']
previews: []
faded: [1870, 1871]
guided: []
independent: [1872, 1873, 1874, 1875, 1876, 1877]
integrated: [1878, 1879, 1880]
---

## Concept: Pad the traces with minus infinity, then take the maximum

Scoring one rollout gives a *trace*: one phrase log-probability per sentence boundary, in order. A trace says how ready the model was to write the phrase at each point of its reasoning.

The monitor turns a trace into a single number by taking its maximum. The reason is what is being detected. A model that suspects a test usually says so once, at one point, and is busy with the task everywhere else. An average over the boundaries would be dominated by the many ordinary points. The maximum asks whether there was ANY point at which the phrase would have fitted.

Rollouts have different numbers of boundaries, so their traces have different lengths. To hold them in one table the short ones are padded, and the filler has to be a value that can never win a maximum. Zero would be a disaster: every log-probability is negative, so a zero would always win. The right filler is minus infinity, written `float("-inf")`, which is below every number:

$$\max(x, -\infty) = x$$

## Worked example

Two rollouts of one prompt gave traces of lengths 3 and 1. The plan: build a table full of minus infinity, write each trace into the left of its row, and take the maximum of everything.

```python
import torch as t
import einops
traces = [t.tensor([-9.0, -4.0, -7.0]),
          t.tensor([-6.0])]
lens = t.tensor([len(q) for q in traces])
table = t.full((len(traces), lens.max().item()),
               float("-inf"))
i = 0
for q in traces:
    table[i, :len(q)] = q
    i += 1
print(table)
# Hidden checks
assert table[0].tolist() == [-9.0, -4.0, -7.0]
assert table[1, 0].item() == -6.0 and table[1, 1].item() == float("-inf")
```

The second row holds its one score and then filler. Now the maximum over both axes:

```python
score = einops.reduce(table, "rollout pos -> ", "max")
print(score)
# Hidden checks
assert score.item() == -4.0
```

The score is −4, from the second boundary of the first rollout. The filler never competed. Had the table been padded with zeros, the answer would have been 0, a score no real boundary produced.

## Faded practice

### q1870
`traces` is a list of 1-D tensors, one per rollout, each holding the phrase log-probability at every sentence boundary of that rollout. The traces have different lengths. Return the highest score of each rollout, shape `(rollouts,)`, by way of one padded table.

```python starter
import torch as t
import einops


def solve(traces):
    lens = t.tensor([len(q) for q in traces])
    table = t.full((len(traces), lens.max().item()), float("_____"))
    i = 0
    for q in traces:
        table[i, :len(q)] = q
        i += 1
    return einops.reduce(table, "rollout pos -> rollout", "max")
```

```python solution
import torch as t
import einops


def solve(traces):
    lens = t.tensor([len(q) for q in traces])
    table = t.full((len(traces), lens.max().item()), float("-inf"))
    i = 0
    for q in traces:
        table[i, :len(q)] = q
        i += 1
    return einops.reduce(table, "rollout pos -> rollout", "max")
```

## Concept: Use the same number of rollouts for every prompt

With traces padded, all the scores for a set of prompts fit in one table of shape `(prompt, rollout, pos)`. The score of a prompt is the maximum over its rollouts and its positions:

$$s(\text{prompt}) = \max_{r < k}\ \max_{j}\ \text{table}[\text{prompt}, r, j]$$

The number k is how many rollouts are used. It matters because a maximum can only grow when more values are added. A prompt scored with eight rollouts has had eight chances to produce a high value, and one scored with a single rollout has had one. So scores are only comparable between prompts that were given the same k, and results are reported per k: the score with one rollout, with two, with four, with eight.

Taking the first k rollouts is a slice on the rollout axis, and both maxima are one reduction: `einops.reduce(table[:, :k], "prompt rollout pos -> prompt", "max")`.

## Worked example

The table below has two prompts, three rollouts each and two positions per rollout. The plan: score every prompt with k = 1, then with k = 3, and compare.

```python
import torch as t
import einops
table = t.tensor([
    [[-9.0, -8.0], [-3.0, -7.0], [-6.0, -6.0]],
    [[-5.0, -4.0], [-9.0, -9.0], [-8.0, -2.0]]])
pattern = "prompt rollout pos -> prompt"
one = einops.reduce(table[:, :1], pattern, "max")
print(one)
# Hidden checks
assert one.tolist() == [-8.0, -4.0]
```

With one rollout the first prompt scores −8 and the second −4. Now all three:

```python
three = einops.reduce(table[:, :3], pattern, "max")
print(three)
# Hidden checks
assert three.tolist() == [-3.0, -2.0]
assert (three >= one).all()
```

Both scores went up, to −3 and −2. Neither could have gone down, because the values seen at k = 1 are still among the values seen at k = 3. This is why a score at k = 3 must not be compared with another prompt's score at k = 1.

## Faded practice

### q1871
`table` has shape `(prompt, rollout, pos)` and holds phrase log-probabilities. Return each prompt's score using only its first `k` rollouts, shape `(prompt,)`.

```python starter
import torch as t
import einops


def solve(table, k):
    return einops.reduce(table[:, :_____], "prompt rollout pos -> prompt", "_____")
```

```python solution
import torch as t
import einops


def solve(table, k):
    return einops.reduce(table[:, :k], "prompt rollout pos -> prompt", "max")
```

## Solo practice

### q1872
`table` has shape `(rollout, pos)` and holds one prompt's phrase log-probabilities, padded with minus infinity. Return the prompt's score as a Python float.

### q1873
`table` has shape `(rollout, pos)` and holds one prompt's phrase log-probabilities, padded with minus infinity. Return the index of the rollout that contains the prompt's highest score, as a Python int.

### q1874
`table` has shape `(rollout, pos)` and holds one prompt's phrase log-probabilities, padded with minus infinity. Return the number of sentence boundaries each rollout really had, shape `(rollout,)`.

### q1875
`table` has shape `(prompt, rollout, pos)` and holds phrase log-probabilities. Return, for each prompt, whether its score from the first `k` rollouts is above the threshold `thr`. Boolean tensor of shape `(prompt,)`.

### q1876
`table` has shape `(prompt, rollout, pos)` and holds phrase log-probabilities. Return how much each prompt's score rises when all its rollouts are used instead of only the first one. Shape `(prompt,)`.

### q1877
`table` has shape `(prompt, rollout, pos)` and holds phrase log-probabilities. Return, for each prompt, whether its score from the first `k` rollouts already equals its score from all rollouts. Boolean tensor of shape `(prompt,)`.

## Integrated practice

### q1878
`traces` is a list of 1-D tensors, one per rollout of a single prompt, each holding the phrase log-probability at every sentence boundary. The traces have different lengths. Return the prompt's score using only the first `k` rollouts, as a Python float.

### q1879
`table` has shape `(prompt, rollout, pos)` and holds phrase log-probabilities. `ks` is a list of rollout counts. Return the prompt scores for each count, shape `(len(ks), prompt)`: row i uses the first `ks[i]` rollouts.

### q1880
A batch of forced sequences for one prompt was scored: `scores`, shape `(N,)`, holds the phrase log-probability of each, and `rollout_of`, shape `(N,)`, holds the index of the rollout each sequence came from. Return the prompt's score using only rollouts with index below `k`, as a Python float.

## Misconceptions

- **Pad short traces with zero.** Log-probabilities are negative, so a zero wins every maximum. Pad with minus infinity.
- **Average the trace.** The signal is one point in a long reasoning. The maximum keeps it and the average buries it.
- **More rollouts just make the score more accurate.** They make it higher. A maximum over more values can only grow.
- **Scores with different numbers of rollouts can be compared.** Only scores at the same k are comparable.
