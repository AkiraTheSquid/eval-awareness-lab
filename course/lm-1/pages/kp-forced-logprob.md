---
kc: monitor.forced-logprob
title: The log-probability of a forced phrase
new_syntax: [torch.log_softmax, torch.log_softmax#dim]
concepts: [log-softmax-over-vocabulary, sum-over-phrase-tokens]
supporting: ['monitor.next-token-logits', 'tensor.stable-probabilities', 'tensor.indexed-selection', 'einops.reduce-model']
previews: []
faded: [1826, 1827]
guided: []
independent: [1828, 1829, 1830, 1831, 1832, 1833]
integrated: [1834, 1835, 1836]
---

## Concept: log_softmax turns a row of scores into log-probabilities

A row of logits is a row of raw scores. To ask "how likely did the model find this token?", the scores have to become probabilities: positive numbers that add up to 1 across the vocabulary. The usual recipe is the *softmax*: exponentiate every score, then divide by the total of the row.

The monitor works with the logarithm of that probability, the *log-probability*. For a row of scores $z$ and a token $i$:

$$\log p_i = z_i - \log \sum_j e^{z_j}$$

`t.log_softmax(logits, dim=-1)` computes this for every row at once. `dim=-1` names the axis whose entries compete with each other, which is the vocabulary axis, the last one. The result has the same shape as `logits`.

Three facts are worth checking on any output. Every entry is at most 0, because a probability is at most 1 and the logarithm of 1 is 0. A more likely token has a log-probability closer to 0. And exponentiating a row gives numbers that add up to 1.

## Worked example

Take two rows of scores over a vocabulary of size 3. The plan: turn each row into log-probabilities, then read off the log-probability of token 2 in the first row. Before running, predict which token of the first row is closest to 0.

```python
import torch as t
logits = t.tensor([[0.0, 1.0, 2.0],
                   [3.0, 3.0, 3.0]])
logp = t.log_softmax(logits, dim=-1)
print(logp)
print(logp[0, 2])
# Hidden checks
assert logp.shape == (2, 3)
assert abs(logp[0, 2].item() + 0.4076) < 1e-3
assert t.allclose(logp[1], t.tensor([-1.0986, -1.0986, -1.0986]), atol=1e-3)
```

Token 2 had the highest score in the first row, so it is closest to 0, at about −0.41. In the second row all three scores are equal, so each token has probability one third, and the logarithm of one third is about −1.10. The scores in that row were all 3, but only their differences mattered.

Now confirm that each row is a set of probabilities once it is exponentiated:

```python
print(logp.exp().sum(dim=-1))
# Hidden checks
assert t.allclose(logp.exp().sum(dim=-1), t.ones(2))
```

Both rows add up to 1. Had the softmax been taken along dimension 0, the columns would add up to 1 instead and tokens would be competing across positions, which means nothing.

## Faded practice

### q1826
`logits` has shape `(n, V)`, one row of scores per position. Return a table of the same shape in which every row holds log-probabilities over the vocabulary.

```python starter
import torch as t


def solve(logits):
    return t._____(logits, _____=-1)
```

```python solution
import torch as t


def solve(logits):
    return t.log_softmax(logits, dim=-1)
```

## Concept: A phrase's log-probability is the sum of its tokens' log-probabilities

The model writes one token at a time. The probability that it writes a whole phrase is therefore a product: the probability of the first phrase token, times the probability of the second given the first, and so on. For a phrase with tokens $w_1, \dots, w_m$ after some earlier text:

$$P(\text{phrase}) = \prod_{j=1}^{m} P(w_j \mid \text{earlier text},\ w_1, \dots, w_{j-1})$$

Taking logarithms turns the product into a sum:

$$\log P(\text{phrase}) = \sum_{j=1}^{m} \log P(w_j \mid \text{earlier text},\ w_1, \dots, w_{j-1})$$

One pass of the model gives every term at once, as long as the phrase is already in the input. Row j of the phrase's rows has read the earlier text and the first j phrase tokens, so it holds exactly the conditional probabilities the formula asks for. This is called *forcing* the phrase: the model never has to choose to write it.

In code there are two steps. First pick one log-probability per phrase token with a pair of index tensors: row j, column $w_j$. Then add them up. With a batch of texts the picked table has shape `(batch, tok)`, and `einops.reduce(per_token, "batch tok -> batch", "sum")` adds over the token axis and keeps one number per text.

## Worked example

Suppose the three rows below are already the rows that score a three-token phrase, and the phrase tokens are 2, 0, 1. The plan: take log-probabilities, pick row 0 column 2, row 1 column 0, row 2 column 1, and add the three numbers.

```python
import torch as t
import einops
rows = t.tensor([[0.0, 1.0, 2.0],
                 [3.0, 0.5, 1.0],
                 [0.2, 4.0, 0.1]])
phrase = t.tensor([2, 0, 1])
logp = t.log_softmax(rows, dim=-1)
per_token = logp[t.arange(3), phrase]
print(per_token)
# Hidden checks
assert t.allclose(per_token, t.tensor([-0.4076, -0.1967, -0.0421]), atol=1e-3)
```

Each of the three tokens was the most likely one in its row, so all three numbers are close to 0. Now the sum:

```python
total = einops.reduce(per_token, "tok -> ", "sum")
print(total)
print(total.exp())
# Hidden checks
assert abs(total.item() + 0.6464) < 1e-3
assert abs(total.exp().item() - 0.5239) < 1e-3
```

The phrase's log-probability is about −0.65. Exponentiating gives 0.52: the model would write exactly this phrase here about half the time. Real phrases score far lower, often below −20, which is why the monitor keeps everything as logarithms: a probability that small is awkward to compare, while its logarithm is an ordinary number.

## Faded practice

### q1827
`per_token` has shape `(b, m)`: for each of b texts, the log-probability of each of the m phrase tokens. Return the log-probability of the whole phrase for each text, shape `(b,)`.

```python starter
import torch as t
import einops


def solve(per_token):
    return einops.reduce(per_token, "batch tok -> _____", "_____")
```

```python solution
import torch as t
import einops


def solve(per_token):
    return einops.reduce(per_token, "batch tok -> batch", "sum")
```

## Solo practice

### q1828
`rows` has shape `(m, V)` and holds raw scores. Return the log-probability that row `r` gives to token `tok`, as a Python float.

### q1829
`rows` has shape `(m, V)` and holds raw scores. Return the probability (not its logarithm) that row `r` gives to token `tok`, as a Python float.

### q1830
`rows` has shape `(m, V)` and holds the raw scores that score an m-token phrase. `phrase` holds the m phrase token ids. Return the log-probability of each phrase token, in order, shape `(m,)`.

### q1831
`rows` has shape `(m, V)` and holds the raw scores that score an m-token phrase. `phrase` holds the m phrase token ids. Return the log-probability of the whole phrase, as a Python float.

### q1832
`logits` has shape `(b, m, V)`: for each of b texts, the raw scores that score the same m-token phrase. `phrase` holds the m phrase token ids. Return the log-probability of the whole phrase in each text, shape `(b,)`.

### q1833
`logits` has shape `(b, m, V)`: for each of b texts, the raw scores that score the same m-token phrase. `phrase` holds the m phrase token ids. Return the index of the text in which the whole phrase is most likely, as a Python int.

## Integrated practice

### q1834
`ids` holds n token ids and `logits`, shape `(n, V)`, is the model's raw output for them. A phrase begins at position `start` and runs to the end of the text. Return the log-probability of the whole phrase given the text before it, as a Python float.

### q1835
`ids` has shape `(b, n)` and `logits` has shape `(b, n, V)`: b texts of n tokens with the model's raw output. In every text a phrase begins at position `start` and runs to the end. Return the log-probability of each text's own phrase given the text before it, shape `(b,)`.

### q1836
`ids` holds n token ids and `logits` has shape `(n, V)`. A phrase begins at position `start` and runs to the end of the text. Return the place within the phrase (0 for its first token) of the token the model found LEAST likely, as a Python int.

## Misconceptions

- **Take the softmax over the position axis.** Tokens compete within one row, so the axis is the vocabulary axis, the last one.
- **Add up the probabilities of the phrase tokens.** Probabilities of successive tokens multiply. It is their logarithms that add.
- **A log-probability near 0 means unlikely.** 0 is the logarithm of probability 1. Unlikely tokens have large negative values.
- **Raw logits can be added in place of log-probabilities.** Logits are only meaningful relative to the rest of their row. Two rows can have the same logit for a token and very different probabilities.
