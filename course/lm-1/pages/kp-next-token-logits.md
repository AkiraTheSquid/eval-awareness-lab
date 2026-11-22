---
kc: monitor.next-token-logits
title: Row p of the logits scores the token at position p + 1
new_syntax: []
concepts: [row-predicts-next, rows-for-a-span]
supporting: ['tensor.indexed-selection', 'torch.argmin-argmax', 'torch.slicing-views']
previews: []
faded: [1815, 1816]
guided: []
independent: [1817, 1818, 1819, 1820, 1821, 1822]
integrated: [1823, 1824, 1825]
---

## Concept: Each row of the logits is about the NEXT token

A language model reads text as a list of whole numbers called *token ids*: each id stands for one piece of text, such as a word or a part of a word. The set of all ids the model knows is its *vocabulary*, of size V.

Give the model n token ids and it returns a table of scores called *logits*, with shape `(n, V)`: one row per position and one column per vocabulary entry. A higher score means the model finds that token more likely.

The row at position p was computed from the tokens at positions 0 up to p, and it scores what the token at position p + 1 could be. So the score the model gave to the token that actually sits at position p is in row p − 1, in the column of that token's id:

$$\text{score of the token at position } p = \text{logits}[p - 1,\ \text{ids}[p]]$$

Two consequences follow. The token at position 0 has no score, because no row comes before it. And the last row scores a token that is not in the input at all: the one the model would write next.

## Worked example

Here are four token ids from a vocabulary of size 3, and a made-up logits table for them. The plan: to find the score the model gave to the token at position 2, first look up which id sits there and which row is one back.

```python
import torch as t
ids = t.tensor([2, 1, 0, 0])
logits = t.tensor([[1.0, 0.0, 2.5],
                   [0.5, 3.0, 1.5],
                   [2.0, 0.2, 4.0],
                   [1.0, 5.0, 1.0]])
p = 2
print(ids[p], logits[p - 1])
# Hidden checks
assert ids[p].item() == 0
assert logits[p - 1].tolist() == [0.5, 3.0, 1.5]
```

The id at position 2 is 0, and the row one back is row 1. The score is the entry of that row in the column of the id:

```python
score = logits[p - 1, ids[p]]
print(score)
# Hidden checks
assert score.item() == 0.5
```

The answer is 0.5. Row 1 was computed from the ids 2 and 1 only, which is why it can be read as a prediction of position 2.

Reading row 2 instead would give 2.0. That is the score row 2 gives to id 0 as a guess for position 3, a different question.

## Faded practice

### q1815
`logits` has shape `(n, V)` and came from one pass of a language model over n token ids. Return the row of scores for the token at position `p`, shape `(V,)`. `p` is at least 1.

```python starter
import torch as t


def solve(logits, p):
    return logits[p - _____]
```

```python solution
import torch as t


def solve(logits, p):
    return logits[p - 1]
```

## Concept: A span of m tokens is scored by m consecutive rows, starting one row early

The monitor in this unit does not score one token. It scores a whole phrase that was placed at the end of the text, for example the tokens of "The user might be testing". Call the position of the phrase's first token `start` and the number of phrase tokens m.

Apply the one-row-back rule to each phrase token in turn. The first phrase token, at position `start`, is scored by row `start − 1`. The last one, at position `start + m − 1`, is scored by row `start + m − 2`. So the rows that score the phrase are the m rows

$$\text{logits}[\text{start} - 1 : \text{start} - 1 + m]$$

The slice begins one row before the phrase and stops one row before the phrase ends. The final row of the table is never used when the phrase is the last thing in the text, because that row scores a token after the phrase.

## Worked example

The text below has six tokens and the phrase is the last three. The plan: work out where the phrase starts and how long it is, then slice the rows that score it.

```python
import torch as t
ids = t.tensor([4, 1, 0, 2, 2, 3])
start = 3
m = len(ids) - start
print(m, start - 1)
# Hidden checks
assert m == 3 and start - 1 == 2
```

There are three phrase tokens, and the first of them is scored by row 2. The made-up logits below hold the row number in every entry, so the printout shows which rows were taken:

```python
logits = t.arange(6.0).reshape(6, 1) * t.ones(6, 5)
rows = logits[start - 1:start - 1 + m]
print(rows.shape)
print(rows[:, 0])
# Hidden checks
assert rows.shape == (3, 5)
assert rows[:, 0].tolist() == [2.0, 3.0, 4.0]
```

Rows 2, 3 and 4 came back. Row 2 scores the first phrase token even though row 2 belongs to the last token before the phrase. Row 5 was left out: it scores whatever would come after the phrase.

## Faded practice

### q1816
`logits` has shape `(n, V)`. A phrase of `m` tokens begins at position `start`. Return the rows that score the phrase tokens, shape `(m, V)`.

```python starter
import torch as t


def solve(logits, start, m):
    return logits[start - _____:start - _____ + m]
```

```python solution
import torch as t


def solve(logits, start, m):
    return logits[start - 1:start - 1 + m]
```

## Solo practice

### q1817
`ids` holds n token ids and `logits`, shape `(n, V)`, is the model's output for them. Return the score the model gave to the token that actually sits at position `p`, as a Python float. `p` is at least 1.

### q1818
`ids` holds n token ids and `logits` has shape `(n, V)`. For every position from 1 to n − 1, find the score the model gave to the token that actually sits there. Return them in order, shape `(n − 1,)`.

### q1819
`logits` has shape `(n, V)`. Return the id of the token the model found most likely for position `p`, as a Python int. `p` is at least 1.

### q1820
`ids` holds n token ids and `logits` has shape `(n, V)`. Count the positions from 1 to n − 1 at which the model's most likely token is the token that actually sits there. Return a Python int.

### q1821
`logits` has shape `(b, n, V)`: b texts of n tokens each. For every text, return the row of scores for its LAST token (the one at position n − 1). Shape `(b, V)`.

### q1822
`logits` has shape `(n, V)`. A phrase begins at position `start` and runs to the end of the text. Return the number of rows of `logits` that score phrase tokens, as a Python int.

## Integrated practice

### q1823
`ids` holds n token ids and `logits` has shape `(n, V)`. A phrase begins at position `start` and runs to the end of the text. Return the score the model gave to each phrase token, in order, shape `(n − start,)`.

### q1824
`ids` holds n token ids and `logits` has shape `(n, V)`. A phrase begins at position `start` and runs to the end of the text. Return True if the model's most likely token matches the actual token at every phrase position, else False.

### q1825
`ids` has shape `(b, n)`: b texts of n token ids. `logits` has shape `(b, n, V)`. In every text a phrase begins at the same position `start`. For each text, return the score the model gave to the FIRST phrase token. Shape `(b,)`.

## Misconceptions

- **Row p scores the token at position p.** Row p has already read the token at position p. It scores the token at position p + 1.
- **The rows for a phrase are the rows at the phrase's own positions.** They start one row before the phrase and stop one row before its end.
- **The last row scores the last token.** The last row scores a token that is not in the input, the one that would come next.
- **The first token has a score in row 0.** Nothing comes before position 0, so no row scores it. Row 0 scores position 1.
