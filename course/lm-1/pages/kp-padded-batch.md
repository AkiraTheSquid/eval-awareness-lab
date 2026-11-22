---
kc: monitor.padded-batch
title: Score many forced sequences in one padded batch
new_syntax: []
concepts: [right-padding-and-mask, per-row-phrase-rows]
supporting: ['monitor.forced-logprob', 'monitor.forced-sequence', 'torch.constructors', 'torch.slice-assignment', 'torch.broadcasting-rules', 'python.control-flow']
previews: []
faded: [1859, 1860]
guided: []
independent: [1861, 1862, 1863, 1864, 1865, 1866]
integrated: [1867, 1868, 1869]
---

## Concept: Pad on the right so every position keeps its index

One prompt gives several rollouts, and each rollout has many sentence boundaries. Every boundary needs its own forced sequence, so a single prompt can need a few hundred sequences. Running the model once per sequence is slow. A graphics card is fast when it is given many sequences at once, stacked as the rows of one table: a *batch*.

A table needs rows of equal length, and forced sequences differ in length because each holds a different amount of reasoning. The fix is *padding*: short rows are filled up with a filler id, the *padding token*, until they match the longest row.

The filler goes on the RIGHT. A language model of this kind only looks backwards: the logits row at position p depends on positions 0 to p and nothing later. Filler placed after the real tokens therefore cannot change any row that scores a real token, and a phrase that starts at position `start` still starts at `start`. Filler on the left would shift every position.

The model also takes a second table of the same shape, the *attention mask*: True where a real token sits and False on filler. For row i with length $\ell_i$ and column j:

$$\text{mask}[i, j] = (j < \ell_i)$$

With the Hugging Face library both tables go into one call, shown for reading only:

```python no-run
logits = model(input_ids=ids,
               attention_mask=mask).logits
```

## Worked example

Two sequences of lengths 3 and 1 are to be stacked, with 0 as the padding id. The plan: make a table full of padding that is as wide as the longest sequence, then write each sequence into the left end of its own row.

```python
import torch as t
seqs = [t.tensor([4, 2, 9]), t.tensor([3])]
lens = t.tensor([len(q) for q in seqs])
ids = t.full((len(seqs), lens.max().item()), 0)
i = 0
for q in seqs:
    ids[i, :len(q)] = q
    i += 1
print(ids)
# Hidden checks
assert ids.tolist() == [[4, 2, 9], [3, 0, 0]]
```

The second row keeps its one real token at position 0, where it was before padding. Now the mask. A column index is compared with each row's length, so the lengths need a trailing axis and the column indices a leading one:

```python
mask = t.arange(3)[None, :] < lens[:, None]
print(mask)
# Hidden checks
assert mask.tolist() == [[True, True, True], [True, False, False]]
```

Row 1 is True only in column 0. The comparison is strict: a row of length 1 has real tokens at positions below 1, which is position 0 alone.

## Faded practice

### q1859
`seqs` is a list of 1-D tensors of token ids with different lengths. Return one table with a row per sequence, as wide as the longest, in which every sequence keeps its positions and the unused entries hold `pad`.

```python starter
import torch as t


def solve(seqs, pad):
    lens = t.tensor([len(q) for q in seqs])
    out = t.full((len(seqs), lens.max().item()), pad)
    i = 0
    for q in seqs:
        out[i, :_____] = q
        i += 1
    return out
```

```python solution
import torch as t


def solve(seqs, pad):
    lens = t.tensor([len(q) for q in seqs])
    out = t.full((len(seqs), lens.max().item()), pad)
    i = 0
    for q in seqs:
        out[i, :len(q)] = q
        i += 1
    return out
```

## Concept: Each row of the batch has its own phrase position

After the batch has been through the model, the logits have shape `(b, n, V)`: b sequences, n positions, V vocabulary entries. The phrase sits at a different place in every row, because each row holds a different amount of reasoning before it. So there is one start per row, a tensor `starts` of shape `(b,)`.

For one sequence the rows that score an m-token phrase were `logits[start - 1 : start - 1 + m]`. A slice cannot change from row to row, so the batch version spells out the positions as a table of shape `(b, m)`:

$$\text{pos}[i, j] = \text{starts}[i] - 1 + j$$

and pairs it with a column of row numbers, shape `(b, 1)`. Indexing with the two together picks, for every row i and every phrase token j, the logits row at position `pos[i, j]` of sequence i. The result has shape `(b, m, V)`: for each sequence, exactly the rows that score its phrase. From there the steps are the ones already known: log-probabilities over the vocabulary, one entry per phrase token, and a sum over the token axis.

## Worked example

The made-up logits below hold the position number in every entry, so the output shows which positions were picked. There are two sequences. The phrase has two tokens and starts at position 1 in the first sequence and position 3 in the second. The plan: build the table of positions, then index with a column of row numbers beside it.

```python
import torch as t
position = t.arange(5.0).reshape(1, 5, 1)
logits = position * t.ones(2, 5, 3)
starts = t.tensor([1, 3])
m = 2
pos = (starts - 1)[:, None] + t.arange(m)[None, :]
print(pos)
# Hidden checks
assert pos.tolist() == [[0, 1], [2, 3]]
```

The first sequence needs positions 0 and 1, the second needs 2 and 3. Now the lookup:

```python
rows = logits[t.arange(2)[:, None], pos]
print(rows.shape)
print(rows[:, :, 0])
# Hidden checks
assert rows.shape == (2, 2, 3)
assert rows[:, :, 0].tolist() == [[0.0, 1.0], [2.0, 3.0]]
```

The shape is `(2, 2, 3)`: two sequences, two phrase tokens, three vocabulary entries. Each sequence got its own pair of positions, both beginning one position before its phrase.

## Faded practice

### q1860
`logits` has shape `(b, n, V)`. In sequence i a phrase of `m` tokens begins at position `starts[i]`. Return the rows that score each sequence's phrase, shape `(b, m, V)`.

```python starter
import torch as t


def solve(logits, starts, m):
    b = len(starts)
    return logits[t.arange(b)[:, None], (starts - _____)[:, None] + t.arange(m)[None, :]]
```

```python solution
import torch as t


def solve(logits, starts, m):
    b = len(starts)
    return logits[t.arange(b)[:, None], (starts - 1)[:, None] + t.arange(m)[None, :]]
```

## Solo practice

### q1861
`lens` holds the length of each of b sequences and `width` is the width of the padded table. Return the attention mask, shape `(b, width)`: True where a real token sits, False on filler.

### q1862
`ids` is a padded table of token ids, shape `(b, n)`, and `pad` is the filler id, which no real token uses. Return the number of real tokens in each row, shape `(b,)`.

### q1863
`logits` has shape `(b, n, V)` and came from a right-padded batch. `lens` holds the number of real tokens in each sequence. For every sequence, return the row of scores for the token that would come NEXT after its last real token. Shape `(b, V)`.

### q1864
`logp` has shape `(b, n, V)` and already holds log-probabilities. In sequence i a phrase begins at position `starts[i]`, and `phrases`, shape `(b, m)`, holds each sequence's m phrase token ids. Return the log-probability of every phrase token, shape `(b, m)`.

### q1865
`befores` is a list of 1-D tensors of token ids with different lengths, and `phrase` is a 1-D tensor of token ids. Return one table in which row i is sequence i followed directly by the phrase, with `pad` in the unused entries on the right.

### q1866
`vals` has shape `(b, n)` and holds one number per position of a padded batch. `mask`, same shape, is True where a real token sits. Return the sum of each row's numbers over its real positions only, shape `(b,)`.

## Integrated practice

### q1867
`ids` has shape `(b, n)` and is a right-padded batch of forced sequences; `logits`, shape `(b, n, V)`, is the model's raw output for it. In sequence i a phrase of `m` tokens begins at position `starts[i]`. Return the log-probability of each sequence's own phrase given the text before it, shape `(b,)`.

### q1868
`prefix`, `reasoning` and `phrase` are 1-D tensors of token ids, and `bounds` is a 1-D tensor of sentence boundaries in the reasoning. Return a right-padded table with one forced sequence per boundary: the prefix, the reasoning up to that boundary, then the phrase. Unused entries hold `pad`.

### q1869
`ids` has shape `(b, n)` and is a right-padded batch of forced sequences, with `pad` as the filler id (no real token uses it). `logits`, shape `(b, n, V)`, is the model's raw output. In every sequence the phrase is the LAST `m` real tokens. Return the log-probability of each sequence's phrase given the text before it, shape `(b,)`.

## Misconceptions

- **Padding on either side is the same.** Filler on the left shifts every position. On the right, positions and the rows that score real tokens are untouched.
- **The last column of the batch scores each sequence's next token.** Only for the longest row. For a shorter row the last column sits on filler.
- **One start position serves the whole batch.** Each row holds a different amount of reasoning, so each row has its own start.
- **The mask is True on the padding.** It is True on real tokens and False on filler.
