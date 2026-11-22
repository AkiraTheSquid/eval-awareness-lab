---
kc: monitor.reasoning-boundaries
title: Cut the reasoning at its end and find the sentence boundaries
new_syntax: [torch.nonzero]
concepts: [first-stop-token, sentence-boundary-positions]
supporting: ['tensor.indexed-selection', 'torch.boolean-masking', 'tensor.row-normalization', 'einops.reduce-model']
previews: []
faded: [1848, 1849]
guided: []
independent: [1850, 1851, 1852, 1853, 1854, 1855]
integrated: [1856, 1857, 1858]
---

## Concept: The reasoning ends at the first stop token

The monitor needs samples of the model's reasoning. A *rollout* is one sampled continuation: give the model the prefix and let it write, choosing each token at random according to its probabilities. Because the choice is random, eight rollouts of one prompt give eight different pieces of reasoning. With the Hugging Face library one call makes them all, and this cell is for reading only:

```python no-run
out = model.generate(
    prefix[None, :], do_sample=True,
    temperature=0.6, top_p=0.95,
    max_new_tokens=300, num_return_sequences=8,
    pad_token_id=pad)
rollouts = out[:, len(prefix):]
```

The result is a table with one row per rollout, and every row has the same width. That is the catch. A rollout whose reasoning finished early is filled up to the full width: first comes the tag that closes the reasoning, `</think>`, then the answer, and after the end of the answer a filler id called the *padding* token.

So the reasoning of one rollout is the part before its first *stop token*, the first position holding the closing tag (or padding). A rollout that hit the length limit has no stop token at all, and its reasoning is the whole row.

To find the first stop position without a loop, write each position's own index where there is a stop token and the row length everywhere else, then take the minimum:

$$n = \min_i \begin{cases} i & \text{if token } i \text{ is a stop token} \\ \text{len} & \text{otherwise} \end{cases}$$

With no stop token every entry is the row length, which is exactly the right answer for an unfinished rollout.

## Worked example

The rollout below has eight tokens, and id 99 is the closing tag. The plan: mark the stop positions, replace each by its index and the rest by 8, take the minimum, and cut there. Predict n before running.

```python
import torch as t
r = t.tensor([10, 11, 12, 13, 99, 40, 41, 99])
stop = r == 99
first = t.where(stop, t.arange(len(r)), len(r))
n = first.min().item()
reasoning = r[:n]
print(n, reasoning)
# Hidden checks
assert n == 4
assert reasoning.tolist() == [10, 11, 12, 13]
```

The tag appears twice, at positions 4 and 7. The minimum picks the first, so the reasoning is the four tokens before it. The same line gives the right result when the tag never appears:

```python
unfinished = t.tensor([10, 11, 12])
stop = unfinished == 99
print(t.where(stop, t.arange(3), 3).min().item())
# Hidden checks
assert t.where(stop, t.arange(3), 3).min().item() == 3
```

No position is a stop token, so every entry is 3 and the reasoning is the whole row. This rollout did not finish, which is worth recording: its reasoning was cut off by the length limit, not by the model.

## Faded practice

### q1848
`r` holds the token ids of one rollout and `stop_id` is the id that closes the reasoning. Return the number of reasoning tokens: the position of the first `stop_id`, or the length of `r` if it never appears. Python int.

```python starter
import torch as t


def solve(r, stop_id):
    stop = r == stop_id
    return t.where(stop, t.arange(len(r)), len(r))._____().item()
```

```python solution
import torch as t


def solve(r, stop_id):
    stop = r == stop_id
    return t.where(stop, t.arange(len(r)), len(r)).min().item()
```

## Concept: A sentence boundary is the position right after a sentence-ending token

Testing the phrase after every single token of the reasoning would cost one forced sequence per token. The paper tests only at *sentence boundaries*: the points where one sentence has just ended and the next has not begun, which is where a new thought such as "The user might be testing" could naturally start.

A boundary is found from the tokens. Some tokens end a sentence: their text ends in a full stop, a question mark, an exclamation mark or a line break. Which ids those are is a property of the vocabulary, so it is computed once and stored as a *lookup table*: a boolean tensor `is_end` of length V, where `is_end[i]` is True when token i ends a sentence.

Indexing the table with the reasoning's ids, `is_end[reasoning]`, gives one True or False per reasoning token. `t.nonzero(hit)` then returns the positions of the True entries as a table with one row per True and one column per axis, so for a 1-D input the positions are its column 0.

A boundary is the position AFTER the sentence-ending token, because the forced sequence takes the first b reasoning tokens and the sentence-ending token must be among them:

$$b = (\text{position of a sentence-ending token}) + 1$$

## Worked example

The vocabulary below has six tokens, and tokens 3 and 5 end a sentence. The reasoning has seven tokens. The plan: look each token up in the table, find the positions that are True, and add 1.

```python
import torch as t
is_end = t.tensor([False, False, False,
                   True, False, True])
reasoning = t.tensor([0, 1, 3, 2, 2, 5, 4])
hit = is_end[reasoning]
print(hit)
# Hidden checks
assert hit.tolist() == [False, False, True, False, False, True, False]
```

Positions 2 and 5 hold sentence-ending tokens. Now the boundaries:

```python
bounds = t.nonzero(hit)[:, 0] + 1
print(bounds)
print(reasoning[:bounds[0]])
# Hidden checks
assert bounds.tolist() == [3, 6]
assert reasoning[:bounds[0]].tolist() == [0, 1, 3]
```

The boundaries are 3 and 6. Taking the first three reasoning tokens gives a piece that ends with token 3, a complete sentence. Without the `+ 1` the piece would be two tokens long and stop just before the sentence ended.

## Faded practice

### q1849
`is_end` is a boolean tensor of length V that is True for token ids that end a sentence. `reasoning` holds token ids. Return the sentence boundaries: for each sentence-ending token, the position right after it. 1-D tensor.

```python starter
import torch as t


def solve(is_end, reasoning):
    return t._____(is_end[reasoning])[:, 0] + _____
```

```python solution
import torch as t


def solve(is_end, reasoning):
    return t.nonzero(is_end[reasoning])[:, 0] + 1
```

## Solo practice

### q1850
`r` holds the token ids of one rollout and `stop_id` is the id that closes the reasoning. Return True if the rollout finished its reasoning (the id appears somewhere), else False.

### q1851
`r` holds the token ids of one rollout. The reasoning ends at the first token equal to `end_id` or to `pad_id`, whichever comes first; if neither appears it is the whole of `r`. Return the reasoning tokens.

### q1852
`R` has shape `(k, T)`: k rollouts of T token ids each. `stop_id` is the id that closes the reasoning. Return the number of reasoning tokens in each rollout (T for a rollout in which the id never appears), shape `(k,)`.

### q1853
`is_end` is a boolean tensor of length V that is True for token ids that end a sentence. `reasoning` holds token ids. Return the number of sentences that end inside the reasoning, as a Python int.

### q1854
`is_end` is a boolean tensor of length V that is True for token ids that end a sentence. `reasoning` holds token ids and contains at least one sentence-ending token. Return the reasoning tokens up to and including the LAST sentence-ending token.

### q1855
`is_end` is a boolean tensor of length V that is True for token ids that end a sentence. `reasoning` holds token ids and `b` is a sentence boundary in it. Return the id of the token that ended the sentence at that boundary, as a Python int.

## Integrated practice

### q1856
`r` holds the token ids of one rollout and `stop_id` closes the reasoning (it may not appear). `is_end` is a boolean tensor of length V that is True for token ids that end a sentence. Return the sentence boundaries of the reasoning only: positions right after each sentence-ending token that comes before the first `stop_id`. 1-D tensor.

### q1857
`R` has shape `(k, T)`: k rollouts of T token ids. `stop_id` closes the reasoning (it may be absent from a row). `is_end` is a boolean tensor of length V that is True for token ids that end a sentence. Return the number of sentence boundaries inside the reasoning of each rollout, shape `(k,)`.

### q1858
`r` holds the token ids of one rollout and `stop_id` closes the reasoning (it may not appear). `is_end` is a boolean tensor of length V that is True for token ids that end a sentence. Return the reasoning tokens before the FIRST sentence boundary, that is, the first sentence of the reasoning. If the reasoning has no sentence-ending token, return all of the reasoning.

## Misconceptions

- **Every rollout row is all reasoning.** A finished rollout is followed by the closing tag, the answer and padding. Cut at the first stop token.
- **Cut at the last stop token.** The answer and the padding come after the first one. The reasoning ends there.
- **A rollout with no stop token has no reasoning.** It has reasoning that ran into the length limit. Its length is the whole row.
- **The boundary is the position of the sentence-ending token.** It is one past it, so that the first b tokens include that token.
