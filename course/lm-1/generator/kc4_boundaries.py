"""monitor.reasoning-boundaries: where the reasoning ends, and where its sentences end."""
from framework import maker

KC = "monitor.reasoning-boundaries"
A = [("lm-reasoning-cut", 1.0)]
AB = [("lm-sentence-boundaries", 1.0)]
ABC = [("lm-sentence-boundaries", 1.0), ("lm-reasoning-cut", 0.5)]
SPEC = dict(
    id=KC, title="Cut the reasoning at its end and find the sentence boundaries",
    concepts=["first-stop-token", "sentence-boundary-positions"],
    supporting=["tensor.indexed-selection", "torch.boolean-masking", "tensor.row-normalization", "einops.reduce-model"],
    new_syntax=["torch.nonzero"],
)

SEGMENTS = [
    dict(title="The reasoning ends at the first stop token", concept="""
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

$$n = \\min_i \\begin{cases} i & \\text{if token } i \\text{ is a stop token} \\\\ \\text{len} & \\text{otherwise} \\end{cases}$$

With no stop token every entry is the row length, which is exactly the right answer for an unfinished rollout.
""", worked="""
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
"""),
    dict(title="A sentence boundary is the position right after a sentence-ending token", concept="""
Testing the phrase after every single token of the reasoning would cost one forced sequence per token. The paper tests only at *sentence boundaries*: the points where one sentence has just ended and the next has not begun, which is where a new thought such as "The user might be testing" could naturally start.

A boundary is found from the tokens. Some tokens end a sentence: their text ends in a full stop, a question mark, an exclamation mark or a line break. Which ids those are is a property of the vocabulary, so it is computed once and stored as a *lookup table*: a boolean tensor `is_end` of length V, where `is_end[i]` is True when token i ends a sentence.

Indexing the table with the reasoning's ids, `is_end[reasoning]`, gives one True or False per reasoning token. `t.nonzero(hit)` then returns the positions of the True entries as a table with one row per True and one column per axis, so for a 1-D input the positions are its column 0.

A boundary is the position AFTER the sentence-ending token, because the forced sequence takes the first b reasoning tokens and the sentence-ending token must be among them:

$$b = (\\text{position of a sentence-ending token}) + 1$$
""", worked="""
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
"""),
]

MISCONCEPTIONS = [
    ("Every rollout row is all reasoning.", "A finished rollout is followed by the closing tag, the answer and padding. Cut at the first stop token."),
    ("Cut at the last stop token.", "The answer and the padding come after the first one. The reasoning ends there."),
    ("A rollout with no stop token has no reasoning.", "It has reasoning that ran into the length limit. Its length is the whole row."),
    ("The boundary is the position of the sentence-ending token.", "It is one past it, so that the first b tokens include that token."),
]

D = maker("rb", KC)
ISE = "is_end = t.tensor([False, False, False, True, False, True])"
DRILLS = [
    D("f1", "faded", 1,
      "`r` holds the token ids of one rollout and `stop_id` is the id that closes the reasoning. Return the number of reasoning tokens: the position of the first `stop_id`, or the length of `r` if it never appears. Python int.",
      "r, stop_id", "Number of tokens before the first stop token.",
      "    stop = r == stop_id\n    return t.where(stop, t.arange(len(r)), len(r)).min().item()",
      [("r = t.tensor([10, 11, 12, 13, 99, 40, 41, 99])\nstop_id = 99", "solve(r, stop_id)"), ("r = t.tensor([10, 11, 12])\nstop_id = 99", "solve(r, stop_id)"),
       ("r = t.tensor([7, 1, 7, 7])\nstop_id = 7", "solve(r, stop_id)"), ("r = t.tensor([1, 2, 3, 4, 5, 6])\nstop_id = 6", "solve(r, stop_id)")],
      ("    stop = r == stop_id\n    return t.where(stop, t.arange(len(r)), 0).max().item()", "r = t.tensor([10, 11, 12, 13, 99, 40, 41, 99])\nstop_id = 99",
       "This found the LAST stop token, so the answer and the first closing tag were counted as reasoning."),
      20, A, scaffold="    stop = r == stop_id\n    return t.where(stop, t.arange(len(r)), len(r))._____().item()"),
    D("f2", "faded", 2,
      "`is_end` is a boolean tensor of length V that is True for token ids that end a sentence. `reasoning` holds token ids. Return the sentence boundaries: for each sentence-ending token, the position right after it. 1-D tensor.",
      "is_end, reasoning", "Positions right after each sentence-ending token.",
      "    return t.nonzero(is_end[reasoning])[:, 0] + 1",
      [(ISE + "\nreasoning = t.tensor([0, 1, 3, 2, 2, 5, 4])", "solve(is_end, reasoning).tolist()"), (ISE + "\nreasoning = t.tensor([3, 3, 0])", "solve(is_end, reasoning).tolist()"),
       (ISE + "\nreasoning = t.tensor([0, 1, 2, 4])", "solve(is_end, reasoning).tolist()"), (ISE + "\nreasoning = t.tensor([4, 4, 4, 5])", "solve(is_end, reasoning).tolist()")],
      ("    return t.nonzero(is_end[reasoning])[:, 0]", ISE + "\nreasoning = t.tensor([0, 1, 3, 2, 2, 5, 4])",
       "These are the positions of the sentence-ending tokens themselves. Taking that many tokens stops just before each sentence ends."),
      25, AB, scaffold="    return t._____(is_end[reasoning])[:, 0] + _____"),
    # ---- solo ------------------------------------------------------------------------------------------------
    D("s1", "solo", 0,
      "`r` holds the token ids of one rollout and `stop_id` is the id that closes the reasoning. Return True if the rollout finished its reasoning (the id appears somewhere), else False.",
      "r, stop_id", "Whether the reasoning was closed.",
      "    return (r == stop_id).any().item()",
      [("r = t.tensor([10, 11, 99, 40])\nstop_id = 99", "solve(r, stop_id)"), ("r = t.tensor([10, 11, 12])\nstop_id = 99", "solve(r, stop_id)"),
       ("r = t.tensor([10, 11, 12, 99])\nstop_id = 99", "solve(r, stop_id)"), ("r = t.tensor([1, 2])\nstop_id = 3", "solve(r, stop_id)")],
      ("    return (r[-1] == stop_id).item()", "r = t.tensor([10, 11, 99, 40])\nstop_id = 99",
       "This looked only at the last token. A finished rollout keeps going after the closing tag, with the answer and then padding."),
      30, A),
    D("s2", "solo", 0,
      "`r` holds the token ids of one rollout. The reasoning ends at the first token equal to `end_id` or to `pad_id`, whichever comes first; if neither appears it is the whole of `r`. Return the reasoning tokens.",
      "r, end_id, pad_id", "Tokens before the first of two stop ids.",
      "    stop = (r == end_id) | (r == pad_id)\n    n = t.where(stop, t.arange(len(r)), len(r)).min().item()\n    return r[:n]",
      [("r = t.tensor([10, 11, 99, 40, 0, 0])\nend_id = 99\npad_id = 0", "solve(r, end_id, pad_id).tolist()"), ("r = t.tensor([10, 11, 12, 0, 0])\nend_id = 99\npad_id = 0", "solve(r, end_id, pad_id).tolist()"),
       ("r = t.tensor([10, 11, 12])\nend_id = 99\npad_id = 0", "solve(r, end_id, pad_id).tolist()"), ("r = t.tensor([99, 5])\nend_id = 99\npad_id = 0", "solve(r, end_id, pad_id).tolist()")],
      ("    stop = r == end_id\n    n = t.where(stop, t.arange(len(r)), len(r)).min().item()\n    return r[:n]", "r = t.tensor([10, 11, 12, 0, 0])\nend_id = 99\npad_id = 0",
       "Only the closing tag was treated as a stop. This rollout has none, so the padding at its end was returned as if the model had reasoned it."),
      45, A, show=1),
    D("s3", "solo", 0,
      "`R` has shape `(k, T)`: k rollouts of T token ids each. `stop_id` is the id that closes the reasoning. Return the number of reasoning tokens in each rollout (T for a rollout in which the id never appears), shape `(k,)`.",
      "R, stop_id", "Reasoning length of every rollout.",
      "    T = len(R[0])\n    return einops.reduce(t.where(R == stop_id, t.arange(T), T), \"rollout pos -> rollout\", \"min\")",
      [("R = t.tensor([[10, 99, 40, 99], [10, 11, 12, 13], [99, 0, 0, 0]])\nstop_id = 99", "solve(R, stop_id).tolist()"),
       ("R = t.tensor([[1, 2, 3], [3, 2, 1]])\nstop_id = 3", "solve(R, stop_id).tolist()"),
       ("R = t.tensor([[5, 5, 5, 5, 9]])\nstop_id = 9", "solve(R, stop_id).tolist()"),
       ("R = t.tensor([[1, 1], [1, 2], [2, 1]])\nstop_id = 2", "solve(R, stop_id).tolist()")],
      ("    return (R != stop_id).sum(dim=1)", "R = t.tensor([[10, 99, 40, 99], [10, 11, 12, 13], [99, 0, 0, 0]])\nstop_id = 99",
       "This counted every token that is not the closing tag, including the ones after it. The reasoning is only the part before the first closing tag."),
      50, A),
    D("s4", "solo", 0,
      "`is_end` is a boolean tensor of length V that is True for token ids that end a sentence. `reasoning` holds token ids. Return the number of sentences that end inside the reasoning, as a Python int.",
      "is_end, reasoning", "How many sentence-ending tokens the reasoning holds.",
      "    return is_end[reasoning].sum().item()",
      [(ISE + "\nreasoning = t.tensor([0, 1, 3, 2, 2, 5, 4])", "solve(is_end, reasoning)"), (ISE + "\nreasoning = t.tensor([3, 3, 3])", "solve(is_end, reasoning)"),
       (ISE + "\nreasoning = t.tensor([0, 1, 2])", "solve(is_end, reasoning)"), (ISE + "\nreasoning = t.tensor([5])", "solve(is_end, reasoning)")],
      ("    return is_end.sum().item()", ISE + "\nreasoning = t.tensor([3, 3, 3])",
       "This counted how many tokens of the vocabulary can end a sentence. The question is about the tokens that occur in this reasoning, and one of them may occur several times or not at all."),
      30, AB, show=1),
    D("s5", "solo", 0,
      "`is_end` is a boolean tensor of length V that is True for token ids that end a sentence. `reasoning` holds token ids and contains at least one sentence-ending token. Return the reasoning tokens up to and including the LAST sentence-ending token.",
      "is_end, reasoning", "The reasoning with its unfinished last sentence removed.",
      "    bounds = t.nonzero(is_end[reasoning])[:, 0] + 1\n    return reasoning[:bounds[-1]]",
      [(ISE + "\nreasoning = t.tensor([0, 1, 3, 2, 2, 5, 4])", "solve(is_end, reasoning).tolist()"), (ISE + "\nreasoning = t.tensor([3, 0, 0])", "solve(is_end, reasoning).tolist()"),
       (ISE + "\nreasoning = t.tensor([1, 5, 2, 3])", "solve(is_end, reasoning).tolist()"), (ISE + "\nreasoning = t.tensor([4, 4, 5, 1, 1])", "solve(is_end, reasoning).tolist()")],
      ("    bounds = t.nonzero(is_end[reasoning])[:, 0]\n    return reasoning[:bounds[-1]]", ISE + "\nreasoning = t.tensor([0, 1, 3, 2, 2, 5, 4])",
       "The cut was made at the sentence-ending token's own position, which leaves that token out. The last sentence then has no ending."),
      45, AB),
    D("s6", "solo", 0,
      "`is_end` is a boolean tensor of length V that is True for token ids that end a sentence. `reasoning` holds token ids and `b` is a sentence boundary in it. Return the id of the token that ended the sentence at that boundary, as a Python int.",
      "is_end, reasoning, b", "The sentence-ending token behind a boundary.",
      "    return reasoning[b - 1].item()",
      [(ISE + "\nreasoning = t.tensor([0, 1, 3, 2, 2, 5, 4])\nb = 3", "solve(is_end, reasoning, b)"), (ISE + "\nreasoning = t.tensor([0, 1, 3, 2, 2, 5, 4])\nb = 6", "solve(is_end, reasoning, b)"),
       (ISE + "\nreasoning = t.tensor([5, 0, 3])\nb = 1", "solve(is_end, reasoning, b)"), (ISE + "\nreasoning = t.tensor([5, 0, 3, 1])\nb = 3", "solve(is_end, reasoning, b)")],
      ("    return reasoning[b].item()", ISE + "\nreasoning = t.tensor([0, 1, 3, 2, 2, 5, 4])\nb = 3",
       "A boundary points at the first token of the NEXT sentence. The token that ended the sentence is the one before it."),
      35, AB),
    # ---- integrated ------------------------------------------------------------------------------------------
    D("i1", "integrated", 0,
      "`r` holds the token ids of one rollout and `stop_id` closes the reasoning (it may not appear). `is_end` is a boolean tensor of length V that is True for token ids that end a sentence. Return the sentence boundaries of the reasoning only: positions right after each sentence-ending token that comes before the first `stop_id`. 1-D tensor.",
      "r, stop_id, is_end", "Sentence boundaries inside the reasoning part of a rollout.",
      "    n = t.where(r == stop_id, t.arange(len(r)), len(r)).min().item()\n    return t.nonzero(is_end[r[:n]])[:, 0] + 1",
      [(ISE + "\nr = t.tensor([0, 3, 1, 5, 4, 2, 3, 3])\nstop_id = 4", "solve(r, stop_id, is_end).tolist()"), (ISE + "\nr = t.tensor([0, 3, 1, 5, 1, 2, 3, 3])\nstop_id = 4", "solve(r, stop_id, is_end).tolist()"),
       (ISE + "\nr = t.tensor([4, 3, 3])\nstop_id = 4", "solve(r, stop_id, is_end).tolist()"), (ISE + "\nr = t.tensor([1, 1, 5, 4, 5])\nstop_id = 4", "solve(r, stop_id, is_end).tolist()")],
      ("    return t.nonzero(is_end[r])[:, 0] + 1", ISE + "\nr = t.tensor([0, 3, 1, 5, 4, 2, 3, 3])\nstop_id = 4",
       "Sentences of the answer, after the closing tag, were counted as reasoning. The monitor is about what the model thinks before it answers."),
      65, ABC),
    D("i2", "integrated", 0,
      "`R` has shape `(k, T)`: k rollouts of T token ids. `stop_id` closes the reasoning (it may be absent from a row). `is_end` is a boolean tensor of length V that is True for token ids that end a sentence. Return the number of sentence boundaries inside the reasoning of each rollout, shape `(k,)`.",
      "R, stop_id, is_end", "Boundaries per rollout, counting the reasoning only.",
      "    T = len(R[0])\n    n = einops.reduce(t.where(R == stop_id, t.arange(T), T), \"rollout pos -> rollout\", \"min\")\n    inside = t.arange(T)[None, :] < n[:, None]\n    return (is_end[R] & inside).sum(dim=1)",
      [(ISE + "\nR = t.tensor([[0, 3, 4, 5, 5], [3, 3, 3, 0, 0], [4, 3, 3, 3, 3]])\nstop_id = 4", "solve(R, stop_id, is_end).tolist()"),
       (ISE + "\nR = t.tensor([[1, 5, 1, 5], [5, 4, 5, 5]])\nstop_id = 4", "solve(R, stop_id, is_end).tolist()"),
       (ISE + "\nR = t.tensor([[0, 1, 2]])\nstop_id = 4", "solve(R, stop_id, is_end).tolist()"),
       (ISE + "\nR = t.tensor([[3, 4], [4, 3], [3, 3]])\nstop_id = 4", "solve(R, stop_id, is_end).tolist()")],
      ("    return is_end[R].sum(dim=1)", ISE + "\nR = t.tensor([[0, 3, 4, 5, 5], [3, 3, 3, 0, 0], [4, 3, 3, 3, 3]])\nstop_id = 4",
       "Sentence endings after the closing tag were counted too. A rollout that answered at length looks as if it had reasoned at length."),
      75, ABC),
    D("i3", "integrated", 0,
      "`r` holds the token ids of one rollout and `stop_id` closes the reasoning (it may not appear). `is_end` is a boolean tensor of length V that is True for token ids that end a sentence. Return the reasoning tokens before the FIRST sentence boundary, that is, the first sentence of the reasoning. If the reasoning has no sentence-ending token, return all of the reasoning.",
      "r, stop_id, is_end", "First sentence of the reasoning.",
      "    n = t.where(r == stop_id, t.arange(len(r)), len(r)).min().item()\n    cot = r[:n]\n    b = t.where(is_end[cot], t.arange(n) + 1, n).min().item()\n    return cot[:b]",
      [(ISE + "\nr = t.tensor([0, 1, 3, 2, 5, 4, 3])\nstop_id = 4", "solve(r, stop_id, is_end).tolist()"), (ISE + "\nr = t.tensor([0, 1, 2, 4, 3])\nstop_id = 4", "solve(r, stop_id, is_end).tolist()"),
       (ISE + "\nr = t.tensor([5, 1, 2])\nstop_id = 4", "solve(r, stop_id, is_end).tolist()"), (ISE + "\nr = t.tensor([1, 2, 0])\nstop_id = 4", "solve(r, stop_id, is_end).tolist()")],
      ("    b = t.where(is_end[r], t.arange(len(r)) + 1, len(r)).min().item()\n    return r[:b]", ISE + "\nr = t.tensor([0, 1, 2, 4, 3])\nstop_id = 4",
       "The search for a sentence ending ran over the whole rollout. This reasoning has no finished sentence, so the first one found was in the answer, and the closing tag came back as part of the reasoning."),
      75, ABC, show=1),
]
