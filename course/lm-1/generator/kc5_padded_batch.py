"""monitor.padded-batch: many forced sequences of different lengths in one pass."""
from framework import maker

KC = "monitor.padded-batch"
A = [("lm-padded-batch", 1.0)]
AF = [("lm-padded-batch", 1.0), ("lm-forced-logprob", 0.5)]
AS = [("lm-padded-batch", 1.0), ("lm-forced-sequence", 0.5)]
SPEC = dict(
    id=KC, title="Score many forced sequences in one padded batch",
    concepts=["right-padding-and-mask", "per-row-phrase-rows"],
    supporting=["monitor.forced-logprob", "monitor.forced-sequence", "torch.constructors", "torch.slice-assignment",
                "torch.broadcasting-rules", "python.control-flow"],
)

SEGMENTS = [
    dict(title="Pad on the right so every position keeps its index", concept="""
One prompt gives several rollouts, and each rollout has many sentence boundaries. Every boundary needs its own forced sequence, so a single prompt can need a few hundred sequences. Running the model once per sequence is slow. A graphics card is fast when it is given many sequences at once, stacked as the rows of one table: a *batch*.

A table needs rows of equal length, and forced sequences differ in length because each holds a different amount of reasoning. The fix is *padding*: short rows are filled up with a filler id, the *padding token*, until they match the longest row.

The filler goes on the RIGHT. A language model of this kind only looks backwards: the logits row at position p depends on positions 0 to p and nothing later. Filler placed after the real tokens therefore cannot change any row that scores a real token, and a phrase that starts at position `start` still starts at `start`. Filler on the left would shift every position.

The model also takes a second table of the same shape, the *attention mask*: True where a real token sits and False on filler. For row i with length $\\ell_i$ and column j:

$$\\text{mask}[i, j] = (j < \\ell_i)$$

With the Hugging Face library both tables go into one call, shown for reading only:

```python no-run
logits = model(input_ids=ids,
               attention_mask=mask).logits
```
""", worked="""
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
"""),
    dict(title="Each row of the batch has its own phrase position", concept="""
After the batch has been through the model, the logits have shape `(b, n, V)`: b sequences, n positions, V vocabulary entries. The phrase sits at a different place in every row, because each row holds a different amount of reasoning before it. So there is one start per row, a tensor `starts` of shape `(b,)`.

For one sequence the rows that score an m-token phrase were `logits[start - 1 : start - 1 + m]`. A slice cannot change from row to row, so the batch version spells out the positions as a table of shape `(b, m)`:

$$\\text{pos}[i, j] = \\text{starts}[i] - 1 + j$$

and pairs it with a column of row numbers, shape `(b, 1)`. Indexing with the two together picks, for every row i and every phrase token j, the logits row at position `pos[i, j]` of sequence i. The result has shape `(b, m, V)`: for each sequence, exactly the rows that score its phrase. From there the steps are the ones already known: log-probabilities over the vocabulary, one entry per phrase token, and a sum over the token axis.
""", worked="""
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
"""),
]

MISCONCEPTIONS = [
    ("Padding on either side is the same.", "Filler on the left shifts every position. On the right, positions and the rows that score real tokens are untouched."),
    ("The last column of the batch scores each sequence's next token.", "Only for the longest row. For a shorter row the last column sits on filler."),
    ("One start position serves the whole batch.", "Each row holds a different amount of reasoning, so each row has its own start."),
    ("The mask is True on the padding.", "It is True on real tokens and False on filler."),
]

D = maker("pb", KC)
SQ = "seqs = [t.tensor([5, 6, 7]), t.tensor([8])]\npad = 0"
SQ2 = "seqs = [t.tensor([1]), t.tensor([2, 3]), t.tensor([4, 5, 6, 7])]\npad = 9"
LG = "logits = t.arange(30.).reshape(2, 5, 3)"
LG3 = "logits = t.arange(24.).reshape(3, 4, 2)"
LS = "logits = t.tensor([[[0., 1.], [2., 0.], [0., 3.], [1., 1.]], [[1., 1.], [0., 4.], [5., 0.], [0., 2.]]])"
LS3 = "logits = t.tensor([[[0., 1., 5.], [2., 0., 1.], [0., 3., 0.], [1., 1., 2.]], [[1., 1., 0.], [0., 4., 1.], [5., 0., 3.], [0., 2., 2.]]])"
PAD = """    lens = t.tensor([len(q) for q in seqs])
    out = t.full((len(seqs), lens.max().item()), pad)
    i = 0
    for q in seqs:
        out[i, {sl}] = q
        i += 1
    return out"""
SCORE = """    b = len(ids)
    rows = t.arange(b)[:, None]
    pos = starts[:, None] + t.arange(m)[None, :]
    logp = t.log_softmax(logits, dim=-1)
    per_token = logp[rows, {p}, ids[rows, pos]]
    return einops.reduce(per_token, "batch tok -> batch", "sum")"""
DRILLS = [
    D("f1", "faded", 1,
      "`seqs` is a list of 1-D tensors of token ids with different lengths. Return one table with a row per sequence, as wide as the longest, in which every sequence keeps its positions and the unused entries hold `pad`.",
      "seqs, pad", "Sequences stacked into one table, filler on the right.",
      PAD.format(sl=":len(q)"),
      [(SQ, "solve(seqs, pad).tolist()"), (SQ2, "solve(seqs, pad).tolist()"),
       ("seqs = [t.tensor([1, 2]), t.tensor([3, 4])]\npad = 0", "solve(seqs, pad).tolist()"), ("seqs = [t.tensor([4]), t.tensor([3, 2, 1])]\npad = 7", "solve(seqs, pad).tolist()")],
      (PAD.format(sl="-len(q):"), SQ,
       "The filler went on the left, so the short sequence's token moved from position 0 to position 2. Every position computed for that sequence beforehand now points at the wrong place."),
      25, A, scaffold=PAD.format(sl=":_____")),
    D("f2", "faded", 2,
      "`logits` has shape `(b, n, V)`. In sequence i a phrase of `m` tokens begins at position `starts[i]`. Return the rows that score each sequence's phrase, shape `(b, m, V)`.",
      "logits, starts, m", "Per-sequence rows that score the phrase.",
      "    b = len(starts)\n    return logits[t.arange(b)[:, None], (starts - 1)[:, None] + t.arange(m)[None, :]]",
      [(LG + "\nstarts = t.tensor([1, 3])\nm = 2", "solve(logits, starts, m).tolist()"), (LG + "\nstarts = t.tensor([2, 2])\nm = 1", "solve(logits, starts, m).tolist()"),
       (LG3 + "\nstarts = t.tensor([1, 2, 3])\nm = 1", "solve(logits, starts, m).tolist()"), (LG3 + "\nstarts = t.tensor([2, 1, 1])\nm = 2", "solve(logits, starts, m).tolist()")],
      ("    b = len(starts)\n    return logits[t.arange(b)[:, None], starts[:, None] + t.arange(m)[None, :]]", LG + "\nstarts = t.tensor([1, 3])\nm = 2",
       "These are the rows at each phrase's own positions. Every one of them scores the token after it, so each sequence's window is one position late."),
      30, A, scaffold="    b = len(starts)\n    return logits[t.arange(b)[:, None], (starts - _____)[:, None] + t.arange(m)[None, :]]"),
    # ---- solo ------------------------------------------------------------------------------------------------
    D("s1", "solo", 0,
      "`lens` holds the length of each of b sequences and `width` is the width of the padded table. Return the attention mask, shape `(b, width)`: True where a real token sits, False on filler.",
      "lens, width", "Attention mask from the lengths.",
      "    return t.arange(width)[None, :] < lens[:, None]",
      [("lens = t.tensor([3, 1])\nwidth = 3", "solve(lens, width).tolist()"), ("lens = t.tensor([1, 2, 4])\nwidth = 4", "solve(lens, width).tolist()"),
       ("lens = t.tensor([2, 2])\nwidth = 5", "solve(lens, width).tolist()"), ("lens = t.tensor([0, 3])\nwidth = 3", "solve(lens, width).tolist()")],
      ("    return t.arange(width)[None, :] <= lens[:, None]", "lens = t.tensor([3, 1])\nwidth = 3",
       "A sequence of length 1 has one real token, at position 0. This also marked position 1, the first piece of filler, as real."),
      35, A),
    D("s2", "solo", 0,
      "`ids` is a padded table of token ids, shape `(b, n)`, and `pad` is the filler id, which no real token uses. Return the number of real tokens in each row, shape `(b,)`.",
      "ids, pad", "Length of each sequence in a padded table.",
      "    return (ids != pad).sum(dim=1)",
      [("ids = t.tensor([[5, 6, 7], [8, 0, 0]])\npad = 0", "solve(ids, pad).tolist()"), ("ids = t.tensor([[1, 9, 9, 9], [2, 3, 9, 9], [4, 5, 6, 7]])\npad = 9", "solve(ids, pad).tolist()"),
       ("ids = t.tensor([[1, 2], [3, 4]])\npad = 0", "solve(ids, pad).tolist()"), ("ids = t.tensor([[4, 7, 7], [3, 2, 7]])\npad = 7", "solve(ids, pad).tolist()")],
      ("    return (ids != pad).sum(dim=0)", "ids = t.tensor([[5, 6, 7], [8, 0, 0]])\npad = 0",
       "This counted down each column, giving how many sequences reach each position. A sequence's length is a count along its own row."),
      30, A),
    D("s3", "solo", 0,
      "`logits` has shape `(b, n, V)` and came from a right-padded batch. `lens` holds the number of real tokens in each sequence. For every sequence, return the row of scores for the token that would come NEXT after its last real token. Shape `(b, V)`.",
      "logits, lens", "Next-token scores for each sequence of a padded batch.",
      "    b = len(lens)\n    return logits[t.arange(b), lens - 1]",
      [(LG + "\nlens = t.tensor([5, 2])", "solve(logits, lens).tolist()"), (LG + "\nlens = t.tensor([1, 4])", "solve(logits, lens).tolist()"),
       (LG3 + "\nlens = t.tensor([4, 1, 2])", "solve(logits, lens).tolist()"), (LG3 + "\nlens = t.tensor([3, 3, 3])", "solve(logits, lens).tolist()")],
      ("    return logits[:, -1]", LG + "\nlens = t.tensor([5, 2])",
       "The last column is right only for the longest sequence. For the short one it sits on filler, and the model's output there is a prediction about padding."),
      40, A),
    D("s4", "solo", 0,
      "`logp` has shape `(b, n, V)` and already holds log-probabilities. In sequence i a phrase begins at position `starts[i]`, and `phrases`, shape `(b, m)`, holds each sequence's m phrase token ids. Return the log-probability of every phrase token, shape `(b, m)`.",
      "logp, starts, phrases", "Log-probability of each phrase token in each sequence.",
      "    b = len(starts)\n    m = len(phrases[0])\n    pos = (starts - 1)[:, None] + t.arange(m)[None, :]\n    return logp[t.arange(b)[:, None], pos, phrases]",
      [(LG.replace("logits", "logp") + "\nstarts = t.tensor([1, 3])\nphrases = t.tensor([[2, 0], [1, 1]])", "solve(logp, starts, phrases).tolist()"),
       (LG.replace("logits", "logp") + "\nstarts = t.tensor([2, 1])\nphrases = t.tensor([[0], [2]])", "solve(logp, starts, phrases).tolist()"),
       (LG3.replace("logits", "logp") + "\nstarts = t.tensor([1, 2, 3])\nphrases = t.tensor([[0], [1], [0]])", "solve(logp, starts, phrases).tolist()"),
       (LG3.replace("logits", "logp") + "\nstarts = t.tensor([2, 2, 1])\nphrases = t.tensor([[1, 1], [0, 1], [1, 0]])", "solve(logp, starts, phrases).tolist()")],
      ("    b = len(starts)\n    m = len(phrases[0])\n    pos = (starts[0] - 1) + t.arange(m)[None, :]\n    return logp[t.arange(b)[:, None], pos, phrases]",
       LG.replace("logits", "logp") + "\nstarts = t.tensor([1, 3])\nphrases = t.tensor([[2, 0], [1, 1]])",
       "The first sequence's start was used for every row. The second sequence has more reasoning before its phrase, so its tokens were looked up in the middle of that reasoning."),
      55, AF),
    D("s5", "solo", 0,
      "`befores` is a list of 1-D tensors of token ids with different lengths, and `phrase` is a 1-D tensor of token ids. Return one table in which row i is sequence i followed directly by the phrase, with `pad` in the unused entries on the right.",
      "befores, phrase, pad", "A padded batch of forced sequences.",
      "    lens = t.tensor([len(q) for q in befores])\n    m = len(phrase)\n    out = t.full((len(befores), lens.max().item() + m), pad)\n    i = 0\n    for q in befores:\n        out[i, :len(q)] = q\n        out[i, len(q):len(q) + m] = phrase\n        i += 1\n    return out",
      [("befores = [t.tensor([5, 6, 7]), t.tensor([8])]\nphrase = t.tensor([1, 2])\npad = 0", "solve(befores, phrase, pad).tolist()"),
       ("befores = [t.tensor([1]), t.tensor([2, 3]), t.tensor([4, 5, 6])]\nphrase = t.tensor([8])\npad = 9", "solve(befores, phrase, pad).tolist()"),
       ("befores = [t.tensor([1, 2]), t.tensor([3, 4])]\nphrase = t.tensor([7, 7])\npad = 0", "solve(befores, phrase, pad).tolist()"),
       ("befores = [t.tensor([4]), t.tensor([3, 2, 1, 5])]\nphrase = t.tensor([6, 6, 6])\npad = 0", "solve(befores, phrase, pad).tolist()")],
      ("    lens = t.tensor([len(q) for q in befores])\n    m = len(phrase)\n    out = t.full((len(befores), lens.max().item() + m), pad)\n    i = 0\n    for q in befores:\n        out[i, :len(q)] = q\n        out[i, -m:] = phrase\n        i += 1\n    return out",
       "befores = [t.tensor([5, 6, 7]), t.tensor([8])]\nphrase = t.tensor([1, 2])\npad = 0",
       "The phrase was written at the end of every row. In the short row that leaves filler between the text and the phrase, so the model reads the phrase after a run of padding instead of after the reasoning."),
      55, AS),
    D("s6", "solo", 0,
      "`vals` has shape `(b, n)` and holds one number per position of a padded batch. `mask`, same shape, is True where a real token sits. Return the sum of each row's numbers over its real positions only, shape `(b,)`.",
      "vals, mask", "Row sums that ignore filler positions.",
      "    return t.where(mask, vals, 0.0).sum(dim=1)",
      [("vals = t.tensor([[1., 2., 3.], [4., 5., 6.]])\nmask = t.tensor([[True, True, True], [True, False, False]])", "solve(vals, mask).tolist()"),
       ("vals = t.tensor([[-1., -2.], [-3., -4.], [-5., -6.]])\nmask = t.tensor([[True, False], [True, True], [False, False]])", "solve(vals, mask).tolist()"),
       ("vals = t.tensor([[2., 2., 2., 2.]])\nmask = t.tensor([[True, True, False, False]])", "solve(vals, mask).tolist()"),
       ("vals = t.tensor([[1., 1.], [1., 1.]])\nmask = t.tensor([[True, True], [True, True]])", "solve(vals, mask).tolist()")],
      ("    return vals.sum(dim=1)", "vals = t.tensor([[1., 2., 3.], [4., 5., 6.]])\nmask = t.tensor([[True, True, True], [True, False, False]])",
       "The numbers the model produced on filler positions were added in. Those numbers are about padding, and a shorter row has more of them."),
      40, A),
    # ---- integrated ------------------------------------------------------------------------------------------
    D("i1", "integrated", 0,
      "`ids` has shape `(b, n)` and is a right-padded batch of forced sequences; `logits`, shape `(b, n, V)`, is the model's raw output for it. In sequence i a phrase of `m` tokens begins at position `starts[i]`. Return the log-probability of each sequence's own phrase given the text before it, shape `(b,)`.",
      "logits, ids, starts, m", "Phrase log-probability for each row of a padded batch.",
      SCORE.format(p="pos - 1"),
      [(LS + "\nids = t.tensor([[0, 1, 0, 0], [1, 1, 0, 1]])\nstarts = t.tensor([1, 2])\nm = 2", "solve(logits, ids, starts, m).tolist()"),
       (LS + "\nids = t.tensor([[0, 0, 1, 1], [1, 0, 1, 0]])\nstarts = t.tensor([3, 1])\nm = 1", "solve(logits, ids, starts, m).tolist()"),
       (LS + "\nids = t.tensor([[1, 1, 1, 0], [0, 0, 0, 0]])\nstarts = t.tensor([1, 1])\nm = 3", "solve(logits, ids, starts, m).tolist()"),
       ("logits = t.zeros(3, 4, 5)\nids = t.tensor([[0, 1, 2, 3], [4, 4, 0, 0], [1, 0, 1, 0]])\nstarts = t.tensor([2, 1, 3])\nm = 1", "solve(logits, ids, starts, m).tolist()")],
      (SCORE.format(p="pos"), LS + "\nids = t.tensor([[0, 1, 0, 0], [1, 1, 0, 1]])\nstarts = t.tensor([1, 2])\nm = 2",
       "Each phrase token was looked up in the logits row at its own position. That row predicts the token after it, so every term of the sum is the probability of the wrong event."),
      80, AF),
    D("i2", "integrated", 0,
      "`prefix`, `reasoning` and `phrase` are 1-D tensors of token ids, and `bounds` is a 1-D tensor of sentence boundaries in the reasoning. Return a right-padded table with one forced sequence per boundary: the prefix, the reasoning up to that boundary, then the phrase. Unused entries hold `pad`.",
      "prefix, reasoning, phrase, bounds, pad", "One padded batch with a forced sequence per boundary.",
      "    width = len(prefix) + bounds.max().item() + len(phrase)\n    out = t.full((len(bounds), width), pad)\n    i = 0\n    for b in bounds.tolist():\n        q = t.cat([prefix, reasoning[:b], phrase])\n        out[i, :len(q)] = q\n        i += 1\n    return out",
      [("prefix = t.tensor([90, 91])\nreasoning = t.tensor([10, 11, 12, 13])\nphrase = t.tensor([70, 71])\nbounds = t.tensor([1, 3])\npad = 0", "solve(prefix, reasoning, phrase, bounds, pad).tolist()"),
       ("prefix = t.tensor([90, 91])\nreasoning = t.tensor([10, 11, 12, 13])\nphrase = t.tensor([70, 71])\nbounds = t.tensor([4, 2, 1])\npad = 0", "solve(prefix, reasoning, phrase, bounds, pad).tolist()"),
       ("prefix = t.tensor([5])\nreasoning = t.tensor([1, 2, 3])\nphrase = t.tensor([8])\nbounds = t.tensor([3])\npad = 9", "solve(prefix, reasoning, phrase, bounds, pad).tolist()"),
       ("prefix = t.tensor([5])\nreasoning = t.tensor([1, 2, 3])\nphrase = t.tensor([8, 8, 8])\nbounds = t.tensor([1, 2])\npad = 9", "solve(prefix, reasoning, phrase, bounds, pad).tolist()")],
      ("    width = len(prefix) + bounds.max().item() + len(phrase)\n    out = t.full((len(bounds), width), pad)\n    i = 0\n    for b in bounds.tolist():\n        q = t.cat([prefix, reasoning[:b]])\n        out[i, :len(q)] = q\n        i += 1\n    return out",
       "prefix = t.tensor([90, 91])\nreasoning = t.tensor([10, 11, 12, 13])\nphrase = t.tensor([70, 71])\nbounds = t.tensor([1, 3])\npad = 0",
       "The rows stop at the boundary and the phrase never went in. The model can only score a phrase that is part of its input."),
      75, AS),
    D("i3", "integrated", 0,
      "`ids` has shape `(b, n)` and is a right-padded batch of forced sequences, with `pad` as the filler id (no real token uses it). `logits`, shape `(b, n, V)`, is the model's raw output. In every sequence the phrase is the LAST `m` real tokens. Return the log-probability of each sequence's phrase given the text before it, shape `(b,)`.",
      "logits, ids, pad, m", "Phrase log-probability when the phrase ends each sequence.",
      "    starts = (ids != pad).sum(dim=1) - m\n" + SCORE.format(p="pos - 1"),
      [(LS3 + "\nids = t.tensor([[0, 1, 0, 2], [1, 1, 0, 1]])\npad = 2\nm = 2", "solve(logits, ids, pad, m).tolist()"),
       (LS3 + "\nids = t.tensor([[0, 0, 2, 2], [1, 0, 1, 2]])\npad = 2\nm = 1", "solve(logits, ids, pad, m).tolist()"),
       (LS3 + "\nids = t.tensor([[1, 1, 1, 0], [0, 0, 0, 0]])\npad = 2\nm = 3", "solve(logits, ids, pad, m).tolist()"),
       ("logits = t.arange(24.).reshape(2, 4, 3)\nids = t.tensor([[0, 1, 1, 2], [1, 0, 2, 2]])\npad = 2\nm = 1", "solve(logits, ids, pad, m).tolist()")],
      ("    starts = len(ids[0]) - m + 0 * ids[:, 0]\n" + SCORE.format(p="pos - 1"), LS3 + "\nids = t.tensor([[0, 1, 0, 2], [1, 1, 0, 1]])\npad = 2\nm = 2",
       "The phrase was assumed to end at the last column. In a shorter row the last columns are filler, so the score there is about padding tokens."),
      85, AF),
]
