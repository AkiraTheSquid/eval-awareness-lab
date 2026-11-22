"""monitor.next-token-logits: which row of the logits scores which token."""
from framework import maker

KC = "monitor.next-token-logits"
A = [("lm-next-token-row", 1.0)]
AI = [("lm-next-token-row", 1.0), ("index-by-tensor", 0.5)]
SPEC = dict(
    id=KC, title="Row p of the logits scores the token at position p + 1",
    concepts=["row-predicts-next", "rows-for-a-span"],
    supporting=["tensor.indexed-selection", "torch.argmin-argmax", "torch.slicing-views"],
)

SEGMENTS = [
    dict(title="Each row of the logits is about the NEXT token", concept="""
A language model reads text as a list of whole numbers called *token ids*: each id stands for one piece of text, such as a word or a part of a word. The set of all ids the model knows is its *vocabulary*, of size V.

Give the model n token ids and it returns a table of scores called *logits*, with shape `(n, V)`: one row per position and one column per vocabulary entry. A higher score means the model finds that token more likely.

The row at position p was computed from the tokens at positions 0 up to p, and it scores what the token at position p + 1 could be. So the score the model gave to the token that actually sits at position p is in row p − 1, in the column of that token's id:

$$\\text{score of the token at position } p = \\text{logits}[p - 1,\\ \\text{ids}[p]]$$

Two consequences follow. The token at position 0 has no score, because no row comes before it. And the last row scores a token that is not in the input at all: the one the model would write next.
""", worked="""
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
"""),
    dict(title="A span of m tokens is scored by m consecutive rows, starting one row early", concept="""
The monitor in this unit does not score one token. It scores a whole phrase that was placed at the end of the text, for example the tokens of "The user might be testing". Call the position of the phrase's first token `start` and the number of phrase tokens m.

Apply the one-row-back rule to each phrase token in turn. The first phrase token, at position `start`, is scored by row `start − 1`. The last one, at position `start + m − 1`, is scored by row `start + m − 2`. So the rows that score the phrase are the m rows

$$\\text{logits}[\\text{start} - 1 : \\text{start} - 1 + m]$$

The slice begins one row before the phrase and stops one row before the phrase ends. The final row of the table is never used when the phrase is the last thing in the text, because that row scores a token after the phrase.
""", worked="""
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
"""),
]

MISCONCEPTIONS = [
    ("Row p scores the token at position p.", "Row p has already read the token at position p. It scores the token at position p + 1."),
    ("The rows for a phrase are the rows at the phrase's own positions.", "They start one row before the phrase and stop one row before its end."),
    ("The last row scores the last token.", "The last row scores a token that is not in the input, the one that would come next."),
    ("The first token has a score in row 0.", "Nothing comes before position 0, so no row scores it. Row 0 scores position 1."),
]

D = maker("nt", KC)
L4 = "logits = t.tensor([[0., 1., 2.], [3., .5, 1.], [.2, 4., .1], [1., 1., 5.]])"
L5 = "logits = t.tensor([[1., 0.], [0., 2.], [3., 0.], [0., 4.], [5., 0.]])"
B2 = "logits = t.tensor([[[1., 2.], [3., 4.], [5., 6.]], [[0., 9.], [8., 0.], [7., 7.]]])"
DRILLS = [
    D("f1", "faded", 1,
      "`logits` has shape `(n, V)` and came from one pass of a language model over n token ids. Return the row of scores for the token at position `p`, shape `(V,)`. `p` is at least 1.",
      "logits, p", "Scores for the token at position p.",
      "    return logits[p - 1]",
      [(L4 + "\np = 1", "solve(logits, p).tolist()"), (L4 + "\np = 3", "solve(logits, p).tolist()"),
       (L5 + "\np = 2", "solve(logits, p).tolist()"), (L5 + "\np = 4", "solve(logits, p).tolist()")],
      ("    return logits[p]", L4 + "\np = 1",
       "This is the row computed after reading position p, so it scores position p + 1. The scores for position p are one row earlier."),
      15, A, scaffold="    return logits[p - _____]"),
    D("f2", "faded", 2,
      "`logits` has shape `(n, V)`. A phrase of `m` tokens begins at position `start`. Return the rows that score the phrase tokens, shape `(m, V)`.",
      "logits, start, m", "Rows that score a phrase of m tokens beginning at start.",
      "    return logits[start - 1:start - 1 + m]",
      [(L5 + "\nstart = 2\nm = 2", "solve(logits, start, m).tolist()"), (L5 + "\nstart = 1\nm = 3", "solve(logits, start, m).tolist()"),
       (L4 + "\nstart = 3\nm = 1", "solve(logits, start, m).tolist()"), (L4 + "\nstart = 1\nm = 3", "solve(logits, start, m).tolist()")],
      ("    return logits[start:start + m]", L5 + "\nstart = 2\nm = 2",
       "These are the rows at the phrase's own positions. Each of them scores the token after it, so the whole window is one row too late."),
      20, A, scaffold="    return logits[start - _____:start - _____ + m]"),
    # ---- solo ------------------------------------------------------------------------------------------------
    D("s1", "solo", 0,
      "`ids` holds n token ids and `logits`, shape `(n, V)`, is the model's output for them. Return the score the model gave to the token that actually sits at position `p`, as a Python float. `p` is at least 1.",
      "logits, ids, p", "Score of the actual token at position p.",
      "    return logits[p - 1, ids[p]].item()",
      [(L4 + "\nids = t.tensor([2, 0, 1, 1])\np = 2", "solve(logits, ids, p)"), (L4 + "\nids = t.tensor([2, 0, 1, 1])\np = 1", "solve(logits, ids, p)"),
       (L4 + "\nids = t.tensor([0, 2, 2, 0])\np = 3", "solve(logits, ids, p)"), (L5 + "\nids = t.tensor([1, 1, 0, 0, 1])\np = 4", "solve(logits, ids, p)")],
      ("    return logits[p, ids[p]].item()", L4 + "\nids = t.tensor([2, 0, 1, 1])\np = 2",
       "Row p was computed after the model had already read the token at position p. The score for that token comes from the row before it."),
      30, AI),
    D("s2", "solo", 0,
      "`ids` holds n token ids and `logits` has shape `(n, V)`. For every position from 1 to n − 1, find the score the model gave to the token that actually sits there. Return them in order, shape `(n − 1,)`.",
      "logits, ids", "Score of each actual token after the first.",
      "    n = len(ids)\n    return logits[t.arange(n - 1), ids[1:]]",
      [(L4 + "\nids = t.tensor([2, 0, 1, 1])", "solve(logits, ids).tolist()"), (L4 + "\nids = t.tensor([0, 2, 2, 0])", "solve(logits, ids).tolist()"),
       (L5 + "\nids = t.tensor([1, 1, 0, 0, 1])", "solve(logits, ids).tolist()"), (L5 + "\nids = t.tensor([0, 0, 0, 0, 0])", "solve(logits, ids).tolist()")],
      ("    n = len(ids)\n    return logits[t.arange(n - 1), ids[:-1]]", L4 + "\nids = t.tensor([2, 0, 1, 1])",
       "This read each row at the column of the token the row had just seen. A row scores the token after it, so the columns must come from the ids shifted by one."),
      40, AI),
    D("s3", "solo", 0,
      "`logits` has shape `(n, V)`. Return the id of the token the model found most likely for position `p`, as a Python int. `p` is at least 1.",
      "logits, p", "Most likely token id for position p.",
      "    return logits[p - 1].argmax().item()",
      [(L4 + "\np = 1", "solve(logits, p)"), (L4 + "\np = 2", "solve(logits, p)"), (L4 + "\np = 3", "solve(logits, p)"), (L5 + "\np = 3", "solve(logits, p)")],
      ("    return logits[p].argmax().item()", L4 + "\np = 1",
       "This is the model's best guess for the position after p. Its guess for position p is made one row earlier."),
      30, A),
    D("s4", "solo", 0,
      "`ids` holds n token ids and `logits` has shape `(n, V)`. Count the positions from 1 to n − 1 at which the model's most likely token is the token that actually sits there. Return a Python int.",
      "logits, ids", "How many next tokens the model's top guess got right.",
      "    return (logits[:-1].argmax(dim=1) == ids[1:]).sum().item()",
      [(L4 + "\nids = t.tensor([2, 2, 0, 1])", "solve(logits, ids)"), (L4 + "\nids = t.tensor([0, 2, 0, 1])", "solve(logits, ids)"),
       (L5 + "\nids = t.tensor([1, 0, 1, 0, 1])", "solve(logits, ids)"), (L5 + "\nids = t.tensor([0, 1, 1, 1, 1])", "solve(logits, ids)")],
      ("    return (logits.argmax(dim=1) == ids).sum().item()", L4 + "\nids = t.tensor([2, 2, 0, 1])",
       "This compared each row's guess with the token the row had already read. The guess in a row is about the token one position later."),
      45, A),
    D("s5", "solo", 0,
      "`logits` has shape `(b, n, V)`: b texts of n tokens each. For every text, return the row of scores for its LAST token (the one at position n − 1). Shape `(b, V)`.",
      "logits", "Scores for the last token of every text.",
      "    return logits[:, -2]",
      [(B2, "solve(logits).tolist()"), ("logits = t.arange(24.).reshape(2, 4, 3)", "solve(logits).tolist()"),
       ("logits = t.arange(8.).reshape(1, 2, 4)", "solve(logits).tolist()"), ("logits = t.arange(18.).reshape(3, 3, 2)", "solve(logits).tolist()")],
      ("    return logits[:, -1]", B2,
       "The final row scores a token that is not in the text yet. The scores for the last token of the text are in the row before the final one."),
      35, A),
    D("s6", "solo", 0,
      "`logits` has shape `(n, V)`. A phrase begins at position `start` and runs to the end of the text. Return the number of rows of `logits` that score phrase tokens, as a Python int.",
      "logits, start", "How many rows score the phrase.",
      "    return len(logits[start - 1:-1])",
      [(L4 + "\nstart = 1", "solve(logits, start)"), (L4 + "\nstart = 3", "solve(logits, start)"),
       (L5 + "\nstart = 2", "solve(logits, start)"), (L5 + "\nstart = 4", "solve(logits, start)")],
      ("    return len(logits[start - 1:])", L4 + "\nstart = 1",
       "This counted the final row too. That row scores a token after the phrase, so the count is one too high."),
      35, A),
    # ---- integrated ------------------------------------------------------------------------------------------
    D("i1", "integrated", 0,
      "`ids` holds n token ids and `logits` has shape `(n, V)`. A phrase begins at position `start` and runs to the end of the text. Return the score the model gave to each phrase token, in order, shape `(n − start,)`.",
      "logits, ids, start", "Score of each phrase token.",
      "    m = len(ids) - start\n    rows = logits[start - 1:start - 1 + m]\n    return rows[t.arange(m), ids[start:]]",
      [(L4 + "\nids = t.tensor([2, 0, 1, 1])\nstart = 2", "solve(logits, ids, start).tolist()"),
       (L4 + "\nids = t.tensor([0, 2, 2, 0])\nstart = 1", "solve(logits, ids, start).tolist()"),
       (L5 + "\nids = t.tensor([1, 1, 0, 0, 1])\nstart = 2", "solve(logits, ids, start).tolist()"),
       (L5 + "\nids = t.tensor([0, 1, 0, 1, 0])\nstart = 4", "solve(logits, ids, start).tolist()")],
      ("    m = len(ids) - start\n    rows = logits[start:]\n    return rows[t.arange(m), ids[start:]]", L4 + "\nids = t.tensor([2, 0, 1, 1])\nstart = 2",
       "The rows were taken at the phrase's own positions, so each phrase token was looked up in the row that predicts the token after it."),
      60, AI),
    D("i2", "integrated", 0,
      "`ids` holds n token ids and `logits` has shape `(n, V)`. A phrase begins at position `start` and runs to the end of the text. Return True if the model's most likely token matches the actual token at every phrase position, else False.",
      "logits, ids, start", "Whether the model's top guess matches every phrase token.",
      "    return (logits[start - 1:-1].argmax(dim=1) == ids[start:]).all().item()",
      [(L4 + "\nids = t.tensor([1, 2, 0, 1])\nstart = 1", "solve(logits, ids, start)"),
       (L4 + "\nids = t.tensor([1, 2, 1, 1])\nstart = 1", "solve(logits, ids, start)"),
       (L5 + "\nids = t.tensor([0, 0, 1, 0, 1])\nstart = 3", "solve(logits, ids, start)"),
       (L5 + "\nids = t.tensor([0, 0, 1, 0, 0])\nstart = 2", "solve(logits, ids, start)")],
      ("    return (logits[start:].argmax(dim=1) == ids[start:]).all().item()", L4 + "\nids = t.tensor([1, 2, 0, 1])\nstart = 1",
       "Each phrase token was compared with the guess for the position after it, so a phrase the model predicted perfectly is reported as a mismatch."),
      65, A),
    D("i3", "integrated", 0,
      "`ids` has shape `(b, n)`: b texts of n token ids. `logits` has shape `(b, n, V)`. In every text a phrase begins at the same position `start`. For each text, return the score the model gave to the FIRST phrase token. Shape `(b,)`.",
      "logits, ids, start", "Score of the first phrase token in every text.",
      "    b = len(ids)\n    return logits[t.arange(b), start - 1, ids[:, start]]",
      [(B2 + "\nids = t.tensor([[0, 1, 1], [1, 0, 0]])\nstart = 1", "solve(logits, ids, start).tolist()"),
       (B2 + "\nids = t.tensor([[0, 1, 1], [1, 0, 0]])\nstart = 2", "solve(logits, ids, start).tolist()"),
       ("logits = t.arange(24.).reshape(2, 4, 3)\nids = t.tensor([[0, 1, 2, 0], [2, 2, 1, 1]])\nstart = 3", "solve(logits, ids, start).tolist()"),
       ("logits = t.arange(24.).reshape(2, 4, 3)\nids = t.tensor([[0, 1, 2, 0], [2, 2, 1, 1]])\nstart = 1", "solve(logits, ids, start).tolist()")],
      ("    b = len(ids)\n    return logits[t.arange(b), start - 1, ids[:, start - 1]]", B2 + "\nids = t.tensor([[0, 1, 1], [1, 0, 0]])\nstart = 1",
       "The row is right, but the column is the id of the token before the phrase. The column must be the id of the phrase token being scored."),
      70, AI),
]
