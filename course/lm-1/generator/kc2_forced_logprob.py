"""monitor.forced-logprob: log_softmax over the vocabulary, then the sum over the phrase tokens."""
from framework import maker

KC = "monitor.forced-logprob"
A = [("lm-forced-logprob", 1.0)]
AR = [("lm-forced-logprob", 1.0), ("einops-reduce", 0.5)]
AN = [("lm-forced-logprob", 1.0), ("lm-next-token-row", 0.5)]
SPEC = dict(
    id=KC, title="The log-probability of a forced phrase",
    concepts=["log-softmax-over-vocabulary", "sum-over-phrase-tokens"],
    supporting=["monitor.next-token-logits", "tensor.stable-probabilities", "tensor.indexed-selection", "einops.reduce-model"],
    new_syntax=["torch.log_softmax", "torch.log_softmax#dim"],
)

SEGMENTS = [
    dict(title="log_softmax turns a row of scores into log-probabilities", concept="""
A row of logits is a row of raw scores. To ask "how likely did the model find this token?", the scores have to become probabilities: positive numbers that add up to 1 across the vocabulary. The usual recipe is the *softmax*: exponentiate every score, then divide by the total of the row.

The monitor works with the logarithm of that probability, the *log-probability*. For a row of scores $z$ and a token $i$:

$$\\log p_i = z_i - \\log \\sum_j e^{z_j}$$

`t.log_softmax(logits, dim=-1)` computes this for every row at once. `dim=-1` names the axis whose entries compete with each other, which is the vocabulary axis, the last one. The result has the same shape as `logits`.

Three facts are worth checking on any output. Every entry is at most 0, because a probability is at most 1 and the logarithm of 1 is 0. A more likely token has a log-probability closer to 0. And exponentiating a row gives numbers that add up to 1.
""", worked="""
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
"""),
    dict(title="A phrase's log-probability is the sum of its tokens' log-probabilities", concept="""
The model writes one token at a time. The probability that it writes a whole phrase is therefore a product: the probability of the first phrase token, times the probability of the second given the first, and so on. For a phrase with tokens $w_1, \\dots, w_m$ after some earlier text:

$$P(\\text{phrase}) = \\prod_{j=1}^{m} P(w_j \\mid \\text{earlier text},\\ w_1, \\dots, w_{j-1})$$

Taking logarithms turns the product into a sum:

$$\\log P(\\text{phrase}) = \\sum_{j=1}^{m} \\log P(w_j \\mid \\text{earlier text},\\ w_1, \\dots, w_{j-1})$$

One pass of the model gives every term at once, as long as the phrase is already in the input. Row j of the phrase's rows has read the earlier text and the first j phrase tokens, so it holds exactly the conditional probabilities the formula asks for. This is called *forcing* the phrase: the model never has to choose to write it.

In code there are two steps. First pick one log-probability per phrase token with a pair of index tensors: row j, column $w_j$. Then add them up. With a batch of texts the picked table has shape `(batch, tok)`, and `einops.reduce(per_token, "batch tok -> batch", "sum")` adds over the token axis and keeps one number per text.
""", worked="""
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
"""),
]

MISCONCEPTIONS = [
    ("Take the softmax over the position axis.", "Tokens compete within one row, so the axis is the vocabulary axis, the last one."),
    ("Add up the probabilities of the phrase tokens.", "Probabilities of successive tokens multiply. It is their logarithms that add."),
    ("A log-probability near 0 means unlikely.", "0 is the logarithm of probability 1. Unlikely tokens have large negative values."),
    ("Raw logits can be added in place of log-probabilities.", "Logits are only meaningful relative to the rest of their row. Two rows can have the same logit for a token and very different probabilities."),
]

D = maker("fl", KC)
R3 = "rows = t.tensor([[0., 1., 2.], [3., .5, 1.], [.2, 4., .1]])"
R2 = "rows = t.tensor([[0., 0.], [2., 0.]])"
L4 = "logits = t.tensor([[0., 1., 2.], [3., .5, 1.], [.2, 4., .1], [1., 1., 5.]])"
L5 = "logits = t.tensor([[1., 0.], [0., 2.], [3., 0.], [0., 4.], [5., 0.]])"
B3 = "logits = t.tensor([[[0., 1., 2.], [3., .5, 1.]], [[2., 2., 2.], [0., 0., 4.]], [[1., 0., 0.], [0., 5., 0.]]])"
DRILLS = [
    D("f1", "faded", 1,
      "`logits` has shape `(n, V)`, one row of scores per position. Return a table of the same shape in which every row holds log-probabilities over the vocabulary.",
      "logits", "Log-probabilities over the vocabulary, row by row.",
      "    return t.log_softmax(logits, dim=-1)",
      [(R3.replace("rows", "logits"), "solve(logits).tolist()"), (R2.replace("rows", "logits"), "solve(logits).tolist()"),
       ("logits = t.tensor([[5., 5., 5., 5.]])", "solve(logits).tolist()"), ("logits = t.tensor([[0., 3.], [3., 0.], [1., 1.]])", "solve(logits).tolist()")],
      ("    return t.log_softmax(logits, dim=0)", R2.replace("rows", "logits"),
       "This made the entries of each COLUMN compete, so one token is compared with itself at other positions. The tokens of one row are the ones that must share a total probability of 1."),
      15, A, scaffold="    return t._____(logits, _____=-1)", show=1),
    D("f2", "faded", 2,
      "`per_token` has shape `(b, m)`: for each of b texts, the log-probability of each of the m phrase tokens. Return the log-probability of the whole phrase for each text, shape `(b,)`.",
      "per_token", "Phrase log-probability for each text.",
      "    return einops.reduce(per_token, \"batch tok -> batch\", \"sum\")",
      [("per_token = t.tensor([[-1., -2.], [-.5, -.5]])", "solve(per_token).tolist()"), ("per_token = t.tensor([[-3., 0., -1.]])", "solve(per_token).tolist()"),
       ("per_token = t.tensor([[-1.], [-2.], [-4.]])", "solve(per_token).tolist()"), ("per_token = t.tensor([[-.25, -.25, -.5], [-2., -2., -2.]])", "solve(per_token).tolist()")],
      ("    return einops.reduce(per_token, \"batch tok -> tok\", \"sum\")", "per_token = t.tensor([[-1., -2.], [-.5, -.5]])",
       "This added over the texts and kept one number per token position, so different texts were mixed together. The sum belongs over the tokens of one text."),
      20, AR, scaffold="    return einops.reduce(per_token, \"batch tok -> _____\", \"_____\")"),
    # ---- solo ------------------------------------------------------------------------------------------------
    D("s1", "solo", 0,
      "`rows` has shape `(m, V)` and holds raw scores. Return the log-probability that row `r` gives to token `tok`, as a Python float.",
      "rows, r, tok", "Log-probability of one token in one row.",
      "    return t.log_softmax(rows, dim=-1)[r, tok].item()",
      [(R3 + "\nr = 0\ntok = 2", "solve(rows, r, tok)"), (R3 + "\nr = 1\ntok = 1", "solve(rows, r, tok)"),
       (R2 + "\nr = 0\ntok = 1", "solve(rows, r, tok)"), (R2 + "\nr = 1\ntok = 0", "solve(rows, r, tok)")],
      ("    return rows[r, tok].item()", R3 + "\nr = 0\ntok = 2",
       "This is the raw score. A raw score says nothing until it is compared with the other scores in its row, and here it is even positive, which no log-probability can be."),
      30, A),
    D("s2", "solo", 0,
      "`rows` has shape `(m, V)` and holds raw scores. Return the probability (not its logarithm) that row `r` gives to token `tok`, as a Python float.",
      "rows, r, tok", "Probability of one token in one row.",
      "    return t.log_softmax(rows, dim=-1)[r, tok].exp().item()",
      [(R2 + "\nr = 0\ntok = 0", "solve(rows, r, tok)"), (R2 + "\nr = 1\ntok = 0", "solve(rows, r, tok)"),
       (R3 + "\nr = 2\ntok = 1", "solve(rows, r, tok)"), ("rows = t.tensor([[1., 1., 1., 1.]])\nr = 0\ntok = 3", "solve(rows, r, tok)")],
      ("    return t.log_softmax(rows, dim=-1)[r, tok].item()", R2 + "\nr = 0\ntok = 0",
       "This is the logarithm of the probability, a negative number. The probability itself is recovered by exponentiating."),
      35, A),
    D("s3", "solo", 0,
      "`rows` has shape `(m, V)` and holds the raw scores that score an m-token phrase. `phrase` holds the m phrase token ids. Return the log-probability of each phrase token, in order, shape `(m,)`.",
      "rows, phrase", "Log-probability of each phrase token.",
      "    logp = t.log_softmax(rows, dim=-1)\n    return logp[t.arange(len(phrase)), phrase]",
      [(R3 + "\nphrase = t.tensor([2, 0, 1])", "solve(rows, phrase).tolist()"), (R3 + "\nphrase = t.tensor([0, 0, 0])", "solve(rows, phrase).tolist()"),
       (R2 + "\nphrase = t.tensor([1, 1])", "solve(rows, phrase).tolist()"), (R2 + "\nphrase = t.tensor([0, 0])", "solve(rows, phrase).tolist()")],
      ("    logp = t.log_softmax(rows, dim=-1)\n    return logp[:, phrase]", R3 + "\nphrase = t.tensor([2, 0, 1])",
       "This kept every row for every phrase token and returned a square table. Token j belongs with row j only, one entry per token."),
      40, A),
    D("s4", "solo", 0,
      "`rows` has shape `(m, V)` and holds the raw scores that score an m-token phrase. `phrase` holds the m phrase token ids. Return the log-probability of the whole phrase, as a Python float.",
      "rows, phrase", "Log-probability of the whole phrase.",
      "    logp = t.log_softmax(rows, dim=-1)\n    per_token = logp[t.arange(len(phrase)), phrase]\n    return einops.reduce(per_token, \"tok -> \", \"sum\").item()",
      [(R3 + "\nphrase = t.tensor([2, 0, 1])", "solve(rows, phrase)"), (R3 + "\nphrase = t.tensor([0, 1, 2])", "solve(rows, phrase)"),
       (R2 + "\nphrase = t.tensor([1, 0])", "solve(rows, phrase)"), (R2 + "\nphrase = t.tensor([0, 1])", "solve(rows, phrase)")],
      ("    p = t.log_softmax(rows, dim=-1).exp()\n    per_token = p[t.arange(len(phrase)), phrase]\n    return einops.reduce(per_token, \"tok -> \", \"sum\").log().item()", R3 + "\nphrase = t.tensor([2, 0, 1])",
       "This added the three probabilities and took the logarithm of the total, which came out above 0. Probabilities of successive tokens multiply, so it is the logarithms that are added."),
      45, AR),
    D("s5", "solo", 0,
      "`logits` has shape `(b, m, V)`: for each of b texts, the raw scores that score the same m-token phrase. `phrase` holds the m phrase token ids. Return the log-probability of the whole phrase in each text, shape `(b,)`.",
      "logits, phrase", "Phrase log-probability in every text.",
      "    logp = t.log_softmax(logits, dim=-1)\n    per_token = logp[:, t.arange(len(phrase)), phrase]\n    return einops.reduce(per_token, \"batch tok -> batch\", \"sum\")",
      [(B3 + "\nphrase = t.tensor([2, 0])", "solve(logits, phrase).tolist()"), (B3 + "\nphrase = t.tensor([1, 1])", "solve(logits, phrase).tolist()"),
       (B3 + "\nphrase = t.tensor([0, 2])", "solve(logits, phrase).tolist()"), ("logits = t.zeros(2, 3, 4)\nphrase = t.tensor([0, 1, 2])", "solve(logits, phrase).tolist()")],
      ("    logp = t.log_softmax(logits, dim=1)\n    per_token = logp[:, t.arange(len(phrase)), phrase]\n    return einops.reduce(per_token, \"batch tok -> batch\", \"sum\")", B3 + "\nphrase = t.tensor([2, 0])",
       "With a batch axis in front, dimension 1 is the position axis. The softmax ran across positions, so a token's probability was shared between rows instead of between vocabulary entries."),
      50, AR),
    D("s6", "solo", 0,
      "`logits` has shape `(b, m, V)`: for each of b texts, the raw scores that score the same m-token phrase. `phrase` holds the m phrase token ids. Return the index of the text in which the whole phrase is most likely, as a Python int.",
      "logits, phrase", "Which text makes the phrase most likely.",
      "    logp = t.log_softmax(logits, dim=-1)\n    per_token = logp[:, t.arange(len(phrase)), phrase]\n    return einops.reduce(per_token, \"batch tok -> batch\", \"sum\").argmax().item()",
      [(B3 + "\nphrase = t.tensor([2, 0])", "solve(logits, phrase)"), (B3 + "\nphrase = t.tensor([1, 1])", "solve(logits, phrase)"),
       ("logits = t.tensor([[[9., 9., 9.], [9., 9., 9.]], [[0., 0., 3.], [3., 0., 0.]]])\nphrase = t.tensor([2, 0])", "solve(logits, phrase)"), (B3 + "\nphrase = t.tensor([2, 2])", "solve(logits, phrase)")],
      ("    per_token = logits[:, t.arange(len(phrase)), phrase]\n    return einops.reduce(per_token, \"batch tok -> batch\", \"sum\").argmax().item()", "logits = t.tensor([[[9., 9., 9.], [9., 9., 9.]], [[0., 0., 3.], [3., 0., 0.]]])\nphrase = t.tensor([2, 0])",
       "Raw scores were added without comparing them to the rest of their rows. A text whose scores are large everywhere wins, even where the phrase tokens are no more likely than any other token."),
      55, AR, show=2),
    # ---- integrated ------------------------------------------------------------------------------------------
    D("i1", "integrated", 0,
      "`ids` holds n token ids and `logits`, shape `(n, V)`, is the model's raw output for them. A phrase begins at position `start` and runs to the end of the text. Return the log-probability of the whole phrase given the text before it, as a Python float.",
      "logits, ids, start", "Log-probability of the phrase that ends the text.",
      "    m = len(ids) - start\n    logp = t.log_softmax(logits[start - 1:start - 1 + m], dim=-1)\n    per_token = logp[t.arange(m), ids[start:]]\n    return einops.reduce(per_token, \"tok -> \", \"sum\").item()",
      [(L4 + "\nids = t.tensor([2, 0, 1, 1])\nstart = 2", "solve(logits, ids, start)"), (L4 + "\nids = t.tensor([0, 2, 0, 1])\nstart = 1", "solve(logits, ids, start)"),
       (L5 + "\nids = t.tensor([1, 1, 0, 1, 0])\nstart = 3", "solve(logits, ids, start)"), (L5 + "\nids = t.tensor([0, 0, 0, 0, 0])\nstart = 4", "solve(logits, ids, start)")],
      ("    m = len(ids) - start\n    logp = t.log_softmax(logits[start:], dim=-1)\n    per_token = logp[t.arange(m), ids[start:]]\n    return einops.reduce(per_token, \"tok -> \", \"sum\").item()", L4 + "\nids = t.tensor([2, 0, 1, 1])\nstart = 2",
       "The rows were taken at the phrase's own positions. Each of those rows predicts the token after it, so every phrase token was scored against the wrong prediction."),
      65, AN),
    D("i2", "integrated", 0,
      "`ids` has shape `(b, n)` and `logits` has shape `(b, n, V)`: b texts of n tokens with the model's raw output. In every text a phrase begins at position `start` and runs to the end. Return the log-probability of each text's own phrase given the text before it, shape `(b,)`.",
      "logits, ids, start", "Phrase log-probability for each text in a batch.",
      "    b = len(ids)\n    m = len(ids[0]) - start\n    logp = t.log_softmax(logits[:, start - 1:start - 1 + m], dim=-1)\n    per_token = logp[t.arange(b)[:, None], t.arange(m)[None, :], ids[:, start:]]\n    return einops.reduce(per_token, \"batch tok -> batch\", \"sum\")",
      [("logits = t.tensor([[[0., 1.], [2., 0.], [0., 3.]], [[1., 1.], [0., 4.], [5., 0.]]])\nids = t.tensor([[0, 0, 1], [1, 1, 0]])\nstart = 1", "solve(logits, ids, start).tolist()"),
       ("logits = t.tensor([[[0., 1.], [2., 0.], [0., 3.]], [[1., 1.], [0., 4.], [5., 0.]]])\nids = t.tensor([[0, 1, 1], [1, 0, 0]])\nstart = 2", "solve(logits, ids, start).tolist()"),
       ("logits = t.zeros(3, 4, 5)\nids = t.tensor([[0, 1, 2, 3], [4, 4, 4, 4], [1, 0, 1, 0]])\nstart = 2", "solve(logits, ids, start).tolist()"),
       ("logits = t.arange(12.).reshape(1, 4, 3)\nids = t.tensor([[2, 2, 0, 1]])\nstart = 1", "solve(logits, ids, start).tolist()")],
      ("    b = len(ids)\n    m = len(ids[0]) - start\n    logp = t.log_softmax(logits[:, start - 1:start - 1 + m], dim=-1)\n    per_token = logp[t.arange(b)[:, None], t.arange(m)[None, :], ids[:, start - 1:-1]]\n    return einops.reduce(per_token, \"batch tok -> batch\", \"sum\")",
       "logits = t.tensor([[[0., 1.], [2., 0.], [0., 3.]], [[1., 1.], [0., 4.], [5., 0.]]])\nids = t.tensor([[0, 0, 1], [1, 1, 0]])\nstart = 1",
       "The rows are right, but the columns came from the tokens the rows had already read. Each row must be read at the id of the token it predicts."),
      75, AN),
    D("i3", "integrated", 0,
      "`ids` holds n token ids and `logits` has shape `(n, V)`. A phrase begins at position `start` and runs to the end of the text. Return the place within the phrase (0 for its first token) of the token the model found LEAST likely, as a Python int.",
      "logits, ids, start", "Which phrase token was least expected.",
      "    m = len(ids) - start\n    logp = t.log_softmax(logits[start - 1:start - 1 + m], dim=-1)\n    return t.argmin(logp[t.arange(m), ids[start:]]).item()",
      [(L4 + "\nids = t.tensor([2, 0, 0, 1])\nstart = 1", "solve(logits, ids, start)"), (L4 + "\nids = t.tensor([0, 2, 1, 0])\nstart = 1", "solve(logits, ids, start)"),
       (L5 + "\nids = t.tensor([1, 1, 0, 0, 0])\nstart = 2", "solve(logits, ids, start)"), (L5 + "\nids = t.tensor([0, 0, 1, 1, 1])\nstart = 1", "solve(logits, ids, start)")],
      ("    m = len(ids) - start\n    rows = logits[start - 1:start - 1 + m]\n    return t.argmin(rows[t.arange(m), ids[start:]]).item()", L5 + "\nids = t.tensor([1, 1, 0, 0, 0])\nstart = 2",
       "Raw scores from different rows were compared directly. A low raw score can still be the best in its own row, so the comparison has to be made after each row is turned into log-probabilities."),
      70, AN, show=2),
]
