"""monitor.prompt-score: one number per prompt, the maximum over rollouts and boundaries."""
from framework import maker

KC = "monitor.prompt-score"
A = [("lm-prompt-score", 1.0)]
AR = [("lm-prompt-score", 1.0), ("einops-reduce", 0.5)]
SPEC = dict(
    id=KC, title="A prompt's score is the maximum over its rollouts and boundaries",
    concepts=["ragged-traces-to-a-table", "max-over-the-first-k-rollouts"],
    supporting=["monitor.forced-logprob", "monitor.reasoning-boundaries", "einops.reduce-model", "torch.constructors", "torch.stack-concat-interleave",
                "torch.slice-assignment", "python.control-flow"],
)

SEGMENTS = [
    dict(title="Pad the traces with minus infinity, then take the maximum", concept="""
Scoring one rollout gives a *trace*: one phrase log-probability per sentence boundary, in order. A trace says how ready the model was to write the phrase at each point of its reasoning.

The monitor turns a trace into a single number by taking its maximum. The reason is what is being detected. A model that suspects a test usually says so once, at one point, and is busy with the task everywhere else. An average over the boundaries would be dominated by the many ordinary points. The maximum asks whether there was ANY point at which the phrase would have fitted.

Rollouts have different numbers of boundaries, so their traces have different lengths. To hold them in one table the short ones are padded, and the filler has to be a value that can never win a maximum. Zero would be a disaster: every log-probability is negative, so a zero would always win. The right filler is minus infinity, written `float("-inf")`, which is below every number:

$$\\max(x, -\\infty) = x$$
""", worked="""
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
"""),
    dict(title="Use the same number of rollouts for every prompt", concept="""
With traces padded, all the scores for a set of prompts fit in one table of shape `(prompt, rollout, pos)`. The score of a prompt is the maximum over its rollouts and its positions:

$$s(\\text{prompt}) = \\max_{r < k}\\ \\max_{j}\\ \\text{table}[\\text{prompt}, r, j]$$

The number k is how many rollouts are used. It matters because a maximum can only grow when more values are added. A prompt scored with eight rollouts has had eight chances to produce a high value, and one scored with a single rollout has had one. So scores are only comparable between prompts that were given the same k, and results are reported per k: the score with one rollout, with two, with four, with eight.

Taking the first k rollouts is a slice on the rollout axis, and both maxima are one reduction: `einops.reduce(table[:, :k], "prompt rollout pos -> prompt", "max")`.
""", worked="""
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
"""),
]

MISCONCEPTIONS = [
    ("Pad short traces with zero.", "Log-probabilities are negative, so a zero wins every maximum. Pad with minus infinity."),
    ("Average the trace.", "The signal is one point in a long reasoning. The maximum keeps it and the average buries it."),
    ("More rollouts just make the score more accurate.", "They make it higher. A maximum over more values can only grow."),
    ("Scores with different numbers of rollouts can be compared.", "Only scores at the same k are comparable."),
]

D = maker("ps", KC)
TR = "traces = [t.tensor([-9., -4., -7.]), t.tensor([-6.])]"
TR2 = "traces = [t.tensor([-2.]), t.tensor([-5., -1.]), t.tensor([-8., -8., -3., -9.])]"
TR3 = "traces = [t.tensor([-7., -7.]), t.tensor([-9.]), t.tensor([-1., -6., -6.])]"
T3 = "table = t.tensor([[[-9., -8.], [-3., -7.], [-6., -6.]], [[-5., -4.], [-9., -9.], [-8., -2.]]])"
T3B = "table = t.tensor([[[-1.], [-5.]], [[-6.], [-2.]], [[-4.], [-4.]]])"
T2 = "table = t.tensor([[-9., -4., -7.], [-6., float('-inf'), float('-inf')]])"
T2B = "table = t.tensor([[-5., float('-inf')], [-8., -1.], [-3., -3.]])"
PADT = """    lens = t.tensor([len(q) for q in traces])
    table = t.full((len(traces), lens.max().item()), {fill})
    i = 0
    for q in traces:
        table[i, :len(q)] = q
        i += 1
"""
DRILLS = [
    D("f1", "faded", 1,
      "`traces` is a list of 1-D tensors, one per rollout, each holding the phrase log-probability at every sentence boundary of that rollout. The traces have different lengths. Return the highest score of each rollout, shape `(rollouts,)`, by way of one padded table.",
      "traces", "Highest score in each rollout.",
      PADT.format(fill='float("-inf")') + '    return einops.reduce(table, "rollout pos -> rollout", "max")',
      [(TR, "solve(traces).tolist()"), (TR2, "solve(traces).tolist()"), (TR3, "solve(traces).tolist()"),
       ("traces = [t.tensor([-1., -2.]), t.tensor([-2., -1.])]", "solve(traces).tolist()")],
      (PADT.format(fill="0.0") + '    return einops.reduce(table, "rollout pos -> rollout", "max")', TR,
       "The short trace was filled up with zeros. Every real score is negative, so the filler won and the rollout was given a score it never produced."),
      30, AR, scaffold=PADT.format(fill='float("_____")') + '    return einops.reduce(table, "rollout pos -> rollout", "max")'),
    D("f2", "faded", 2,
      "`table` has shape `(prompt, rollout, pos)` and holds phrase log-probabilities. Return each prompt's score using only its first `k` rollouts, shape `(prompt,)`.",
      "table, k", "Prompt scores from the first k rollouts.",
      '    return einops.reduce(table[:, :k], "prompt rollout pos -> prompt", "max")',
      [(T3 + "\nk = 1", "solve(table, k).tolist()"), (T3 + "\nk = 2", "solve(table, k).tolist()"),
       (T3B + "\nk = 1", "solve(table, k).tolist()"), (T3B + "\nk = 2", "solve(table, k).tolist()")],
      ('    return einops.reduce(table[:, :k], "prompt rollout pos -> prompt", "mean")', T3 + "\nk = 1",
       "This averaged over the boundaries. A single point where the phrase fits well is diluted by all the ordinary points around it."),
      25, AR, scaffold='    return einops.reduce(table[:, :_____], "prompt rollout pos -> prompt", "_____")'),
    # ---- solo ------------------------------------------------------------------------------------------------
    D("s1", "solo", 0,
      "`table` has shape `(rollout, pos)` and holds one prompt's phrase log-probabilities, padded with minus infinity. Return the prompt's score as a Python float.",
      "table", "Score of one prompt.",
      '    return einops.reduce(table, "rollout pos -> ", "max").item()',
      [(T2, "solve(table)"), (T2B, "solve(table)"), ("table = t.tensor([[-7., -7.], [-9., -8.]])", "solve(table)"), ("table = t.tensor([[-2., -6., -6.]])", "solve(table)")],
      ('    return einops.reduce(table[:, 0], "rollout -> ", "max").item()', T2,
       "Only the first boundary of each rollout was looked at. The point where the phrase fits best is usually later in the reasoning."),
      30, AR),
    D("s2", "solo", 0,
      "`table` has shape `(rollout, pos)` and holds one prompt's phrase log-probabilities, padded with minus infinity. Return the index of the rollout that contains the prompt's highest score, as a Python int.",
      "table", "Which rollout produced the prompt's score.",
      '    return t.argmax(einops.reduce(table, "rollout pos -> rollout", "max")).item()',
      [(T2B, "solve(table)"), (T2, "solve(table)"), ("table = t.tensor([[-7., -7.], [-9., -8.], [-9., -1.]])", "solve(table)"), ("table = t.tensor([[-5., -6.], [-4., -9.]])", "solve(table)")],
      ("    return t.argmax(table).item()", T2B,
       "This is the place of the highest score when all the entries are counted in one long line. It mixes up the rollout with the position inside it."),
      40, AR),
    D("s3", "solo", 0,
      "`table` has shape `(rollout, pos)` and holds one prompt's phrase log-probabilities, padded with minus infinity. Return the number of sentence boundaries each rollout really had, shape `(rollout,)`.",
      "table", "How many real scores each rollout has.",
      '    return (table != float("-inf")).sum(dim=1)',
      [(T2, "solve(table).tolist()"), (T2B, "solve(table).tolist()"), ("table = t.tensor([[-7., -7.], [-9., -8.]])", "solve(table).tolist()"),
       ("table = t.tensor([[-2., float('-inf'), float('-inf')], [-1., -1., -1.], [-4., -4., float('-inf')]])", "solve(table).tolist()")],
      ('    return (table != float("-inf")).sum(dim=0)', T2,
       "This counted down the columns, giving how many rollouts reached each boundary. The count wanted runs along each rollout's own row."),
      35, A),
    D("s4", "solo", 0,
      "`table` has shape `(prompt, rollout, pos)` and holds phrase log-probabilities. Return, for each prompt, whether its score from the first `k` rollouts is above the threshold `thr`. Boolean tensor of shape `(prompt,)`.",
      "table, k, thr", "Which prompts the monitor flags at a threshold.",
      '    return einops.reduce(table[:, :k], "prompt rollout pos -> prompt", "max") > thr',
      [(T3 + "\nk = 1\nthr = -5.0", "solve(table, k, thr).tolist()"), (T3 + "\nk = 3\nthr = -2.5", "solve(table, k, thr).tolist()"),
       (T3B + "\nk = 1\nthr = -3.0", "solve(table, k, thr).tolist()"), (T3B + "\nk = 2\nthr = -3.0", "solve(table, k, thr).tolist()")],
      ('    return einops.reduce(table, "prompt rollout pos -> prompt", "max") > thr', T3 + "\nk = 1\nthr = -5.0",
       "Every rollout was used although only the first k were asked for. Extra rollouts can only raise a maximum, so a prompt was flagged that is below the threshold at this k."),
      40, AR),
    D("s5", "solo", 0,
      "`table` has shape `(prompt, rollout, pos)` and holds phrase log-probabilities. Return how much each prompt's score rises when all its rollouts are used instead of only the first one. Shape `(prompt,)`.",
      "table", "Gain in score from using every rollout.",
      '    every = einops.reduce(table, "prompt rollout pos -> prompt", "max")\n    first = einops.reduce(table[:, :1], "prompt rollout pos -> prompt", "max")\n    return every - first',
      [(T3, "solve(table).tolist()"), (T3B, "solve(table).tolist()"), ("table = t.tensor([[[-1., -2.], [-3., -4.]]])", "solve(table).tolist()"),
       ("table = t.tensor([[[-6.], [-5.], [-4.]], [[-4.], [-5.], [-6.]]])", "solve(table).tolist()")],
      ('    every = einops.reduce(table, "prompt rollout pos -> prompt", "max")\n    last = einops.reduce(table[:, -1:], "prompt rollout pos -> prompt", "max")\n    return every - last', T3,
       "The comparison was made against the last rollout. The baseline is the score a prompt has with a single rollout, the first one."),
      50, AR),
    D("s6", "solo", 0,
      "`table` has shape `(prompt, rollout, pos)` and holds phrase log-probabilities. Return, for each prompt, whether its score from the first `k` rollouts already equals its score from all rollouts. Boolean tensor of shape `(prompt,)`.",
      "table, k", "Whether the first k rollouts already contain the best score.",
      '    part = einops.reduce(table[:, :k], "prompt rollout pos -> prompt", "max")\n    return part == einops.reduce(table, "prompt rollout pos -> prompt", "max")',
      [(T3 + "\nk = 2", "solve(table, k).tolist()"), (T3 + "\nk = 1", "solve(table, k).tolist()"), (T3B + "\nk = 2", "solve(table, k).tolist()"), (T3 + "\nk = 3", "solve(table, k).tolist()")],
      ('    part = einops.reduce(table[:, k - 1:k], "prompt rollout pos -> prompt", "max")\n    return part == einops.reduce(table, "prompt rollout pos -> prompt", "max")', T3B + "\nk = 2",
       "Only the k-th rollout was looked at. The score at k uses all of the first k rollouts, so a best value in an earlier one still counts."),
      50, AR, show=2),
    # ---- integrated ------------------------------------------------------------------------------------------
    D("i1", "integrated", 0,
      "`traces` is a list of 1-D tensors, one per rollout of a single prompt, each holding the phrase log-probability at every sentence boundary. The traces have different lengths. Return the prompt's score using only the first `k` rollouts, as a Python float.",
      "traces, k", "Score of one prompt from ragged traces at k rollouts.",
      PADT.format(fill='float("-inf")') + '    return einops.reduce(table[:k], "rollout pos -> ", "max").item()',
      [(TR2 + "\nk = 2", "solve(traces, k)"), (TR2 + "\nk = 1", "solve(traces, k)"), (TR + "\nk = 2", "solve(traces, k)"), (TR3 + "\nk = 2", "solve(traces, k)")],
      ('    return einops.reduce(traces[k - 1], "pos -> ", "max").item()', TR3 + "\nk = 2",
       "Only the k-th rollout was scored. The score at k uses all of the first k rollouts, and here the best value is in an earlier one."),
      65, AR, show=3),
    D("i2", "integrated", 0,
      "`table` has shape `(prompt, rollout, pos)` and holds phrase log-probabilities. `ks` is a list of rollout counts. Return the prompt scores for each count, shape `(len(ks), prompt)`: row i uses the first `ks[i]` rollouts.",
      "table, ks", "Prompt scores at several rollout counts.",
      '    return t.stack([einops.reduce(table[:, :k], "prompt rollout pos -> prompt", "max") for k in ks])',
      [(T3 + "\nks = [1, 2, 3]", "solve(table, ks).tolist()"), (T3 + "\nks = [3, 1]", "solve(table, ks).tolist()"), (T3B + "\nks = [1, 2]", "solve(table, ks).tolist()"), (T3B + "\nks = [2]", "solve(table, ks).tolist()")],
      ('    return t.stack([einops.reduce(table[:, k - 1:k], "prompt rollout pos -> prompt", "max") for k in ks])', T3 + "\nks = [1, 2, 3]",
       "Each row used the k-th rollout alone. Scores computed this way can fall as k grows, which a maximum over the first k rollouts never does."),
      70, AR),
    D("i3", "integrated", 0,
      "A batch of forced sequences for one prompt was scored: `scores`, shape `(N,)`, holds the phrase log-probability of each, and `rollout_of`, shape `(N,)`, holds the index of the rollout each sequence came from. Return the prompt's score using only rollouts with index below `k`, as a Python float.",
      "scores, rollout_of, k", "Prompt score from a flat batch of forced sequences.",
      '    return t.where(rollout_of < k, scores, float("-inf")).max().item()',
      [("scores = t.tensor([-9., -7., -4., -6., -2.])\nrollout_of = t.tensor([0, 0, 0, 1, 2])\nk = 2", "solve(scores, rollout_of, k)"),
       ("scores = t.tensor([-9., -7., -4., -6., -2.])\nrollout_of = t.tensor([0, 0, 0, 1, 2])\nk = 3", "solve(scores, rollout_of, k)"),
       ("scores = t.tensor([-9., -4., -7., -6., -2.])\nrollout_of = t.tensor([0, 1, 1, 2, 2])\nk = 1", "solve(scores, rollout_of, k)"),
       ("scores = t.tensor([-3., -8., -1.])\nrollout_of = t.tensor([2, 0, 1])\nk = 1", "solve(scores, rollout_of, k)")],
      ("    return scores[:k].max().item()", "scores = t.tensor([-9., -7., -4., -6., -2.])\nrollout_of = t.tensor([0, 0, 0, 1, 2])\nk = 2",
       "The first k SEQUENCES were taken. One rollout contributes a sequence per sentence boundary, so the first k sequences are only the start of the first rollout."),
      70, A),
]
