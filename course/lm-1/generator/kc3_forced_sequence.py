"""monitor.forced-sequence: prefix + reasoning so far + phrase, with the phrase tokenised in context."""
from framework import maker

KC = "monitor.forced-sequence"
A = [("lm-forced-sequence", 1.0)]
AN = [("lm-forced-sequence", 1.0), ("lm-next-token-row", 0.5)]
SPEC = dict(
    id=KC, title="Build the forced sequence and tokenise the phrase in context",
    concepts=["prefix-reasoning-phrase", "tokenise-in-context"],
    supporting=["monitor.next-token-logits", "torch.stack-concat-interleave", "python.control-flow", "tensor.row-normalization"],
)

SEGMENTS = [
    dict(title="The forced sequence is the prefix, the reasoning so far, then the phrase", concept="""
The monitor asks one question many times: "at this point in the model's reasoning, how likely is it to write the phrase next?" Each time, the model is given a sequence made of three pieces of token ids, joined end to end.

1. The *prefix*: the user's prompt wrapped in the model's chat format, ending at the place where the model's reasoning begins. With the Hugging Face library this is the text from `apply_chat_template(..., add_generation_prompt=True)` followed by the reasoning tag `<think>` and a line break.
2. The *reasoning so far*: the first b tokens of the reasoning the model wrote, where b is the point being tested.
3. The *phrase*: the tokens of the fixed text, such as "The user might be testing".

Joining 1-D tensors end to end is `t.cat([prefix, reasoning[:b], phrase])`. The phrase then starts at position

$$\\text{start} = \\text{len(prefix)} + b$$

and `start` is what decides which rows of the logits are read: the rows from `start − 1` on.

The real calls that produce the prefix look like this. They need the model's tokenizer, so this cell is for reading only:

```python no-run
messages = [{"role": "user", "content": user_prompt}]
text = tok.apply_chat_template(
    messages, tokenize=False,
    add_generation_prompt=True) + "<think>\\n"
prefix = t.tensor(
    tok.encode(text, add_special_tokens=False))
```

`add_special_tokens=False` matters: the chat template has already written the special tokens, and letting `encode` add its own would put a second start-of-text token in front.
""", worked="""
Use small made-up ids so the pieces stay visible: the prefix is three tokens, the reasoning has five, the phrase two. Test the point after the first two reasoning tokens, so b is 2. The plan: join the three pieces, then compute where the phrase starts.

```python
import torch as t
prefix = t.tensor([90, 91, 92])
reasoning = t.tensor([10, 11, 12, 13, 14])
phrase = t.tensor([70, 71])
b = 2
seq = t.cat([prefix, reasoning[:b], phrase])
print(seq)
# Hidden checks
assert seq.tolist() == [90, 91, 92, 10, 11, 70, 71]
```

The sequence holds only the first two reasoning tokens. Tokens 12, 13 and 14 are left out on purpose: the question is what the model would write at that point, before it has seen the rest of its own reasoning. Now the place where the phrase starts:

```python
start = len(prefix) + b
print(start, seq[start:])
# Hidden checks
assert start == 5 and seq[start:].tolist() == [70, 71]
```

The phrase starts at position 5, after three prefix tokens and two reasoning tokens. The rows that score it are therefore rows 4 and 5.
"""),
    dict(title="Tokenise the phrase together with the text before it", concept="""
Turning text into token ids is not done word by word in isolation. Most tokenizers fold a leading space into the token that follows it, so the same word becomes a different id depending on what comes before. In the Qwen3 tokenizer, "The" right after a line break is token 785, while " The" after a sentence (with its space) is token 576.

So encoding the phrase on its own and gluing the ids on gives the wrong tokens in the middle of a paragraph. The safe procedure is to encode the whole text at once and cut the phrase's ids out of the result.

1. Write the full text: everything before the phrase, a space unless that text ends in a line break, then the phrase.
2. Encode the full text. Call the result `full`.
3. Let n be the number of ids in the encoding of the text before the phrase, `before`. The phrase's ids are `full[n:]`.

Step 3 relies on the first n ids of `full` being exactly `before`. That is usually true and occasionally false: adding text can change how the last few characters before it are split. So the procedure checks it, `(full[:n] == before).all()`, and treats a failure as a case to handle instead of scoring the wrong tokens silently.
""", worked="""
Suppose the text before the phrase encodes to `before`, and the full text encodes to `full`. The plan: compare the first n ids of `full` with `before`, and only then slice off the phrase.

```python
import torch as t
before = t.tensor([90, 91, 10, 11])
full = t.tensor([90, 91, 10, 11, 576, 71])
n = len(before)
unchanged = (full[:n] == before).all()
phrase = full[n:]
print(unchanged, phrase)
# Hidden checks
assert unchanged.item() is True
assert phrase.tolist() == [576, 71]
```

The first four ids match, so the last two are the phrase as the tokenizer writes it in this context. Now a case where the join changed the token before the phrase (11 became 15):

```python
full_changed = t.tensor([90, 91, 10, 15, 71])
print((full_changed[:n] == before).all())
# Hidden checks
assert (full_changed[:n] == before).all().item() is False
```

Here `full_changed[n:]` would be a single id, 71, and the model would be scored on a sequence that is not the text it wrote. The check catches this before any score is computed.
"""),
]

MISCONCEPTIONS = [
    ("Include the whole reasoning before the phrase.", "The test is about one point in the reasoning, so only the tokens up to that point go in."),
    ("The phrase always has the same token ids.", "Its first token depends on whether a space or a line break comes before it."),
    ("Encode the phrase alone and append it.", "That gives the start-of-line form of the first word. Encode the joined text and slice."),
    ("Slicing the joined encoding is always safe.", "Only when the ids before the phrase are unchanged. Check them."),
]

D = maker("fs", KC)
PRE = "prefix = t.tensor([90, 91, 92])\nreasoning = t.tensor([10, 11, 12, 13, 14])\nphrase = t.tensor([70, 71])"
PRE2 = "prefix = t.tensor([5])\nreasoning = t.tensor([1, 2, 3])\nphrase = t.tensor([8, 8, 9])"
BF = "before = t.tensor([90, 91, 10, 11])"
DRILLS = [
    D("f1", "faded", 1,
      "`prefix`, `reasoning` and `phrase` are 1-D tensors of token ids. Return the sequence that tests the point after the first `b` reasoning tokens: the prefix, those `b` tokens, then the phrase.",
      "prefix, reasoning, phrase, b", "Prefix, first b reasoning tokens, phrase.",
      "    return t.cat([prefix, reasoning[:b], phrase])",
      [(PRE + "\nb = 2", "solve(prefix, reasoning, phrase, b).tolist()"), (PRE + "\nb = 0", "solve(prefix, reasoning, phrase, b).tolist()"),
       (PRE2 + "\nb = 3", "solve(prefix, reasoning, phrase, b).tolist()"), (PRE2 + "\nb = 1", "solve(prefix, reasoning, phrase, b).tolist()")],
      ("    return t.cat([prefix, reasoning, phrase])", PRE + "\nb = 2",
       "The whole reasoning went in, so the phrase was placed at the end of the reasoning instead of at the point being tested."),
      15, A, scaffold="    return t.cat([prefix, reasoning[:_____], phrase])"),
    D("f2", "faded", 2,
      "`before` holds the token ids of the text before a phrase, and `full` the ids of that text and the phrase encoded together. Return True if the ids before the phrase are unchanged in `full`, else False.",
      "before, full", "Whether the ids before the phrase survived the joint encoding.",
      "    return (full[:len(before)] == before).all().item()",
      [(BF + "\nfull = t.tensor([90, 91, 10, 11, 576, 71])", "solve(before, full)"), (BF + "\nfull = t.tensor([90, 91, 10, 15, 71])", "solve(before, full)"),
       ("before = t.tensor([3])\nfull = t.tensor([3, 4])", "solve(before, full)"), ("before = t.tensor([3, 3])\nfull = t.tensor([3, 4, 3])", "solve(before, full)")],
      ("    return (full[-len(before):] == before).all().item()", BF + "\nfull = t.tensor([90, 91, 10, 11, 576, 71])",
       "This compared the END of the joint encoding with the earlier ids. The earlier text sits at the beginning."),
      20, A, scaffold="    return (full[:_____] == before)._____().item()"),
    # ---- solo ------------------------------------------------------------------------------------------------
    D("s1", "solo", 0,
      "`prefix`, `reasoning` and `phrase` are 1-D tensors of token ids. A forced sequence is built to test the point after the first `b` reasoning tokens. Return the index of the logits row that scores the FIRST phrase token, as a Python int.",
      "prefix, reasoning, phrase, b", "Row that scores the first phrase token.",
      "    return len(prefix) + b - 1",
      [(PRE + "\nb = 2", "solve(prefix, reasoning, phrase, b)"), (PRE + "\nb = 5", "solve(prefix, reasoning, phrase, b)"),
       (PRE2 + "\nb = 0", "solve(prefix, reasoning, phrase, b)"), (PRE2 + "\nb = 3", "solve(prefix, reasoning, phrase, b)")],
      ("    return len(prefix) + b", PRE + "\nb = 2",
       "This is the position of the first phrase token. The row that scores it is the one before."),
      30, AN),
    D("s2", "solo", 0,
      "`prefix`, `reasoning` and `phrase` are 1-D tensors of token ids, and `bounds` is a 1-D tensor of points in the reasoning to test. Return the total length of the forced sequence for each point, as a tensor shaped like `bounds`.",
      "prefix, reasoning, phrase, bounds", "Length of each forced sequence.",
      "    return len(prefix) + bounds + len(phrase)",
      [(PRE + "\nbounds = t.tensor([0, 2, 5])", "solve(prefix, reasoning, phrase, bounds).tolist()"), (PRE + "\nbounds = t.tensor([3])", "solve(prefix, reasoning, phrase, bounds).tolist()"),
       (PRE2 + "\nbounds = t.tensor([1, 2])", "solve(prefix, reasoning, phrase, bounds).tolist()"), (PRE2 + "\nbounds = t.tensor([0, 1, 2, 3])", "solve(prefix, reasoning, phrase, bounds).tolist()")],
      ("    return len(prefix) + len(reasoning) + len(phrase) + 0 * bounds", PRE + "\nbounds = t.tensor([0, 2, 5])",
       "Every sequence was given the whole reasoning. A sequence holds only the reasoning up to its own point, so earlier points make shorter sequences."),
      30, A),
    D("s3", "solo", 0,
      "`before` holds the token ids of the text before a phrase, and `full` the ids of that text and the phrase encoded together. The ids before the phrase are unchanged in `full`. Return the phrase's token ids.",
      "before, full", "Phrase ids cut out of the joint encoding.",
      "    return full[len(before):]",
      [(BF + "\nfull = t.tensor([90, 91, 10, 11, 576, 71])", "solve(before, full).tolist()"), (BF + "\nfull = t.tensor([90, 91, 10, 11, 785])", "solve(before, full).tolist()"),
       ("before = t.tensor([3])\nfull = t.tensor([3, 4, 5, 6])", "solve(before, full).tolist()"), ("before = t.tensor([1, 2])\nfull = t.tensor([1, 2, 2, 1])", "solve(before, full).tolist()")],
      ("    return full[-2:]", BF + "\nfull = t.tensor([90, 91, 10, 11, 785])",
       "This assumed the phrase is always two tokens long. How many tokens a phrase takes depends on the context, so the cut must be made where the earlier ids end."),
      30, A, show=1),
    D("s4", "solo", 0,
      "`before` holds the token ids of the text before a phrase, and `full` the ids of that text and the phrase encoded together. `full` is at least as long as `before`. Return the first position at which `full` differs from `before`, or the length of `before` if they agree everywhere, as a Python int.",
      "before, full", "First position where the joint encoding departs from the earlier ids.",
      "    n = len(before)\n    differs = full[:n] != before\n    return t.where(differs, t.arange(n), n).min().item()",
      [(BF + "\nfull = t.tensor([90, 91, 10, 11, 576, 71])", "solve(before, full)"), (BF + "\nfull = t.tensor([90, 91, 10, 15, 71])", "solve(before, full)"),
       (BF + "\nfull = t.tensor([90, 7, 10, 7, 71])", "solve(before, full)"), ("before = t.tensor([4, 4])\nfull = t.tensor([5, 4, 4])", "solve(before, full)")],
      ("    n = len(before)\n    differs = full[:n] != before\n    return differs.sum().item()", BF + "\nfull = t.tensor([90, 7, 10, 7, 71])",
       "This counted how many positions differ. The question is where the first difference is, which is a position, not a count."),
      45, A, show=2),
    D("s5", "solo", 0,
      "`before` holds the token ids of the text before a phrase. `full_a` and `full_b` are joint encodings of that text with two different phrases, and the earlier ids are unchanged in both. Return True if the two phrases take the same number of tokens, else False.",
      "before, full_a, full_b", "Whether two phrases have equally many tokens here.",
      "    return len(full_a) == len(full_b)",
      [(BF + "\nfull_a = t.tensor([90, 91, 10, 11, 576, 71])\nfull_b = t.tensor([90, 91, 10, 11, 576, 72])", "solve(before, full_a, full_b)"),
       (BF + "\nfull_a = t.tensor([90, 91, 10, 11, 576, 71])\nfull_b = t.tensor([90, 91, 10, 11, 576])", "solve(before, full_a, full_b)"),
       ("before = t.tensor([1])\nfull_a = t.tensor([1, 2, 3])\nfull_b = t.tensor([1, 3, 2])", "solve(before, full_a, full_b)"),
       ("before = t.tensor([1])\nfull_a = t.tensor([1, 2])\nfull_b = t.tensor([1, 2, 2, 2])", "solve(before, full_a, full_b)")],
      ("    return t.equal(full_a, full_b)", BF + "\nfull_a = t.tensor([90, 91, 10, 11, 576, 71])\nfull_b = t.tensor([90, 91, 10, 11, 576, 72])",
       "This asked whether the two phrases are the same tokens. Two different phrases can still be equally long, which is what makes their scores comparable."),
      35, A),
    D("s6", "solo", 0,
      "`prefix` and `reasoning` are 1-D tensors of token ids. Return the ids of the text that comes before the phrase when the point after the first `b` reasoning tokens is tested.",
      "prefix, reasoning, b", "Ids before the phrase at point b.",
      "    return t.cat([prefix, reasoning[:b]])",
      [(PRE + "\nb = 2", "solve(prefix, reasoning, b).tolist()"), (PRE + "\nb = 0", "solve(prefix, reasoning, b).tolist()"),
       (PRE2 + "\nb = 3", "solve(prefix, reasoning, b).tolist()"), (PRE2 + "\nb = 2", "solve(prefix, reasoning, b).tolist()")],
      ("    return t.cat([prefix, reasoning[b:]])", PRE + "\nb = 2",
       "This kept the reasoning AFTER the point. The model must see only what it had written up to that point."),
      30, A),
    # ---- integrated ------------------------------------------------------------------------------------------
    D("i1", "integrated", 0,
      "`before` holds the token ids of the text before a phrase, `full` the ids of that text and the phrase encoded together, and `alone` the ids of the phrase encoded on its own. Return the phrase ids cut from `full` when the earlier ids are unchanged in it; otherwise return `alone`.",
      "before, full, alone", "Phrase ids in context, falling back to the stand-alone ids.",
      "    n = len(before)\n    if (full[:n] == before).all():\n        return full[n:]\n    return alone",
      [(BF + "\nfull = t.tensor([90, 91, 10, 11, 576, 71])\nalone = t.tensor([785, 71])", "solve(before, full, alone).tolist()"),
       (BF + "\nfull = t.tensor([90, 91, 10, 15, 71])\nalone = t.tensor([785, 71])", "solve(before, full, alone).tolist()"),
       ("before = t.tensor([1, 2])\nfull = t.tensor([1, 2, 3])\nalone = t.tensor([4])", "solve(before, full, alone).tolist()"),
       ("before = t.tensor([1, 2])\nfull = t.tensor([2, 2, 3])\nalone = t.tensor([4])", "solve(before, full, alone).tolist()")],
      ("    return alone", BF + "\nfull = t.tensor([90, 91, 10, 11, 576, 71])\nalone = t.tensor([785, 71])",
       "The stand-alone ids start with the start-of-line form of the first word. In the middle of a paragraph the tokenizer writes that word with a different id, and the joint encoding has it."),
      60, A),
    D("i2", "integrated", 0,
      "`prefix` and `reasoning` are 1-D tensors of token ids. `full` holds the ids of the prefix, the first `b` reasoning tokens and a phrase, encoded together, and the ids before the phrase are unchanged in it. Return the indices of the logits rows that score the phrase tokens, as a 1-D tensor.",
      "prefix, reasoning, b, full", "Rows of the logits that score the phrase.",
      "    start = len(prefix) + b\n    m = len(full) - start\n    return t.arange(start - 1, start - 1 + m)",
      [("prefix = t.tensor([90, 91, 92])\nreasoning = t.tensor([10, 11, 12])\nb = 2\nfull = t.tensor([90, 91, 92, 10, 11, 576, 71])", "solve(prefix, reasoning, b, full).tolist()"),
       ("prefix = t.tensor([90, 91, 92])\nreasoning = t.tensor([10, 11, 12])\nb = 0\nfull = t.tensor([90, 91, 92, 785, 71, 72])", "solve(prefix, reasoning, b, full).tolist()"),
       ("prefix = t.tensor([5])\nreasoning = t.tensor([1, 2, 3])\nb = 3\nfull = t.tensor([5, 1, 2, 3, 8])", "solve(prefix, reasoning, b, full).tolist()"),
       ("prefix = t.tensor([5, 5])\nreasoning = t.tensor([1, 2, 3])\nb = 1\nfull = t.tensor([5, 5, 1, 8, 9])", "solve(prefix, reasoning, b, full).tolist()")],
      ("    start = len(prefix) + b\n    m = len(full) - start\n    return t.arange(start, start + m)", "prefix = t.tensor([90, 91, 92])\nreasoning = t.tensor([10, 11, 12])\nb = 2\nfull = t.tensor([90, 91, 92, 10, 11, 576, 71])",
       "These are the phrase's own positions. The last of them is the final row, which scores a token after the phrase, and the row that scores the first phrase token is missing."),
      65, AN),
    D("i3", "integrated", 0,
      "`prefix` and `reasoning` are 1-D tensors of token ids. `full` holds the ids of the prefix, the first `b` reasoning tokens and a phrase encoded together, and `alone` the ids of the phrase encoded on its own. Return the forced sequence to give the model: `full` when the ids before the phrase are unchanged in it; otherwise the ids before the phrase followed by `alone`.",
      "prefix, reasoning, b, full, alone", "Forced sequence, with the fallback when the joint encoding changed earlier ids.",
      "    before = t.cat([prefix, reasoning[:b]])\n    n = len(before)\n    if (full[:n] == before).all():\n        return full\n    return t.cat([before, alone])",
      [("prefix = t.tensor([90, 91])\nreasoning = t.tensor([10, 11, 12])\nb = 2\nfull = t.tensor([90, 91, 10, 11, 576, 71])\nalone = t.tensor([785, 71])", "solve(prefix, reasoning, b, full, alone).tolist()"),
       ("prefix = t.tensor([90, 91])\nreasoning = t.tensor([10, 11, 12])\nb = 2\nfull = t.tensor([90, 91, 10, 15, 71])\nalone = t.tensor([785, 71])", "solve(prefix, reasoning, b, full, alone).tolist()"),
       ("prefix = t.tensor([5])\nreasoning = t.tensor([1, 2, 3])\nb = 0\nfull = t.tensor([5, 8])\nalone = t.tensor([8])", "solve(prefix, reasoning, b, full, alone).tolist()"),
       ("prefix = t.tensor([5])\nreasoning = t.tensor([1, 2, 3])\nb = 3\nfull = t.tensor([5, 1, 2, 4, 9])\nalone = t.tensor([8])", "solve(prefix, reasoning, b, full, alone).tolist()")],
      ("    return full", "prefix = t.tensor([90, 91])\nreasoning = t.tensor([10, 11, 12])\nb = 2\nfull = t.tensor([90, 91, 10, 15, 71])\nalone = t.tensor([785, 71])",
       "The joint encoding changed a token of the reasoning here, so the model would be scored on text it never wrote. The earlier ids have to be kept as they were."),
      70, A, show=1),
]
