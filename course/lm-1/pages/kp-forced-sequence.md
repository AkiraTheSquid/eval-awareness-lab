---
kc: monitor.forced-sequence
title: Build the forced sequence and tokenise the phrase in context
new_syntax: []
concepts: [prefix-reasoning-phrase, tokenise-in-context]
supporting: ['monitor.next-token-logits', 'torch.stack-concat-interleave', 'python.control-flow', 'tensor.row-normalization']
previews: []
faded: [1837, 1838]
guided: []
independent: [1839, 1840, 1841, 1842, 1843, 1844]
integrated: [1845, 1846, 1847]
---

## Concept: The forced sequence is the prefix, the reasoning so far, then the phrase

The monitor asks one question many times: "at this point in the model's reasoning, how likely is it to write the phrase next?" Each time, the model is given a sequence made of three pieces of token ids, joined end to end.

1. The *prefix*: the user's prompt wrapped in the model's chat format, ending at the place where the model's reasoning begins. With the Hugging Face library this is the text from `apply_chat_template(..., add_generation_prompt=True)` followed by the reasoning tag `<think>` and a line break.
2. The *reasoning so far*: the first b tokens of the reasoning the model wrote, where b is the point being tested.
3. The *phrase*: the tokens of the fixed text, such as "The user might be testing".

Joining 1-D tensors end to end is `t.cat([prefix, reasoning[:b], phrase])`. The phrase then starts at position

$$\text{start} = \text{len(prefix)} + b$$

and `start` is what decides which rows of the logits are read: the rows from `start − 1` on.

The real calls that produce the prefix look like this. They need the model's tokenizer, so this cell is for reading only:

```python no-run
messages = [{"role": "user", "content": user_prompt}]
text = tok.apply_chat_template(
    messages, tokenize=False,
    add_generation_prompt=True) + "<think>\n"
prefix = t.tensor(
    tok.encode(text, add_special_tokens=False))
```

`add_special_tokens=False` matters: the chat template has already written the special tokens, and letting `encode` add its own would put a second start-of-text token in front.

## Worked example

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

## Faded practice

### q1837
`prefix`, `reasoning` and `phrase` are 1-D tensors of token ids. Return the sequence that tests the point after the first `b` reasoning tokens: the prefix, those `b` tokens, then the phrase.

```python starter
import torch as t


def solve(prefix, reasoning, phrase, b):
    return t.cat([prefix, reasoning[:_____], phrase])
```

```python solution
import torch as t


def solve(prefix, reasoning, phrase, b):
    return t.cat([prefix, reasoning[:b], phrase])
```

## Concept: Tokenise the phrase together with the text before it

Turning text into token ids is not done word by word in isolation. Most tokenizers fold a leading space into the token that follows it, so the same word becomes a different id depending on what comes before. In the Qwen3 tokenizer, "The" right after a line break is token 785, while " The" after a sentence (with its space) is token 576.

So encoding the phrase on its own and gluing the ids on gives the wrong tokens in the middle of a paragraph. The safe procedure is to encode the whole text at once and cut the phrase's ids out of the result.

1. Write the full text: everything before the phrase, a space unless that text ends in a line break, then the phrase.
2. Encode the full text. Call the result `full`.
3. Let n be the number of ids in the encoding of the text before the phrase, `before`. The phrase's ids are `full[n:]`.

Step 3 relies on the first n ids of `full` being exactly `before`. That is usually true and occasionally false: adding text can change how the last few characters before it are split. So the procedure checks it, `(full[:n] == before).all()`, and treats a failure as a case to handle instead of scoring the wrong tokens silently.

## Worked example

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

## Faded practice

### q1838
`before` holds the token ids of the text before a phrase, and `full` the ids of that text and the phrase encoded together. Return True if the ids before the phrase are unchanged in `full`, else False.

```python starter
import torch as t


def solve(before, full):
    return (full[:_____] == before)._____().item()
```

```python solution
import torch as t


def solve(before, full):
    return (full[:len(before)] == before).all().item()
```

## Solo practice

### q1839
`prefix`, `reasoning` and `phrase` are 1-D tensors of token ids. A forced sequence is built to test the point after the first `b` reasoning tokens. Return the index of the logits row that scores the FIRST phrase token, as a Python int.

### q1840
`prefix`, `reasoning` and `phrase` are 1-D tensors of token ids, and `bounds` is a 1-D tensor of points in the reasoning to test. Return the total length of the forced sequence for each point, as a tensor shaped like `bounds`.

### q1841
`before` holds the token ids of the text before a phrase, and `full` the ids of that text and the phrase encoded together. The ids before the phrase are unchanged in `full`. Return the phrase's token ids.

### q1842
`before` holds the token ids of the text before a phrase, and `full` the ids of that text and the phrase encoded together. `full` is at least as long as `before`. Return the first position at which `full` differs from `before`, or the length of `before` if they agree everywhere, as a Python int.

### q1843
`before` holds the token ids of the text before a phrase. `full_a` and `full_b` are joint encodings of that text with two different phrases, and the earlier ids are unchanged in both. Return True if the two phrases take the same number of tokens, else False.

### q1844
`prefix` and `reasoning` are 1-D tensors of token ids. Return the ids of the text that comes before the phrase when the point after the first `b` reasoning tokens is tested.

## Integrated practice

### q1845
`before` holds the token ids of the text before a phrase, `full` the ids of that text and the phrase encoded together, and `alone` the ids of the phrase encoded on its own. Return the phrase ids cut from `full` when the earlier ids are unchanged in it; otherwise return `alone`.

### q1846
`prefix` and `reasoning` are 1-D tensors of token ids. `full` holds the ids of the prefix, the first `b` reasoning tokens and a phrase, encoded together, and the ids before the phrase are unchanged in it. Return the indices of the logits rows that score the phrase tokens, as a 1-D tensor.

### q1847
`prefix` and `reasoning` are 1-D tensors of token ids. `full` holds the ids of the prefix, the first `b` reasoning tokens and a phrase encoded together, and `alone` the ids of the phrase encoded on its own. Return the forced sequence to give the model: `full` when the ids before the phrase are unchanged in it; otherwise the ids before the phrase followed by `alone`.

## Misconceptions

- **Include the whole reasoning before the phrase.** The test is about one point in the reasoning, so only the tokens up to that point go in.
- **The phrase always has the same token ids.** Its first token depends on whether a space or a line break comes before it.
- **Encode the phrase alone and append it.** That gives the start-of-line form of the first word. Encode the joined text and slice.
- **Slicing the joined encoding is always safe.** Only when the ids before the phrase are unchanged. Check them.
