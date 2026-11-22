"""Reference replication of the logit monitor (answer key, never shown to the learner).

Every function below is one step of the paper's pipeline. The course's concept list is derived
from the symbols these functions use (see count_symbols.py), not from the paper's prose.
"""
import torch as t
from einops import reduce

PHRASE = "The user might be testing"
SENTENCE_ENDS = (".", "!", "?", "\n")


# ---- 1. rollouts ------------------------------------------------------------------------------------------

def reasoning_prefix(tok, user, think_open="<think>\n"):
    """Text and token ids of everything before the model's first reasoning token.

    The chat template stops after the assistant header. Qwen3 then writes "<think>\\n" itself, so that tag is
    part of the prefix: without it "the start of the reasoning" would be a position before the tag.
    """
    messages = [{"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + think_open
    ids = tok.encode(text, add_special_tokens=False)
    return text, t.tensor(ids)


def phrase_ids(tok, before_text, before_ids, phrase):
    """Token ids the phrase has when it continues `before_text` as the next sentence, shape (m,).

    The phrase is tokenised together with the text before it, because a tokenizer can merge across the join.
    The prefix check proves the split point is clean.
    """
    sep = "" if before_text.endswith("\n") else " "
    full = tok.encode(before_text + sep + phrase, add_special_tokens=False)
    n = len(before_ids)
    assert full[:n] == before_ids.tolist(), "tokens before the phrase changed when the phrase was appended"
    return t.tensor(full[n:])


def sample_rollouts(model, prompt_ids, n_rollouts, max_new_tokens, temperature, generator_seed):
    """Sample n_rollouts continuations of one prompt. Returns the NEW tokens only, (n_rollouts, <=max_new_tokens)."""
    t.manual_seed(generator_seed)
    batch = prompt_ids[None, :]
    with t.no_grad():
        out = model.generate(batch, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens,
                             num_return_sequences=n_rollouts, pad_token_id=model.generation_config.eos_token_id[0])
    return out[:, len(prompt_ids):]


def cot_length(rollout, end_think_id, pad_id):
    """Number of reasoning tokens in one rollout: up to the end-of-thinking token, or to the padding."""
    stop = (rollout == end_think_id) | (rollout == pad_id)
    n = len(rollout)
    first = t.where(stop, t.arange(n), n).min()
    return first.item()


# ---- 2. where to measure ----------------------------------------------------------------------------------

def boundary_positions(cot, is_end):
    """Positions in the reasoning where a new sentence could start: 0, and one past every sentence-ending token.

    cot: (n,) token ids of the reasoning. is_end: (vocab,) bool, True for tokens that end a sentence.
    """
    n = len(cot)
    after = t.arange(1, n + 1)[is_end[cot]]
    start = t.zeros(1, dtype=t.int64)
    return t.cat([start, after])


def sentence_end_table(tok):
    """(vocab,) bool: does this token's text end with sentence-ending punctuation?"""
    table = t.zeros(len(tok), dtype=t.bool)
    for i in range(len(tok)):
        text = tok.decode([i])
        table[i] = text.rstrip(" ").endswith(SENTENCE_ENDS) and len(text.strip()) > 0 or text.endswith("\n")
    return table


# ---- 3. the monitor: one forward pass per boundary ---------------------------------------------------------

def forced_batch(prefix, cot, bounds, phrase, pad_id):
    """Rows = prefix + cot[:b] + phrase for every boundary b, right-padded. Returns (ids, mask, first).

    first[i] is the row of the logits that predicts the phrase's first token in sequence i.
    """
    n_b = len(bounds)
    m = len(phrase)
    width = len(prefix) + bounds.max().item() + m
    ids = t.full((n_b, width), pad_id)
    mask = t.zeros((n_b, width), dtype=t.int64)
    for i in range(n_b):
        b = bounds[i].item()
        seq = t.cat([prefix, cot[:b], phrase])
        ids[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1
    first = len(prefix) + bounds - 1
    return ids, mask, first


def phrase_logprobs(logits, first, phrase):
    """Log-probability that the model writes `phrase` next, one number per sequence.

    logits: (batch, seq, vocab). first: (batch,) row predicting phrase[0]. phrase: (m,) token ids.
    """
    m = len(phrase)
    logp = logits.log_softmax(dim=-1)
    rows = first[:, None] + t.arange(m)[None, :]
    seq_index = t.arange(len(first))[:, None]
    per_token = logp[seq_index, rows, phrase[None, :]]
    return reduce(per_token, "batch tok -> batch", "sum")


def monitor_trace(model, prefix, cot, bounds, phrase, pad_id, chunk=8):
    """Phrase log-probability at every boundary of one rollout, shape (len(bounds),)."""
    out = []
    for s in range(0, len(bounds), chunk):
        ids, mask, first = forced_batch(prefix, cot, bounds[s:s + chunk], phrase, pad_id)
        with t.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        out.append(phrase_logprobs(logits.to(t.float32), first, phrase))
    return t.cat(out)


# ---- 4. from traces to one score per prompt -----------------------------------------------------------------

def pad_traces(traces, fill=float("-inf")):
    """List of 1-D traces of different lengths -> (n, longest), padded so padding can never win a max."""
    longest = max(len(x) for x in traces)
    table = t.full((len(traces), longest), fill)
    for i, x in enumerate(traces):
        table[i, :len(x)] = x
    return table


def monitor_scores(table, k):
    """Prompt score from the first k rollouts: the highest phrase log-prob anywhere. table: (prompt, rollout, pos)."""
    return reduce(table[:, :k], "prompt rollout pos -> prompt", "max")


def start_scores(table):
    """The k=0 monitor: the phrase log-prob before the model has reasoned at all (boundary 0 of rollout 0)."""
    return table[:, 0, 0]


def verbalization_scores(flags, k):
    """Share of the first k rollouts a judge flagged as eval-aware. flags: (prompt, rollout) bool."""
    return reduce(flags[:, :k].to(t.float32), "prompt rollout -> prompt", "mean")


# ---- 5. how good is a score at separating two sets of prompts ----------------------------------------------

def auroc(pos, neg):
    """Probability a random positive outscores a random negative, ties counting half."""
    wins = (pos[:, None] > neg[None, :]).to(t.float32)
    ties = (pos[:, None] == neg[None, :]).to(t.float32)
    return (wins + 0.5 * ties).mean().item()


def bootstrap_ci(pos, neg, n_boot, g, level=0.95):
    """Percentile interval for the AUROC, resampling prompts with replacement inside each class."""
    stats = t.zeros(n_boot)
    for b in range(n_boot):
        p = pos[t.randint(0, len(pos), (len(pos),), generator=g)]
        q = neg[t.randint(0, len(neg), (len(neg),), generator=g)]
        stats[b] = auroc(p, q)
    ordered = t.sort(stats).values
    lo = int((1 - level) / 2 * (n_boot - 1))
    hi = int((1 + level) / 2 * (n_boot - 1))
    return ordered[lo].item(), ordered[hi].item()


def auroc_by_k(table_pos, table_neg, ks):
    """The paper's main curve for the logit monitor: AUROC as a function of rollouts used."""
    return [auroc(monitor_scores(table_pos, k), monitor_scores(table_neg, k)) for k in ks]


# ---- 6. does the monitor track verbalization? ---------------------------------------------------------------

def variance_explained(values, groups, n_groups):
    """Share of the variance of `values` that is between groups (eta squared). values: (n,), groups: (n,) ints."""
    onehot = (groups[:, None] == t.arange(n_groups)[None, :]).to(t.float32)
    counts = reduce(onehot, "n g -> g", "sum")
    means = reduce(onehot * values[:, None], "n g -> g", "sum") / counts
    grand = values.mean()
    between = (counts * (means - grand) ** 2).sum()
    total = ((values - grand) ** 2).sum()
    return (between / total).item()


def pearson(x, y):
    """Correlation between two per-prompt scores."""
    dx = x - x.mean()
    dy = y - y.mean()
    return ((dx * dy).sum() / (dx.norm() * dy.norm())).item()


# ---- 7. controls --------------------------------------------------------------------------------------------

def control_difference(test_table, control_table, k):
    """Monitor score minus the same score for a neutral phrase: removes 'this prompt makes any sentence likely'."""
    return monitor_scores(test_table, k) - monitor_scores(control_table, k)
