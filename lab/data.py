"""Fortress (evaluation) vs WildChat (deployment) prompts, matched on
length; the same rule as reference/rerun_source.py.

usage: python lab/data.py        (writes runs/pairs/ and the held-out split)

For each Fortress prompt, in a seeded random order, take the unused
WildChat prompt closest in token length; keep the pair if the lengths
agree within max(2, 10%). Qwen3 0.6B / 1.7B / 4B share one tokenizer,
so one pairing serves every model (E12 needs the same pairs).

What is committed: prompt ids, token lengths and sha256 of the text,
never WildChat text (docs: data hygiene). The text is re-read from the
pinned dataset revisions by `load_pairs`.
"""
import bisect
import hashlib
import json
import os
import random

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

EXCLUDED = {828, 861, 869, 901, 906, 921, 964, 371, 422, 424, 836}
MAX_PROMPT = 1500
LENGTH_TOL = 0.10
SEED = 0
TOKENIZER = "Qwen/Qwen3-1.7B"
PAIRS_PATH = "runs/pairs/pairs.json"
HELDOUT_PATH = "research/heldout_ids.json"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def fortress():
    f = hf_hub_download(
        "ScaleAI/fortress_public", "data/train-00000-of-00001.parquet",
        repo_type="dataset",
        revision="0c096becbc75bb12065c8059a53960c7f0d4d35c")
    return sorted((str(r["ID"]), r["adversarial_prompt"])
                  for r in pq.read_table(f).to_pylist()
                  if r["ID"] not in EXCLUDED)


def wildchat(max_chars, toxic=False):
    """First-turn English WildChat prompts from shard 0, deduplicated.
    toxic=False is the reference's deployment set; toxic=True gives
    E7's toxic set."""
    w = hf_hub_download(
        "allenai/WildChat-1M", "data/train-00000-of-00014.parquet",
        repo_type="dataset",
        revision="7d6490e462285cf85d91eabea0f9a954fbddcd1f")
    cols = ["conversation_hash", "turn", "language", "toxic",
            "conversation"]
    out, seen = [], set()
    for r in pq.read_table(w, columns=cols).to_pylist():
        if r["language"] != "English" or r["turn"] != 1 \
                or r["toxic"] != toxic:
            continue
        text = r["conversation"][0]["content"]
        if not text or not text.strip() or text in seen \
                or len(text) > max_chars:
            continue
        seen.add(text)
        out.append((r["conversation_hash"], text))
    return sorted(out)


def match(f_len, w_len, seed=SEED):
    """[(fortress index, wildchat index)] in a random order, so the
    first N pairs are a random sample."""
    free = sorted((n, i) for i, n in enumerate(w_len) if n <= MAX_PROMPT)
    rng = random.Random(seed)
    pairs = []
    for fi in rng.sample(range(len(f_len)), len(f_len)):
        n = f_len[fi]
        if n > MAX_PROMPT or not free:
            continue
        j = bisect.bisect_left(free, (n, -1))
        c = min((c for c in (j - 1, j) if 0 <= c < len(free)),
                key=lambda c: abs(free[c][0] - n))
        if abs(free[c][0] - n) <= max(2, LENGTH_TOL * n):
            pairs.append((fi, free.pop(c)[1]))
    return pairs


def check_pairs(pairs, f_len, w_len):
    """Check 1 of docs/01-method.md section 6."""
    fs, ws = [a for a, _ in pairs], [b for _, b in pairs]
    assert len(set(fs)) == len(fs) and len(set(ws)) == len(ws), \
        "a prompt is used twice"
    assert all(abs(f_len[a] - w_len[b]) <= max(2, LENGTH_TOL * f_len[a])
               for a, b in pairs), "pair outside tolerance"


def build(tok):
    fort = fortress()
    wild = wildchat(max(len(x) for _, x in fort))
    n_tok = lambda xs: [len(x) for x in
                        tok(xs, add_special_tokens=False)["input_ids"]]
    f_len, w_len = n_tok([x for _, x in fort]), n_tok([x for _, x in wild])
    pairs = match(f_len, w_len)
    check_pairs(pairs, f_len, w_len)
    rows = [{"pair": p, "eval_id": fort[a][0], "deploy_id": wild[b][0],
             "eval_len": f_len[a], "deploy_len": w_len[b],
             "eval_sha": sha(fort[a][1]), "deploy_sha": sha(wild[b][1])}
            for p, (a, b) in enumerate(pairs)]
    return rows, fort, wild


def split(rows, seed=SEED):
    """Held-out half for confirmation runs (docs/05 section 1)."""
    ids = [r["pair"] for r in rows]
    held = sorted(random.Random(seed + 1).sample(ids, len(ids) // 2))
    return held


def load_pairs(tok=None):
    """[(pair row, eval text, deploy text)], text checked by hash."""
    rows = json.load(open(PAIRS_PATH))["pairs"]
    fort = dict(fortress())
    wild = dict(wildchat(max(len(x) for x in fort.values())))
    out = []
    for r in rows:
        e, d = fort[r["eval_id"]], wild[r["deploy_id"]]
        assert sha(e) == r["eval_sha"] and sha(d) == r["deploy_sha"]
        out.append((r, e, d))
    return out


def held_out():
    return set(json.load(open(HELDOUT_PATH))["pairs"])


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rows, fort, wild = build(tok)
    os.makedirs(os.path.dirname(PAIRS_PATH), exist_ok=True)
    json.dump({"tokenizer": TOKENIZER, "seed": SEED,
               "length_tol": LENGTH_TOL, "max_prompt": MAX_PROMPT,
               "n_fortress": len(fort), "n_wildchat_pool": len(wild),
               "pairs": rows}, open(PAIRS_PATH, "w"), indent=0)
    if not os.path.exists(HELDOUT_PATH):
        json.dump({"seed": SEED + 1, "note": "pair indices reserved for "
                   "confirmation runs; pilots never use them",
                   "pairs": split(rows)}, open(HELDOUT_PATH, "w"))
    print(f"fortress {len(fort)}, wildchat pool {len(wild)}, "
          f"matched pairs {len(rows)}")
