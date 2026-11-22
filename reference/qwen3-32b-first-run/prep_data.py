"""Build the two prompt sets johntrob14 uses for the primary contrast: fortress_adv vs wildchat_benign.

Fortress: same repo, revision and excluded ids as their config. WildChat: same repo, revision, filters and
seed, but drawn from shard 0 only (231 MB of 3.4 GB), so the 489 sampled prompts are NOT their 489.
"""
import json
import random
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]
EXCLUDED = {828, 861, 869, 901, 906, 921, 964, 371, 422, 424, 836}

f = hf_hub_download("ScaleAI/fortress_public", "data/train-00000-of-00001.parquet", repo_type="dataset",
                    revision="0c096becbc75bb12065c8059a53960c7f0d4d35c")
rows = pq.read_table(f).to_pylist()
print("fortress columns:", list(rows[0].keys()))
fort = sorted((r["ID"], r["adversarial_prompt"]) for r in rows if r["ID"] not in EXCLUDED)
max_chars = max(len(r["adversarial_prompt"]) for r in rows)
print("fortress_adv", len(fort), "max chars", max_chars)

w = hf_hub_download("allenai/WildChat-1M", "data/train-00000-of-00014.parquet", repo_type="dataset",
                    revision="7d6490e462285cf85d91eabea0f9a954fbddcd1f")
t = pq.read_table(w, columns=["conversation_hash", "turn", "language", "toxic", "conversation"])
pool, seen = [], set()
for r in t.to_pylist():
    if r["language"] != "English" or r["turn"] != 1 or r["toxic"]:
        continue
    text = r["conversation"][0]["content"]
    if not text or not text.strip() or text in seen or len(text) > max_chars:
        continue
    seen.add(text)
    pool.append((r["conversation_hash"], text))
pool.sort()
picked = random.Random("20260926|wildchat_benign").sample(pool, 489)
print("wildchat pool", len(pool), "picked", len(picked))
json.dump({"fortress_adv": [{"id": i, "text": x} for i, x in fort],
           "wildchat_benign": [{"id": i, "text": x} for i, x in sorted(picked)]}, open(OUT, "w"))
