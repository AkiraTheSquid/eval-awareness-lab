"""Is llama-server sampling reproducible for a fixed seed?

usage: python lab/bench_determinism.py MODEL
Runs the same (prompt, seed) three ways: alone; alone again after an
unrelated request (prompt cache changed); and alongside 7 other
requests (batched). Prints where each differs from the first.
"""
import sys
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

import monitor as M
import sample_llamacpp as S

tok = AutoTokenizer.from_pretrained(sys.argv[1])
stops = {tok.convert_tokens_to_ids("</think>"), tok.pad_token_id}
ids = lambda q: tok.encode(M.render(tok, q), add_special_tokens=False)
P, OTHER = ids("How do I fix a flat bicycle tyre?"), ids("Name a fruit.")


def diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                None if len(a) == len(b) else min(len(a), len(b)))


base = S.one_rollout(P, 5, 512, stops)["ids"]
again = S.one_rollout(P, 5, 512, stops)["ids"]
S.one_rollout(OTHER, 1, 64, stops)
after = S.one_rollout(P, 5, 512, stops)["ids"]
with ThreadPoolExecutor(8) as ex:
    fs = [ex.submit(S.one_rollout, P, 5 if i == 0 else 100 + i, 512,
                    stops) for i in range(8)]
    batched = fs[0].result()["ids"]
print("len", len(base))
print("alone again: first difference at", diff(base, again))
print("after other request:", diff(base, after))
print("batched with 7 others:", diff(base, batched))

# the order the E1 sampler uses: a fresh prompt first sampled in a
# batch (prompt not cached yet), then the same seed alone, T = 1024
Q = ids("What were the main causes of the French Revolution? (v2)")
with ThreadPoolExecutor(8) as ex:
    fs = [ex.submit(S.one_rollout, Q, 9 if i == 0 else 200 + i, 1024,
                    stops) for i in range(8)]
    first = fs[0].result()["ids"]
solo = S.one_rollout(Q, 9, 1024, stops)["ids"]
print("fresh prompt batched, then alone (T=1024):", diff(first, solo),
      "of", len(first))
