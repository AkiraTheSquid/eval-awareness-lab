"""GPU sampling + CPU scoring on a few prompts: speed and fidelity.

usage: python lab/bench_pipeline.py Qwen/Qwen3-1.7B R T out_dir
Needs llama-server running the matching BF16 GGUF (see STATUS.md).
Reports sampling tok/s, CPU scoring s/rollout, a determinism check
(same seed twice), and how closely llama.cpp's next-token
probabilities match the HF model's along one rollout.
"""
import json
import os
import sys
import time

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

import monitor as M
import sample_llamacpp as S

t.set_num_threads(12)
NAME, R, T, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
    sys.argv[4]
os.makedirs(OUT, exist_ok=True)
QUESTIONS = [
    "How do I fix a flat bicycle tyre?",
    "What were the main causes of the French Revolution?",
    "Write a regular expression that matches an email address.",
]
tok = AutoTokenizer.from_pretrained(NAME)
stop_ids = {tok.convert_tokens_to_ids("</think>"), tok.pad_token_id,
            tok.convert_tokens_to_ids("<|im_end|>")}
jobs = []
for i, q in enumerate(QUESTIONS):
    text = M.render(tok, q)
    jobs.append(dict(key=f"q{i}", seed=i, prefix_text=text,
                     prefix_ids=tok.encode(text,
                                           add_special_tokens=False)))
path = os.path.join(OUT, "rollouts.jsonl")
if os.path.exists(path):
    os.remove(path)
t0 = time.time()
S.run(jobs, path, R, T, stop_ids)
t_samp = time.time() - t0
rows = [json.loads(x) for x in open(path)]
n_tok = sum(len(r["ids"]) for x in rows for r in x["rollouts"])
fin = sum(r["finished"] for x in rows for r in x["rollouts"])
print(f"SAMPLING {n_tok} tokens in {t_samp:.0f} s = "
      f"{n_tok / t_samp:.0f} tok/s; finished {fin}/{len(rows) * R}")

# determinism: rerun rollout 0 of prompt 0 alone, same seed
j = jobs[0]
again = S.one_rollout(j["prefix_ids"], j["seed"] * 1000, T, stop_ids)
same = again["ids"] == rows[0]["rollouts"][0]["ids"]
print("DETERMINISM same seed -> same ids:", same)

model = AutoModelForCausalLM.from_pretrained(NAME, dtype=t.float32)
model.eval()
is_end = M.sentence_end_table(tok)

# fidelity: HF vs llama.cpp next-token probs along rollout 0
r0 = rows[0]["rollouts"][0]["ids"][:200]
lc = S.post("/completion", dict(
    prompt=j["prefix_ids"] + r0, n_predict=1, n_probs=5,
    temperature=0.0))
with t.no_grad():
    lg = model(t.tensor([j["prefix_ids"] + r0])).logits[0, -1]
hf = lg.float().log_softmax(-1)
top = lc["completion_probabilities"][0]
top = top.get("top_logprobs") or top.get("top_probs")
gaps = [abs(hf[p["id"]].item() - p["logprob"]) for p in top]
print("FIDELITY top-5 next-token log-prob gap after 200 tokens:",
      round(max(gaps), 4), "nats; HF top-5",
      hf.topk(5).indices.tolist(), "llama.cpp",
      [p["id"] for p in top])

secs, nb = [], []
for x, j in zip(rows, jobs):
    for r in x["rollouts"]:
        bounds = M.boundary_positions(r["ids"], is_end)
        s = time.time()
        M.score_rollout(model, tok, j["prefix_text"], j["prefix_ids"],
                        r["ids"], bounds, [M.PRIMARY, M.CONTROL])
        secs.append(time.time() - s)
        nb.append(len(bounds))
        print(f"scored {x['key']} len {len(r['ids'])} bounds "
              f"{len(bounds)} in {secs[-1]:.1f} s", flush=True)
print(f"SCORING mean {sum(secs) / len(secs):.1f} s/rollout, "
      f"mean bounds {sum(nb) / len(nb):.0f}, "
      f"mean cot {n_tok / len(secs):.0f} tokens")
