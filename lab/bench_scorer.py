"""Old vs new scorer on real rollouts: agreement and wall time.

usage: python lab/bench_scorer.py Qwen/Qwen3-0.6B 512 [n_rollouts]
Samples rollouts on harmless questions, scores both phrases at every
boundary with (a) the reference's per-boundary scorer and (b)
monitor.score_rollout, and checks both against slow_score.
"""
import json
import sys
import time

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

import monitor as M

t.set_num_threads(12)
NAME = sys.argv[1]
T = int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
QUESTIONS = [
    "How do I fix a flat bicycle tyre?",
    "What were the main causes of the French Revolution?",
    "Write a regular expression that matches an email address.",
]
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=t.float32)
model.eval()
PAD = tok.pad_token_id
is_end = M.sentence_end_table(tok)
eos = model.generation_config.eos_token_id
stops = {tok.convert_tokens_to_ids("</think>"), PAD}
stops |= set(eos if isinstance(eos, list) else [eos])


@t.no_grad()
def ref_trace(prefix_text, prefix, cot, bounds, chunk=4):
    """The reference's scorer (reference/rerun_source.py `trace`):
    a padded pass over prefix + cot[:b] + phrase per boundary."""
    jobs = []
    for b in bounds:
        base = prefix + cot[:b]
        text = prefix_text + tok.decode(cot[:b])
        p = M.phrase_after(tok, base, text, M.PRIMARY)
        c = M.phrase_after(tok, base, text, M.CONTROL)
        assert len(p) == len(c) and p[:-1] == c[:-1]
        jobs.append((base + p, len(base), c[-1]))
    te, ak = [], []
    for s in range(0, len(jobs), chunk):
        part = jobs[s:s + chunk]
        width = max(len(q) for q, _, _ in part)
        ids = t.full((len(part), width), PAD)
        mask = t.zeros((len(part), width), dtype=t.int64)
        for i, (q, _, _) in enumerate(part):
            ids[i, :len(q)] = t.tensor(q)
            mask[i, :len(q)] = 1
        h = model.model(input_ids=ids,
                        attention_mask=mask).last_hidden_state
        for row, (q, st, swap) in zip(h, part):
            lp = model.lm_head(row[st - 1:len(q) - 1]).log_softmax(-1)
            ph = t.tensor(q[st:])
            te.append(lp[t.arange(len(ph)), ph].sum().item())
            ph[-1] = swap
            ak.append(lp[t.arange(len(ph)), ph].sum().item())
    return te, ak


rows = []
for qi, q in enumerate(QUESTIONS[:N]):
    prefix_text = M.render(tok, q)
    prefix = tok.encode(prefix_text, add_special_tokens=False)
    t.manual_seed(qi)
    t0 = time.time()
    with t.no_grad():
        gen = model.generate(
            t.tensor([prefix]), do_sample=True, temperature=0.6,
            top_p=0.95, top_k=20, max_new_tokens=T,
            pad_token_id=PAD)[0, len(prefix):].tolist()
    t_gen = time.time() - t0
    n = next((i for i, x in enumerate(gen) if x in stops), len(gen))
    cot = gen[:n]
    bounds = M.boundary_positions(cot, is_end)

    t0 = time.time()
    ref_t, ref_a = ref_trace(prefix_text, prefix, cot, bounds)
    t_ref = time.time() - t0
    t0 = time.time()
    new = M.score_rollout(model, tok, prefix_text, prefix, cot,
                          bounds, [M.PRIMARY, M.CONTROL])
    t_new = time.time() - t0

    gap_ref = max(abs(a - b) for a, b in
                  zip(ref_t + ref_a,
                      new[M.PRIMARY] + new[M.CONTROL]))
    # slow scorer on up to 4 boundaries spread over the rollout
    pick = sorted({0, len(bounds) // 3, 2 * len(bounds) // 3,
                   len(bounds) - 1})
    gap_slow = 0.0
    for i in pick:
        b = bounds[i]
        base = prefix + cot[:b]
        text = prefix_text + tok.decode(cot[:b])
        for ph in (M.PRIMARY, M.CONTROL):
            seq = base + M.phrase_after(tok, base, text, ph)
            s = M.slow_score(model, seq, len(base))
            gap_slow = max(gap_slow, abs(s - new[ph][i]))
    row = dict(question=qi, prompt_tokens=len(prefix), cot_tokens=n,
               boundaries=len(bounds), gen_s=round(t_gen, 1),
               gen_tok_per_s=round(len(gen) / t_gen, 1),
               ref_score_s=round(t_ref, 2),
               new_score_s=round(t_new, 2),
               speedup=round(t_ref / t_new, 1),
               max_gap_new_vs_ref=gap_ref,
               max_gap_new_vs_slow=gap_slow)
    print(json.dumps(row), flush=True)
    rows.append(row)
print("WORST gap vs ref", max(r["max_gap_new_vs_ref"] for r in rows),
      "vs slow", max(r["max_gap_new_vs_slow"] for r in rows))
