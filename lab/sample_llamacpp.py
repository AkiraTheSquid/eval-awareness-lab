"""Sample reasoning rollouts on the GPU through llama.cpp's server.

Why: on this PC, HF `generate` on the CPU is bound by memory bandwidth
(~6 tok/s for 0.6B); llama-server with Vulkan on the RX 6800 XT gives
~66 tok/s for 1.7B BF16 on one slot (~340 with 8, not reproducible). Rollouts are
saved as token ids and scored afterwards with the HF model on the CPU
(monitor.score_rollout), so only sampling moves to the GPU.

Fidelity choices (docs/01-method.md section 3):
- BF16 GGUF, i.e. the released weights, not an 8-bit copy.
- The prompt is sent as token ids from the HF tokenizer, so the server
  never applies its own chat template or tokenisation.
- Sampler chain set explicitly to HF's order: top_k, then temperature,
  then top_p; min_p 0 (the server defaults to 0.05 and tempers last).
- Server runs on Windows; WSL reaches it through Windows' curl.exe
  because WSL's NAT networking cannot see Windows localhost.

Reproducibility (check 6): run the server with ONE slot (-np 1) and
cache_prompt off. lab/bench_determinism.py shows that a prompt first
processed alongside other requests (-np 8), or reused from the prompt
cache, diverges from a solo run of the same seed after a few hundred
tokens (GPU reductions change with batch shape and KV layout). One
slot gives ~66 tok/s for 1.7B, still faster than CPU scoring consumes
rollouts (~53 tok/s), so the end-to-end time is unchanged.
"""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

CURL = "/mnt/c/Windows/System32/curl.exe"
URL = "http://127.0.0.1:8089"
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                samplers=["top_k", "temperature", "top_p"],
                repeat_penalty=1.0, cache_prompt=False)


def post(path, body):
    out = subprocess.run(
        [CURL, "-s", "-X", "POST", URL + path, "-H",
         "Content-Type: application/json", "--data-binary", "@-"],
        input=json.dumps(body).encode(), capture_output=True,
        check=True)
    return json.loads(out.stdout)


def one_rollout(prefix_ids, seed, T, stop_ids):
    r = post("/completion", dict(
        prompt=prefix_ids, n_predict=T, seed=seed,
        return_tokens=True, stop=["</think>"], **SAMPLING))
    ids = r["tokens"]
    cut = next((i for i, x in enumerate(ids) if x in stop_ids),
                len(ids))
    finished = r["stop_type"] in ("eos", "word") or cut < len(ids)
    return {"seed": seed, "ids": ids[:cut], "finished": finished,
            "stop_type": r["stop_type"]}


def sample_prompt(prefix_ids, seed, R, T, stop_ids, pool):
    """R rollouts of one prompt, run concurrently on the server's
    slots. Rollout r uses seed * 1000 + r so it is reproducible on its
    own (the reference seeded once per prompt; HF and llama.cpp RNGs
    differ anyway, so exact reuse of those streams is impossible)."""
    futs = [pool.submit(one_rollout, prefix_ids, seed * 1000 + r, T,
                        stop_ids) for r in range(R)]
    return [f.result() for f in futs]


def run(jobs, out_path, R, T, stop_ids, slots=1):
    """jobs: dicts with key, prefix_ids, seed (+ anything to keep).
    Appends one JSON line per finished prompt; skips done keys."""
    try:
        done = {json.loads(x)["key"] for x in open(out_path)}
    except FileNotFoundError:
        done = set()
    todo = [j for j in jobs if j["key"] not in done]
    # two prompts in flight keeps all slots busy across prompt ends
    with ThreadPoolExecutor(slots) as pool, \
            ThreadPoolExecutor(2) as outer, open(out_path, "a") as fh:
        futs = [(j, outer.submit(sample_prompt, j["prefix_ids"],
                                 j["seed"], R, T, stop_ids, pool))
                for j in todo]
        for j, f in futs:
            rec = {k: v for k, v in j.items() if k != "prefix_ids"}
            rec["n_prompt"] = len(j["prefix_ids"])
            rec["rollouts"] = f.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print("sampled", j["key"], flush=True)
