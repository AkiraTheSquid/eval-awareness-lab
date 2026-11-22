# Status (Windows agent keeps this current)

## Environment
- WSL2 Ubuntu, repo at `~/eval-awareness-lab`, uv venv `.venv` (torch 2.15 CPU, transformers 5.24).
  Torch capped at 12 threads.
- Sampling runs on the GPU through llama.cpp b12633 (Vulkan, RX 6800 XT) with BF16 GGUFs in
  `C:\Users\ethan\gguf\` (the released weights, converted, not quantised). Server command:
  `llama-server.exe -m Qwen3-1.7B-BF16.gguf -ngl 99 -np 1 -c 4096 --host 127.0.0.1 --port 8089 --no-webui`.
  WSL reaches it through `curl.exe` (`lab/sample_llamacpp.py`). Scoring runs on the CPU with the HF model in fp32.
- No system changes made: no ROCm, no drivers. GPU route C from `docs/00` (llama.cpp) works for sampling.
  Torch-on-GPU (route B) was not tried.

## Method changes (docs/01-method.md), say-so list
1. **Scorer** (`lab/monitor.py: score_rollout`). The method is the same; only the arithmetic is
   reorganised. One KV-cached pass covers prompt + reasoning. Then every (boundary, phrase) branch is
   packed into one pass, using a 4-D mask so each branch sees only the context before its boundary.
   It matches the slow scorer to 3.4e-5 nats (check 3 limit 0.15) and is 6 to 12× faster than the
   reference `trace`. The speedup grows with rollout length.
2. **Sampling** uses llama.cpp, not HF `generate`. The sampler settings are the same (top_k 20 →
   temperature 0.6 → top_p 0.95, min_p 0). The prompt goes in as HF token ids. Fidelity check: after
   200 reasoning tokens, llama.cpp's top-5 next tokens are the same as HF fp32's, with a max log-prob gap
   of 0.012 nats (`lab/bench_pipeline.py`). Seeds: rollout r of a prompt uses seed·1000 + r.
3. **Determinism (check 6)** holds only with one server slot and the prompt cache off. With 8 slots,
   or with a cached prompt, the same seed diverges after 119–264 tokens (`lab/bench_determinism.py`).
   All runs use `-np 1`, `cache_prompt=false`.
4. **"Primary − control"**: §5 of the method says "same boundary, then max". The reference run's
   0.448 used max(primary) − max(control). `lab/analysis.py` reports both.

## Speed table (Phase 0 benchmark)
| step | model | measured | command |
|---|---|---|---|
| CPU matmul | – | ~420 GFLOPS at 12 threads | `lab/bench_matmul.py` |
| sampling, HF CPU | 0.6B fp32 | ~5.6 tok/s | (earlier session) |
| sampling, llama.cpp GPU, 1 slot | 1.7B BF16 | ~66 tok/s | `lab/bench_pipeline.py` |
| sampling, llama.cpp GPU, 8 slots | 1.7B BF16 | 340 tok/s (not reproducible, unused) | `lab/bench_pipeline.py Qwen/Qwen3-1.7B 8 1024 /tmp/pipe17` |
| scoring, reference `trace` vs new | 0.6B fp32 | 614-token rollout: 54.6 s → 4.6 s | `lab/bench_scorer.py` |
| scoring, new | 1.7B fp32 | 16.5 s/rollout (mean 879 tokens, 59 boundaries) | `lab/bench_pipeline.py` |
| finished within T=1024 | 1.7B | 14/24 rollouts | same |

**E1 on 1.7B (100 pairs, R = 8, T = 1,024): about 7.5 h.** CPU scoring is the bottleneck
(1,600 × 16.5 s); GPU sampling at ~66 tok/s keeps ahead and runs at the same time. The reference
scorer would have taken roughly 100 h.

## Current run
None. E1 on 1.7B (about 7.5 h) is waiting for Seth's go-ahead. Machine sharing: see `C:SERSETHANGENT-COORDRESOURCES.MD`. THIS AGENT USES 10 THREADS AND AT MOST 14 GB; WSL NOW HAS 24 GB.

## Predictions written before each run
In `research/predictions.csv`: P003–P005 (E1 on 1.7B), written before any E1 result.

## Latest result per experiment
- **E0** (done): `lab/e0_reference.py` reproduces all 13 AUROCs of the Qwen3-32B reference
  summary to 3 decimals. Table, plot and variance split are in `runs/e0/`. Variance of the per-rollout
  max: between prompts 55.8, within prompts 9.2, so about 86% of the variance sits between prompts
  (first look at Q2). The same-boundary primary − control score is 0.423 [0.33, 0.52] at k = 4.
- **X1, X2 pilots** (reference data): see `research/journal.md`. Both predictions missed. X3 was opened.
- **Pairs**: `lab/data.py` → `runs/pairs/pairs.json` has 474 matched pairs (Fortress 489, WildChat pool 13,621).
  Check 1 passes. The held-out half (237 pairs) is in `research/heldout_ids.json`. E1 uses the
  exploration half.
- **E1 smoke, Qwen3-0.6B** (10 pairs, R = 8, T = 1,024; `python lab/e1.py sample|checks|report Qwen/Qwen3-0.6B 10 8 1024`).
  Checks 2 to 6 pass: scorer agreement 3.8e-5 nats on 3 real rollouts; planted signal rose 6/6 in
  the prompt and 6/6 in the reasoning; same seed gives the same ids. 140/160 rollouts finished; 0
  keyword-flagged. AUROCs on 10 pairs are too wide to read (monitor k = 8: 0.43 [0.17, 0.72]). They
  only show that the pipeline runs. Table: `runs/e1/Qwen3-0.6B_N10_R8_T1024/table.md`.
