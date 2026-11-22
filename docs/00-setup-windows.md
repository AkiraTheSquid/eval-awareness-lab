# 00 · Setup on the Windows PC

Goal of this step: a Python environment that can load Qwen3 models, sample reasoning, and score
phrases, plus a measured speed table so every later run has an honest time estimate.

## 1. The hardware problem in one paragraph

The GPU is an AMD RX 6800 XT (RDNA2, `gfx1030`, 16 GB). PyTorch's GPU support on AMD is ROCm.
As of late 2026, AMD's official PyTorch wheels for Windows list RX 9000 / RX 7000 / Ryzen AI parts,
not RDNA2, and AMD's WSL support targets RDNA3 and newer. Community builds for `gfx1030` on Windows
exist but are unofficial. So: **build everything on the CPU path first** (it always works and is
enough for Qwen3-0.6B and 1.7B), then try the GPU routes in section 4 with a time box. Never block an
experiment on the GPU.

Sources: [AMD Windows PyTorch release notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-2.html),
[ROCm/TheRock discussion on gfx1030 for Windows](https://github.com/ROCm/TheRock/discussions/1612),
[community gfx1030 Windows build](https://github.com/ssubedir/RCOm-windows-gfx1030),
[AMD WSL support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html).

## 2. Environment (CPU path)

Use PowerShell. `uv` is the simplest way to get a clean Python on Windows.

```powershell
winget install --id=astral-sh.uv -e          # ask Seth first if winget is not already allowed
cd <this repo>
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers accelerate einops numpy pyarrow huggingface_hub scikit-learn matplotlib tqdm
```

Hugging Face:

- Put the cache on a drive with at least 40 GB free: `setx HF_HOME D:\hf` (any drive; record it in `STATUS.md`).
- `hf auth login` with a read token. `ScaleAI/fortress_public` may be gated: if the download says
  you must accept terms, stop and ask Seth which HF account to use. Do not create accounts.
- `allenai/WildChat-1M`: only one shard is needed (`data/train-00000-of-00014.parquet`).
- Pin the dataset revisions used by the reference run (see `reference/rerun_source.py`, the data cell)
  so results are comparable.

Disk for weights (bf16 on disk): Qwen3-0.6B ≈ 1.5 GB, 1.7B ≈ 4 GB, 4B ≈ 8 GB, 8B ≈ 16 GB.

RAM rule (32 GB box, 24 GB budget for us): fp32 needs 4 bytes per parameter, bf16 needs 2.
0.6B and 1.7B fit in fp32; 4B fits in fp32 (≈16 GB) only if nothing else is big; 8B only in bf16.

## 3. Phase 0 benchmark (required before any experiment)

Write `lab/bench.py` and run it. It must record, for each of Qwen3-0.6B and Qwen3-1.7B
(and 4B if RAM allows), in both fp32 and bf16:

| measurement | how |
|---|---|
| load time and peak RAM | `psutil.Process().memory_info().rss` after load |
| sampling speed | tokens/second for `generate` with R = 1 and R = 8 sequences, 256 new tokens, `temperature=0.6, top_p=0.95, top_k=20` |
| scoring speed | seconds for one forward pass over a batch of 8 sequences of 1,000 tokens |

Why both dtypes: the Ryzen 5700G (Zen 3) has no native bf16 matrix instructions, so bf16 can be
*slower* than fp32 on this CPU. Measure, do not guess. Prefer fp32 for scoring when it fits, because
the experiments compare small log-probability differences.

Put the table in `STATUS.md` with the exact command. Then estimate, for the core replication
(experiment E1: 100 length-matched pairs, R = 8 rollouts, T = 1,024 reasoning tokens, both phrases at
every sentence boundary), how many hours each model size would take. Ask Seth before starting
anything over 30 minutes.

## 4. GPU routes (optional, 45 minutes each, stop at the first that works)

Each route counts as working only if it passes the **CPU agreement check**: score the same 3
sequences (prompt + reasoning + phrase) on CPU fp32 and on the GPU route; the summed phrase
log-probabilities must agree within 0.05 nats. Record the result either way in `STATUS.md`.

**A. llama.cpp with Vulkan, for sampling only.** Vulkan works on RDNA2 under Windows without ROCm.
Download a prebuilt Windows Vulkan release of llama.cpp (a zip, no installer) and a Q8_0 GGUF of the
Qwen3 model. Run `llama-server` and call the raw `/completion` endpoint with `prompt` set to the exact
rendered prefix string from `docs/01-method.md` (do not let the server apply its own chat template),
the same sampling settings, and a per-request seed. This only speeds up *sampling* the reasoning; the
returned text is then tokenised and scored with the Hugging Face model on CPU as usual. Caveat to
record in every run that uses it: rollouts come from an 8-bit quantised copy of the model, while
scoring uses the full-precision one. The agreement check above does not apply here (it is not a
scorer); instead, check that the reasoning looks like the HF model's on 3 prompts.

**B. Community PyTorch ROCm build for gfx1030 on Windows** (link above). Unofficial. Install it into a
*separate* venv (`.venv-rocm`) so the CPU venv stays intact. Working means
`torch.cuda.is_available()` is true, a 4096×4096 matmul matches CPU, and the agreement check passes.
With 16 GB VRAM: 4B in bf16 fits; 8B in bf16 does not (no bitsandbytes on AMD).

**C. torch-directml.** Lowest priority: op coverage gaps make it unreliable for `generate`.

Do **not** set up WSL2 + ROCm, install drivers, or change Windows features without Seth's explicit yes.
