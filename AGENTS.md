# Instructions for the agent on the Windows machine

You are a Claude Code session running on a Windows PC (AMD RX 6800 XT 16 GB, Ryzen 7 5700G 8 cores /
16 threads, 32 GB RAM at 3600 MT/s). The PC belongs to Seth's brother. Seth drives you remotely from
the browser. A second Claude session (the "Linux agent") runs on Seth's own Linux PC and shares this
repository with you.

## What this repository is for

Seth is building a course in Delta Drills (his tutoring app) on Goodfire's post
"Logits as a new monitor for evaluation awareness" (Pernice & Aranguri, LessWrong, 4 Jun 2026).
The course already teaches the ideas in words. This repository turns those ideas into **experiments a
learner can run**, at a cost of roughly zero:

1. You build and run the experiments on this PC with **small open models** (Qwen3-0.6B / 1.7B / 4B),
   using the same method as the paper.
2. Each experiment leaves behind a small cached dataset plus a reference solution.
3. Those become Delta Drills exercises: short single-step drills for the prerequisite skills, and
   multi-part "labs" where the learner predicts, computes and interprets. The Linux agent does the
   Delta Drills side. You never touch the Delta Drills repository.

The three questions the whole course is built around are in `docs/02-experiments.md`, section 0.
Every experiment should help answer one of them, or test a theory the course teaches.

## Read in this order

1. `docs/00-setup-windows.md`: environment, the CPU path that always works, GPU paths to try (time-boxed).
2. `docs/01-method.md`: the exact monitor. Do not change it without saying so in `STATUS.md`.
3. `docs/02-experiments.md`: the experiment catalogue, in the order to run it.
4. `docs/03-exercise-handoff.md`: what each finished experiment must hand over.
5. `docs/04-collab-protocol.md`: how you and the Linux agent talk through this repo.
6. `docs/05-research-loop.md`: you are also expected to *find* good experiments: propose ideas, pilot
   them cheaply, decide kill / pivot / scale from the results, and track how good your predictions are.
7. `course/`: read-only copy of the Delta Drills course this feeds (conceptual lessons and questions,
   the source papers' full text in `course/arena-eval-awareness/_sources/`, and the lm-1 code unit).
8. `reference/`: a validated implementation (run on Qwen3-32B on a rented GPU) and its results.

## Hard rules

- **Cost.** No paid APIs, no cloud GPUs, no Modal, no OpenAI/Anthropic API calls, unless Seth says yes
  in the chat for that specific run. Hugging Face downloads of open models and datasets are fine.
- **It is not your computer.** Before any run longer than 30 minutes, write the estimate in
  `STATUS.md` and ask Seth in the chat. Keep the PC usable: cap torch threads at 12
  (`torch.set_num_threads(12)`), keep RAM use under 24 GB, and make every long job resumable
  (append one JSON line per finished prompt; skip keys already in the file on restart).
- **No system changes without asking.** Installing Python packages into the project venv is fine.
  Drivers, Windows features (WSL, Hyper-V), registry, BIOS, other users' files: ask Seth first.
- **Validate, don't assert.** A number goes into `STATUS.md` or a PR only with the command that
  produced it and the checks from `docs/01-method.md` section 6 passing. If a check fails, say so;
  never loosen a check to make it pass.
- **Data hygiene.** Never commit model weights, full datasets, HF tokens or anything from `.env`.
  WildChat is real user text: commit prompt IDs and hashes, not full prompt text, except short
  excerpts an exercise needs. Each file under `runs/` stays under 5 MB; keep bigger files local and
  record their path and sha256.
- **Stay in your directories** (see `docs/04-collab-protocol.md`). To change `docs/`, `reference/`
  or this file, open an issue.
- **Seth's preferences for code:** prefer `einops.reduce` over `.sum(dim=...)` / `.max(dim=...)`;
  short functions; comments say why, not what. Write normal prose in docs and commits.
