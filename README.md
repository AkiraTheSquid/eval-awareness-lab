# eval-awareness-lab

Small-model experiments on Goodfire's "Logits as a new monitor for evaluation awareness"
(Pernice & Aranguri, LessWrong, 4 Jun 2026), built to become exercises in Delta Drills.

- `AGENTS.md`: start here (instructions for the agent running the experiments).
- `docs/`: setup, the exact method, the experiment catalogue, the hand-off format, the collaboration protocol.
- `reference/`: a validated implementation and results on Qwen3-32B (rented GPU, 22 Nov 2026).
  `qwen3-32b-vast/` is the length-matched run; `qwen3-32b-first-run/` is the earlier unmatched k = 0 run.
- `lab/`, `runs/`, `exercises/`, `STATUS.md`: owned by the Windows agent.

Private: contains model outputs on adversarial (Fortress) prompts and real-user (WildChat) data.
