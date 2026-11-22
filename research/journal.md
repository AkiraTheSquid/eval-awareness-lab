# Research journal (newest first; format: docs/05-research-loop.md §2 step 5)

## 2026-11-22 · X1 and X2 on the Qwen3-32B reference rollouts (no model time)
Command: `python lab/x1_x2.py` -> `runs/pilots/x1_x2_qwen3-32b-vast.json`. 67 matched pairs, R = 4.

- **X1** (P001, p = 0.6, miss). AUROC(asking) − AUROC(testing), paired bootstrap: k = 0 +0.049
  [−0.012, +0.107]; k = 1 +0.060 [−0.000, +0.118]; k = 4 +0.093 [+0.028, +0.160]. From the prompt
  alone, the control phrase's edge is weak and within noise. It roughly doubles as sampled reasoning
  is added. **Decision: pivot.** "Is the asking advantage a property of the reasoning?" moves into E5 on
  Qwen3-1.7B, which re-scores the same rollouts with more phrases.
- **X2** (P002, p = 0.5, miss, wrong sign). The winning boundary sits at a median of 0.54 of the way
  through the reasoning for eval prompts and 0.40 for deploy prompts (difference +0.148 [+0.004, +0.276]).
  Boundary 0 is a rollout's maximum in only 8% of rollouts. **Decision: pivot** to X3: does the eval
  score drift upward through the reasoning, and does the drift survive subtracting the control phrase?
  X3 is tested on the 1.7B E1 rollouts. This is not evidence yet: one interval only just clears 0, on
  67 pairs.
- What this taught me about choosing experiments: both seed ideas cost seconds and both came out
  differently from the prediction. On cached data, running the pilot is cheaper than arguing about
  the prior. [exercise] X1 makes a good "predict where the control phrase's advantage comes from"
  drill.
