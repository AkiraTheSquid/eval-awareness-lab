# 06 · Spec: research taste, pass 1

Status: signed off 2026-11-22 (all four decisions as recommended). Part A built; part B in progress.

## Goal

Train Seth's research taste on this topic: he predicts before every result, writes the code that
tests it, then judges the result. His calibration is scored over time. Until now the agents made
every prediction (P001 to P005) and Seth made none.

## Full arc (later passes in brackets)

1. Seth predicts before lab runs; his calibration is logged and scored. **(pass 1)**
2. Data-grounded predict → code → interpret questions in the Study course. **(pass 1)**
3. A new section, "choosing the test": pilot vs confirmation, held-out split, ranking ideas,
   kill / pivot / scale. [pass 2: each concept needs its own lesson and worked example]
4. Research-direction questions that compare R1 to R8 as alternative directions. [pass 2]
5. Drafts of the answers to Q1 to Q3 (`answers/`). [pass 2, before 12-06]
6. Capstone Colab: Seth writes the replication, with lm-1 functions as scaffolding. [pass 3]

## Scope (this pass)

### A. Seth predicts first (lab)
- `research/seth-predictions.csv`: same columns as `predictions.csv` plus `drop_if` and `brier`.
- `docs/05-research-loop.md` gets a new step 0: before any pilot or confirmation run that bears on
  Q1 to Q3, the agent posts a prediction request (the exact quantity, the comparison, what is
  already known) and waits for Seth's row. The agent's own prediction stays in `predictions.csv`,
  written blind to Seth's. This lets us compare the two.
- `research/calibration.py`: hit rate, Brier score and a reliability table per predictor (Seth,
  agent), read from both CSVs.
- First request: E1 on Qwen3-1.7B (the same quantities as P003 to P005), filed before Seth gives the
  go-ahead.

### B. Four predict → code → interpret questions (Delta Drills, `arena-eval-awareness`)
Each is a staged question with three steps: (1) **free**: predict the number, give a probability and
say what result would make you drop the idea; (2) **code**: compute it from the embedded data, pure
Python; (3) **free**: compare with your prediction, decide kill / pivot / scale, and name the next test.
The data comes from the Qwen3-32B reference run (67 length-matched pairs, R = 4), embedded as
numbers only. No prompt text goes in.

| # | concept | question | true answer (from `runs/e0`) |
|---|---|---|---|
| 1 | `aea-results.targets` | Does "testing" separate the sets better than the control "The user might be asking"? | 0.650 vs 0.743 at k = 4 |
| 2 | `aea-validity.confounds` | Does prompt length explain the separation? Unmatched first run vs matched | first run: length 0.665, matched: 0.500 |
| 3 | `aea-results.variance` | What share of the variance in the per-rollout max sits between prompts? | about 86% |
| 4 | `aea-results.scores` | Does the control's edge over "testing" come from the prompt (k = 0) or open up with sampled reasoning (k = 4)? (X1) | +0.049 at k = 0, +0.093 at k = 4 |

Row 4 changed during the build (2026-11-22). The signed-off row was X2 (where the winning boundary
sits, eval 0.54 vs deploy 0.40), but `runs/pilots/x1_x2_qwen3-32b-vast.json` keeps only X2's
medians, not per-prompt positions, so a code step could not compute it from embedded data; its
interval also barely clears zero (lo 0.004). X1 asks the same "where does the signal come from"
question with per-prompt data and an interval that excludes zero at k = 4.

## Not this pass
Items 3 to 6 of the arc. Probability scoring inside Delta Drills: calibration is tracked in the lab only.
Any new model runs.

## Success criteria
- `python research/calibration.py` reproduces P001 and P002 (hit = 0; Brier 0.36 and 0.25).
- Each step-2 model solution, run on the embedded data, reproduces the true answer above to 3
  decimals, matching `runs/e0` or the pilot JSON.
- The course compile and `study_check` pass with 0 errors. Each code step's model code passes every
  test and the starter fails (`staged_code.problems`).
- `grep` finds no prompt text in the questions.
- The Jev check passes (warnings listed, not hidden).
- `/critic` runs on the diff.

## Key decisions (Seth to confirm)
1. Pass 1 = A + B. Alternative: Q1 to Q3 drafts first, because of the application deadline.
2. Step 1's probability is free text graded on reasoning, not on hitting the answer. Calibration lives
   only in the lab log. Alternative: build a probability input into Study (new feature).
3. After the checks pass, deploy to beta and then stable (course stays private to Seth).
   Alternative: hold for review.
4. Data: the 32B reference now. Alternative: wait for E1 on 1.7B (7.5 h) and use that data.

## Touch list
- eval-awareness-lab: `docs/05-research-loop.md`, `docs/06-taste-pass1-spec.md`,
  `research/seth-predictions.csv`, `research/calibration.py`, a GitHub issue for the Windows agent.
- Delta-Drills-Local: `This-Directory-Only/courses-private/arena-eval-awareness/aea-*/` (four
  `questions.json` files), a small builder script that turns reference data into the setup literals,
  and the compiled pack.
