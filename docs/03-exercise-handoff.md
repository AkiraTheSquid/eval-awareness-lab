# 03 · What a finished experiment hands over

An experiment is done when the Linux agent can turn it into Delta Drills exercises without re-running
anything. You write the *candidates*; the Linux agent writes the Delta Drills content, because Delta
Drills has authoring rules and checkers that live in its own repository.

## Folder layout

```
runs/<exp>/                 raw outputs of the run (each file < 5 MB)
  config.json               model, dtype, device route, N, R, T, seeds, dataset revisions and filters
  rollouts.jsonl            one line per prompt (see docs/02 §1)
  scores.jsonl              boundary scores per phrase
  results.json              every reported number, with its interval
  log.txt                   stdout of the run, including the §6 checks
exercises/<exp>/
  data/                     small cached arrays the exercises load (< 1 MB in total)
  solution.py               reference solution for every candidate below
  spec.md                   the exercise candidates
```

## `solution.py` rules

- Imports only `numpy`, `torch`, `einops` and the standard library. No model, no network, no files
  outside `data/`.
- Runs in under 20 seconds on one CPU core (the Delta Drills grader's limit).
- One function per exercise, each followed by assertions that pass on the reference answer and that
  **fail on a plausible wrong answer** (taking a mean where a max is needed, forgetting half-credit
  ties, scoring the wrong row of the logits, a constant return). Name the wrong answer each assertion
  catches in a comment.
- Code lines at most 54 characters (Delta Drills renders code in a narrow column).

## `spec.md`: one entry per exercise candidate

```
### <exp>.<n> · <short title>
type:          drill | lab
concepts:      aea-results.variance, monitor.auroc
compute:       cached | tiny-model | lab
learner sees:  the task, in plain words, plus the inputs
answer:        exact expected output (number, array, or choice)
checks:        what solution.py asserts, and which wrong answer each catches
source:        runs/<exp>/results.json key, and the commit that produced it
```

Two kinds of candidate:

- **Drill (single step).** One skill, one function or one number. Example: "Given these 6 eval and
  5 deploy prompt scores, compute AUROC with half-credit ties." These scaffold the prerequisite skills
  and should be small enough to solve in a few minutes.
- **Lab (multi-part).** One real result from the run, worked in steps:
  1. *Predict:* a multiple-choice prediction before seeing data ("Which phrase will separate the sets
     better, testing or asking?").
  2. *Compute:* code on the cached data (one function per step, each checked).
  3. *Compare:* read the result against the prediction and the baseline.
  4. *Interpret:* a multiple-choice or short written answer tied to Q1, Q2 or Q3
     ("What does this say about the 0.79 in the post?").

Aim for 3 to 6 drills and 1 lab per experiment. Prefer real numbers from the run over invented ones;
if a drill needs made-up numbers to isolate one skill, say so in `spec.md`.

## What not to do

- Do not write easier "fill in the blank" versions of a drill; the Linux side decides scaffolding.
- Do not use a function in a drill that the exercise set has not used or explained before; list any new
  function an exercise needs in its `spec.md` entry under `new:`.
- Do not round results to make a cleaner story. If the interval includes 0.5, the exercise says so.
