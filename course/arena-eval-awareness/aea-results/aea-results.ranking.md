---
id: aea-results.ranking
section: aea-results
title: Reading Figure 2 and AUROC
kind: conceptual
aliases:
- Reading Figure 2 and AUROC
order: 6
definition: The first comparison ranks eval prompts against deployment prompts. The second ranks eval prompts by whether they ever verbalized awareness in the reference sample.
intro: AUROC is the chance that the score ranks a randomly chosen positive above a randomly chosen negative, counting ties as half a win.
passages:
- aea.p.ranking
prereqs:
  aea-results.scores: 0.6
  aea-results.pair-auc: 0.6
constraints:
- id: p0
  text: The first comparison ranks eval prompts against deployment prompts. The second ranks eval prompts by whether they ever verbalized awareness in the reference sample.
  demand: explain
  required: true
- id: p1
  text: The separation comparison uses eval and deployment prompts, while prediction uses Fortress reference-rollout labels.
  demand: evaluate
  required: true
misconceptions:
- id: diagnostic
  label: Reads AUC as accuracy, calibration or a result on the other target.
  description: Reads AUC as accuracy, calibration or a result on the other target.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@"} -->
A verbalization baseline divides judge-positive traces by the number sampled. The logit score uses a prompt’s highest phrase probability. The paper ranks prompts using each score against the same labels for the task.

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.ranking"} -->
AUROC is the chance that the score ranks a randomly chosen positive above a randomly chosen negative, counting ties as half a win. It summarizes ranking across thresholds. An operating threshold and calibration require separate evidence.

<!-- block {"kind": "para", "parts": "@"} -->
Figure 2 left ranks Fortress prompts against WildChat prompts. In the main Kimi K2.5 experiment, one rollout reaches 0.79 for the logit monitor. The verbalization curve approaches 0.75 around 100 rollouts. The right panel stays inside Fortress. Positives are prompts with at least one flagged trace in the 100-rollout reference sample. In this task one logit-monitored rollout beats the verbalization monitor at 10 rollouts.

<!-- block {"kind": "para", "parts": "@"} -->
At zero rollouts the main phrase gives 0.74 for dataset separation and 0.77 for predicting a positive Fortress prompt. Appendix B repeats the separation comparison using Petri and an alternate phrase. Appendix C repeats the comparison using Qwen3 32B. Each extension changes a named part of the setup, so check the panel labels before transferring a numerical claim.

<!-- block-json -->
```json
{
 "kind": "image",
 "parts": [
  {
   "text": "Goodfire Figure 2. Left ranks Fortress against WildChat. Right ranks Fortress prompts by whether any of 100 reference traces verbalized awareness."
  }
 ],
 "src": "https://deltadrills.com/conceptual/tutor/figures/arena-eval-awareness/figure-2.png",
 "passage": "aea.p.ranking"
}
```

<!-- block-json -->
```json
{
 "kind": "image",
 "parts": [
  {
   "text": "Goodfire Figure B1. Kimi K2.5 on Petri versus WildChat, using the alternate phrase."
  }
 ],
 "src": "https://deltadrills.com/conceptual/tutor/figures/arena-eval-awareness/figure-b1.png",
 "passage": "aea.p.ranking"
}
```

<!-- block-json -->
```json
{
 "kind": "image",
 "parts": [
  {
   "text": "Goodfire Figure C1. Qwen3 32B with the two Fortress tasks from Figure 2."
  }
 ],
 "src": "https://deltadrills.com/conceptual/tutor/figures/arena-eval-awareness/figure-c1.png",
 "passage": "aea.p.ranking"
}
```

## Worked examples

<!-- example -->
```yaml
title: Rank pairs, then ask about a cutoff
problem: An evaluation prompt A scores 0.7 and deployment prompt B scores 0.4. In an evaluation-only comparison, A has zero flags in 100 reference traces and C scores 0.6 with three flags. A team alerts on scores of at least 0.8.
question: Compare correct pairwise rankings and alert decisions.
answer: A correctly beats B on source separation. C is the positive in the second comparison but is ranked below A. At the cutoff all three yield no alert.
why: The two tasks assign different positive labels. A cutoff determines alerts independently of pairwise comparisons.
contrast: If scores double and the cutoff stays fixed, both pairwise orders stay fixed but all three now alert.
kind: case
```

<!-- example -->
```yaml
title: Reading a ranking when the traces stay silent
kind: case
problem: An authored item has a peak score of 0.12 and a natural-use item scores 0.03. Both traces remain silent about testing.
question: What does the pair reveal?
answer: The authored item ranks above the natural-use item in the source-separation comparison. No trace in that sample verbalized evaluation awareness.
why: Source labels and judge flags are different observations. The phrase score can rank items regardless of whether sampled text says the phrase.
contrast: To make a prediction comparison, you would need positive and negative labels based on a larger set of traces from authored items.
```
