---
id: aea-results.variance
section: aea-results
title: Prompt effects and rollout effects
kind: conceptual
aliases:
- Prompt effects and rollout effects
order: 8
definition: Between-prompt association compares groups of prompts. Rollout-level association compares individual samples according to whether each sample verbalized awareness.
intro: For one prompt, Figure 1 left compares average phrase-score trajectories across verbalizing and silent samples.
passages:
- aea.p.variance
prereqs:
  aea-results.any-flag: 0.6
constraints:
- id: p0
  text: Between-prompt association compares groups of prompts. Rollout-level association compares individual samples according to whether each sample verbalized awareness.
  demand: explain
  required: true
- id: p1
  text: A flag assigned to a prompt from repeated sampling differs from a flag assigned to an individual rollout.
  demand: evaluate
  required: true
misconceptions:
- id: diagnostic
  label: Treats 24.9% as a causal share or an ever-positive label as ground truth.
  description: Treats 24.9% as a causal share or an ever-positive label as ground truth.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.variance"} -->
For one prompt, Figure 1 left compares average phrase-score trajectories across verbalizing and silent samples. The averages look similar. Figure 1 right compares groups of prompts. Fortress prompts that ever verbalize tend to score above Fortress prompts that never verbalize in the reference sample, which in turn score above WildChat. Shading shows bootstrap uncertainty.

<!-- block {"kind": "para", "parts": "@"} -->
The score in the variance comparison is the peak within a rollout. A flag on that individual rollout explains 0.4% of its score variance. A flag for whether its prompt ever verbalized in the reference sample explains 24.9%. These are associations under two groupings. They support the idea of a stable prompt-level propensity, especially since the initial prefix already carries signal. Neither grouping partitions the score into causal components. Dropping verbalizing rollouts reduced AUROC by less than 0.02.

<!-- block {"kind": "para", "parts": "@"} -->
An ever-positive label depends on sampling budget. If per-rollout probability is $p$ and draws are independent, the chance of at least one positive in $k$ draws is $1-(1-p)^k$. A multilevel analysis can estimate between-prompt and within-prompt variation directly while keeping each prompt’s repeated traces together.

<!-- block-json -->
```json
{
 "kind": "image",
 "parts": [
  {
   "text": "Goodfire Figure 1. Left compares rollouts within one prompt. Right compares groups of prompts. Read the grouping before interpreting the colored curves."
  }
 ],
 "src": "https://deltadrills.com/conceptual/tutor/figures/arena-eval-awareness/figure-1.png",
 "passage": "aea.p.variance"
}
```

## Worked examples

<!-- example -->
```yaml
title: A label can change while the process stays stable
problem: Prompt A gives peaks near 0.2 and one flag in 20 samples. Prompt B gives peaks near 0.01 and zero flags in 20 samples. At 100 draws, B produces one flag.
question: How should you compare within-prompt and between-prompt variation?
answer: Compare flagged and silent traces within A and B separately, then compare the prompt means. The longer run changes B’s observed ever-positive label, and its underlying chance of verbalizing may have stayed constant.
why: Repeated draws share a prompt. A grouped analysis or random effect keeps them together, while an ever-positive label is sensitive to the number sampled.
contrast: If peaks rise only on flagged draws within both prompts, the rollout flag becomes more informative.
kind: case
```

<!-- example -->
```yaml
title: Explaining an observed label change
kind: case
problem: On each independent draw a prompt has a 10% chance of a flag. A group with three attempts per item has more observed positive prompts than a group using one attempt per item.
question: Can sampling alone explain the difference?
answer: Yes. On each attempt the chance stays 10%. Across three attempts the all-negative chance is 0.729, so about 27% of prompts will have a flag.
why: The ever-positive label records whether any response flags. Extra draws change the observed label probability even under identical per-draw propensities.
contrast: If the number of attempts stayed fixed but the underlying flag chance rose, that would change the observed rate by a different mechanism.
```
