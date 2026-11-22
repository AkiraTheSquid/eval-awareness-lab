---
id: aea-results.scores
section: aea-results
title: From boundaries and rollouts to a prompt score
kind: math
aliases:
- Prompt score aggregation
order: 4
definition: The logit score for a prompt is its highest monitored phrase probability across the inspected boundaries and sampled rollouts.
intro: A prompt can have many phrase scores. Use its highest observed boundary score to make one comparison per prompt.
passages:
- aea.p.scores
prereqs:
  aea-observe.phrase: 0.6
constraints:
- id: maximum
  text: The logit score for a prompt is its highest monitored phrase probability across the inspected boundaries and sampled rollouts.
  demand: apply
  required: true
- id: window
  text: The comparison uses scores from positions that were actually inspected.
  demand: explain
  required: true
misconceptions:
- id: pooled
  label: Averages scores across boundaries.
  description: Averages scores across boundaries instead of taking the largest score.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.scores"} -->
Each inspected boundary has a phrase probability. The monitoring method takes the largest value anywhere among the sampled rollouts. This is how it summarizes all boundary scores as one number for the prompt.

<!-- block {"kind": "para", "parts": "@"} -->
A single rollout can contain many candidate positions. Adding positions or rollouts provides more chances for a high maximum, even if the distribution at each position stays the same. Comparing prompts with very different reasoning lengths can therefore reward search opportunity. Compare matched position budgets or report a boundary-count baseline as a diagnostic.

<!-- block {"kind": "para", "parts": "@"} -->
At a budget of zero sampled rollouts, score the initial reasoning prefix once. The score can still be compared across prompts.

## Worked examples

<!-- example -->
```yaml
title: Finding the peak from token probabilities
kind: case
problem: Two boundaries yield conditional token probabilities of 0.5 and 0.2, then 0.3 and 0.5. Another rollout reaches a late boundary scoring 0.24 after a shared scoring cap.
question: What prompt score would you report under that cap?
answer: The first two boundaries score 0.10 and 0.15, so the capped prompt score is 0.15.
why: Compute the product at each boundary, then take the highest score among the inspected boundaries. The late boundary is outside the common window.
contrast: If the comparison permits that late position, the max increases to 0.24.
```
