---
id: aea-results.any-flag
title: Probability of at least one flag
aliases:
- Probability of at least one flag
kind: math
section: aea-results
order: 7
definition: For a fixed per-draw positive probability and independent draws, the probability of any flag is one minus the probability that all draws miss.
intro: An ever-positive label can change when you sample more, even if nothing about the prompt or model changes. Suppose each independently sampled rollout has the same chance of producing a positive judge flag.
passages:
- aea.p.guide-budget
prereqs: {}
procedural_kcs: []
constraints:
- id: mechanism
  text: For a fixed per-draw positive probability and independent draws, the probability of any flag is one minus the probability that all draws miss.
  demand: explain
  required: true
misconceptions:
- id: calculation
  label: Adds per-draw flag chances without accounting for draws that both flag.
  description: Adds per-draw flag chances without accounting for draws that both flag.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.guide-budget"} -->
An ever-positive label can change when you sample more, even if nothing about the prompt or model changes. Suppose each independently sampled rollout has the same chance of producing a positive judge flag.

<!-- block {"kind": "para", "parts": "@"} -->
The probability of no flag on one rollout is one minus its positive probability. Multiply this miss probability once for each independent rollout to get the chance that all miss. Subtract that result from one to get the probability of any positive flag.

<!-- block {"kind": "para", "parts": "@"} -->
A larger sample budget improves the chance of observing a rare flag under these assumptions. It changes the observation process, even when the per-rollout propensity stays fixed.

## Worked examples

<!-- example -->
```yaml
title: Computing the chance of at least one flag
problem: Every independent rollout from a fixed prompt has a 25% chance of a flag.
question: What chance does a pair of rollouts have of at least one flag?
answer: Both miss with probability 0.5625, so any positive occurs with probability 0.4375.
why: Each miss has probability 0.75. Independence lets us multiply the two miss chances. The all-miss event is the complement of any flag.
contrast: If the two samples shared an identical continuation, the observations would coincide and the chance would stay 0.25.
kind: case
```
