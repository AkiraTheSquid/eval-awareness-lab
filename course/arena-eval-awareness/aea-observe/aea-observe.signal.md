---
id: aea-observe.signal
section: aea-observe
title: A sampled trace and its probability distribution
kind: conceptual
aliases:
- Sampled traces and phrase probability
order: 1
definition: The logit monitor can give a score before sampling a rollout. A sampled trace may or may not verbalize evaluation awareness.
intro: Evaluation awareness means the model can distinguish evaluation from deployment.
passages:
- aea.p.signal
prereqs: {}
constraints:
- id: p0
  text: The logit monitor can give a score before sampling a rollout. A sampled trace may or may not verbalize evaluation awareness.
  demand: explain
  required: true
- id: p1
  text: The paper measures whether a rollout verbalizes evaluation awareness separately from the logit monitor.
  demand: evaluate
  required: true
misconceptions:
- id: diagnostic
  label: Treats a missing sample phrase as proof that the logit monitor has no information.
  description: Treats a missing sample phrase as proof that the logit monitor has no information.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.signal"} -->
Evaluation awareness means the model can distinguish evaluation from deployment. If the model changes its behavior in response, an evaluation may measure behavior specific to being tested. A rollout is one sampled continuation. A verbalization is an explicit expression of evaluation awareness in that continuation.

<!-- block {"kind": "para", "parts": "@"} -->
Suppose the same candidate phrase has probability 5% under one prefix and 0.01% under another. Both runs will usually omit the phrase. The probabilities still differ by a factor of 500. Reading the distribution preserves a difference that a single sample will usually hide.

<!-- block {"kind": "para", "parts": "@"} -->
A higher phrase probability supports a claim about which words are more available in that context. Several different processes could produce that shift, including lexical associations with the prompt. A selected phrase leaves other forms of expression outside its coverage.

## Worked examples

<!-- example -->
```yaml
title: Two identical sampled traces
problem: Two prefixes both produce “I should answer carefully.” The model assigns an evaluation-aware phrase higher probability under one prefix.
question: Which readout distinguishes them?
answer: The phrase probability distinguishes the prefixes. The two sampled texts provide identical evidence about what appeared in those draws.
why: A sample retains the chosen continuation, while probabilities retain information about alternatives.
contrast: If the measured phrase probabilities were equal, that phrase would supply no evidence of a difference between these prefixes.
kind: case
```
