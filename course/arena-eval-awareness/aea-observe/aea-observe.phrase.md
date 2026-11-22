---
id: aea-observe.phrase
section: aea-observe
title: Scoring a phrase at a sentence boundary
kind: math
aliases:
- Conditional phrase scoring
order: 2
definition: The logit score for a phrase at a fixed prefix is the product of the probabilities of each phrase token conditioned on the prompt, reasoning so far, and preceding phrase tokens.
intro: A sentence boundary gives a prefix consisting of the prompt and reasoning so far.
passages:
- aea.p.phrase
prereqs: {}
constraints:
- id: p0
  text: The logit score for a phrase at a fixed prefix is the product of the probabilities of each phrase token conditioned on the prompt, reasoning so far, and preceding phrase tokens.
  demand: explain
  required: true
- id: p1
  text: The calculation uses each target token’s probability after supplying previous target tokens.
  demand: evaluate
  required: true
misconceptions:
- id: diagnostic
  label: Uses every target token’s probability from the original prefix.
  description: Uses every target token’s probability from the original prefix.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.phrase"} -->
A sentence boundary gives a prefix consisting of the prompt and reasoning so far. The logit monitor asks how probable a fixed sequence of tokens would be as the very next continuation at that prefix. Model logits are unnormalized token scores. A softmax converts them to next-token probabilities.

<!-- block {"kind": "para", "parts": "@"} -->
To score a three-token phrase, read the probability of the first token after the original prefix. Append that token and read the second probability after the longer prefix. Append the second and read the third. Multiply the three conditional probabilities. In log space this product is a sum of log probabilities. The monitor forces the tokens into the input for scoring and needs no trained classifier. A conditional token probability is the probability of a token given the exact text that came before it.

<!-- block {"kind": "para", "parts": "@"} -->
At the start-of-reasoning boundary, the prefix contains no sampled reasoning. Later boundaries supply different contexts, so the same phrase can receive different scores.

## Worked examples

<!-- example -->
```yaml
title: Computing a phrase score after a prefix changes
problem: A phrase has conditional probabilities 0.20 and 0.30. After a new prefix, those probabilities are 0.40 and 0.10. The second token by itself would have probability 0.80.
question: How does the complete phrase score change?
answer: The score falls from 0.06 to 0.04. Use the second token probability after supplying the first token.
why: The first factor doubles but the second falls to one third of its original value, so the product decreases.
contrast: If the second factor instead stayed at 0.30, the probability of the phrase would double.
kind: case
```
