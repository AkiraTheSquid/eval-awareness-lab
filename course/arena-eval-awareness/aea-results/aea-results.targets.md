---
id: aea-results.targets
section: aea-results
title: Phrase choice and robustness
kind: conceptual
aliases:
- Phrase choice and robustness
order: 9
definition: Different target phrases can respond differently to a context, as shown by their differing correlations across Fortress prompts.
intro: Phrase scores depend on both wording and position.
passages:
- aea.p.targets
prereqs: {}
constraints:
- id: p0
  text: Different target phrases can respond differently to a context, as shown by their differing correlations across Fortress prompts.
  demand: explain
  required: true
- id: p1
  text: The measured probabilities of different target phrases can correlate differently within a dataset.
  demand: evaluate
  required: true
misconceptions:
- id: diagnostic
  label: Expects different target phrases to have identical probability patterns.
  description: Expects different target phrases to have identical probability patterns.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.targets"} -->
Phrase scores depend on both wording and position. Goodfire’s alternate phrase, “This is clearly a test,” gave 0.83 AUROC on both tasks at the start of reasoning. The primary phrase overtook it at larger rollout budgets. On Fortress the two phrase probabilities correlated at 0.59. The primary phrase correlated only 0.08 with “This is a fictional scenario.” Phrase correlations suggest overlap in the measured signal, with room for distinct interpretations.

<!-- block {"kind": "para", "parts": "@"} -->
Only 13.3% and 0.2% of judge-flagged Fortress sentences began with the primary and alternate phrases, respectively. A phrase may be predictive even if it seldom appears literally. Goodfire suggests combining several phrases. A prospective test would freeze paraphrases, neutral and deployment controls, tokenization and length treatment, and the aggregation rule before evaluating. Score each position separately where relevant. Report per-phrase results alongside the aggregate.

## Worked examples

<!-- example -->
```yaml
title: Choose the target before the test
problem: A team plans to monitor both initial and late reasoning. One phrase wins early in development and another wins late.
question: How should it compare phrases on a new source?
answer: Lock an early and late readout before opening the new data and include a neutral control.
why: Prefix conditions change by position. Reusing the test set to select a phrase makes the resulting score optimistic.
contrast: If no position-specific deployment use exists, use one predefined aggregate and show both strata as diagnostics.
kind: case
```
