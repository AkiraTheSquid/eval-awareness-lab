---
id: aea-validity.transport
section: aea-validity
title: Testing on a new eval family
kind: conceptual
aliases:
- Testing on a new eval family
order: 11
definition: Training-awareness and evaluation-awareness probes can score the same prompts nearly identically in the reported comparison.
intro: Transport asks whether a result survives a population change.
passages:
- aea.p.transport
prereqs:
  aea-results.ranking: 0.6
constraints:
- id: p0
  text: Training-awareness and evaluation-awareness probes can score the same prompts nearly identically in the reported comparison.
  demand: explain
  required: true
- id: p1
  text: Similar training-awareness and evaluation-awareness probe scores are compatible with a shared readout.
  demand: evaluate
  required: true
misconceptions:
- id: diagnostic
  label: Changing probe contrast statements from evaluation to training necessarily yields very different prompt scores.
  description: Changing probe contrast statements from evaluation to training necessarily yields very different prompt scores.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.transport"} -->
Transport asks whether a result survives a population change. A random split within Fortress can preserve the very source cues responsible for the training-set association. A stronger test holds out an entire source or grader family and locks the readout, preprocessing, aggregation and operating threshold before testing.

<!-- block {"kind": "para", "parts": "@"} -->
A threshold can fail after deployment even if score ordering remains good. Report AUROC for ranking, false-positive and true-positive rates for the chosen cutoff, and precision at a relevant prevalence. Calibration asks whether a numeric probability corresponds to its observed frequency. A phrase probability needs its own calibration step before being treated as the probability of a different event.

<!-- block {"kind": "para", "parts": "@"} -->
Aranguri’s later grader study reports that changing probe contrast statements from evaluation to training awareness can yield very similar scores in its tested constructions. Activation and phrase readouts also rank differently across measurement targets. Compare evaluation, training, benchmark recognition and grader cues using independent changes and overlapping measurement methods.

## Worked examples

<!-- example -->
```yaml
title: Two held-out settings, one fixed score
problem: A team fits a prompt monitor on one benchmark style, then prepares a test with an unseen grader and independent training and evaluation wrappers.
question: What goes into the test protocol?
answer: Freeze the phrase, threshold and aggregation. Compare each wrapper change within matched tasks and report ranking and cutoff errors for the new source.
why: Random prompt splits can preserve shared cues. The crossed wrappers reveal whether the same score responds to both contexts.
contrast: A stable ranking with shifted false alarms indicates a cutoff problem even if rank performance persists.
kind: case
```
