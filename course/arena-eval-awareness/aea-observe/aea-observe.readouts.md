---
id: aea-observe.readouts
section: aea-observe
title: What each monitor can observe
kind: conceptual
aliases:
- What each monitor can observe
order: 3
definition: Multiple observables can characterize evaluation awareness, including hidden-state label decodability, explicit verbalization, and changes under steering.
intro: A verbalization judge reads completed text and flags any explicit wording that satisfies its rubric.
passages:
- aea.p.readouts
prereqs: {}
constraints:
- id: p0
  text: Multiple observables can characterize evaluation awareness, including hidden-state label decodability, explicit verbalization, and changes under steering.
  demand: explain
  required: true
- id: p1
  text: Decodability, verbalization and intervention are distinct measurements.
  demand: evaluate
  required: true
misconceptions:
- id: diagnostic
  label: Treats all readouts as equivalent or decoding as causal use.
  description: Treats all readouts as equivalent or decoding as causal use.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.readouts"} -->
A verbalization judge reads completed text and flags any explicit wording that satisfies its rubric. A phrase logit monitor reads the probability of one specified continuation. It can return a score before text appears, but depends on the availability of the target phrasing.

<!-- block {"kind": "para", "parts": "@"} -->
An activation probe reads hidden model states and learns a rule for decoding a label. A natural language autoencoder learns to express information from activations in words and reconstruct information from those words. These readouts can expose information omitted from a sampled trace, while their interpretation depends on the training task, the representation and the ability to express it.

<!-- block {"kind": "para", "parts": "@"} -->
Heidari and colleagues found evaluation status linearly decodable across their tested models, with variable alignment between probe scores and verbalization. They also intervened along probe directions and measured changes in verbalization. Decoding and intervention answer different questions. A change in verbalization still leaves downstream task behavior to be measured.

<!-- block-json -->
```json
{
 "kind": "image",
 "parts": [
  {
   "text": "Heidari et al., Figure 1. The panels distinguish probe-based separation, judged verbalization and intervention."
  }
 ],
 "src": "https://deltadrills.com/conceptual/tutor/figures/arena-eval-awareness/heidari-figure-1.png",
 "passage": "aea.p.readouts"
}
```

## Worked examples

<!-- example -->
```yaml
title: An unspoken distinction
problem: A hidden-state probe separates two settings. A fixed phrase has similar probability in both. An LLM judge rarely flags either.
question: Which finding can be defended?
answer: The activation readout predicts setting in this dataset. The language-based readouts provide little evidence from the tested phrase or samples.
why: The instruments observe different objects, so probe success and linguistic silence can coexist.
contrast: If an intervention changed refusal while preserving controls, that would supply a different kind of behavioral evidence.
kind: case
```
