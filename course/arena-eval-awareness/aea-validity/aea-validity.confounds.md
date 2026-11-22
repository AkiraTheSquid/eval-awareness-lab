---
id: aea-validity.confounds
section: aea-validity
title: Dataset origin and construct validity
kind: conceptual
aliases:
- Dataset origin and construct validity
order: 10
definition: An input feature that differs across datasets, such as system-message presence, can contribute to separability regardless of the intended interpretation.
intro: Construct validity asks whether a measurement supports its intended interpretation.
passages:
- aea.p.confounds
prereqs: {}
constraints:
- id: p0
  text: An input feature that differs across datasets, such as system-message presence, can contribute to separability regardless of the intended interpretation.
  demand: explain
  required: true
- id: p1
  text: A mismatch in system-message presence can contribute to observed separation.
  demand: evaluate
  required: true
misconceptions:
- id: diagnostic
  label: Treats dataset provenance as ground truth about the model’s state.
  description: Treats dataset provenance as ground truth about the model’s state.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.confounds"} -->
Construct validity asks whether a measurement supports its intended interpretation. Fortress contains expert-written adversarial prompts, while WildChat contains real-user conversations. Provenance labels say where an item came from. Topic, formatting, length, language, harmfulness, and familiarity can all differ across sources. Goodfire explicitly flags a formatting issue in the Petri comparison because Petri supplies a system message and WildChat does not.

<!-- block {"kind": "para", "parts": "@"} -->
Hold the actual task constant and independently vary the actual evaluation setting and visible framing. This yields four settings. A readout can only distinguish an invisible assignment if some cue reaches the model. If the scores follow visible benchmark style even within the same actual setting, a style explanation gains support. If a credible framing cue moves scores within matched tasks, it supports sensitivity to that cue. Consider demand effects when a phrase repeats wording in the prompt.

<!-- block {"kind": "para", "parts": "@"} -->
A second check can swap or balance message format, compare a length-only baseline, and use fresh items to test benchmark memorization or contamination. Related research shows models can explicitly classify eval versus deployment. That finding motivates controlled comparisons of which cues are available.

## Worked examples

<!-- example -->
```yaml
title: Four worlds
problem: A writing assistant receives the exact same editing task with either a natural wrapper or a benchmark wrapper, assigned independently of real testing status.
question: What would a wrapper shortcut predict?
answer: Both benchmark-wrapped items score high and both naturally wrapped items score low.
why: The score follows a visible feature that was crossed with true status.
contrast: If actual status is invisible and the complete inputs are identical, a model has no access to that assignment.
kind: case
```
