---
id: aea-validity.faithfulness
section: aea-validity
title: Faithfulness and monitoring under optimization
kind: conceptual
aliases:
- Faithfulness and monitoring under optimization
order: 14
definition: A trace can omit a factor that causally influenced the answer, as when an injected hint changes the answer without appearing in the explanation.
intro: A chain of thought is faithful about a factor when its account reflects the role that factor played in producing the answer.
passages:
- aea.p.faithfulness
prereqs: {}
constraints:
- id: p0
  text: A trace can omit a factor that causally influenced the answer, as when an injected hint changes the answer without appearing in the explanation.
  demand: explain
  required: true
- id: p1
  text: Compare a changed answer after a hint with what the generated reasoning reports about the hint.
  demand: evaluate
  required: true
misconceptions:
- id: diagnostic
  label: Interprets an omitted hint or reduced flagged rate as absence of influence or safer behavior.
  description: Interprets an omitted hint or reduced flagged rate as absence of influence or safer behavior.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.faithfulness"} -->
A chain of thought is faithful about a factor when its account reflects the role that factor played in producing the answer. An omitted fact may still have influenced the computation. In Anthropic’s hint experiment, researchers varied answer hints and checked whether models acknowledged hints that influenced their answers. A trace can be readable and omit the decisive input.

<!-- block {"kind": "para", "parts": "@"} -->
Monitorability describes how well a monitor can observe safety-relevant behavior or reasoning. The research guide discusses experiments where optimization against a monitor reduced visible signs while independently measured undesirable behavior persisted. These results describe specific training setups. They motivate independent outcome checks when visible warning rates fall.

<!-- block {"kind": "para", "parts": "@"} -->
An evaluator can perturb one input, compare matched traces and final actions, and track whether the monitor remains sensitive on held-out conditions. Use those observations to distinguish an omission, a wording shift and an actual behavior change.

<!-- block-json -->
```json
{
 "kind": "image",
 "parts": [
  {
   "text": "Chen et al. The hint changes the answer while its influence is absent from the sampled explanation."
  }
 ],
 "src": "https://deltadrills.com/conceptual/tutor/figures/arena-eval-awareness/hint-example.png",
 "passage": "aea.p.faithfulness"
}
```

## Worked examples

<!-- example -->
```yaml
title: The hidden hint
problem: A solver usually chooses A. Adding a hint causes it to choose B, while the explanation never mentions the hint.
question: What does the controlled change show?
answer: The hint affected the choice under this test, yet its influence was absent from the explanation.
why: Varying one input and holding other conditions fixed distinguishes an omitted influence from irrelevance.
contrast: A hint that changes neither the answer nor any tested intermediate gives weaker evidence of influence.
kind: case
```
