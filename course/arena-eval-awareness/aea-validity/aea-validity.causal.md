---
id: aea-validity.causal
section: aea-validity
title: From association to effects on behavior
kind: conceptual
aliases:
- From association to effects on behavior
order: 12
definition: Changing evaluation-aware reasoning can affect downstream refusal, as measured by controlled removal or insertion in Kimi K2.5.
intro: A readout can predict context without affecting behavior.
passages:
- aea.p.causal
prereqs: {}
constraints:
- id: p0
  text: Changing evaluation-aware reasoning can affect downstream refusal, as measured by controlled removal or insertion in Kimi K2.5.
  demand: explain
  required: true
- id: p1
  text: The Kimi K2.5 intervention result is about refusal on Fortress.
  demand: evaluate
  required: true
misconceptions:
- id: diagnostic
  label: Assumes editing evaluation-related reasoning cannot affect refusal.
  description: Assumes editing evaluation-related reasoning cannot affect refusal.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.causal"} -->
A readout can predict context without affecting behavior. An observational comparison between flagged and silent rollouts does not determine how a model would answer if the reasoning were changed. To test that, randomly assign a controlled modification, hold other input features fixed, and measure both the putative intermediate state and a downstream action.

<!-- block {"kind": "para", "parts": "@"} -->
Goodfire’s companion study added evaluation-aware reasoning and observed increased refusals for Kimi K2.5 on Fortress under tested conditions. The research guide describes broader intervention work reporting small average shifts in other setups. Outcomes can differ across models, tasks and manipulation details. A neutral continuation, alternative wording and checks for topic or persona effects help interpret the causal result.

<!-- block {"kind": "para", "parts": "@"} -->
Prompt rewrites can reduce conspicuous cues and help evaluators identify unrealistic items cheaply. Verify that rewriting preserves the task’s difficulty, content and intended behavior. An intervention on a probe direction also needs controls because it can change other features besides the one the probe reads.

<!-- block-json -->
```json
{
 "kind": "image",
 "parts": [
  {
   "text": "Aranguri and Bloom. The controlled reasoning-continuation comparison measures refusal downstream."
  }
 ],
 "src": "https://deltadrills.com/conceptual/tutor/figures/arena-eval-awareness/prefill-refusal.png",
 "passage": "aea.p.causal"
}
```

<!-- block-json -->
```json
{
 "kind": "image",
 "parts": [
  {
   "text": "Aranguri and Bloom. Prompt cue removal changes the observed verbalization and refusal rates in the tested setup."
  }
 ],
 "src": "https://deltadrills.com/conceptual/tutor/figures/arena-eval-awareness/prompt-rewrites.png",
 "passage": "aea.p.causal"
}
```

## Worked examples

<!-- example -->
```yaml
title: Intervene with a control
problem: A team prepends a suspected awareness statement to reasoning and observes more cautious answers. The statement also lengthens reasoning and changes tone.
question: What comparison would strengthen the causal interpretation?
answer: Randomize awareness and neutral prefill conditions matched on length and tone, and compare downstream caution.
why: Matching the control reduces alternative explanations introduced by the intervention.
contrast: If both conditions move caution equally, common editing or style effects become plausible.
kind: case
```
