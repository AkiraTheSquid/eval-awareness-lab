---
id: aea-validity.strategy
section: aea-validity
title: Awareness, gaming and deception
kind: conceptual
aliases:
- Awareness, gaming and deception
order: 13
definition: Evaluation-awareness experiments can test whether a model distinguishes evaluation from deployment, and whether it identifies the purpose of the evaluation.
intro: Situational awareness covers facts about the model and its circumstances.
passages:
- aea.p.strategy
prereqs: {}
constraints:
- id: p0
  text: Evaluation-awareness experiments can test whether a model distinguishes evaluation from deployment, and whether it identifies the purpose of the evaluation.
  demand: explain
  required: true
- id: p1
  text: Judging which setting a model recognizes is distinct from identifying what the evaluation is measuring.
  demand: evaluate
  required: true
misconceptions:
- id: diagnostic
  label: Infers a deceptive objective from a high awareness score or a behavioral switch.
  description: Infers a deceptive objective from a high awareness score or a behavioral switch.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.strategy"} -->
Situational awareness covers facts about the model and its circumstances. Evaluation awareness concerns the test-versus-use distinction. Strategy additionally requires evidence that behavior responds to incentives. An objective claim requires evidence about what the strategy serves.

<!-- block {"kind": "para", "parts": "@"} -->
Evaluation gaming means adapting to score well under perceived evaluation. Sandbagging is strategic underperformance. Alignment faking involves apparent compliance in a monitored or training context while preserving a conflicting preference. Sleeper-agent experiments study trigger-dependent policies. Scheming adds covert pursuit of a conflicting objective. Reward hacking exploits a scoring gap and can occur without recognizing a formal evaluation.

<!-- block {"kind": "para", "parts": "@"} -->
The research guide uses constructed cases to show that these behaviors are possible under particular incentives, training or prompts. To distinguish them in an unfamiliar study, identify what the model could observe, what changed about incentives, and what actions changed. Compare brittle keyword policies, familiarity, task difficulty and ordinary behavior variation before inferring strategy.

## Worked examples

<!-- example -->
```yaml
title: A switch without an objective
problem: A model performs worse after a specific phrase that labels the prompt a benchmark. Its scores fall even when incentives stay constant.
question: What is the strongest supported observation?
answer: Performance changes under that phrase. To establish sandbagging, test whether underperformance is selective relative to demonstrated capability and tracks an incentive to conceal it.
why: A framing response could result from a learned trigger or changed task interpretation.
contrast: If the model deliberately withholds correct answers only when revealing capability harms an objective, the strategic interpretation gains support.
kind: case
```
