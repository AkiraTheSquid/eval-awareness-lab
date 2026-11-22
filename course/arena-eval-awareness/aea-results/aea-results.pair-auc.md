---
id: aea-results.pair-auc
title: Pairwise calculation of AUROC
aliases:
- Pairwise calculation of AUROC
kind: math
section: aea-results
order: 5
definition: AUROC summarizes how often a scoring rule ranks a randomly chosen positive item above a randomly chosen negative item.
intro: AUROC measures ordering across positive and negative groups. In the paper’s separation task, positive means evaluation source. In its prediction task, positive means the prompt ever verbalized in the reference samples. Identify the labels before counting pairs.
passages:
- aea.p.guide-ranking
- aea.p.fawcett-ranking
- aea.p.fawcett-ties
prereqs: {}
procedural_kcs: []
constraints:
- id: mechanism
  text: AUROC summarizes how often a scoring rule ranks a randomly chosen positive item above a randomly chosen negative item.
  demand: explain
  required: true
misconceptions:
- id: calculation
  label: Counts a positive–negative tie as a full win.
  description: Counts a positive–negative tie as a full win.
lesson: '@body'
examples: '@body'
---

## Lesson

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.fawcett-ranking"} -->
AUROC measures ordering across positive and negative groups. In the paper’s separation task, positive means evaluation source. In its prediction task, positive means the prompt ever verbalized in the reference samples. Identify the labels before counting pairs.

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.fawcett-ties"} -->
Take each positive score together with each negative score. Assign one point to a higher positive score, half a point to a tie, and zero to a lower positive score. Divide total points by the number of cross-group pairs. This compares rankings without choosing an alert threshold.

<!-- block {"kind": "para", "parts": "@", "passage": "aea.p.guide-ranking"} -->
Adding the same constant to every score leaves every comparison unchanged. A fixed cutoff can still issue a different number of alerts.

## Worked examples

<!-- example -->
```yaml
title: Calculate the pairwise score
problem: Two eval prompts score 0.65 and 0.85. Two deployment prompts score 0.55 and 0.65.
question: What is the ranking AUROC?
answer: The two eval items earn 1.5 and 2 points, for 3.5 points across four pairs. AUROC is 0.875.
why: The lower eval item wins once and ties once. The higher eval item wins both comparisons.
contrast: If the lower eval score dropped below 0.55, it would earn zero and AUROC would be 0.50.
kind: case
```
