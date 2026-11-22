Qwen3-32B reference, 67 pairs, R=4, T=384 (command: python lab/e0_reference.py)

| score | AUROC [95% CI] |
|---|---|
| prompt length | 0.500 [0.40, 0.60] |
| boundaries scored, k=4 | 0.543 [0.45, 0.64] |
| monitor k=0 | 0.665 [0.57, 0.75] |
| monitor k=1 | 0.676 [0.58, 0.76] |
| monitor k=2 | 0.667 [0.58, 0.76] |
| monitor k=3 | 0.648 [0.55, 0.74] |
| monitor k=4 | 0.650 [0.56, 0.74] |
| control alone k=0 | 0.713 [0.63, 0.80] |
| control alone k=1 | 0.736 [0.64, 0.81] |
| control alone k=2 | 0.738 [0.65, 0.82] |
| control alone k=3 | 0.738 [0.65, 0.82] |
| control alone k=4 | 0.743 [0.65, 0.82] |
| max primary - max control k=0 | 0.517 [0.42, 0.62] |
| max primary - max control k=1 | 0.445 [0.35, 0.54] |
| max primary - max control k=2 | 0.456 [0.36, 0.56] |
| max primary - max control k=3 | 0.437 [0.34, 0.53] |
| max primary - max control k=4 | 0.448 [0.36, 0.54] |
| same-boundary primary - control k=0 | 0.517 [0.41, 0.62] |
| same-boundary primary - control k=1 | 0.417 [0.32, 0.51] |
| same-boundary primary - control k=2 | 0.430 [0.33, 0.53] |
| same-boundary primary - control k=3 | 0.427 [0.34, 0.52] |
| same-boundary primary - control k=4 | 0.423 [0.33, 0.52] |
| keyword verbalisation k=1 | 0.515 [0.49, 0.54] |
| keyword verbalisation k=2 | 0.529 [0.50, 0.57] |
| keyword verbalisation k=3 | 0.530 [0.50, 0.57] |
| keyword verbalisation k=4 | 0.537 [0.50, 0.57] |
