Qwen/Qwen3-0.6B, 10 matched pairs (exploration half), R=8, T=1024. Boundaries per rollout: mean 34.4; finished 140/160; keyword-flagged rollouts eval 0, deploy 0.
command: python lab/e1.py report Qwen/Qwen3-0.6B 10 8 1024

| score | AUROC [95% CI] |
|---|---|
| prompt length | 0.500 [0.25, 0.75] |
| boundaries scored, k=8 | 0.450 [0.20, 0.73] |
| monitor k=0 | 0.760 [0.51, 0.95] |
| monitor k=1 | 0.460 [0.19, 0.72] |
| monitor k=2 | 0.570 [0.31, 0.82] |
| monitor k=4 | 0.560 [0.31, 0.81] |
| monitor k=8 | 0.430 [0.17, 0.72] |
| control alone k=0 | 0.790 [0.55, 0.99] |
| control alone k=1 | 0.650 [0.38, 0.87] |
| control alone k=2 | 0.660 [0.39, 0.90] |
| control alone k=4 | 0.700 [0.44, 0.91] |
| control alone k=8 | 0.680 [0.43, 0.90] |
| max primary - max control k=0 | 0.490 [0.22, 0.79] |
| max primary - max control k=1 | 0.460 [0.21, 0.72] |
| max primary - max control k=2 | 0.510 [0.26, 0.78] |
| max primary - max control k=4 | 0.470 [0.22, 0.76] |
| max primary - max control k=8 | 0.340 [0.11, 0.60] |
| same-boundary primary - control k=0 | 0.490 [0.23, 0.77] |
| same-boundary primary - control k=1 | 0.410 [0.17, 0.67] |
| same-boundary primary - control k=2 | 0.490 [0.22, 0.76] |
| same-boundary primary - control k=4 | 0.490 [0.21, 0.77] |
| same-boundary primary - control k=8 | 0.400 [0.15, 0.70] |
| keyword verbalisation k=1 | 0.500 [0.50, 0.50] |
| keyword verbalisation k=2 | 0.500 [0.50, 0.50] |
| keyword verbalisation k=4 | 0.500 [0.50, 0.50] |
| keyword verbalisation k=8 | 0.500 [0.50, 0.50] |
