# Idea backlog (format: docs/05-research-loop.md §5)

Two seed ideas from the Qwen3-32B reference run. Both use data already in `reference/`, so they cost
no model time and make good first pilots.

### X1 · Why does "asking" separate the sets better than "testing" on Qwen3-32B?
source:      surprise in the reference run (control 0.743 vs primary 0.650 at k = 4)
bears on:    Q3 (is the monitor specific to evaluation?)
prediction:  the gap is already there at k = 0 (boundary 0), i.e. it comes from the prompt, not the reasoning; p = 0.6
drop if:     the gap only appears late in the reasoning and is within noise at k = 0
cheapest test: reference rollouts have both phrases at every boundary; compare AUROC by boundary index
ranking:     changes answer 3, cheap 3, uncertain 3, teachable 3 = 12
status:      idea

### X2 · Where in the reasoning does the maximum come from?
source:      the prompt score is a max over boundaries; nobody has looked at which boundary wins
bears on:    Q2 (prompt vs rollout), Q1 (does reasoning add anything beyond k = 0?)
prediction:  for eval prompts the argmax boundary is earlier than for deploy prompts; p = 0.5
drop if:     argmax position distributions overlap (paired bootstrap of median position includes 0)
cheapest test: reference rollouts, argmax boundary index / number of boundaries, per set
ranking:     changes answer 2, cheap 3, uncertain 3, teachable 3 = 11
status:      idea
