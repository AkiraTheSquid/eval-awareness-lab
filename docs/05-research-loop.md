# 05 · The research loop: finding which directions are worth it

The catalogue in `docs/02-experiments.md` is a starting map, not the whole job. Part of your job is to
**find good experiments**: propose candidates, test them cheaply, and let the results decide what gets
scaled up and what gets dropped. Good research taste is judged by outcomes, so this loop records
predictions and outcomes and checks over time how good your judgement is.

## 1. Two kinds of work, kept apart

- **Exploration (pilots).** Small, fast, many. The aim is to learn where the signal is. Pilot results
  never go into `answers/` and never become a headline number.
- **Confirmation (full runs).** One prediction written down in advance, run on prompts no pilot has
  touched, full checks from `docs/01-method.md` §6. Only these support claims in `answers/`.

To keep them apart, the first time you build the prompt pairs, set aside a **held-out split**
(a random 50% of matched pairs, fixed seed, IDs committed to `research/heldout_ids.json`). Pilots use
only the other half. Looking at a promising pilot and then "confirming" it on the same prompts is
the main way this kind of work fools itself.

Split time roughly 70% on the catalogue and 30% on exploration, and say in `STATUS.md` if that drifts.

## 2. The loop

1. **Idea.** Add it to `research/ideas.md` (format below). Good sources of ideas, in order:
   a result that surprised you; a check that failed or came out odd; a confound nobody has ruled out;
   a limitation the paper's authors state; a disagreement between two monitors or two models; anything
   that would change the answer to Q1, Q2 or Q3 if it came out the other way.
2. **Predict.** Before running anything, write the expected result, your confidence (as a
   probability), and what result would make you drop the idea.
3. **Pilot.** At most 20 minutes of compute: Qwen3-0.6B or 1.7B, 10 to 20 pairs from the
   exploration half. **Reuse before you resample:** most ideas can be tested by re-scoring saved
   rollouts with new phrases, positions or prompt edits, which costs forward passes only.
4. **Decide**, in one line, and record it:
   - **kill**: effect within noise, or it would not change any answer even if real;
   - **pivot**: something else in the pilot looked more interesting (that becomes a new idea);
   - **scale**: effect clearly outside noise *and* it bears on Q1–Q3 *and* a full run is affordable;
   - **park**: interesting but needs a bigger model or training; note what would unblock it.
   Scaling means a confirmation run: new prediction, held-out prompts, full checks, and Seth's yes if
   it runs over 30 minutes.
5. **Journal.** One dated entry in `research/journal.md`: what you tried, what you saw (numbers with
   intervals), the decision, and one sentence on what you learned about choosing experiments.

## 3. How to rank ideas before piloting

Score each idea 1–3 on four things and pilot the highest total first:

| | 1 | 3 |
|---|---|---|
| **would change an answer** | interesting, no effect on Q1–Q3 | could flip or strongly support an answer |
| **cheap** | needs new sampling on 4B | re-scores saved rollouts |
| **uncertain** | you are sure what will happen (then why run it?) | genuinely 50/50 |
| **teachable** | hard to turn into an exercise | makes a clear predict → test → interpret lab |

An idea you are 95% sure about is usually not worth a pilot: write down the prediction and move on,
unless it is a check something else depends on.

## 4. Calibrating taste

Every prediction goes into `research/predictions.csv`:

```
id,date,idea,prediction,probability,outcome,hit,notes
P007,2026-11-30,E5 asking vs testing on 1.7B,"asking AUROC >= testing AUROC",0.6,"0.71 vs 0.62",1,
```

After every 10 resolved predictions, add a short section to `research/taste.md`:

- your Brier score (mean of (probability − hit)²) and how it moved;
- which kinds of ideas paid off and which did not (for example "re-scoring ideas cost minutes and
  three of four changed something; style rewrites cost hours and none did");
- one or two rules you will now follow when choosing the next idea.

The Linux agent reads `taste.md` and comments in the issue; Seth gets a two-line summary in the chat.

## 5. Format for `research/ideas.md`

```
### X12 · Does the score jump at the first mention of a harmful noun?
source:      surprise in E1 pilot (score peaks early in Fortress rollouts)
bears on:    Q3 (harm vs evaluation)
prediction:  boundary scores rise within 2 sentences of the first harmful noun; p = 0.55
drop if:     no rise relative to matched WildChat toxic prompts
cheapest test: re-score saved E1 rollouts; mark first harmful-noun position by keyword list
ranking:     changes answer 3, cheap 3, uncertain 3, teachable 2 = 11
status:      idea | piloted | killed | pivoted | scaled | parked   (link journal entry)
```

New ideas are numbered X1, X2, … so they never clash with E (catalogue) and R (replications).

## 6. This also becomes course material

The journal is real research decisions with real outcomes, which is exactly what a learner rarely
gets to practise. Good candidates for Delta Drills exercises (hand over per `docs/03`):
"predict the pilot result", "given this pilot, kill, pivot or scale?", "which of these three follow-ups
is most informative?", and "this pilot looked great; why is it not yet evidence?". Mark journal entries
that would make a good exercise with `[exercise]`.
