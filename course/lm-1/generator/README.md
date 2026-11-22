# logit_monitor_course

## Purpose
- Generator for the ARENA course unit `lm-1`, "Logit monitor for evaluation awareness": the concepts
  and drills that lead up to a from-scratch replication of Goodfire's LessWrong post "Logits as a
  new monitor for evaluation awareness".
- **Status 2026-11-21: BUILT, not committed, not deployed.** Eight concepts `monitor.*`, 88 drills
  (q1815-1902), pages, registry rows, atoms and edges are applied to the working tree.
- The conceptual half of the topic (no code) is the private Study module `arena-eval-awareness`.
  This unit is the code half and adds no multiple-choice questions.

## Owns
- The source of truth for the unit's lesson pages (`Local_Deployed_Shared/lessons/logit-monitor/`),
  its drills, its registry rows and its atoms. The pages are GENERATED from here; edit the modules in this folder, never the pages.

## Does NOT own
- The drill machinery. It lives in `scripts/refusal_course/framework.py` and is loaded here as a
  private module instance; this folder never edits it.
- The capstone Colab notebook where the learner writes the real replication.
- Any backend code.

## Key Files
- `framework.py`: loads the refusal course's framework under a private module name and repoints
  its ids file, lesson id and topic at this unit. Wraps `render_kp` so a page can declare
  `new_syntax` (the refusal renderer hard-codes an empty list). `maker(prefix, kc)` returns a short
  Drill constructor; `_src` adds `import einops` to a drill that uses it.
- `build.py`: `--check` executes every drill and checks every solution symbol is owned upstream;
  `--apply` also writes bank rows, overrides, tags, registry, graph and pages.
- `wiring.py`: registry rows (prereqs, encompassing weights), atoms, graph edges.
- `symcheck.py`: reports who owns each symbol a snippet uses, before a drill is written.
- `kc1_next_token.py` ... `kc8_baselines.py`: one concept each (lesson segments + 11 drills).
- `ids.json`: question ids and CSV block position. Written by `--apply`; do not edit.

## Data & External Dependencies
- `scripts/refusal_course/framework.py` (Drill, prepare, CSV block, overrides, tags, page renderer).
- Writes on `--apply`: `kc_registry.json`, `glossary.js`, `placement_time_caps.json`,
  `curated_additions.csv`, `curated_overrides.jsonl`, `question_atom_tags.jsonl`,
  `arena_drillable_v1.json`.
- Needs the torch venv: `This-Directory-Only/backend/.venv/bin/python3`.

## How It Works (Flow)
1. `build.py --check`: every drill's answer runs on its cases, the near miss must differ on its
   anchor case, every solution symbol must be owned by the concept or an ancestor.
2. `build.py --apply`: CSV block, export (mints ids), overrides, tags, registry, graph, pages.
3. Then, all under the venv python: `scripts/compile_lessons.py`, `scripts/build_qmatrix.py`,
   `scripts/build_symbol_prereqs.py`, `This-Directory-Only/scripts/export_kc_atom_crosswalk.py`,
   `scripts/validate_lessons.py --coverage`, `Local_Deployed_Shared/pipeline/audit_question_bank.py
   --gate`, `scripts/audit_solution_prereqs.py --new`, `scripts/audit_arena_grounding.py --new`.

## Invariants & Constraints
- **Directness (Seth, 2026-11-21):** a concept exists only if the replication's code needs it.
  Derive the concept list from the functions the code uses and how often, most-used first. Do not
  add concepts from the paper's prose that no code step requires.
- Reductions over named axes use `einops.reduce` (Seth's preference); `log_softmax`, token
  selection, pairwise comparison, sorting and resampling stay plain torch.
- Never author or scaffold Faded drills beyond the one stub per segment the validator demands.
- The paper has NO official code release. Function counts come from third-party code or from our
  own reference implementation; say which whenever a count is quoted.
- Toy numbers in lesson examples are labelled as toy until a real sweep exists.
- Drills are tensor-only. Hugging Face calls (`apply_chat_template`, `generate`, the model call)
  appear only in `python no-run` fences in lesson prose.
- Symbols with no ARENA notebook use are avoided: `multinomial`, `isin`, `logsumexp`. Sampling
  internals are not taught; the replication calls `generate`.
- A near miss's shown setup must be the setup of case `show` (default 0). The framework does not
  check this; the bank gate does (`wrong_example_matches_correct`).
- Lesson code lines: 54 columns at most. A drill must not reuse a worked example's data.

## Extension Points
- Add a concept as `kcN_<slug>.py` exporting `KC`, `SPEC`, `SEGMENTS`, `MISCONCEPTIONS`, `DRILLS`,
  the shape the refusal course's concept modules use.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Function counts have no authoritative source** — `ACTIVE`
  - When it happens: prioritising concepts by how often the paper's code uses a function.
  - Symptom: counts reflect whoever wrote the code being counted.
  - Root cause: the authors released no code.
  - Prevention/fix: count from third-party replications and ARENA 4.3, keep our own reference as
    the answer key only. Decision pending with Seth.
  - Status: `ACTIVE`.

## Recent Changes
- 2026-11-21: Unit built: `build.py`, `wiring.py`, `symcheck.py`, eight `kc*.py` modules; applied
  to the tree (q1815-1902). Validator, bank gate, prereq and grounding ratchets pass with 0 new.
- 2026-11-21: Folder created with `framework.py` only. Build paused for Seth's decision on where
  the function counts come from.
