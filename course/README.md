# Course snapshot (read-only copy, 22 Nov 2026)

The Delta Drills material these experiments feed into, copied here so the Windows agent can read it.
The source of truth stays in the Delta Drills repository; edits made here are not picked up.
To change course content, open an issue for the Linux agent.

- `arena-eval-awareness/`: the conceptual course (14 concepts in 3 sections: `aea-observe`,
  `aea-results`, `aea-validity`). Each concept is `<section>/<concept>.md` (lesson) plus
  `<concept>.questions.json`. `_sources/` holds the full texts of the papers and posts it is built on
  (Goodfire post and companion paper, graders post, Needham, Heidari, Chen, Fawcett, research guide).
  Use these for the related-work replications in `docs/02-experiments.md` §2b.
- `lm-1/`: the code unit (8 implementation skills, `monitor.*`). `pages/` are the generated lesson
  pages, `generator/` is the script that builds them, `drills.csv` holds its 88 drills.
