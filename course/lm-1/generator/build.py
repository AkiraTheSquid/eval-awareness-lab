"""Build the logit-monitor unit: python build.py [--check | --apply]  (run under the torch venv).

--check  execute every drill (reference passes, starter fails, near miss differs) and report every
         symbol a solution uses that no upstream lesson teaches
--apply  the same, then: CSV rows, export to mint ids, overrides, tags, registry, graph, clocks,
         glossary, lesson pages
"""
import importlib
import json
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import framework as F  # noqa: E402
import wiring as W  # noqa: E402
from symcheck import report  # noqa: E402

KC_MODULES = ["kc1_next_token", "kc2_forced_logprob", "kc3_forced_sequence", "kc4_boundaries", "kc5_padded_batch", "kc6_prompt_score", "kc7_auroc", "kc8_baselines"]
EXPORT = F.ROOT / "Local_Deployed_Shared/pipeline/export_questions_json.py"


def load():
    mods = [importlib.import_module(m) for m in KC_MODULES]
    return mods, [d for m in mods for d in m.DRILLS]


def symbol_problems(mods, prepared):
    """Symbols a solution needs that neither an upstream lesson nor this unit's pages declare."""
    own_by_kc, bad = {}, []
    for m in mods:   # a page may use what it declares and what its unit prerequisites declare
        own = set(m.SPEC.get("new_syntax", []))
        for p in W.KCS[m.KC]["prereqs"]:
            own |= own_by_kc.get(p, set())
        own_by_kc[m.KC] = own
    for m in mods:
        outside = [p for p in W.upstream(m.KC) if p not in W.KCS]
        for d in m.DRILLS:
            for sym, owner, status in report(prepared[d.key]["answer"], outside, own_by_kc[m.KC]):
                if status != "ok":
                    bad.append(f"{d.key}: {sym} {status} (owner {owner})")
    return bad


def main(mode):
    mods, drills = load()
    assert len({d.key for d in drills}) == len(drills), "duplicate drill key"
    prepared = {d.key: F.prepare(d) for d in drills}
    for d in drills:
        p = prepared[d.key]
        print(f"ok {d.key:8} {d.rung:10} cases={len(p['test_cases'])} near-miss fails {p['wrong_fail_cases']}/{len(d.cases)}")
    bad = symbol_problems(mods, prepared)
    print("\n".join(bad) or "symbols: every solution symbol is taught upstream")
    if mode != "--apply":
        return
    assert not bad, "fix the symbol problems first"
    F.upsert_csv(drills, prepared)
    F.drop_overrides()
    subprocess.run([sys.executable, str(EXPORT)], check=True)
    ids = F.resolve_ids(drills)
    F.write_overrides(drills, prepared, ids)
    F.write_tags(drills, ids)
    kc_ids = [m.KC for m in mods]
    atoms = {a for d in drills for a, _ in d.atoms if a in W.ATOMS}
    W.apply_registry(kc_ids)
    W.apply_graph(atoms, [(ids[d.key]["id"], d.atoms[0][0]) for d in drills])
    W.apply_caps(kc_ids)
    W.apply_glossary(kc_ids)
    for m in mods:
        spec = dict(m.SPEC, segments=m.SEGMENTS, misconceptions=m.MISCONCEPTIONS)
        path = F.LESSONS / F.FOLDER / f"kp-{spec['id'].split('.', 1)[1]}.md"
        path.parent.mkdir(exist_ok=True)
        path.write_text(F.render_kp(spec, drills, ids, prepared))
        print("wrote", path)
    subprocess.run([sys.executable, str(EXPORT)], check=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode not in ("--check", "--apply") or len(sys.argv) > 2:
        sys.exit(f"usage: build.py [--check | --apply] (got {sys.argv[1:]})")
    main(mode)
