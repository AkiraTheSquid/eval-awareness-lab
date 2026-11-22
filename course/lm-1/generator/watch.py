"""watch.py — health checks for logit_monitor_course

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL. No torch needed: the checks read
source text, they do not import the framework (which imports the refusal course's data).
"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../..'))


def _read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


def check_imports():
    src = _read("framework.py")
    compile(src, "framework.py", "exec")
    shared = os.path.join(HERE, "..", "refusal_course", "framework.py")
    assert os.path.exists(shared), "the refusal framework this unit loads is missing"


def check_public_api():
    src = _read("framework.py")
    for name in ("Drill", "prepare", "upsert_csv", "resolve_ids", "write_overrides", "write_tags", "def render_kp"):
        assert name in src, f"framework.py no longer exposes {name}"


def check_invariants():
    src = _read("framework.py")
    assert 'LESSON_ID = "lm-1"' in src, "lesson id changed: the CSV block and registry rows key on it"
    assert 'R.IDS_PATH = HERE / "ids.json"' in src, "ids must live in THIS folder, not the refusal course's"
    assert "refusal_course" not in "".join(
        f for f in os.listdir(HERE) if f.endswith(".json")), "no refusal state may be copied here"


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
