"""Which lesson owns each symbol a piece of drill code uses, and is that lesson upstream?

    report(code, prereqs=[...], own=[...])  -> list of (symbol, owner, status)

status: ok (owner is a prerequisite or declared by this unit), UNOWNED (no page teaches it),
OFF-BRANCH (a page teaches it, but it is not upstream of the given prerequisites).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_solution_prereqs as A  # noqa: E402
from solution_symbols import collect  # noqa: E402

_DECLARED, _PAGES = A.declaring_kcs()
_RANK = A.lesson_order(_PAGES)
_BEFORE = A.ancestors()


def closure(prereqs):
    out = set(prereqs)
    for k in prereqs:
        out |= _BEFORE.get(k, set())
    return out


def report(code, prereqs, own=(), known=()):
    reach = closure(prereqs)
    rows = []
    for sym in sorted(collect(code, set(known))):
        if sym in A.SELF_DEFINED:
            continue
        if any(a in own for a in A.aliases(sym)) or sym in own:
            rows.append((sym, "(this unit)", "ok"))
            continue
        owner = A.owner_of(sym, _DECLARED, _RANK)
        status = "UNOWNED" if owner is None else ("ok" if owner in reach else "OFF-BRANCH")
        rows.append((sym, owner, status))
    return rows
