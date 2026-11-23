"""Calibration of research predictions: Seth's vs the agents' (docs/05-research-loop.md §4).

Reads research/predictions.csv (agents) and research/seth-predictions.csv (Seth). A row counts once
its `hit` is 0 or 1. Seth's rows marked late (written after the result existed) are left out of his
score, because a prediction made with the answer visible measures nothing.

    python research/calibration.py
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def load(path: Path, skip_late: bool) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for r in csv.DictReader(path.open(newline="")):
        if r.get("hit", "").strip() not in ("0", "1"):
            continue  # not resolved yet
        if skip_late and r.get("late", "").strip().lower() in ("1", "true", "yes", "late"):
            continue
        try:
            p = float(r["probability"])
        except ValueError:
            sys.exit(f"{path.name} {r.get('id')}: probability {r['probability']!r} is not a number")
        if not 0.0 <= p <= 1.0:
            sys.exit(f"{path.name} {r.get('id')}: probability {p} is outside 0 to 1")
        rows.append({**r, "p": p, "y": int(r["hit"])})
    return rows


def brier(rows: list[dict]) -> float:
    # mean squared gap between the stated probability and what happened
    return sum((r["p"] - r["y"]) ** 2 for r in rows) / len(rows)


def report(name: str, rows: list[dict]) -> None:
    print(f"\n## {name}: {len(rows)} resolved")
    if not rows:
        return
    hits = sum(r["y"] for r in rows)
    mean_p = sum(r["p"] for r in rows) / len(rows)
    print(f"hit rate {hits}/{len(rows)} = {hits / len(rows):.2f}; mean stated probability {mean_p:.2f}; "
          f"Brier {brier(rows):.3f} (always saying 0.5 scores 0.250)")
    print("per row: " + ", ".join(f"{r['id']} {(r['p'] - r['y']) ** 2:.3f}" for r in rows))
    print("| stated p | n | hit rate |")
    print("|---|---|---|")
    for lo, hi in BINS:
        b = [r for r in rows if lo <= r["p"] < hi]
        if b:
            print(f"| {lo:.1f}-{min(hi, 1):.1f} | {len(b)} | {sum(r['y'] for r in b) / len(b):.2f} |")


def head_to_head(seth: list[dict], agent: list[dict]) -> None:
    by_id = {r["id"]: r for r in agent}
    pairs = [(s, by_id[s["ref"]]) for s in seth if s.get("ref") in by_id]
    print(f"\n## Head to head on the same questions: {len(pairs)}")
    if pairs:
        print(f"Seth Brier {brier([s for s, _ in pairs]):.3f}, agent Brier {brier([a for _, a in pairs]):.3f}")


def main() -> int:
    agent = load(HERE / "predictions.csv", skip_late=False)
    seth = load(HERE / "seth-predictions.csv", skip_late=True)
    report("Seth", seth)
    report("Agents", agent)
    head_to_head(seth, agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
