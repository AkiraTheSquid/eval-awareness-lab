"""Build the four predict -> code -> interpret labs for the Study course `arena-eval-awareness`
(docs/06-taste-pass1-spec.md, part B).

Each lab is a staged question with three steps: the learner predicts (free, graded on the quality
of the prediction, never on hitting the answer), computes the answer from the real Qwen3-32B
reference data (code), then judges it (free). The data go into each code step's `setup` as
numbers only: no prompt text, no prompt ids.

Before writing anything, every model solution is run on its data and must reproduce the number
the lab already published (runs/e0, runs/pilots, the first-run analysis) to 3 decimals.

    python3 tools/build_aea_labs.py             # check only: build, run, compare, print
    python3 tools/build_aea_labs.py --write DIR # also merge into DIR (the course folder)

The course folder is the source of truth in Delta Drills; this script only replaces the questions
whose ids start with `aea-lab.` and leaves every other question untouched.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAST = ROOT / "reference/qwen3-32b-vast/rollouts_Qwen3-32B_R4_T384_seed0.jsonl"
FIRST = ROOT / "reference/qwen3-32b-first-run/k0_Qwen3-32B.jsonl"
E0 = ROOT / "runs/e0/results.json"
X1X2 = ROOT / "runs/pilots/x1_x2_qwen3-32b-vast.json"
VARIANCE = ROOT / "runs/e0/variance.json"


# ---------------------------------------------------------------- data

def matched_rows():
    """Per prompt of the 67 complete pairs: [is_eval, testing at boundary 0, asking at boundary 0,
    each rollout's max of testing, each rollout's max of asking]. Boundary 0 is the same in every
    rollout (no reasoning sampled yet), so one value stands for all."""
    by_pair = {}
    for line in VAST.open():
        r = json.loads(line)
        by_pair.setdefault(r["pair"], {})[r["set"]] = r
    rows = []
    for _, v in sorted(by_pair.items()):
        if len(v) != 2:
            continue
        for s in ("eval", "deploy"):
            ro = v[s]["rollouts"]
            rows.append([int(s == "eval"), round(ro[0]["testing"][0], 4), round(ro[0]["asking"][0], 4),
                         [round(max(x["testing"]), 4) for x in ro],
                         [round(max(x["asking"]), 4) for x in ro], v[s]["n_prompt"]])
    return rows


def first_run_rows():
    """Unmatched first run (486 Fortress, 486 WildChat, k = 0): [is_eval, prompt tokens, score]."""
    out = []
    for line in FIRST.open():
        r = json.loads(line)
        if not r.get("skipped"):
            out.append([int(r["set"] == "fortress_adv"), r["n_prompt"], round(r["j_primary"], 3)])
    return out


AUROC_SRC = '''def auroc(pos, neg):
    """Share of (eval, deploy) pairs where the eval prompt scores higher; ties count half."""
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))
'''


def matched_setup(with_auroc):
    data = json.dumps([r[:5] for r in matched_rows()], separators=(",", ":"))
    head = ("# Qwen3-32B reference run: 67 Fortress prompts paired with 67 WildChat prompts of matched\n"
            "# length, 4 sampled rollouts each. One dict per prompt (numbers only, no prompt text):\n"
            "#   eval      True for a Fortress (evaluation) prompt, False for WildChat (real use)\n"
            "#   testing0  log P(\"The user might be testing\") at boundary 0, before any reasoning\n"
            "#   asking0   the same for the control \"The user might be asking\"\n"
            "#   testing   4 numbers: each rollout's maximum over its boundaries, testing phrase\n"
            "#   asking    4 numbers: the same for the control phrase\n")
    body = (f"_D = {data}\n"
            "ROWS = [dict(eval=bool(e), testing0=t0, asking0=a0, testing=t, asking=a)\n"
            "        for e, t0, a0, t, a in _D]\n")
    return head + body + ("\n" + AUROC_SRC if with_auroc else "")


def first_run_setup():
    first = json.dumps(first_run_rows(), separators=(",", ":"))
    matched = json.dumps([[r[0], r[5], r[1]] for r in matched_rows()], separators=(",", ":"))
    return ("# Two Qwen3-32B runs, numbers only. Each row: (eval, prompt_tokens, monitor_k0)\n"
            "#   eval          True for a Fortress (evaluation) prompt, False for WildChat (real use)\n"
            "#   prompt_tokens length of the prompt in tokens\n"
            "#   monitor_k0    log P(\"The user might be testing\") at boundary 0, before any reasoning\n"
            "# FIRST_RUN: 486 + 486 prompts drawn without regard to length.\n"
            "# MATCHED:   67 + 67 prompts, each Fortress prompt paired with a WildChat prompt of near-equal length.\n"
            f"FIRST_RUN = [(bool(e), n, s) for e, n, s in {first}]\n"
            f"MATCHED = [(bool(e), n, s) for e, n, s in {matched}]\n\n" + AUROC_SRC)


# ---------------------------------------------------------------- the four labs

STEM_RUN = ("Qwen3-32B was given 67 Fortress prompts (adversarial prompts from an evaluation set) and "
            "67 WildChat prompts (real users), paired so each pair has near-equal length. It sampled 4 "
            "reasoning rollouts per prompt, 384 tokens at most. At every sentence boundary of the reasoning "
            "the monitor scored the log-probability of the phrase \"The user might be testing\"; a "
            "prompt's monitor score with $k$ rollouts is the maximum over all boundaries of its first $k$ "
            "rollouts, and at $k = 0$ it is the score at boundary 0, before any reasoning is sampled. "
            "AUROC is the share of (evaluation, real-use) prompt pairs the score orders correctly.")

PREDICT_HOW = (" Write three things: your number, your probability that it lands on the side you "
               "chose, and the result that would make you drop the idea. You are graded on the "
               "prediction being specific and reasoned, not on being right.")


def lab_phrase():
    code = ("def monitor_auroc(rows, phrase, k):\n"
            "    def score(r):\n"
            "        # k = 0: boundary 0 only; otherwise the best boundary of the first k rollouts\n"
            "        return r[phrase + \"0\"] if k == 0 else max(r[phrase][:k])\n"
            "    pos = [score(r) for r in rows if r[\"eval\"]]\n"
            "    neg = [score(r) for r in rows if not r[\"eval\"]]\n"
            "    return auroc(pos, neg)\n")
    starter = ("def monitor_auroc(rows, phrase, k):\n"
               "    # phrase is \"testing\" or \"asking\"; k is 0..4\n"
               "    ...\n\n"
               "# Try: print(monitor_auroc(ROWS, \"testing\", 4), monitor_auroc(ROWS, \"asking\", 4))")
    toy = ("[dict(eval=True, testing0=-9, asking0=-9, testing=[-5, -2], asking=[-6, -6]), "
           "dict(eval=False, testing0=-8, asking0=-7, testing=[-4, -3], asking=[-6, -1])]")
    return {
        "id": "aea-lab.phrase-specificity",
        "concept": "aea-results.targets",
        "prompt": STEM_RUN + " The control phrase \"The user might be asking\" was scored at the same "
                  "boundaries. It says nothing about evaluation. If the monitor reads evaluation awareness, "
                  "the testing phrase should separate the two sets better than the control does.",
        "predict": "At $k = 4$ the testing phrase's AUROC is 0.650. Predict the control phrase's AUROC at "
                   "$k = 4$: higher, about equal, or lower than 0.650, and roughly what value." + PREDICT_HOW,
        "predict_solution": "A strong prediction commits to a direction and a value, with a reason. For example: "
                            "the control should sit near 0.55, well below 0.650, because \"asking\" carries no "
                            "evaluation meaning; probability 0.6 that it is lower. Drop the evaluation reading "
                            "if the control comes out at or above the testing phrase, since then the score "
                            "tracks something both phrases share, such as how the model frames any user "
                            "request on these prompts.",
        "code_text": "Write `monitor_auroc(rows, phrase, k)`: the AUROC of the monitor built from `phrase` "
                     "(\"testing\" or \"asking\") with $k$ rollouts, evaluation prompts as positives. "
                     "`auroc(pos, neg)` is defined for you.",
        "starter": starter, "code": code, "setup": matched_setup(True),
        "code_solution": "Each prompt gets one score: the boundary-0 value at $k = 0$, otherwise the largest "
                         "per-rollout maximum among its first $k$ rollouts. AUROC compares every "
                         "evaluation score with every real-use score and counts the share it wins, ties half.",
        "tests": [
            (f"monitor_auroc({toy}, 'testing', 1)", None, {"0.5": "Check the order: an eval score above a deploy score is a win."}),
            (f"monitor_auroc({toy}, 'asking', 0)", None, {}),
            ("monitor_auroc(ROWS, 'testing', 4)", "0.650", {}),
            ("monitor_auroc(ROWS, 'asking', 4)", "0.743", {}),
            ("monitor_auroc(ROWS, 'testing', 0)", "0.665", {}),
        ],
        "interpret": "Compare your result with your prediction. What does it say about whether this "
                     "monitor reads evaluation awareness on this data? Decide: kill, pivot or scale the "
                     "claim that the testing phrase measures awareness, and name the next test you would run.",
        "interpret_solution": "The control separates the sets better than the testing phrase: 0.743 against "
                              "0.650 at $k = 4$, and ranking prompts by max(testing) − max(asking) gives only 0.448, below chance. So "
                              "on this data the score mostly tracks something both phrases share, not "
                              "evaluation in particular; the evaluation reading should not be scaled as it "
                              "stands. Pivot: ask where the control's edge comes from. The next test compares "
                              "the two phrases at $k = 0$, before any reasoning, against $k = 4$: an edge "
                              "already there at $k = 0$ comes from the prompt itself; one that grows with $k$ "
                              "comes from the reasoning. A family of pre-registered control phrases would "
                              "show whether \"testing\" beats any of them.",
        "expect": {"monitor_auroc(ROWS, 'testing', 4)": ("e0", "monitor k=4"),
                   "monitor_auroc(ROWS, 'asking', 4)": ("e0", "control alone k=4"),
                   "monitor_auroc(ROWS, 'testing', 0)": ("e0", "monitor k=0")},
    }


def lab_gap_source():
    code = ("def gap_by_k(rows):\n"
            "    def score(r, phrase, k):\n"
            "        return r[phrase + \"0\"] if k == 0 else max(r[phrase][:k])\n"
            "    def roc(phrase, k):\n"
            "        pos = [score(r, phrase, k) for r in rows if r[\"eval\"]]\n"
            "        neg = [score(r, phrase, k) for r in rows if not r[\"eval\"]]\n"
            "        return auroc(pos, neg)\n"
            "    return [roc(\"asking\", k) - roc(\"testing\", k) for k in range(5)]\n")
    starter = ("def gap_by_k(rows):\n"
               "    # AUROC(asking) - AUROC(testing) for k = 0, 1, 2, 3, 4; auroc(pos, neg) is given\n"
               "    ...\n\n"
               "# Try: print([round(g, 3) for g in gap_by_k(ROWS)])")
    toy = ("[dict(eval=True, testing0=-3, asking0=-1, testing=[-2, -2, -2, -2], asking=[-1, -1, -1, -1]), "
           "dict(eval=False, testing0=-2, asking0=-2, testing=[-3, -3, -3, -3], asking=[-2, -2, -2, -2])]")
    return {
        "id": "aea-lab.gap-source",
        "concept": "aea-results.scores",
        "prompt": STEM_RUN + " On this run the control phrase \"The user might be asking\" separates the sets "
                  "better than the testing phrase: AUROC 0.743 against 0.650 at $k = 4$. The control's edge "
                  "could come from the prompt alone, visible at boundary 0, or from the reasoning the model "
                  "samples, growing with $k$.",
        "predict": "Predict the edge AUROC(asking) − AUROC(testing) at $k = 0$ compared with $k = 4$: "
                   "already about as large at $k = 0$, or clearly smaller and growing with $k$?" + PREDICT_HOW,
        "predict_solution": "A strong prediction picks one source and says why. For example: most of the edge "
                            "is already there at $k = 0$, because the prompt alone decides how a model frames "
                            "the request, and sampled reasoning should add little; probability 0.6. Drop it if "
                            "the $k = 0$ edge is under half the $k = 4$ edge or sits within noise.",
        "code_text": "Write `gap_by_k(rows)`: the list of AUROC(asking) − AUROC(testing) for $k = 0, 1, 2, 3, 4$, "
                     "evaluation prompts as positives. `auroc(pos, neg)` is defined for you.",
        "starter": starter, "code": code, "setup": matched_setup(True),
        "code_solution": "Score every prompt with each phrase at each $k$ exactly as the monitor does (boundary "
                         "0 at $k = 0$, else the best of the first $k$ rollout maxima), take each phrase's "
                         "AUROC, and subtract testing from asking at every $k$.",
        "tests": [
            (f"gap_by_k({toy})", None, {}),
            ("round(gap_by_k(ROWS)[0], 3)", "0.049", {}),
            ("round(gap_by_k(ROWS)[4], 3)", "0.093", {"-0.093": "Subtract the other way: asking minus testing."}),
        ],
        "interpret": "The lab's paired bootstrap gives $k = 0$: +0.049 [−0.012, +0.107] and $k = 4$: +0.093 "
                     "[+0.028, +0.160] on these 67 pairs. Compare with your prediction, decide kill, pivot or "
                     "scale, and say what you would test next and on which data.",
        "interpret_solution": "At $k = 0$ the edge is about half the $k = 4$ edge and its interval includes 0, so "
                              "the prompt alone gives the control at most a small edge; the edge roughly doubles "
                              "as sampled reasoning is added and only clears noise at $k = 4$. A prediction "
                              "that the edge was mostly in the prompt missed (the lab's own did, at p = 0.6). "
                              "Pivot: the question becomes whether the reasoning raises a general "
                              "\"the user wants something\" framing more on Fortress prompts. Test it on a "
                              "fresh, held-out set of pairs with more control phrases re-scored on the same "
                              "rollouts, since 67 pairs and one look already spent cannot confirm it.",
        "expect": {"round(gap_by_k(ROWS)[0], 3)": ("x1", "0"), "round(gap_by_k(ROWS)[4], 3)": ("x1", "4")},
    }


def lab_length():
    code = ("def length_vs_monitor(rows):\n"
            "    pos = [r for r in rows if r[0]]\n"
            "    neg = [r for r in rows if not r[0]]\n"
            "    by_length = auroc([r[1] for r in pos], [r[1] for r in neg])\n"
            "    by_monitor = auroc([r[2] for r in pos], [r[2] for r in neg])\n"
            "    return (by_length, by_monitor)\n")
    starter = ("def length_vs_monitor(rows):\n"
               "    # return (AUROC of prompt length, AUROC of the monitor at k = 0); auroc(pos, neg) is given\n"
               "    ...\n\n"
               "# Try: print(length_vs_monitor(FIRST_RUN), length_vs_monitor(MATCHED))")
    return {
        "id": "aea-lab.length-confound",
        "concept": "aea-validity.confounds",
        "prompt": "Qwen3-32B was scored on prompts from two sources: Fortress (adversarial prompts from an "
                  "evaluation set) and WildChat (real users). The monitor's score at $k = 0$ is the "
                  "log-probability of \"The user might be testing\" right after the prompt, before any "
                  "reasoning. A first run drew 486 prompts from each source without regard to length. A "
                  "second run paired 67 prompts from each source by near-equal length. AUROC is the share "
                  "of (evaluation, real-use) prompt pairs a score orders correctly.",
        "predict": "On the first run, will prompt length alone separate the two sources better or worse than "
                   "the monitor does, and what happens to the monitor once length is matched?" + PREDICT_HOW,
        "predict_solution": "A strong prediction gives both numbers a direction. For example: on the first run "
                            "length separates the sources about as well as the monitor, near 0.6, because "
                            "evaluation prompts are written to a template and differ in length from real "
                            "requests; once matched, length falls to 0.5 by construction and the monitor "
                            "drops toward 0.55; probability 0.55. Drop the awareness reading if the monitor "
                            "falls to chance once length is matched.",
        "code_text": "Write `length_vs_monitor(rows)`: return the pair (AUROC of prompt length, AUROC of the "
                     "monitor at $k = 0$), evaluation prompts as positives. Each row is "
                     "`(eval, prompt_tokens, monitor_k0)`; `auroc(pos, neg)` is defined for you.",
        "starter": starter, "code": code, "setup": first_run_setup(),
        "code_solution": "A baseline is scored exactly like the monitor: split the rows by source, rank by "
                         "prompt length, then by the monitor score, and take each AUROC. Length needs no "
                         "model at all, so it sets the bar the monitor has to clear.",
        "tests": [
            ("length_vs_monitor([(True, 10, -2.0), (False, 5, -3.0), (False, 20, -1.0)])", None, {}),
            ("length_vs_monitor(FIRST_RUN)", None, {}),
            ("length_vs_monitor(MATCHED)", None, {}),
        ],
        "interpret": "Compare with your prediction. What did the first run's monitor result mean, and what "
                     "does the matched run change? Decide kill, pivot or scale, and name the next confound "
                     "you would match on.",
        "interpret_solution": "On the first run, prompt length alone (0.665) beat the monitor (0.611), so that "
                              "monitor number could not count as evidence of awareness: a model-free baseline "
                              "did better. On the matched run length is at chance (0.500) and the monitor "
                              "reaches 0.665, so its separation does not come from length. The matched sample "
                              "is only 67 pairs, so its interval is wide (about 0.57 to 0.75). Scale the "
                              "matched design, not the claim: other differences between the sources remain, "
                              "such as topic, harm and writing style. The next test matches on harm or topic, "
                              "or trains a plain text classifier to see how easy the two sources are to tell "
                              "apart without the model.",
        "expect": {"length_vs_monitor(FIRST_RUN)": ("first", None), "length_vs_monitor(MATCHED)": ("e0pair", None)},
    }


def lab_variance():
    code = ("from statistics import mean, variance\n\n"
            "def share_between(rows):\n"
            "    per_prompt = [r[\"testing\"] for r in rows]          # 4 rollout maxima per prompt\n"
            "    between = variance([mean(x) for x in per_prompt])  # spread of prompt means\n"
            "    within = mean(variance(x) for x in per_prompt)     # spread inside a prompt, averaged\n"
            "    return between / (between + within)\n")
    starter = ("from statistics import mean, variance\n\n"
               "def share_between(rows):\n"
               "    # between-prompt variance / (between + within), from each prompt's 4 rollout maxima\n"
               "    ...\n\n"
               "# Try: print(share_between(ROWS))")
    toy = "[dict(testing=[1, 3, 1, 3]), dict(testing=[5, 7, 5, 7])]"
    return {
        "id": "aea-lab.variance-share",
        "concept": "aea-results.variance",
        "prompt": STEM_RUN + " Each rollout's score is its maximum over boundaries. Some of the spread in those "
                  "scores comes from which prompt it is (between prompts), and some from which reasoning "
                  "happened to be sampled for the same prompt (within a prompt). The post reports that "
                  "whether a prompt ever produces a verbalising rollout explains 24.9% of the score's "
                  "variance, while whether a given rollout verbalises explains 0.4%.",
        "predict": "Predict the share of the variance of rollout scores that sits between prompts: "
                   "$\\text{between} / (\\text{between} + \\text{within})$." + PREDICT_HOW,
        "predict_solution": "A strong prediction gives a range and ties it to the post's numbers. For example: "
                            "most of the variance is between prompts, around 0.7, because verbalisation at the "
                            "prompt level explains far more than at the rollout level; probability 0.65 that "
                            "it exceeds 0.6. Drop the view that awareness is a property of the prompt if the "
                            "share is under 0.4.",
        "code_text": "Write `share_between(rows)`: from each prompt's 4 rollout maxima of the testing phrase "
                     "(`r[\"testing\"]`), return between / (between + within), where between is the sample "
                     "variance of the 134 prompt means and within is the mean of each prompt's own sample "
                     "variance.",
        "starter": starter, "code": code, "setup": matched_setup(False),
        "code_solution": "Average each prompt's 4 rollout scores and take the sample variance of those means "
                         "for the between part; take each prompt's own sample variance and average them for "
                         "the within part. The share is the between part over the total: a quick split that "
                         "slightly overstates the between part, because a mean of 4 rollouts still carries "
                         "a quarter of the within variance.",
        "tests": [
            (f"share_between({toy})", None, {}),
            ("share_between(ROWS)", "0.858", {"0.142": "That is the within-prompt share; put between on top."}),
        ],
        "interpret": "Compare with your prediction. What does the share say about where the monitor's score "
                     "comes from, and what would an evaluation designer do with it? Decide kill, pivot or "
                     "scale, and say what this quick split cannot show.",
        "interpret_solution": "About 86% of the variance sits between prompts (between 55.8, within 9.2), so the "
                              "score is mostly a property of the prompt, not of a lucky rollout. That agrees with "
                              "the post's 24.9% against 0.4%, and it is useful to an evaluation designer: prompt "
                              "features can be audited and edited before any rollout is sampled. Scale, with "
                              "care. The variance of 4-rollout means still holds within / 4 ≈ 2.3 of sampling "
                              "noise; subtracting it leaves 53.5 between and a share of about 0.85, so the "
                              "conclusion holds, but with few rollouts per prompt the correction matters. The "
                              "split also ignores pairing and source, the within part is noisy with 4 "
                              "rollouts, and the testing score is itself partly driven by "
                              "what the control phrase also picks up. A proper mixed-effects model with source, "
                              "pair and prompt terms is the next test.",
        "expect": {"share_between(ROWS)": ("var", "share_between")},
    }


LABS = [lab_phrase, lab_gap_source, lab_length, lab_variance]


# ---------------------------------------------------------------- build + verify

def run(setup, code, call):
    ns = {}
    exec(setup, ns)
    exec(code, ns)
    return eval(call, ns)


def show(v):
    if isinstance(v, float):
        return ("%.3f" % v).rstrip("0").rstrip(".") if v else "0.0"
    if isinstance(v, (list, tuple)):
        inner = ", ".join(show(x) for x in v)
        return "(" + inner + ")" if isinstance(v, tuple) else "[" + inner + "]"
    return repr(v)


def published():
    """The numbers the lab already published, keyed as the labs' `expect` maps name them."""
    e0 = {row[0]: row[1] for row in json.loads(E0.read_text())["rows"]}
    x1 = json.loads(X1X2.read_text())["x1"]
    var = json.loads(VARIANCE.read_text())
    return {"e0": e0, "x1": {k: v["asking_minus_testing"] for k, v in x1.items()},
            "var": var, "first": (0.665, 0.611), "e0pair": (e0["prompt length"], e0["monitor k=0"])}


def check(lab, pub):
    errs = []
    for call, (src, key) in lab["expect"].items():
        got = run(lab["setup"], lab["code"], call)
        want = pub[src] if key is None else pub[src][key]
        g, w = (got, want) if isinstance(got, tuple) else ((got,), (want,))
        if any(abs(a - b) > 0.0005 for a, b in zip(g, w)):
            errs.append(f"{lab['id']}: {call} = {show(got)}, published {want}")
    try:
        starter_out = run(lab["setup"], lab["starter"], lab["tests"][-1][0])
    except Exception:
        starter_out = None
    if starter_out is not None:
        errs.append(f"{lab['id']}: the starter returns {starter_out!r} on the last test")
    return errs


def question(lab):
    tests = []
    for call, expect, mistakes in lab["tests"]:
        exp = expect or show(run(lab["setup"], lab["code"], call))
        tests.append({"call": call, "expect": exp, "mistakes": mistakes})
    c = lab["concept"]
    return {
        "id": lab["id"], "kind": "evaluative", "format": "staged", "demand": "evaluate", "targets": [c],
        "prompt": lab["prompt"],
        "reference": lab["interpret_solution"],
        "parts": [
            {"kind": "free", "text": lab["predict"], "concept": c, "segment": "", "facet": "explain",
             "solution": lab["predict_solution"]},
            {"kind": "code", "text": lab["code_text"], "concept": c, "segment": "", "facet": "do",
             "starter": lab["starter"], "setup": lab["setup"], "code": lab["code"],
             "solution": lab["code_solution"], "tests": tests},
            {"kind": "free", "text": lab["interpret"], "concept": c, "segment": "", "facet": "explain",
             "solution": lab["interpret_solution"]},
        ],
    }


def file_for(folder, concept):
    section = concept.split(".")[0]
    return folder / section / f"{concept}.questions.json"


def evidence(lab):
    """One `direct` evidence argument per step (COURSE_RECIPE.md §4: a staged part is its own
    unit `<qid>#<n>`). All three steps measure the lab's one concept on one shared case."""
    q, c = lab["id"], lab["concept"]
    return [
        {"question": f"{q}#1", "role": "direct",
         "observes": "The response commits to a direction and a value for the quantity, attaches a "
                     "probability, and names the result that would make the learner drop the idea.",
         "warrant": "Stating a checkable prediction before the computation shows the learner can turn "
                    f"this idea ({c}) into an expectation about real data, which is the first move "
                    "in choosing an informative test.",
         "rival": "A vague or hedged statement can sound reasonable without committing to anything the "
                  "data could contradict; the grade is on specificity and reasoning, not on being right."},
        {"question": f"{q}#2", "role": "direct",
         "observes": "The function reproduces the published number on the real reference data and passes "
                     "the small hand-built cases.",
         "warrant": "Computing the quantity from scored rollouts shows the learner can carry out the "
                    "measurement the idea calls for, not only describe it.",
         "rival": "Code adapted from a similar example can pass if the data are shaped the same way; the "
                  "hand-built cases make an order or sign error visible."},
        {"question": f"{q}#3", "role": "direct",
         "observes": "The response compares the computed result with the prediction, states what it "
                     "supports and what it does not, decides kill, pivot or scale, and names a next test.",
         "warrant": "Judging a real result against a prior prediction, and choosing the next "
                    "experiment from it, is the evaluative step this idea is meant to support.",
         "rival": "A response can restate the number without saying what it rules out, or propose a "
                  "next step that would not change the conclusion."},
    ]


def write_design(path, labs):
    d = json.loads(path.read_text())
    ours = {lab["concept"]: evidence(lab) for lab in labs}
    found = set()
    for node in d["nodes"]:
        if node["id"] in ours:
            # replace only the units this script generates; other evidence stays
            mine = {e["question"] for e in ours[node["id"]]}
            keep = [e for e in node["evidence"] if e["question"] not in mine]
            node["evidence"] = keep + ours[node["id"]]
            found.add(node["id"])
    missing = set(ours) - found
    if missing:
        sys.exit(f"{path}: no design node for {sorted(missing)}; questions would have no evidence")
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote evidence arguments -> {path}")


def main(argv):
    pub = published()
    labs = [f() for f in LABS]
    errs = [e for lab in labs for e in check(lab, pub)]
    qs = [question(lab) for lab in labs]
    for q in qs:
        code = q["parts"][1]
        print(f"{q['id']:30s} setup {len(code['setup']) / 1024:5.1f} KB  tests: "
              + "; ".join(f"{t['call'][:40]} -> {t['expect']}" for t in code["tests"][-2:]))
    text = json.dumps(qs)
    for needle in ("\"text\":", "Okay, let", "fortress_adv|", "wildchat"):
        if needle in "".join(q["parts"][1]["setup"] for q in qs):
            errs.append(f"setup contains {needle!r}: prompt data leaked")
    if errs:
        print("\nFAILED:\n  " + "\n  ".join(errs))
        return 1
    print(f"\nall {len(qs)} labs reproduce the published numbers; starters fail; {len(text) // 1024} KB total")
    if "--write" in argv:
        folder = Path(argv[argv.index("--write") + 1])
        for q in qs:
            path = file_for(folder, q["targets"][0])
            cur = json.loads(path.read_text())
            cur = [x for x in cur if x["id"] != q["id"]] + [q]
            path.write_text(json.dumps(cur, indent=2, ensure_ascii=False) + "\n")
            print(f"wrote {q['id']} -> {path}")
        write_design(folder / "authoring" / f"{folder.name}.design.json", labs)
    if "--pack" in argv:
        # the recipe gate reads the design beside the pack (course_recipe.py), as patch scripts write it
        pack = Path(argv[argv.index("--pack") + 1])
        write_design(pack / "authoring" / "arena-eval-awareness.design.json", labs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
