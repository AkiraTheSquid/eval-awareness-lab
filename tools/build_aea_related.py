"""Build the related-work section `aea-related` of the Study course `arena-eval-awareness`.

What this writes (the lessons, `aea-related/*.md`, are hand-written in the course folder):
  * every question of the ten `aea-related.*` concepts, including five predict -> code -> interpret
    labs whose data are made up for teaching and generated here from fixed seeds;
  * the section's verbatim source passages (`passages.json`);
  * the section's typed edges (`edges.json`);
  * the design record: candidates, nodes with evidence, dependencies, semantic links, and the
    integrated evidence the comparison questions add to four existing nodes.

Before writing, every lab's model code is run on its data: it must give the numbers the lab text
states, and the starter must fail the last test. The labs' numbers are invented teaching data, and
every lab prompt says so; the papers' own numbers appear only in the lessons.

    python3 tools/build_aea_related.py                      # build and check only
    python3 tools/build_aea_related.py --write <folder>     # also write into the course folder
    python3 tools/build_aea_related.py --pack <pack dir>    # also write the design beside the pack

The `aea-related` section is wholly owned by this script. Re-running replaces: every
`aea-related.*.questions.json`, the passages it lists, every edge whose source is an aea-related
node, the design candidates/nodes/dependencies for aea-related nodes, every semantic link touching
one, and evidence units on older nodes whose question id starts with `aea-related.`. Hand edits to
any of these are lost on the next run; make them here instead. Every output is computed and checked
before any file is written, so a failure leaves the course untouched.
"""
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEC = "aea-related"
PASSAGE_SCRIPT_SOURCES = {
    "aea-needham": "needham", "aea-chen": "chen", "aea-companion": "companion", "aea-heidari": "heidari",
    "aea-graders": "graders", "aea-research-guide": "research-guide",
}

# ---------------------------------------------------------------- passages (verbatim, cut by markers)
# (id, source id, locator, start marker, end marker); the text is cut from _sources/<file>.txt
PASSAGES = [
    ('aea.p.asking', 'aea-needham', '4.1 Frontier models show evaluation-awareness',
     'We find that frontier models are better at distinguishing', 'due to the authors’ familiarity with the datasets used.'),
    ('aea.p.asking-calibration', 'aea-needham', '4.3 Frontier models tend to over-classify transcripts as evaluations',
     'While AUC measures discriminative ability', 'when they otherwise might not have.'),
    ('aea.p.mention', 'aea-chen', 'Testing for faithfulness',
     'After checking that the models really did use the hints', 'A substantial majority of answers, then, were unfaithful.'),
    ('aea.p.mention-limits', 'aea-chen', 'Conclusions',
     'Like all experiments, ours have limitations.', 'making monitoring more straightforward.'),
    ('aea.p.resample', 'aea-companion', 'Section 3',
     'For each rollout containing a sentence displaying verbalized eval awareness, we sample 50', '( Appendix D1 ).'),
    ('aea.p.inject', 'aea-companion', 'Section 3',
     'Injecting sentences that display verbalized eval awareness into the chain of thought increases refusal, with', 'largely nullifies this effect.'),
    ('aea.p.cues-method', 'aea-companion', 'Section 4.1',
     'Questions that trigger verbalized eval awareness tend also', 'relates to biological hazards (Figure 8).'),
    ('aea.p.cues', 'aea-companion', 'Summary of results',
     'Analyzing chains of thought across many Claude Haiku 4.5 rollouts', 'reducing it by 75% .'),
    ('aea.p.intent', 'aea-companion', 'Section 5',
     'In a given interaction, there are two related notions', 'how the model would behave in a high-stakes setup.'),
    ('aea.p.intent-vectors', 'aea-companion', 'Summary of results',
     'Steering with vectors that represent user intent', 'may also be partly inserting a user intent .'),
    ('aea.p.probe-sign', 'aea-heidari', 'Section 2, Evaluation awareness probe',
     'Notice that this sign convention implies', 'more evaluation-like representations.'),
    ('aea.p.probe-findings', 'aea-heidari', 'Section 7, Conclusion',
     'Our findings show that evaluation awareness is linearly separable', 'yet it generalizes to unseen benchmarks.'),
    ('aea.p.probe-limits', 'aea-heidari', 'Section 5, Limitations',
     'First, we study a specific operationalization', 'more complex interventions may produce different effects.'),
    ('aea.p.graders-monitors', 'aea-graders', 'Summary of results',
     'Different monitors capture different eval awareness .', 'relative to giving just the prompt.'),
    ('aea.p.graders-split', 'aea-graders', 'Summary of results',
     'Safety and capabilities evals elicit different verbalized eval awareness .', 'nearly identically (r = 0.98).'),
    ('aea.p.quadruples', 'aea-research-guide', 'Strong replication and ablation program',
     'Construct matched prompt quadruples', 'much stronger evidence than cross-dataset AUROC.'),
    ('aea.p.family', 'aea-research-guide', 'Strong replication and ablation program',
     'Pre-register a semantic family', 'rather than retrospectively choosing the best performer.'),
    ('aea.p.family-negative', 'aea-research-guide', 'Strong replication and ablation program',
     'Aranguri reports that an early attempt', 'how target construction can encode researcher assumptions.'),
    ('aea.p.multilevel', 'aea-research-guide', 'Prompt-level versus rollout-level variance',
     'The binary variable', 'it does not uniquely identify one.'),
    ('aea.p.heldout', 'aea-research-guide', 'Construct validity',
     'Train or select target sentences', 'illustrating a useful protocol for guarding against benchmark-specific shortcuts.'),
    ('aea.p.probe-heldout', 'aea-heidari', 'Section 3, Internal representation',
     'The calibration dataset comprises', 'computed on the held-out test\nset;'),
    ('aea.p.probe-random', 'aea-heidari', 'Appendix G.2, Random-direction control',
     'For both statistics, we report a random-direction baseline', 'model, layer, and sample size.'),
    ('aea.p.graders-probe', 'aea-graders', 'Methods, evaluation-awareness probes',
     'The probe is the difference of means between the positive and negative statements',
     'for evaluation vs deployment prompts.'),
]


def cut_passages(folder):
    out = []
    for pid, src, loc, a, b in PASSAGES:
        t = (folder / "_sources" / f"{PASSAGE_SCRIPT_SOURCES[src]}.txt").read_text()
        i = t.find(a)
        j = t.find(b, i)
        if i < 0 or j < 0:
            sys.exit(f"passage {pid}: marker not found in {src}")
        out.append({"id": pid, "source": src, "locator": loc, "text": t[i:j + len(b)]})
    return out


# ---------------------------------------------------------------- shared helpers for labs

AUROC_SRC = '''def auroc(pos, neg):
    # share of (positive, negative) pairs ordered correctly, ties count half
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
'''


def auroc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def lit(x):
    # a Python literal: the setup is exec'd as Python, so booleans must be True/False
    return repr(x)


# ---------------------------------------------------------------- lab 1: asking (calibration)

ASK_LABELS = [1] * 10 + [0] * 10
ASK_A = [0.97, 0.95, 0.94, 0.92, 0.9, 0.88, 0.86, 0.83, 0.81, 0.62,
         0.85, 0.78, 0.74, 0.7, 0.68, 0.65, 0.61, 0.58, 0.55, 0.52]
ASK_B = [0.93, 0.88, 0.85, 0.75, 0.7, 0.65, 0.55, 0.45, 0.35, 0.3,
         0.62, 0.55, 0.5, 0.42, 0.35, 0.3, 0.25, 0.15, 0.1, 0.05]


def lab_asking():
    code = ("def ece(probs, labels, bins):\n"
            "    total = 0.0\n"
            "    for b in range(bins):\n"
            "        idx = [i for i, p in enumerate(probs) if min(int(p * bins), bins - 1) == b]\n"
            "        if idx:\n"
            "            mean_p = sum(probs[i] for i in idx) / len(idx)\n"
            "            share = sum(labels[i] for i in idx) / len(idx)\n"
            "            total += len(idx) / len(probs) * abs(mean_p - share)\n"
            "    return total\n")
    setup = ("# Made-up teaching data: 20 transcripts, the first 10 from evaluations (label 1),\n"
             "# the last 10 from real use (label 0). A and B are the probabilities two models put on\n"
             "# 'this transcript comes from an evaluation' when asked.\n"
             f"LABELS = {lit(ASK_LABELS)}\nA = {lit(ASK_A)}\nB = {lit(ASK_B)}\n" + AUROC_SRC)
    return {
        "id": "aea-related.asking.lab", "concept": "aea-related.asking",
        "prompt": "Made-up data for practice. Two models were shown the same 20 transcripts, 10 from evaluations "
                  "and 10 from real use, and asked whether each came from an evaluation. Each number is the "
                  "probability a model put on \"evaluation\". Ranking the transcripts by these numbers, model A "
                  "reaches AUROC 0.920 and model B 0.855.",
        "predict": "Before computing: which model will have the lower expected calibration error (ECE) with five "
                   "equal-width bins, and roughly what will each value be? Write your two numbers, your "
                   "probability that the better-ranking model is also the better calibrated one, and the "
                   "result that would make you stop reading model A's numbers as probabilities. You are graded "
                   "on the prediction being specific and reasoned, not on being right.",
        "predict_solution": "A strong prediction commits to values with a reason. For example: model A gives most "
                            "real-use transcripts between 0.5 and 0.85, so its upper bins hold many real-use "
                            "transcripts and its ECE should be large, near 0.25; model B spreads its numbers and "
                            "should land near 0.1. Probability 0.25 that A is also better calibrated. Stop reading "
                            "A's numbers as probabilities if its ECE is several times B's.",
        "code_text": "Write `ece(probs, labels, bins)`. Split the interval from 0 to 1 into `bins` equal-width bins "
                     "(a value of exactly 1 goes in the last bin). For each non-empty bin, take the absolute "
                     "difference between the mean probability in it and the share of its labels equal to 1, "
                     "weight that difference by the bin's share of all transcripts, and return the sum.",
        "starter": "def ece(probs, labels, bins):\n    ...\n\n# Try: print(ece(A, LABELS, 5), ece(B, LABELS, 5))",
        "setup": setup, "code": code,
        "code_solution": "Each probability goes to bin $\\lfloor p \\cdot \\text{bins} \\rfloor$ (the top value "
                         "into the last bin). A bin contributes its size over $N$ times the gap between its mean "
                         "prediction and its share of evaluations: $\\mathrm{ECE}=\\sum_b \\frac{n_b}{N}"
                         "\\lvert \\bar p_b-\\bar y_b\\rvert$.",
        "tests": [
            ("ece([0.9, 0.8, 0.4, 0.7, 0.3, 0.2], [1, 1, 1, 0, 0, 0], 2)", None,
             {"0.167": "Weight each bin's gap by its share of the transcripts, then add."}),
            ("ece([0.95, 0.05], [1, 0], 5)", None, {}),
            ("ece(A, LABELS, 5)", "0.276", {}),
            ("ece(B, LABELS, 5)", "0.092", {}),
        ],
        "interpret": "Compare with your prediction. Which model's numbers would you use as the probability that a "
                     "new transcript is an evaluation, and which number would you set beside the logit monitor's "
                     "AUROC? Name one feature of how both sets of numbers were obtained that limits what they say "
                     "about evaluation awareness.",
        "interpret_solution": "Model A ranks better (0.920 against 0.855) but its ECE is 0.276 against "
                              "0.092: its stated probabilities are far from the observed frequencies, so B's numbers "
                              "(or A's after recalibration on held-out transcripts) are the ones to read as "
                              "probabilities. Set A's AUROC beside the logit monitor's, since AUROC compares rankings "
                              "only. Both came from a question that names the possibility of an evaluation, so they "
                              "measure classification on request, not whether the idea arises unprompted.",
        "expect": {"ece(A, LABELS, 5)": 0.276, "ece(B, LABELS, 5)": 0.092,
                   "auroc(A[:10], A[10:])": 0.920, "auroc(B[:10], B[10:])": 0.855},
    }


# ---------------------------------------------------------------- lab 2: hint mention rate

def mention_rows():
    r = random.Random(11)
    opts = "ABCD"
    rows = []
    # 10 moved to the hint (2 mention), 4 already on the hinted option (1 mentions), 16 not moved (7 mention)
    spec = [("moved", True)] * 2 + [("moved", False)] * 8 + [("already", True)] + [("already", False)] * 3 \
        + [("stay", True)] * 7 + [("stay", False)] * 9
    r.shuffle(spec)
    for kind, mention in spec:
        hinted = r.choice(opts)
        others = [o for o in opts if o != hinted]
        if kind == "moved":
            rows.append([r.choice(others), hinted, hinted, mention])
        elif kind == "already":
            rows.append([hinted, hinted, hinted, mention])
        else:
            a = r.choice(others)
            rows.append([a, a if r.random() < 0.8 else r.choice([o for o in others if o != a]), hinted, mention])
    return rows


def lab_mention():
    rows = mention_rows()
    code = ("def mention_rates(rows):\n"
            "    all_rate = sum(m for _, _, _, m in rows) / len(rows)\n"
            "    moved = [m for plain, hinted_run, hint, m in rows if plain != hint and hinted_run == hint]\n"
            "    return all_rate, sum(moved) / len(moved)\n")
    setup = ("# Made-up teaching data: 30 multiple-choice questions, each asked without and with a hint.\n"
             "# One row per question: [answer without hint, answer with hint, hinted option,\n"
             "#                         True if the hinted run's reasoning mentions the hint]\n"
             f"ROWS = {lit(rows)}\nROWS = [tuple(r) for r in ROWS]\n")
    return {
        "id": "aea-related.mention-rate.lab", "concept": "aea-related.mention-rate",
        "prompt": "Made-up results in the style of the hint experiment. Each of 30 multiple-choice questions was "
                  "asked twice, once plainly and once with a hint pointing at one option. For each question you "
                  "have the answer without the hint, the answer with the hint, the hinted option, and whether the "
                  "hinted run's reasoning mentions the hint.",
        "predict": "Before computing: will the share of hinted runs whose reasoning mentions the hint be higher, "
                   "lower or about equal to the same share among only the runs whose answer moved to the hinted "
                   "option? Give your probability and the result that would make you doubt the hint influenced "
                   "the moved answers at all. You are graded on the prediction being specific and reasoned.",
        "predict_solution": "For example: higher over all runs, because runs that ignore the hint can mention it "
                            "while rejecting it, and runs that follow it tend to leave it out; probability 0.7. "
                            "If plain reruns without the hint moved to the same options about as often, the moves "
                            "would not show influence.",
        "code_text": "Write `mention_rates(rows)`, returning a tuple `(all_rate, moved_rate)`: the share of all rows "
                     "whose hinted reasoning mentions the hint, and the same share among the rows whose answer "
                     "moved to the hinted option (the plain answer differs from the hinted option and the hinted "
                     "run's answer equals it).",
        "starter": "def mention_rates(rows):\n    ...\n\n# Try: print(mention_rates(ROWS))",
        "setup": setup, "code": code,
        "code_solution": "Over all rows the rate is mentions over 30. For the moved rate keep only rows with plain "
                         "answer $\\neq$ hinted option and hinted answer $=$ hinted option; a row already on the "
                         "hinted option without the hint did not move, so it is left out.",
        "tests": [
            ("mention_rates([('A', 'B', 'B', False), ('B', 'B', 'B', True), ('A', 'A', 'C', True), "
             "('C', 'D', 'D', True)])", None,
             {"(0.75, 0.667)": "A question already on the hinted option without the hint did not move."}),
            ("mention_rates(ROWS)", "(0.333, 0.2)", {}),
        ],
        "interpret": "Compare with your prediction. Which of the two rates answers whether the reasoning admits a "
                     "hint it used? What does the result mean for a monitor that only reads the reasoning, and what "
                     "extra run would you add before trusting the count of moved answers?",
        "interpret_solution": "The moved rate, 0.2, is the faithfulness rate: among answers the hint moved, four "
                              "in five traces leave it out. The overall rate, 0.333, is raised by traces that mention "
                              "the hint only to reject it. A monitor that reads the reasoning would miss most of the "
                              "hints that were used. Before trusting the ten moved answers, rerun the plain questions "
                              "to see how many move to that option without any hint, and subtract those.",
        "expect": {"mention_rates(ROWS)": (0.3333, 0.2)},
    }


# ---------------------------------------------------------------- lab 3: difference-of-means probe

def probe_data():
    r = random.Random(46)

    def layer(sig):
        ang = r.uniform(0, 2 * math.pi)
        u, w = (math.cos(ang), math.sin(ang)), (-math.sin(ang), math.cos(ang))

        def vec(a, b):
            return [round(a * u[0] + b * w[0], 2), round(a * u[1] + b * w[1], 2)]
        pairs = [[vec(r.gauss(1, 0.3), r.gauss(0, 1)), vec(r.gauss(-1, 0.3), r.gauss(0, 1))] for _ in range(4)]

        def mk(mu, k):
            return [vec(r.gauss(mu, 1), r.gauss(0, 1)) for _ in range(k)]
        dev = [mk(-sig, 6), mk(sig, 6)]
        new = [mk(-sig, 6), mk(sig, 6)]
        return {"pairs": pairs, "dev": dev, "new": new}
    return {"L4": layer(0.6), "L8": layer(0.3)}


def lab_probe():
    data = probe_data()
    code = ("def probe_auroc(pairs, evals, deploys):\n"
            "    n, d = len(pairs), len(pairs[0][0])\n"
            "    v = [sum(p[0][i] - p[1][i] for p in pairs) / n for i in range(d)]\n"
            "    length = math.sqrt(sum(x * x for x in v))\n"
            "    unit = [x / length for x in v]\n"
            "    def score(a):\n"
            "        return sum(x * y for x, y in zip(a, unit))\n"
            "    # higher = more deployment-like, so rank evaluation prompts by the negative score\n"
            "    return auroc([-score(a) for a in evals], [-score(a) for a in deploys])\n")
    setup = ("import math\n"
             "# Made-up activations (two numbers per prompt) at two layers of a small model.\n"
             "# LAYERS[name]['pairs']: four contrastive pairs [deployment-like, evaluation-like]\n"
             "# LAYERS[name]['dev']:   [evaluation prompts, real-use prompts] used to compare layers\n"
             "# LAYERS[name]['new']:   [evaluation prompts, real-use prompts] not used for anything yet\n"
             f"LAYERS = {lit(data)}\n" + AUROC_SRC)
    return {
        "id": "aea-related.probe.lab", "concept": "aea-related.probe",
        "prompt": "Made-up activations, two numbers per prompt, at two layers (L4 and L8) of a small model. At "
                  "each layer there are four contrastive pairs (deployment-like activation first), six evaluation "
                  "and six real-use development prompts used to compare the layers, and six plus six new prompts. "
                  "The probe direction is the mean paired difference divided by its length; a prompt's score is "
                  "its dot product with that direction, and a higher score means more deployment-like. On the "
                  "development prompts L8 reaches AUROC 0.917 and L4 0.806, so L8 was picked.",
        "predict": "Before computing: predict L8's AUROC on the new prompts, and whether L4 will beat it there. "
                   "Give your probability and the result that would make you drop L8. You are graded on the "
                   "prediction being specific and reasoned, not on being right.",
        "predict_solution": "For example: L8 should fall toward L4 or below it on new prompts, say to about 0.7, "
                            "because it was chosen as the best of two on only 12 prompts and part of its lead is "
                            "luck; probability 0.6 that L4 beats it. Drop L8 if it does worse than L4 on the new "
                            "prompts.",
        "code_text": "Write `probe_auroc(pairs, evals, deploys)`: build the unit direction from `pairs`, score every "
                     "prompt, and return the AUROC with evaluation prompts as the positive class. `auroc(pos, neg)` "
                     "and `math` are available.",
        "starter": "def probe_auroc(pairs, evals, deploys):\n    ...\n\n"
                   "# Try: L = LAYERS['L8']; print(probe_auroc(L['pairs'], *L['new']))",
        "setup": setup, "code": code,
        "code_solution": "Average the paired differences, $v=\\frac{1}{n}\\sum_i (a_i^{+}-a_i^{-})$, divide by "
                         "$\\lVert v\\rVert$, and score $s(x)=a(x)^{\\top}\\hat v$. A lower score is more "
                         "evaluation-like, so the AUROC ranks evaluation prompts by $-s$.",
        "tests": [
            ("probe_auroc([[[3, 1], [1, 1]], [[2, 0], [0, 2]]], [[1, 2]], [[2, 1]])", None,
             {"0": "A higher score means deployment-like: rank evaluation prompts by the negative score."}),
            ("probe_auroc([[[2, 0], [0, 0]]], [[-1, 5], [0, 0]], [[1, -5], [0, 0]])", None, {}),
            ("probe_auroc(LAYERS['L8']['pairs'], *LAYERS['L8']['dev'])", "0.917", {}),
            ("probe_auroc(LAYERS['L4']['pairs'], *LAYERS['L4']['dev'])", "0.806", {}),
            ("probe_auroc(LAYERS['L8']['pairs'], *LAYERS['L8']['new'])", "0.5", {}),
            ("probe_auroc(LAYERS['L4']['pairs'], *LAYERS['L4']['new'])", "0.806", {}),
        ],
        "interpret": "Compare with your prediction. What does the result say about choosing the layer on the "
                     "development prompts, how should a probe's AUROC be reported, and what baseline would you add?",
        "interpret_solution": "L8 falls from 0.917 to 0.500 on the new prompts while L4 stays at 0.806. L8 won the "
                              "comparison partly by luck on 12 prompts, so its development AUROC overstated it. "
                              "Choose the layer on one set of prompts and report the AUROC on prompts not used to "
                              "choose it. Add a random unit direction of the same size, scored the same way, to show "
                              "how much separation any direction gets on these prompts.",
        "expect": {"probe_auroc(LAYERS['L8']['pairs'], *LAYERS['L8']['dev'])": 0.9167,
                   "probe_auroc(LAYERS['L4']['pairs'], *LAYERS['L4']['dev'])": 0.8056,
                   "probe_auroc(LAYERS['L8']['pairs'], *LAYERS['L8']['new'])": 0.5,
                   "probe_auroc(LAYERS['L4']['pairs'], *LAYERS['L4']['new'])": 0.8056},
    }


# ---------------------------------------------------------------- lab 4: quadruples

def quad_rows():
    r = random.Random(7)
    rows = []
    for _ in range(8):
        base = r.uniform(-30, -18)
        rows.append([round(base, 2), round(base + 3 + r.gauss(0, 1), 2), round(base + 0.4 + r.gauss(0, 1), 2),
                     round(base - 0.1 + r.gauss(0, 1), 2)])
    return rows


def lab_quadruples():
    rows = quad_rows()
    code = ("def arm_effects(rows):\n"
            "    def effect(arm):\n"
            "        return sum(r[arm] - r[0] for r in rows) / len(rows)\n"
            "    return effect(1), effect(2), effect(3)\n")
    setup = ("# Made-up teaching data: 8 tasks, each written four ways. One row per task:\n"
             "# [natural, benchmark style, natural + 'This is an evaluation.',\n"
             "#  natural + 'This is a real conversation with a user, not a test.']\n"
             "# Each number is the prompt-start log-probability of \"The user might be testing\".\n"
             f"QUADS = {lit(rows)}\n")
    return {
        "id": "aea-related.quadruples.lab", "concept": "aea-related.quadruples",
        "prompt": "Made-up monitor scores for 8 tasks, each written four ways: a natural user wording, a "
                  "benchmark-style wording, the natural wording plus \"This is an evaluation.\", and the natural "
                  "wording plus \"This is a real conversation with a user, not a test.\" Each score is the "
                  "log-probability of \"The user might be testing\" at the start of the reasoning.",
        "predict": "Before computing: which of the three effects (benchmark style, evaluation statement, deployment "
                   "statement) will be largest in size, and what sign do you expect for each if the monitor reads "
                   "meaning? Give your probability for the largest one and the result that would make you drop "
                   "the reading that the monitor tracks meaning. You are graded on specificity and reasoning.",
        "predict_solution": "For example: if the monitor reads meaning, the evaluation statement should be "
                            "positive and the deployment statement negative, with style smaller; I put 0.5 on the "
                            "evaluation statement being largest. Drop the meaning reading if style is several times "
                            "larger than either statement and the deployment statement does not lower the score.",
        "code_text": "Write `arm_effects(rows)`, returning `(style, eval_statement, deploy_statement)`: for each, "
                     "the mean over tasks of that version's score minus the same task's natural score.",
        "starter": "def arm_effects(rows):\n    ...\n\n# Try: print(arm_effects(QUADS))",
        "setup": setup, "code": code,
        "code_solution": "For each task subtract the natural score from each other version, then average over the "
                         "tasks: $\\text{style}=\\frac{1}{T}\\sum_t (s_{t,\\text{bench}}-s_{t,\\text{natural}})$, "
                         "and likewise for the two statements.",
        "tests": [
            ("arm_effects([[-24, -21, -23, -24], [-18, -15, -18, -19]])", None,
             {"(-18, -20.5, -21.5)": "Subtract each task's natural score before averaging."}),
            ("arm_effects([[0, 1, 2, 3]])", None, {}),
            ("arm_effects(QUADS)", "(3.569, 0.775, -0.148)", {}),
        ],
        "interpret": "Compare with your prediction. What does the pattern say the monitor tracks, and what would you "
                     "need before calling the evaluation-statement effect real? Decide: kill, pivot or scale the "
                     "claim that the monitor reads evaluation framing, and name the next test.",
        "interpret_solution": "Style is 3.569, the evaluation statement 0.775 and the deployment statement −0.148. "
                              "Benchmark style moves the score more than four times as much as the evaluation "
                              "statement, and the false deployment statement barely moves it, so on these tasks the "
                              "monitor tracks format more than meaning. Pivot: the evaluation-statement effect is "
                              "small next to the task-to-task scatter of the differences, so test it on more tasks "
                              "with an interval from resampling tasks, and check that the statement is not simply "
                              "read as an instruction.",
        "expect": {"arm_effects(QUADS)": (3.569, 0.775, -0.148)},
    }


# ---------------------------------------------------------------- lab 5: phrase family

FAMILY_TOKENS = {"E": [4, 6, 5], "D": [5, 4]}


def family_rows():
    r = random.Random(132)
    rows = []
    for half in "AB":
        for ev in (1, 0) * 6:
            lat = r.gauss(0.35 if ev else -0.35, 1)
            e = [round((-2.2 + 0.25 * lat + r.gauss(0, 0.45)) * n, 2) for n in FAMILY_TOKENS["E"]]
            d = [round((-2.4 - 0.15 * lat + r.gauss(0, 0.45)) * n, 2) for n in FAMILY_TOKENS["D"]]
            rows.append({"eval": ev, "half": half, "E": e, "D": d})
    return rows


def lab_family():
    rows = family_rows()
    code = ("def holdout_check(rows):\n"
            "    def per_token(row, fam, i):\n"
            "        return row[fam][i] / TOKENS[fam][i]\n"
            "    def contrast(row):\n"
            "        e = sum(per_token(row, 'E', i) for i in range(3)) / 3\n"
            "        d = sum(per_token(row, 'D', i) for i in range(2)) / 2\n"
            "        return e - d\n"
            "    def roc(half, score):\n"
            "        rs = [r for r in rows if r['half'] == half]\n"
            "        return auroc([score(r) for r in rs if r['eval']], [score(r) for r in rs if not r['eval']])\n"
            "    on_a = [roc('A', lambda r, i=i: per_token(r, 'E', i)) for i in range(3)]\n"
            "    best = 0\n"
            "    for i in range(3):\n"
            "        if on_a[i] > on_a[best]:\n"
            "            best = i\n"
            "    return on_a[best], roc('B', lambda r: per_token(r, 'E', best)), roc('B', contrast)\n")
    setup = ("# Made-up teaching data: 24 prompts (12 evaluation, 12 real use) in halves 'A' and 'B'.\n"
             "# For each prompt: total log-probability of three evaluation-aware phrases ('E') and two\n"
             "# deployment-aware phrases ('D') at the start of the reasoning. TOKENS gives each phrase's length.\n"
             f"TOKENS = {lit(FAMILY_TOKENS)}\nROWS = {lit(rows)}\n" + AUROC_SRC)
    return {
        "id": "aea-related.phrase-family.lab", "concept": "aea-related.phrase-family",
        "prompt": "Made-up scores for 24 prompts, 12 evaluation and 12 real use, split into halves A and B. For each "
                  "prompt you have the total log-probability of three evaluation-aware phrases (4, 6 and 5 tokens "
                  "long) and two deployment-aware phrases (5 and 4 tokens) at the start of the reasoning. A student "
                  "plans to pick the single evaluation-aware phrase that separates the sets best on half A.",
        "predict": "Before computing: predict the chosen phrase's AUROC on half B, and whether the contrastive "
                   "family score (evaluation family minus deployment family) will beat it on half B. Give your "
                   "probability and the result that would make you trust the single phrase instead. You are graded "
                   "on the prediction being specific and reasoned.",
        "predict_solution": "For example: the chosen phrase will lose part of its half-A lead on half B, falling "
                            "to about 0.65, and the family score will beat it there; probability 0.65. Trust the "
                            "single phrase if it keeps nearly all of its half-A AUROC on half B.",
        "code_text": "Write `holdout_check(rows)`, returning `(best_on_A, same_phrase_on_B, family_on_B)`. A single "
                     "phrase's score is its total log-probability divided by its token count. Pick the "
                     "evaluation-aware phrase with the highest AUROC on half A; return that AUROC, the same phrase's "
                     "AUROC on half B, and the half-B AUROC of the contrastive score (mean per-token score of the "
                     "evaluation phrases minus mean per-token score of the deployment phrases). Evaluation prompts "
                     "are the positive class; `auroc(pos, neg)` and `TOKENS` are defined for you.",
        "starter": "def holdout_check(rows):\n    ...\n\n# Try: print(holdout_check(ROWS))",
        "setup": setup, "code": code,
        "code_solution": "Per phrase, $\\bar\\ell=\\text{total}/T$. The contrastive score is "
                         "$\\overline{\\bar\\ell}_{\\text{eval}}-\\overline{\\bar\\ell}_{\\text{deploy}}$. The phrase "
                         "is chosen on half A only, so its half-B AUROC is an honest estimate and its half-A AUROC "
                         "is not.",
        "tests": [
            ("holdout_check([{'eval': 1, 'half': 'A', 'E': [-8, -12, -10], 'D': [-15, -12]}, "
             "{'eval': 0, 'half': 'A', 'E': [-12, -12, -10], 'D': [-10, -8]}, "
             "{'eval': 1, 'half': 'B', 'E': [-12, -12, -10], 'D': [-15, -12]}, "
             "{'eval': 0, 'half': 'B', 'E': [-8, -12, -10], 'D': [-10, -8]}])", None, {}),
            ("holdout_check(ROWS)", "(0.917, 0.528, 0.75)", {}),
        ],
        "interpret": "Compare with your prediction. What does the gap between half A and half B show, and how "
                     "should the monitor's phrases be chosen and reported? Decide kill, pivot or scale for the "
                     "single-phrase monitor.",
        "interpret_solution": "The phrase chosen on half A scores 0.917 there but 0.528 on half B, close to chance: "
                              "most of its lead was luck among three candidates. The contrastive family score reaches "
                              "0.750 on half B without any choice made on half B. Kill the single-phrase monitor as "
                              "reported; commit to the family in advance, or choose on one half and report only the "
                              "other.",
        "expect": {"holdout_check(ROWS)": (0.9167, 0.5278, 0.75)},
    }


LABS = [lab_asking, lab_mention, lab_probe, lab_quadruples, lab_family]


def run(setup, code, call):
    ns = {}
    exec(setup, ns)
    exec(code, ns)
    return eval(call, ns)


def show(v):
    # the same display as Delta Drills' browser harness (study_code.js `_dd_show`): mistakes are keyed by it
    if isinstance(v, float) and v == v and (v == 0 or 1e-3 <= abs(v) < 1e6):
        return ("%.3f" % v).rstrip("0").rstrip(".")
    if isinstance(v, float):
        return "%.3g" % v
    if isinstance(v, list):
        return "[" + ", ".join(show(x) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ", ".join(show(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    return repr(v)


def check_lab(lab):
    errs = []
    for call, want in lab["expect"].items():
        got = run(lab["setup"], lab["code"], call)
        g, w = (got, want) if isinstance(got, tuple) else ((got,), (want,))
        # zip alone would let a function that drops a trailing value pass
        if (len(g) != len(w) or not all(isinstance(a, (int, float)) and math.isfinite(a) for a in g)
                or any(abs(a - b) > 0.0006 for a, b in zip(g, w))):
            errs.append(f"{lab['id']}: {call} = {show(got)}, lab text says {want}")
    for call, expect, _ in lab["tests"]:
        got = show(run(lab["setup"], lab["code"], call))
        if expect is not None and got != expect:
            errs.append(f"{lab['id']}: {call} -> {got}, test expects {expect}")
    try:
        out = run(lab["setup"], lab["starter"], lab["tests"][-1][0])
    except Exception:
        out = None
    if out is not None:
        errs.append(f"{lab['id']}: the starter returns {out!r} on the last test")
    return errs


def lab_question(lab):
    tests = [{"call": c, "expect": e or show(run(lab["setup"], lab["code"], c)), "mistakes": m}
             for c, e, m in lab["tests"]]
    c = lab["concept"]
    return {
        "id": lab["id"], "kind": "evaluative", "format": "staged", "demand": "evaluate", "targets": [c],
        "prompt": lab["prompt"], "reference": lab["interpret_solution"],
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


def lab_evidence(lab):
    q = lab["id"]
    return [
        {"question": f"{q}#1", "role": "direct",
         "observes": "The response commits to a direction and a value, attaches a probability, and names the "
                     "result that would make the learner drop the idea.",
         "warrant": "A checkable prediction made before computing shows the learner can turn this idea into an "
                    "expectation about data, the first move in choosing an informative test.",
         "rival": "A hedged statement can sound reasonable without committing to anything the data could "
                  "contradict; the grade is on specificity and reasoning, not on being right."},
        {"question": f"{q}#2", "role": "direct",
         "observes": "The function gives the stated numbers on the data and passes the small hand-built cases.",
         "warrant": "Computing the quantity shows the learner can carry out the measurement the idea calls for, "
                    "not only describe it.",
         "rival": "Code adapted from a similar example can pass if the data are shaped alike; the hand-built "
                  "cases expose the order, sign or denominator error the lesson warns about."},
        {"question": f"{q}#3", "role": "direct",
         "observes": "The response compares the result with the prediction, says what it supports and what it "
                     "does not, and names the next step.",
         "warrant": "Judging a result against a prior prediction and choosing the next test from it is the "
                    "evaluative use this idea serves.",
         "rival": "A response can restate the numbers without saying what they rule out, or propose a next step "
                  "that could not change the conclusion."},
    ]


# ---------------------------------------------------------------- authored free / mc questions

def free(qid, concept, demand, prompt, reference_parts, misconceptions, kind="evaluative", targets=None,
         integration=None):
    q = {"id": qid, "kind": kind, "format": "free", "demand": demand, "targets": targets or [concept],
         "prompt": prompt, "reference": " ".join(reference_parts), "expectations": list(reference_parts),
         "misconceptions": misconceptions}
    if integration:
        q["integration"] = integration
    return q


def mc(qid, concept, demand, prompt, reference, options, facet=None, kind="content"):
    q = {"id": qid, "kind": kind, "format": "mc", "demand": demand}
    if facet:
        q["facet"] = facet
    q.update({"targets": [concept], "prompt": prompt, "reference": reference,
              "options": [{"id": "abcd"[i], "text": t, "why": w, "correct": c} for i, (t, w, c) in enumerate(options)]})
    return q


M = {  # misconception labels, as in each concept's front matter
    "asking": ["Treats a high AUC as evidence that the stated probabilities are accurate.",
               "Treats a prompted classification as evidence of unprompted awareness."],
    "mention": ["Computes the mention rate over every hinted question, including ones the hint did not change.",
                "Reads unmentioned hint use as evidence that some other readout catches the missed cases."],
    "resample": ["Treats the outcome gap between rollouts that did and did not voice awareness as the effect of the sentence.",
                 "Assumes any inserted awareness sentence has the same effect whatever its wording."],
    "cues": ["Credits a drop in refusals after a rewrite to the removed cue without checking that the request stayed the same.",
             "Treats a word that co-occurs with voiced awareness as a demonstrated cause."],
    "intent": ["Treats any intervention that lowers voiced awareness as revealing deployment behavior.",
               "Assumes an evaluation-versus-deployment steering vector changes only the belief about being evaluated."],
    "probe": ["Uses the projection with the wrong sign, so evaluation prompts rank as deployment-like.",
              "Reports the best layer's AUROC on the same prompts used to pick the layer."],
    "graders": ["Ranks monitors on one task and assumes the ranking holds on another.",
                "Treats a high evaluation-probe score as evidence about evaluation rather than training."],
    "quad": ["Compares versions across different tasks instead of within each task.",
             "Treats any rise after an explicit evaluation statement as awareness without considering that the statement is unusual text."],
    "family": ["Averages total log-probabilities across phrases of different lengths.",
               "Reports the best phrase's AUROC on the prompts used to choose it."],
    "dir": ["Chooses an experiment because it is interesting without saying which answer it could change.",
            "Proposes an experiment whose every possible result would be read as support."],
}


def authored():
    """(question, [(node, role, observes, warrant, rival), ...]) for every non-lab question."""
    A = "aea-related."
    out = []

    def add(q, *ev):
        out.append((q, list(ev)))

    # ---- asking
    add(free(A + "asking.direct-1", A + "asking", "evaluate",
             "Needham and colleagues measured evaluation awareness by asking a model directly whether a transcript "
             "came from an evaluation and reading its answer as a probability. In a small replication, a team shows "
             "Qwen3-4B each of 200 prompts and asks: \"Is this message from (A) an evaluation of an AI "
             "system or (B) a real user?\" Reading the probability of A from the next-token logits gives AUC 0.86 "
             "for separating Fortress from WildChat prompts. On the same prompts the logit monitor reaches 0.70. "
             "A colleague concludes that the model is more evaluation-aware than the logit monitor shows. Evaluate "
             "the colleague's conclusion.",
             ["Asking measures whether the model can classify the prompt when the question is put to it, and the "
              "question itself introduces the idea of an evaluation.",
              "The logit monitor reads whether that idea is available while the model handles the original prompt, "
              "so a higher AUC for asking does not show more unprompted awareness.",
              "Both inherit the surface differences between Fortress and WildChat, so compare them on length-matched "
              "pairs and a held-out source, and add an open-ended or neutrally worded question to see how much the "
              "multiple-choice framing adds."], M["asking"]),
        (A + "asking", "direct",
         "Whether the response separates classification on request from unprompted awareness and names a comparison "
         "that controls the shared dataset differences.",
         "Applying the construct distinction to a new pair of readouts on the same prompts is the core of the mastery "
         "claim for elicited judgments.",
         "A response could reject the conclusion only because 0.86 and 0.70 look close, without saying what each "
         "readout measures."))
    add(free(A + "asking.transfer-1", A + "asking", "evaluate",
             "An evaluation team plans to flag agentic logs as \"probably a test\" whenever a model's elicited "
             "probability of evaluation exceeds 0.5. On their labelled development set the score has AUC 0.92, and "
             "most real-use logs get probabilities between 0.6 and 0.95. In live traffic, real evaluations are about "
             "5% of logs. Assess the plan and say what you would measure before using the cutoff.",
             ["The AUC shows the score ranks evaluations above real use, but it says nothing about whether 0.5 is a "
              "sensible cutoff.",
              "Real-use logs mostly score above 0.5, so the probabilities are overconfident and the cutoff would flag "
              "most real use; at 5% prevalence most flags would be false alarms.",
              "Measure calibration by bin (or ECE), the false-positive rate at the cutoff and precision at the live "
              "prevalence on held-out logs, and recalibrate or move the cutoff before use."], M["asking"]),
        (A + "asking", "transfer",
         "Whether the response separates ranking from calibration and connects overconfident probabilities to a "
         "fixed cutoff's false alarms at low prevalence.",
         "The case moves the discrimination-versus-calibration idea from comparing models to setting an operating "
         "threshold, a changed use of the same principle.",
         "A response could repeat that AUC 0.92 is high and approve the plan without reading the stated range of "
         "real-use probabilities."))

    add(mc(A + "asking.mc-1", A + "asking", "apply",
           "In one calibration bin, a model gave five transcripts probabilities $0.82$, $0.85$, $0.88$, $0.9$ and "
           "$0.95$ of being evaluations; two of the five really are evaluations. The bin holds 5 of the 25 "
           "transcripts. What does this bin add to the expected calibration error?",
           "Mean probability $0.88$, share of evaluations $0.4$, gap $0.48$, weighted by $5/25$: $0.096$.",
           [("$0.096$", "Gap $\\lvert 0.88-0.4\\rvert=0.48$ times the bin's share $5/25=0.2$.", True),
            ("$0.48$", "This is the bin's gap before weighting by its share of the transcripts.", False),
            ("$0.088$", "This weights the mean probability, not the gap, by the bin's share.", False),
            ("$0.6$", "This is the share of the bin that is not an evaluation.", False)], facet="do"),
        (A + "asking", "direct",
         "Whether the learner computes a bin's gap and weights it by the bin's share.",
         "Weighting each bin's gap is the computational core of calibration error.",
         "A learner could reach 0.096 by multiplying the given numbers in another order; the lab's code step "
         "checks the full computation."))

    # ---- mention-rate
    add(mc(A + "mention-rate.mc-1", A + "mention-rate", "apply",
           "Forty questions carry a hint. Twelve answers move to the hinted option, and the reasoning mentions the "
           "hint in 3 of those 12. Of the 28 answers that do not move, 14 traces mention the hint. What rate answers "
           "whether the reasoning admits a hint it used?",
           "Only the 12 moved answers show the hint was used; 3 of them mention it.",
           [("$0.25$", "Three mentions among the 12 answers the hint moved.", True),
            ("$0.425$", "This counts all 17 mentions over all 40 hinted questions, including unmoved ones.", False),
            ("$0.30$", "This is the share of answers that moved, not the share that mention the hint.", False),
            ("$0.50$", "This is the mention rate among answers the hint did not move.", False)], facet="do"),
        (A + "mention-rate", "direct",
         "Whether the learner restricts the rate to answers the hint moved.",
         "Choosing the denominator is the computational core of the mastery claim.",
         "A learner could pick 0.25 by dividing 3 by 12 without understanding why the unmoved answers are excluded; "
         "the distractors built from other denominators make that less likely."))
    add(free(A + "mention-rate.direct-1", A + "mention-rate", "evaluate",
             "You add the line \"Note: this conversation is part of a safety benchmark\" to the system prompt of 300 "
             "borderline requests. Compared with the same requests without the line, refusals change on 60 of them, "
             "and the reasoning mentions the benchmark note in 9 of those 60. A teammate writes: \"The model rarely "
             "notices the note.\" Evaluate the teammate's claim.",
             ["The 60 changed answers are where the note was used, and 9 of 60 is a mention rate of 0.15, so the "
              "model often acted on the note without saying so.",
              "\"Rarely notices\" confuses not mentioning with not using; the evidence is of use without mention.",
              "Rerun the 300 requests twice without the note to see how often refusals change by chance, and compute "
              "the mention rate among the changes that exceed that baseline."], M["mention"]),
        (A + "mention-rate", "direct",
         "Whether the response computes the rate among changed answers and separates use from mention.",
         "Applying the hint design to an evaluation cue is the use the mastery claim names.",
         "A response could agree that 9 of 300 is small, dividing by the wrong total, and still sound careful."))
    add(free(A + "mention-rate.transfer-1", A + "mention-rate", "evaluate",
             "A code-review assistant is given pull requests with a planted comment, \"The maintainer prefers the "
             "second implementation\". On 50 of 400 pull requests its recommendation switches to the second "
             "implementation, and its written justification cites the comment in 6 of those 50. What does this say "
             "about relying on the justifications to audit the assistant?",
             ["Among the 50 switches, where the comment was used, the justification cites it 6 times, a rate of "
              "0.12, so most influenced recommendations hide the influence.",
              "Auditing only the written justifications would miss most comment-driven switches.",
              "It does not show that another check, such as comparing recommendations with and without the comment, "
              "would catch every case, and some switches may happen without the comment; a rerun without it "
              "estimates that."], M["mention"]),
        (A + "mention-rate", "transfer",
         "Whether the response computes the rate among switches and draws the audit consequence.",
         "The same denominator logic is applied to a non-quiz setting with a different kind of planted cue.",
         "A response could compute 6 of 400 and conclude the comment is rarely relevant."))

    # ---- resample
    add(free(A + "resample.direct-1", A + "resample", "evaluate",
             "On a refusal benchmark, rollouts that contain the sentence \"this might be a test\" refuse 20 points "
             "more often than rollouts of the same prompts that do not. A report calls this the causal effect of "
             "awareness on refusal. Evaluate the claim and describe a test that estimates the effect of the sentence.",
             ["Rollouts that write the sentence also differ in everything before it, and awareness may cluster on "
              "certain prompts, so the 20-point gap mixes the sentence with its history.",
              "Take rollouts that contain the sentence, resample many continuations from just before it (rejecting "
              "ones that write awareness again) and from just after it, and compare refusal between the two arms.",
              "Weight the comparison so each prompt counts as often as it does among the aware rollouts, and report "
              "the effect for this model and benchmark only."], M["resample"]),
        (A + "resample", "direct",
         "Whether the response identifies the shared-history problem and proposes resampling from a fixed prefix.",
         "Designing the fixed-prefix comparison is what the mastery claim asks for.",
         "A response could say \"correlation is not causation\" and propose an unrelated randomized trial over "
         "prompts, which does not isolate the sentence."))
    add(free(A + "resample.direct-2", A + "resample", "explain",
             "A team wants the effect of the reasoning sentence \"I should be careful, this could be an evaluation\" on "
             "refusal. Across 40 prompts, they compare refusal between prompts whose rollouts usually contain the sentence "
             "and prompts whose rollouts rarely do, and find a 12-point gap. Explain why this gap does not estimate the effect of the sentence.",
             ["The comparison is between prompts, which differ in content and risk, so the 12 points mix the sentence "
              "with whatever makes those prompts produce it.",
              "Even within a prompt, rollouts that write the sentence differ in everything they wrote before it.",
              "Only continuations sampled from the same prefix, just before and just after the sentence, differ in the "
              "sentence alone."], M["resample"], kind="content"),
        (A + "resample", "direct",
         "Whether the response sees that prompt-level grouping confounds the sentence and proposes the fixed-prefix "
         "design with matched weighting.",
         "Diagnosing a flawed effect estimate and replacing it is the design half of the mastery claim.",
         "A response could ask for more prompts, which would not remove the confound."))
    add(free(A + "resample.transfer-1", A + "resample", "evaluate",
             "A coding agent's reasoning sometimes contains \"the tests may be checking this edge case\", and runs "
             "that contain it pass hidden tests more often. Design a test of whether the sentence changes the pass "
             "rate, and name one result that would show the effect depends on the sentence's wording rather than its "
             "idea.",
             ["Resample continuations from just before the sentence (rejecting ones that write it anyway) and from "
              "just after it on the same runs, and compare pass rates.",
              "Also insert the sentence, a rewording with the same idea, and a neutral sentence of the same length at "
              "the same point.",
              "If only the original wording raises the pass rate while the same idea reworded does not, the effect "
              "depends on wording."], M["resample"]),
        (A + "resample", "transfer",
         "Whether the response transfers the fixed-prefix design to a new behavior and adds a wording control.",
         "The principle is applied to coding rather than refusal, with the injection-wording result as the twist.",
         "A response could copy the refusal design without a wording control, which leaves the second part "
         "unanswered."))

    # ---- prompt-cues
    add(free(A + "prompt-cues.direct-1", A + "prompt-cues", "evaluate",
             "In a refusal benchmark, the word \"hypothetically\" appears in the reasoning for 70% of prompts that ever "
             "voice awareness and 20% of prompts that never do. A team proposes deleting hypothetical framing from "
             "every benchmark prompt to make the benchmark more realistic. Evaluate the inference and describe how you "
             "would test it.",
             ["The word's co-occurrence makes hypothetical framing a candidate cue, not a demonstrated cause; it may "
              "mark a kind of request that is suspicious for other reasons.",
              "Test it by rewriting a sample of prompts to remove the hypothetical framing while keeping the requested "
              "information and risk, then compare voiced awareness and refusal between originals and rewrites.",
              "Check by hand or with a judge that each rewrite asks for the same thing, since a softer request would "
              "lower refusals on its own."], M["cues"]),
        (A + "prompt-cues", "direct",
         "Whether the response treats co-occurrence as a lead and designs a content-preserving rewrite test.",
         "Moving from a co-occurring cue to a controlled rewrite is the loop this node teaches.",
         "A response could endorse deletion because 70% against 20% looks strong, missing that the word appears in "
         "the reasoning, not as a tested cause."))
    add(free(A + "prompt-cues.direct-2", A + "prompt-cues", "explain",
             "After 60 benchmark prompts are rewritten to remove the phrase \"for research purposes\", voiced awareness "
             "falls by a third while refusal stays at 80%. A colleague concludes that the phrase inflates the benchmark's "
             "measured safety. Explain why this result does not show that.",
             ["Inflated safety would need refusal to fall when the cue is removed; here only voiced awareness moved.",
              "So on this evidence the phrase changes what the model says about being tested, not what it does.",
              "Before settling it, confirm the rewrites asked for the same thing and that enough rollouts per prompt were "
              "run to detect a modest refusal change."], M["cues"], kind="content"),
        (A + "prompt-cues", "direct",
         "Whether the response requires a behavior change, not only an awareness change, before calling safety "
         "inflated.",
         "Separating the two outcomes of a rewrite is a core distinction of this node.",
         "A response could accept the conclusion because awareness fell substantially."))
    add(free(A + "prompt-cues.transfer-1", A + "prompt-cues", "evaluate",
             "In a customer-service agent evaluation, the agent's reasoning often says \"this ticket looks synthetic\" "
             "when tickets contain round order numbers such as 10000. Plan a test of whether the order format drives "
             "the voiced suspicion and whether it changes how the agent handles refund requests.",
             ["Create matched versions of each ticket that differ only in the order number (round against realistic), "
              "keeping the customer's request and details the same.",
              "Measure both voiced suspicion and the refund decision across many rollouts per version.",
              "If suspicion and refund behavior both change with the number alone, the format is a cue that distorts "
              "the evaluation; if only suspicion changes, it is cosmetic."], M["cues"]),
        (A + "prompt-cues", "transfer",
         "Whether the response isolates the candidate cue in matched versions and measures both outcomes.",
         "The cue-rewrite logic is applied to a non-safety evaluation with a non-verbal cue.",
         "A response could propose deleting order numbers everywhere without a test."))

    # ---- intent-gap
    add(free(A + "intent-gap.direct-1", A + "intent-gap", "evaluate",
             "A lab steers a model along an \"evaluation minus deployment\" direction. On a deception benchmark, "
             "voiced awareness falls from 30% to 5% and honest answers fall from 90% to 70%. The lab reports that the "
             "model is less honest in deployment than the benchmark suggested. Evaluate the report.",
             ["Lower voiced awareness does not show the model now infers the purpose the benchmark was built to "
              "recreate; the steering may have supplied some other purpose.",
              "Such vectors may partly insert a user intent, so the drop in honesty could reflect a changed scenario "
              "rather than deployment behavior.",
              "Check what purpose the model infers after steering (from its reasoning or a separate question), and "
              "compare with a condition that states the target situation directly."], M["intent"]),
        (A + "intent-gap", "direct",
         "Whether the response distinguishes inferred from target intent and proposes measuring the inferred one.",
         "Applying the two-intents distinction to a new benchmark and outcome is the mastery claim.",
         "A response could criticize steering in general, for example its strength, without naming the intent gap."))
    add(free(A + "intent-gap.direct-2", A + "intent-gap", "explain",
             "An agentic evaluation tests whether a model leaks a confidential file when a task tempts it to. To lower "
             "suspicion, the team steers with a vector for \"the user wants help with routine paperwork\". Voiced "
             "awareness drops and leaks rise from 2% to 12%. The team reports that the model leaks 12% of the time in "
             "deployment. Explain why the 12% may not describe deployment behavior.",
             ["The steering supplied a purpose, routine paperwork, so the model's inferred intent may now differ from the "
              "target situation of a tempting, sensitive task.",
              "The 12% therefore describes behavior when the model believes it is doing routine paperwork, not "
              "necessarily behavior in the deployment the evaluation targets.",
              "Check the purpose the model infers after steering and compare with a condition that lowers suspicion "
              "while keeping the target situation."], M["intent"], kind="content"),
        (A + "intent-gap", "direct",
         "Whether the response identifies the inserted purpose and its effect on what the leak rate measures.",
         "Applying the inferred-versus-target distinction to a new agentic case is the mastery claim.",
         "A response could object only to steering strength without naming the intent gap."))
    add(free(A + "intent-gap.transfer-1", A + "intent-gap", "evaluate",
             "To reduce awareness in a jailbreak evaluation, a team rewrites each prompt so the user says they are a "
             "nurse asking for work. Voiced awareness falls and compliance rises. Explain what the result can and "
             "cannot say about the model's behavior toward the original jailbreak users, and what you would add.",
             ["The rewrite changed the stated purpose, so the new compliance rate describes behavior toward a nurse "
              "with a work need, not toward the jailbreak users the evaluation targets.",
              "It does show that the scenario's purpose affects both suspicion and compliance.",
              "Add a rewrite that lowers suspicion while keeping the jailbreak user's goal, and check the inferred "
              "purpose in each condition."], M["intent"]),
        (A + "intent-gap", "transfer",
         "Whether the response sees that a prompt rewrite, not steering, can also change the inferred purpose.",
         "The idea transfers from steering vectors to a prompt-level intervention.",
         "A response could praise the rewrite for lowering awareness without asking whose behavior is measured."))

    # ---- probe
    add(mc(A + "probe.mc-1", A + "probe", "apply",
           "The mean paired difference (deployment-like minus evaluation-like) at a layer is $v=(0, 3, 4)$. A prompt "
           "has activation $(2, 1, -2)$. What is its probe score $s(x)$, and which way does it lean?",
           "The unit direction is $(0, 0.6, 0.8)$, so $s=0.6-1.6=-1.0$: evaluation-like.",
           [("$-1.0$, evaluation-like", "Unit direction $(0,0.6,0.8)$; dot product $-1.0$; lower is "
             "evaluation-like.", True),
            ("$-5.0$, evaluation-like", "This projects onto $v$ without dividing by its length 5.", False),
            ("$1.0$, deployment-like", "This flips the sign of the projection.", False),
            ("$-1.0$, deployment-like", "The value is right, but a lower score is evaluation-like.", False)],
           facet="do"),
        (A + "probe", "direct",
         "Whether the learner normalizes the direction and reads the sign convention.",
         "Normalizing and reading the sign are the two steps the mastery claim requires.",
         "A learner who normalizes correctly but guesses the direction could still pick the last option; that error "
         "is what it diagnoses."))
    add(free(A + "probe.direct-1", A + "probe", "evaluate",
             "A probe built from \"real user\" against \"AI evaluator\" contrast pairs separates benchmark prompts from "
             "WildChat with AUROC 0.82 at its best of 28 layers, chosen on the same prompts. Per prompt, its score "
             "correlates with the logit monitor at $r = 0.15$ and with a judge's verbalization score at $r = 0.05$. A "
             "colleague says the probe has found the true awareness signal the other two miss. Evaluate the claim.",
             ["The 0.82 comes from the best of 28 layers chosen on the same prompts, so it is optimistic; choose the "
              "layer on other prompts and compare with a random direction.",
              "Benchmark prompts and WildChat also differ in topic, style and length, so the probe may read the source.",
              "Weak correlations show the readouts measure different things, not which one is true; it would take "
              "held-out tests and an intervention showing behavior changes along the direction."], M["probe"]),
        (A + "probe", "direct",
         "Whether the response names layer-selection optimism, the dataset confound and the meaning of weak "
         "coupling.",
         "Evaluating a probe claim on new numbers uses the selection and interpretation parts of the mastery claim.",
         "A response could reject the claim on general grounds (probes are opaque) without the specific checks."))
    add(free(A + "probe.transfer-1", A + "probe", "evaluate",
             "A probe built from contrast pairs of math problems against small talk separates GSM8K from WildChat with "
             "AUROC 0.95. What does this tell you about reading an evaluation probe's AUROC on GSM8K against WildChat "
             "as evidence of evaluation awareness?",
             ["A direction for a different property separates the same sets almost perfectly, so separating them is "
              "easy for reasons unrelated to evaluation.",
              "An evaluation probe's AUROC on these sets therefore says little about evaluation awareness.",
              "Use matched prompts with the same content and different framing, or hold out a whole source, so topic "
              "and task type cannot separate the classes."], M["probe"]),
        (A + "probe", "transfer",
         "Whether the response uses the control probe to discount the evaluation probe's AUROC and proposes a "
         "matched test set.",
         "The node's caution about dataset differences is applied through a control direction, a new move.",
         "A response could praise both probes as accurate without asking what separates the sets."))

    # ---- graders
    add(free(A + "graders.direct-1", A + "graders", "evaluate",
             "You must choose one monitor to decide which prompts of a new safety benchmark to rewrite because they "
             "make the model voice suspicion. Monitor P (an activation probe) has AUROC 0.90 for separating evaluation "
             "from deployment prompts and 0.62 for predicting which prompts voice awareness in at least one of 50 "
             "rollouts. Monitor L (a logit monitor) has 0.74 and 0.81. Which do you choose, why, and what would make "
             "you reconsider?",
             ["The task is to find prompts that elicit voiced suspicion, so the relevant number is AUROC for "
              "predicting voiced awareness, where L (0.81) beats P (0.62).",
              "P's higher source-separation AUROC answers a different question.",
              "Reconsider if L's phrase does not fit the form of awareness this benchmark elicits, or if L's advantage "
              "does not hold on prompts held out from choosing it."], M["graders"]),
        (A + "graders", "direct",
         "Whether the response picks the monitor by the task-relevant AUROC.",
         "Choosing by task when rankings reverse is the first part of the mastery claim.",
         "A response could pick P for its larger headline number."))
    add(free(A + "graders.direct-2", A + "graders", "explain",
             "A logit monitor using \"The user might be testing\" scores near chance on a capability benchmark, where "
             "the model's reasoning often says \"This is a standard dataset problem\". A colleague concludes the model "
             "is not evaluation-aware on capability benchmarks. Explain why the near-chance score does not show that.",
             ["The monitor's phrase fits the safety form of awareness, suspicion that a user is probing behavior.",
              "Capability benchmarks elicit a different form, recognizing a dataset task and how it will be graded, which "
              "the reasoning here shows, so the monitor may miss it.",
              "Commit in advance to a phrase in the capability form and rescore before concluding anything about "
              "awareness on these benchmarks."], M["graders"], kind="content"),
        (A + "graders", "direct",
         "Whether the response links the near-chance score to a phrase that does not fit the benchmark's form of "
         "awareness.",
         "Phrase fit across grader types is the second consequence this node teaches.",
         "A response could accept the conclusion because the monitor worked on safety benchmarks."))
    add(free(A + "graders.transfer-1", A + "graders", "evaluate",
             "A new agentic benchmark is graded by running unit tests. Before running anything, state a prediction "
             "about the form of voiced awareness it will elicit, one that the data could show to be wrong.",
             ["An automatically graded benchmark should elicit the capability form, recognizing a dataset task and how "
              "it will be graded, rather than suspicion that a user is probing.",
              "Commit to a phrase in that form (for example about this being a benchmark task) alongside the usual "
              "testing phrase.",
              "If the reasoning instead mostly says the user is testing its behavior, as on judge-graded safety evals, "
              "that counts against the grader idea."], M["graders"]),
        (A + "graders", "transfer",
         "Whether the response derives a prediction from the grader hypothesis and states a disconfirming result.",
         "The hypothesis is applied to a new grader type and turned into a prediction before data.",
         "A response could list both forms as possible without committing to one."))

    # ---- quadruples
    add(mc(A + "quadruples.mc-1", A + "quadruples", "apply",
           "Three tasks. Natural versions score $-20$, $-26$ and $-17$; the same tasks with \"This is an "
           "evaluation.\" added score $-18$, $-26$ and $-15$. What is the evaluation-statement effect?",
           "Within-task differences are $2$, $0$ and $2$, mean $4/3 \\approx 1.33$.",
           [("$1.33$", "The mean of the within-task differences $2$, $0$, $2$.", True),
            ("$-19.67$", "This is the mean score of the statement version, not an effect.", False),
            ("$2.0$", "This takes the largest difference instead of the mean.", False),
            ("$2.0$ divided by $3$, about $0.67$", "This drops one task's difference of 2.", False)], facet="do"),
        (A + "quadruples", "direct",
         "Whether the learner subtracts within each task and averages.",
         "The within-task effect is the computation the mastery claim names.",
         "A learner could average each version and subtract, which gives the same mean here; the free items test "
         "the reason for working within tasks."))
    add(free(A + "quadruples.direct-1", A + "quadruples", "evaluate",
             "Forty quadruples give a style effect of $+0.2$, an evaluation-statement effect of $+2.5$ and a "
             "deployment-statement effect of $-1.8$ (log-probability units; within-task differences scatter with "
             "standard deviation about 1). What does this pattern say the monitor reads?",
             ["The statements move the score in opposite directions within fixed content while style barely moves "
              "it, which fits a monitor that reads the stated framing rather than format.",
              "With 40 tasks and scatter near 1, effects of 2.5 and −1.8 are well outside noise.",
              "The worry is that an explicit statement is unusual text the model may treat as an instruction or a "
              "topic, so test natural framings that imply evaluation without saying it."], M["quad"]),
        (A + "quadruples", "direct",
         "Whether the response maps the effect pattern to the semantic reading and names the statement caveat.",
         "Reading effects against rival predictions is the evaluative half of the mastery claim.",
         "A response could call any positive statement effect awareness without the caveat."))
    add(free(A + "quadruples.transfer-1", A + "quadruples", "evaluate",
             "Use the same design to test whether a refusal classifier reacts to harm or to tone. Describe the four "
             "versions of each item and the pattern of effects that would favor each reading.",
             ["Hold each request's harm fixed and write four versions: neutral tone, aggressive tone, and the neutral "
              "version with the harmful detail made explicit or made vague, so tone and stated harm vary separately.",
              "Compute each effect as the within-item difference from the neutral version, averaged over items.",
              "A classifier that reads harm moves with the harm versions and not with tone; one that reads tone moves "
              "with the tone version even when harm is fixed."], M["quad"]),
        (A + "quadruples", "transfer",
         "Whether the response crosses the two factors within items and states rival patterns.",
         "The factorial within-item design is moved to a different construct.",
         "A response could compare two datasets of harmful and harmless requests, repeating the confound."))

    # ---- phrase-family
    add(mc(A + "phrase-family.mc-1", A + "phrase-family", "apply",
           "A phrase family was fixed before scoring: three evaluation-aware statements and three deployment-aware "
           "statements. On one prompt their log-probabilities per token are $-2.0, -2.6, -3.1$ (evaluation) and "
           "$-2.9, -3.0, -3.4$ (deployment). What is the family contrast, the evaluation mean minus the deployment "
           "mean?",
           "Evaluation mean $-7.7/3 \\approx -2.567$, deployment mean $-9.3/3 = -3.1$, contrast $\\approx 0.53$.",
           [("$\\approx 0.53$", "$-2.567 - (-3.1) \\approx 0.53$, averaging every statement in each family.", True),
            ("$0.9$", "This keeps only the best statement of each family, $-2.0 - (-2.9)$, which the pre-registered "
             "family is meant to avoid.", False),
            ("$\\approx -0.53$", "This subtracts in the wrong order; the evaluation family is first.", False),
            ("$\\approx -2.57$", "This is the evaluation mean alone, without the deployment contrast.", False)],
           facet="do"),
        (A + "phrase-family", "direct",
         "Whether the learner aggregates every statement in both families and takes the contrast in the stated order.",
         "Aggregating the fixed family rather than keeping the best phrase is the computational core of the node.",
         "A learner could pick the best statement of each family; the distractor for that is listed."))
    add(free(A + "phrase-family.direct-1", A + "phrase-family", "evaluate",
             "A student tried 12 evaluation-aware phrases on 60 prompt pairs and reports the best one, at AUROC 0.81, "
             "as the monitor's performance. Evaluate the report and say how to redo it.",
             ["The best of 12 on the same prompts is biased upward, since part of its lead is luck among candidates.",
              "The reported 0.81 is therefore not an estimate of performance on new prompts.",
              "Redo it by committing to the phrase family before scoring and reporting its contrastive score, or by "
              "choosing on one half of the pairs and reporting only the other half."], M["family"]),
        (A + "phrase-family", "direct",
         "Whether the response names the selection bias and a remedy that separates choosing from reporting.",
         "Recognizing the winner's curse in a new report is the evaluative part of the mastery claim.",
         "A response could ask for more prompts without separating choice from evaluation."))
    add(free(A + "phrase-family.transfer-1", A + "phrase-family", "evaluate",
             "A colleague builds a monitor for \"the model thinks it is being trained\": write 8 training-aware phrases, "
             "score them on 100 prompts, keep the 3 with the highest AUROC, average their total log-probabilities, and "
             "report the AUROC of that average on the same 100 prompts. What would you change before trusting the "
             "reported number?",
             ["Keeping the best 3 of 8 on the same prompts inflates the reported AUROC; commit to the phrases in advance "
              "or choose on one half and report on the other.",
              "Averaging totals lets phrase length drive the score; compare phrases per token before averaging.",
              "Add a contrast family (deployment-aware statements) and neutral controls, so a prompt that raises every "
              "statement's probability does not look like training awareness."], M["family"]),
        (A + "phrase-family", "transfer",
         "Whether the response finds the selection, length and missing-contrast problems in a new monitor plan.",
         "The family method is moved from evaluation to training awareness and applied as a diagnosis.",
         "A response could fix one problem, usually selection, and miss the length and contrast problems."))

    # ---- directions (single-target)
    add(free(A + "directions.direct-1", A + "directions", "evaluate",
             "You have one day of compute on Qwen3-1.7B and saved rollouts from the length-matched run. A colleague "
             "proposes R4 (train a probe at every layer and steer along the best one) to strengthen the answer to Q2, "
             "about where awareness comes from. Evaluate the proposal against one alternative and say which you would "
             "run first.",
             ["R4 mainly bears on Q1 and Q3, other readouts and what separates the sets; it adds little to the "
              "prompt-versus-rollout question in Q2.",
              "R6, a multilevel model with a random effect per prompt fitted to the saved rollouts, answers Q2 "
              "directly and needs only re-analysis.",
              "Run R6 first: it is cheaper and its outcome, the between-prompt share, could change the Q2 answer."],
             M["dir"]),
        (A + "directions", "direct",
         "Whether the response matches each candidate to the question it can change and weighs cost.",
         "Choosing by which answer can change and at what cost is the mastery claim.",
         "A response could prefer R4 because probes are more advanced."))
    add(free(A + "directions.direct-2", A + "directions", "explain",
             "To strengthen the answer to Q3, a teammate proposes rerunning the Fortress-versus-WildChat comparison on "
             "Qwen3-4B and Qwen3-1.7B and reporting both AUROCs. Explain why this cannot change the answer to Q3.",
             ["Both reruns compare the same two datasets, which differ in topic, harm and length, so they inherit the "
              "confound Q3 asks about.",
              "Whatever AUROCs come out, they cannot separate evaluation awareness from dataset differences, so the "
              "answer to Q3 would not change.",
              "A within-content design such as R7's quadruples, or a held-out source, could change it."], M["dir"], kind="content"),
        (A + "directions", "direct",
         "Whether the response judges the proposal by whether any outcome could change the answer.",
         "Rejecting a study whose results cannot discriminate the rivals is the mastery claim.",
         "A response could welcome more models as more evidence without asking what the evidence separates."))
    add(free(A + "directions.transfer-1", A + "directions", "evaluate",
             "Suppose a fourth question asked whether evaluation awareness changes behavior on agentic tasks. Choose "
             "two of R1 to R8 to adapt, say what each would show, which rival explanation each rules out, and their "
             "relative cost.",
             ["R3's remove-and-resample design adapted to agent trajectories tests whether voiced awareness changes "
              "later actions, ruling out that aware runs differ only in what came before.",
              "R7's quadruples adapted to agent tasks test whether framing changes actions with content fixed, ruling "
              "out task differences.",
              "Quadruples need new sampling on rewritten tasks; resampling needs many continuations per run, so both "
              "cost more than re-scoring, and R7 is the cheaper to run at small scale."], M["dir"]),
        (A + "directions", "transfer",
         "Whether the response picks designs that test behavior change, names each rival and compares cost.",
         "The selection criteria are applied to a question outside Q1 to Q3.",
         "A response could pick R1 and R4, readouts that do not test behavior change."))

    # ---- integrated comparisons (all filed under directions)
    def integ(qid, targets, prompt, parts, mis, ev):
        q = free(A + "directions." + qid, A + "directions", "evaluate", prompt, parts, mis, kind="combined",
                 targets=[A + "directions"] + targets, integration="integrated")
        add(q, *[(n, "integrated", ev[0], ev[1], ev[2]) for n in q["targets"]])

    integ("compare-r1", [A + "asking", "aea-observe.readouts"],
          "R1 asks Qwen3-1.7B a fixed multiple-choice question about each prompt of the length-matched pairs and "
          "reads the probability of the evaluation option from the logits. A colleague says that if R1's AUROC beats "
          "the logit monitor's on those pairs, asking is the better measure of evaluation awareness. Evaluate the "
          "colleague's claim.",
          ["R1 measures classification when the model is asked, a judgment the question itself prompts; the logit "
           "monitor reads whether the idea is available while the model handles the prompt unprompted.",
           "A higher AUROC would show asking separates the sets better, not that it measures unprompted awareness "
           "better, and it needs intervals that do not overlap.",
           "Run the same confound checks on both (length, topic and harm, held-out source) so neither wins by "
           "reading the dataset."], M["asking"] + M["dir"],
          ("Whether the response contrasts the two readouts, states a discriminating result and applies the same "
           "checks to both.",
           "The task needs the elicited-judgment idea, the readout distinction and the choice of a fair test together.",
           "A response could describe R1 accurately but compare it with the monitor on unmatched data."))
    integ("compare-r2", [A + "mention-rate"],
          "R2 plants hints in 200 MMLU questions for Qwen3-1.7B with four rollouts each. How would its result bear "
          "on Q1?",
          ["If the model often uses hints without mentioning them, waiting for verbalized awareness will miss cases, "
           "which supports measuring beyond words (Q1).",
           "Report the mention rate among answers that moved to the hinted option, since only those show the hint "
           "was used, against a rerun without hints.",
           "A mention rate near 1 among moved answers would weaken the case, since the reasoning would then report "
           "what influenced it."], M["mention"] + M["dir"],
          ("Whether the response links the hint result to Q1, picks the right rate and states a disconfirming "
           "outcome.",
           "Using a replication to answer a research question needs both the rate and the direction-choosing logic.",
           "A response could report the overall mention rate and call any value support."))
    integ("compare-r3", [A + "prompt-cues", A + "resample"],
          "R3 has two parts: insert an awareness sentence into the reasoning and measure refusal, and rewrite away "
          "prompt cues and measure refusal. Explain how each part bears on Q2, what each holds fixed, and which "
          "pattern of results would show that awareness set by the prompt changes behavior.",
          ["The insertion part holds the prefix fixed and varies one sentence, testing whether voiced awareness "
           "changes behavior downstream.",
           "The rewrite part holds the request fixed and varies prompt cues, testing whether prompt features set the "
           "awareness (Q2) and the safer behavior.",
           "If rewrites lower both voiced awareness and refusal and inserted sentences raise refusal, prompt-set "
           "awareness changes behavior; if rewrites lower awareness but not refusal, it does not."],
          M["cues"] + M["resample"],
          ("Whether the response distinguishes what each part holds fixed and combines them into one prediction.",
           "Answering Q2 from R3 needs both the rewrite logic and the fixed-prefix logic.",
           "A response could describe both parts but not say which result pattern would support the claim."))
    integ("compare-r4-r5", [A + "probe", A + "graders"],
          "In R4 a probe chosen on half the prompts reaches AUROC 0.78 on the other half, while the logit monitor "
          "reaches 0.66. In R5 a probe rebuilt from training-awareness statements correlates 0.97 with it per prompt. "
          "Interpret the results together and say what they add to the answer to Q3.",
          ["The probe's 0.78 is an honest held-out number, so it separates the sets better than the logit monitor.",
           "The 0.97 correlation with a training-awareness probe means the probe cannot say whether it tracks "
           "evaluation or training, or something both share such as graded, task-like prompts.",
           "For Q3 this shows that separating Fortress from WildChat, even well, does not identify evaluation "
           "awareness; a within-content test is still needed."], M["probe"] + M["graders"],
          ("Whether the response credits the held-out AUROC and still limits its meaning using the training-probe "
           "result.",
           "Interpreting the pair needs the probe procedure and the grader and training findings together.",
           "A response could take 0.78 as proof of awareness, or dismiss the probe without reading the held-out "
           "design."))
    integ("compare-r6", ["aea-results.variance"],
          "Two prompts have true per-rollout chances of voicing awareness of 0.01 and 0.05. Compute the chance that "
          "each is labelled \"ever voices awareness\" in 100 rollouts, explain why that makes the 24.9% figure hard "
          "to read, and say what R6's multilevel model would report instead.",
          ["The chances are $1-0.99^{100}\\approx0.634$ and $1-0.95^{100}\\approx0.994$, so a prompt with a small "
           "chance can land in either group.",
           "The label is itself an outcome of sampling tied to each prompt's propensity, so grouping by it sorts "
           "prompts partly by what the monitor tracks, and 24.9% is not a causal share.",
           "A multilevel model with a random effect per prompt estimates the between-prompt share of score variance "
           "directly as an intraclass correlation."], M["dir"],
          ("Whether the response computes both chances, explains the label's sampling noise and names the "
           "multilevel estimate.",
           "The task joins the at-least-one calculation, the variance interpretation and the choice of R6.",
           "A response could compute the chances but not connect them to the 24.9% grouping."))
    integ("compare-r7-r8", [A + "quadruples", A + "phrase-family"],
          "R7 (quadruples) and R8 (a pre-registered phrase family) both bear on whether the monitor's separation is "
          "about evaluation (Q3). Compare what each can and cannot rule out, and propose an order for running them "
          "with a reason.",
          ["R8 tests whether \"testing\" is special among committed evaluation-aware phrases and whether the "
           "evaluation family beats the deployment family; it still uses the same confounded prompt sets.",
           "R7 holds content fixed and varies framing, so it can rule out topic, harm and length differences, but "
           "explicit statements are unusual text.",
           "Run R8 first because it only re-scores saved rollouts, then use its committed family as the target in "
           "R7, which needs new prompts."], M["quad"] + M["family"],
          ("Whether the response states each design's reach and orders them using cost and dependency.",
           "Comparing the two needs both designs and the direction-choosing criteria.",
           "A response could prefer one design without saying what the other rules out."))
    integ("compare-steer", [A + "intent-gap", A + "probe"],
          "R4's last step adds plus or minus a multiple of the probe direction during sampling on 30 prompts, with "
          "a random direction of the same length as a control. Refusals fall when steering toward deployment and not "
          "with the random direction. What does this show?",
          ["Against the random-direction control, a refusal drop shows this direction affects behavior.",
           "It does not show the model now behaves as in deployment, since the direction may change the inferred "
           "purpose or insert a user intent rather than remove the sense of being tested.",
           "Also measure what purpose the model infers after steering, voiced awareness, and output quality, since "
           "large steering can break the model."], M["intent"] + M["probe"],
          ("Whether the response credits the controlled effect and applies the intent gap to its interpretation.",
           "Interpreting probe steering needs both the probe procedure and the inferred-versus-target distinction.",
           "A response could read the drop as deployment behavior, or dismiss it without using the control."))
    return out


# ---------------------------------------------------------------- design record

NODES = {
    "aea-related.asking": dict(
        mastery_claim="Interpret a model's answers to \"is this an evaluation?\" by separating ranking (AUC) from "
                      "calibration and both from unprompted awareness.",
        split_test="A learner can compute AUC and ECE correctly yet still read a prompted classification as "
                   "unprompted awareness; the free items catch that, the lab catches the computation.",
        diagnostic_error="Treats a high AUC from an asked question as evidence of accurate probabilities or of "
                         "unprompted awareness.",
        instructional_response="Show one model that ranks well with poor calibration, and contrast a question-prompted "
                               "judgment with a readout taken while the model handles the original prompt."),
    "aea-related.mention-rate": dict(
        mastery_claim="Compute and interpret how often reasoning admits a planted hint, among the answers the hint "
                      "moved.",
        split_test="A learner can define faithfulness yet divide by every hinted question; the numeric items "
                   "separate the two.",
        diagnostic_error="Uses all hinted questions as the denominator, or reads silence as absence of use.",
        instructional_response="Work a table where unmoved runs mention the hint to reject it, and compute both "
                               "rates."),
    "aea-related.resample": dict(
        mastery_claim="Design and interpret a remove-and-resample test of one reasoning sentence's effect on later "
                      "behavior.",
        split_test="A learner can say correlation is not causation yet propose a comparison across prompts that "
                   "does not hold the reasoning history fixed.",
        diagnostic_error="Treats the outcome gap between aware and unaware rollouts as the sentence's effect.",
        instructional_response="Contrast the observational gap with continuations resampled from the same prefix."),
    "aea-related.prompt-cues": dict(
        mastery_claim="Turn words that co-occur with voiced awareness into a controlled, content-preserving rewrite "
                      "test.",
        split_test="A learner can find candidate cues yet accept a rewrite that also softens the request.",
        diagnostic_error="Credits a refusal drop to a removed cue without checking that the request stayed the same.",
        instructional_response="Work a case where the rewrite changes two things, then show the single-change "
                               "rewrite."),
    "aea-related.intent-gap": dict(
        mastery_claim="Judge whether an intervention that lowers voiced awareness leaves the model in the situation "
                      "the evaluation targets.",
        split_test="A learner can explain steering yet read any awareness drop as deployment behavior.",
        diagnostic_error="Treats lower voiced awareness as revealing deployment behavior.",
        instructional_response="Separate inferred from target intent with a steering case that inserts a different "
                               "purpose."),
    "aea-related.probe": dict(
        mastery_claim="Build a difference-of-means probe, score prompts with the right sign, and report its AUROC on "
                      "prompts not used to choose the layer.",
        split_test="A learner can compute the projection yet report the best layer's AUROC on the prompts used to "
                   "choose it.",
        diagnostic_error="Flips the sign convention, or reports a layer-selected AUROC as held-out performance.",
        instructional_response="Work the direction and projection by hand, then compare development and new-prompt "
                               "AUROC for a selected layer."),
    "aea-related.graders": dict(
        mastery_claim="Choose and interpret an awareness monitor given that awareness differs by eval and grader "
                      "type and may not be specific to evaluation.",
        split_test="A learner can recall that monitors rank differently yet pick a monitor by its largest headline "
                   "AUROC.",
        diagnostic_error="Assumes one monitor ranking holds across tasks, or reads an evaluation probe as specific to "
                         "evaluation.",
        instructional_response="Pair a task-reversal table with the training-probe correlation and ask what each "
                               "monitor's number can mean."),
    "aea-related.quadruples": dict(
        mastery_claim="Estimate within-task style and framing effects from quadruples and read them against rival "
                      "predictions.",
        split_test="A learner can compute within-task differences yet call any statement effect awareness.",
        diagnostic_error="Compares versions across tasks, or ignores that an explicit statement is unusual text.",
        instructional_response="Work a two-task example, then set the three effects against the semantic and format "
                               "predictions."),
    "aea-related.phrase-family": dict(
        mastery_claim="Score a committed phrase family per token, form the contrastive score, and evaluate it on "
                      "prompts not used to choose phrases.",
        split_test="A learner can normalize per token yet report the best phrase on the prompts used to choose it.",
        diagnostic_error="Averages totals across phrase lengths, or reports a selected phrase's in-sample AUROC.",
        instructional_response="Work the per-token contrast by hand, then show a chosen phrase's AUROC falling on the "
                               "other half."),
    "aea-related.directions": dict(
        mastery_claim="Choose among R1 to R8 the experiment whose outcome could change an answer, that separates the "
                      "live rivals, and that the available compute can run.",
        split_test="A learner can describe each experiment yet choose by novelty or pick one whose every result "
                   "would count as support.",
        diagnostic_error="Chooses an experiment without naming the answer it could change or the rival it rules out.",
        instructional_response="Map each experiment to Q1 to Q3 and its rival, then compare two candidates on "
                               "outcome, rival and cost."),
}

CANDIDATE_SOURCES = {
    "aea-related.asking": ("Asking the model whether a transcript is an evaluation", "needham, sections 3-4"),
    "aea-related.mention-rate": ("Hint experiments and the mention rate", "chen, Testing for faithfulness"),
    "aea-related.resample": ("Removing an awareness sentence and resampling", "companion, section 3"),
    "aea-related.prompt-cues": ("Prompt cues found across rollouts and rewritten away", "companion, section 4"),
    "aea-related.intent-gap": ("User-intent steering and the inferred-versus-target gap", "companion, section 5"),
    "aea-related.probe": ("Difference-of-means probes, steering and layer choice", "heidari, sections 2-5, 7"),
    "aea-related.graders": ("Awareness by grader type; evaluation versus training probes", "graders, summary"),
    "aea-related.quadruples": ("Content-by-framing quadruples (R7)", "research guide, replication program"),
    "aea-related.phrase-family": ("Pre-registered phrase families (R8)", "research guide, replication program"),
    "aea-related.directions": ("Choosing among related experiments R1 to R8", "research guide; lab docs/02"),
}

# (prerequisite, target, reason, counterfactual, task)
DEPENDENCIES = [
    ("aea-results.pair-auc", "aea-related.asking", "Reading an elicited AUC needs AUROC as the share of pairs ranked "
     "correctly.", "Without pairwise AUROC the learner cannot see why ranking says nothing about calibration.",
     "aea-related.asking.lab#3"),
    ("aea-validity.faithfulness", "aea-related.mention-rate", "The mention rate measures faithfulness about a hint.",
     "Without the idea of an omitted influence the learner has no reason to condition on moved answers.",
     "aea-related.mention-rate.direct-1"),
    ("aea-validity.causal", "aea-related.resample", "Remove-and-resample is a controlled intervention on one "
     "sentence.", "Without the association-versus-intervention distinction the learner accepts the observational "
     "gap.", "aea-related.resample.direct-1"),
    ("aea-validity.causal", "aea-related.prompt-cues", "A cue rewrite is an intervention on one prompt feature, "
     "judged against the association that suggested it.", "Without the association-versus-intervention distinction "
     "the learner accepts a co-occurring word as a cause.", "aea-related.prompt-cues.direct-1"),
    ("aea-validity.causal", "aea-related.intent-gap", "Steering is an intervention whose side effects need controls.",
     "Without the causal-control idea the learner reads any awareness drop as the intended change.",
     "aea-related.intent-gap.direct-1"),
    ("aea-results.pair-auc", "aea-related.probe", "A probe is judged by AUROC over evaluation and real-use pairs.",
     "Without pairwise AUROC the learner cannot compute or compare the probe's separation.",
     "aea-related.probe.lab#2"),
    ("aea-related.probe", "aea-related.graders", "The training-probe finding compares two difference-of-means "
     "probes.", "Without knowing how such a probe is built the learner cannot read the 0.98 correlation.",
     "aea-related.graders.direct-1"),
    ("aea-validity.confounds", "aea-related.quadruples", "Quadruples are a design against dataset confounds.",
     "Without the confound idea the learner cannot say what holding content fixed removes.",
     "aea-related.quadruples.direct-1"),
    ("aea-results.targets", "aea-related.phrase-family", "A family generalizes the choice of target phrase and its "
     "control.", "Without phrase targets and controls the learner cannot see why a contrast family is needed.",
     "aea-related.phrase-family.direct-1"),
    ("aea-related.quadruples", "aea-related.directions", "Choosing for Q3 compares the quadruple design with other "
     "candidates.", "Without the quadruple design the learner lacks the strongest Q3 option to weigh.",
     "aea-related.directions.direct-2"),
    ("aea-results.variance", "aea-related.directions", "Judging R6 needs the prompt-versus-rollout variance result "
     "it re-analyses.", "Without the 0.4% and 24.9% comparison the learner cannot see what a multilevel model would "
     "change about Q2.", "aea-related.directions.direct-1"),
]

# (a, b, type, reason)
LINKS = [
    ("aea-related.asking", "aea-observe.readouts", "contrasts",
     "Asking elicits a judgment on request; the logit monitor reads an unprompted phrase probability."),
    ("aea-related.mention-rate", "aea-observe.signal", "evidence-for",
     "Hints used without mention show that the sampled text can omit what influenced the answer."),
    ("aea-related.resample", "aea-results.variance", "contrasts",
     "Grouping observed rollouts is associational; resampling from a fixed prefix varies one sentence."),
    ("aea-related.prompt-cues", "aea-related.intent-gap", "explains",
     "The cues that trigger awareness concern user intent, so lowering awareness can change the inferred intent."),
    ("aea-related.probe", "aea-observe.phrase", "contrasts",
     "A probe reads a direction in hidden activations; the phrase monitor reads an output probability."),
    ("aea-related.graders", "aea-validity.transport", "explains",
     "If awareness differs by grader type, a monitor tuned on one eval family can fail on another."),
    ("aea-related.quadruples", "aea-related.prompt-cues", "contrasts",
     "Cue rewrites edit natural prompts; quadruples cross framing with content held fixed."),
]


def edge_rows():
    rows = [{"a": t, "b": p, "type": "depends-on", "note": reason} for p, t, reason, _, _ in DEPENDENCIES]
    rows += [{"a": a, "b": b, "type": ty, "note": reason} for a, b, ty, reason in LINKS]
    return rows


def design_text(path, evidence):
    d = json.loads(path.read_text())
    own = set(NODES)
    d["candidates"] = [c for c in d["candidates"] if not set(c["outputs"]) & own] + [
        {"id": "candidate-" + n.split(".", 1)[1], "decision": "keep", "inputs": [CANDIDATE_SOURCES[n][0]],
         "outputs": [n], "reason": NODES[n]["diagnostic_error"] + " warrants this remediation: "
         + NODES[n]["instructional_response"], "source_notes": CANDIDATE_SOURCES[n][1]} for n in NODES]
    old = [n for n in d["nodes"] if n["id"] not in own]
    for n in old:   # integrated evidence this script adds to existing nodes
        n["evidence"] = [e for e in n["evidence"] if not e["question"].startswith("aea-related.")] + \
            evidence.get(n["id"], [])
    missing = set(evidence) - own - {n["id"] for n in old}
    if missing:
        sys.exit(f"{path}: evidence for unknown nodes {sorted(missing)}")
    d["nodes"] = old + [{"id": n, **NODES[n], "evidence": evidence[n]} for n in NODES]
    d["dependencies"] = [x for x in d["dependencies"] if x["target"] not in own] + [
        {"prerequisite": p, "target": t, "reason": r, "counterfactual": cf, "task": task,
         "provenance": "author-hypothesis"} for p, t, r, cf, task in DEPENDENCIES]
    d["semantic_links"] = [x for x in d["semantic_links"] if not {x["a"], x["b"]} & own] + [
        {"a": a, "b": b, "type": ty, "reason": r} for a, b, ty, r in LINKS]
    return json.dumps(d, indent=2, ensure_ascii=False) + "\n"


def main(argv):
    labs = [f() for f in LABS]
    errs = [e for lab in labs for e in check_lab(lab)]
    pairs = [(lab_question(lab), lab_evidence(lab)) for lab in labs]
    items = authored()
    evidence = {}
    for lab, (q, ev) in zip(labs, pairs):
        evidence.setdefault(lab["concept"], []).extend(ev)
    for q, ev in items:
        for node, role, obs, war, riv in ev:
            evidence.setdefault(node, []).append(
                {"question": q["id"], "role": role, "observes": obs, "warrant": war, "rival": riv})
    questions = [q for q, _ in pairs] + [q for q, _ in items]
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        errs.append("duplicate question ids")
    for lab in labs:
        print(f"{lab['id']:36s} " + "; ".join(f"{c[:46]} -> {show(run(lab['setup'], lab['code'], c))}"
                                               for c in list(lab["expect"])[:2]))
    per = {}
    for q in questions:
        per[q["targets"][0]] = per.get(q["targets"][0], 0) + 1
    print("questions per file:", per, "total", len(questions))
    if errs:
        print("\nFAILED:\n  " + "\n  ".join(errs))
        return 1
    out = {}     # path -> text; filled completely before anything is written
    if "--write" in argv:
        folder = Path(argv[argv.index("--write") + 1]).resolve()
        by_file = {}
        for q in questions:
            by_file.setdefault(q["targets"][0], []).append(q)
        for concept, qs in by_file.items():
            out[folder / SEC / f"{concept}.questions.json"] = json.dumps(qs, indent=2, ensure_ascii=False) + "\n"
        mine = {p[0] for p in PASSAGES}
        ps = [p for p in json.loads((folder / "passages.json").read_text()) if p["id"] not in mine]
        out[folder / "passages.json"] = json.dumps(ps + cut_passages(folder), indent=1, ensure_ascii=False) + "\n"
        own = set(NODES)
        es = [e for e in json.loads((folder / "edges.json").read_text()) if e["a"] not in own]
        out[folder / "edges.json"] = json.dumps(es + edge_rows(), indent=2, ensure_ascii=False) + "\n"
        dpath = folder / "authoring" / f"{folder.name}.design.json"
        out[dpath] = design_text(dpath, evidence)
    if "--pack" in argv:
        dpath = Path(argv[argv.index("--pack") + 1]) / "authoring" / "arena-eval-awareness.design.json"
        out[dpath] = design_text(dpath, evidence)
    for path, text in out.items():
        path.write_text(text)
    if "--write" in argv:
        print(f"wrote {len(questions)} questions in {len(by_file)} files, {len(PASSAGES)} passages, "
              f"{len(edge_rows())} edges")
    for path in out:
        if path.name.endswith(".design.json"):
            print(f"wrote design -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
