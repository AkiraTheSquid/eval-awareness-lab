"""Registry rows, atom-graph nodes and edges, clocks and glossary lines for the logit-monitor unit (idempotent)."""
import json
import re

import framework as F

LESSON_TITLE = "Logit monitor for evaluation awareness (Goodfire 2026)"
LESSON_ROW = {"id": F.LESSON_ID, "topic": F.TOPIC, "title": LESSON_TITLE, "subtopic_key": f"{F.TOPIC}: {F.LESSON_ID}"}
SUB = f"{F.TOPIC}: {F.LESSON_ID}"

KCS = {
    "monitor.next-token-logits": dict(
        title="Row p of the logits scores the token at position p + 1",
        prereqs=["tensor.indexed-selection", "torch.argmin-argmax", "torch.slicing-views"],
        encompassing={"tensor.indexed-selection": 0.3}, secs=600),
    "monitor.forced-logprob": dict(
        title="The log-probability of a forced phrase",
        prereqs=["monitor.next-token-logits", "tensor.stable-probabilities", "tensor.indexed-selection", "einops.reduce-model"],
        encompassing={"monitor.next-token-logits": 0.4, "tensor.stable-probabilities": 0.2, "einops.reduce-model": 0.2}, secs=600),
    "monitor.forced-sequence": dict(
        title="Build the forced sequence and tokenise the phrase in context",
        prereqs=["monitor.next-token-logits", "torch.stack-concat-interleave", "python.control-flow", "tensor.row-normalization"],
        encompassing={"torch.stack-concat-interleave": 0.2}, secs=600),
    "monitor.reasoning-boundaries": dict(
        title="Cut the reasoning at its end and find the sentence boundaries",
        prereqs=["tensor.indexed-selection", "torch.boolean-masking", "tensor.row-normalization", "einops.reduce-model"],
        encompassing={"tensor.indexed-selection": 0.2, "torch.boolean-masking": 0.2}, secs=600),
    "monitor.padded-batch": dict(
        title="Score many forced sequences in one padded batch",
        prereqs=["monitor.forced-logprob", "monitor.forced-sequence", "torch.constructors", "torch.slice-assignment",
                 "torch.broadcasting-rules", "python.control-flow"],
        encompassing={"monitor.forced-logprob": 0.4, "monitor.forced-sequence": 0.2, "torch.broadcasting-rules": 0.2}, secs=900),
    "monitor.prompt-score": dict(
        title="One score per prompt: the maximum over rollouts and positions",
        prereqs=["monitor.forced-logprob", "monitor.reasoning-boundaries", "einops.reduce-model", "torch.constructors", "torch.stack-concat-interleave",
                 "torch.slice-assignment", "python.control-flow"],
        encompassing={"einops.reduce-model": 0.3}, secs=600),
    "monitor.auroc": dict(
        title="AUROC with a bootstrap interval",
        prereqs=["monitor.prompt-score", "torch.broadcasting-rules", "torch.boolean-masking", "torch.random-seeding"],
        encompassing={"monitor.prompt-score": 0.2, "torch.broadcasting-rules": 0.2}, secs=900),
    "monitor.baselines": dict(
        title="Baselines and confounds: verbalisation, a control phrase, prompt length",
        prereqs=["monitor.auroc", "monitor.prompt-score", "tensor.row-normalization"],
        encompassing={"monitor.auroc": 0.4, "monitor.prompt-score": 0.3}, secs=900),
}

D = "Logit-monitor unit: "
ATOMS = {  # id: (title, description)
    "lm-next-token-row": ("Pick the logits row that predicts a position", D + "row p scores token p + 1; rows for a span."),
    "lm-forced-logprob": ("Log-probability of forced tokens", D + "log_softmax over the vocabulary, paired index, sum over the phrase."),
    "lm-forced-sequence": ("Assemble prefix, reasoning and phrase", D + "concatenate the pieces, find where the phrase starts, tokenise in context."),
    "lm-reasoning-cut": ("Cut reasoning at the first stop token", D + "first stop index, unfinished rollouts."),
    "lm-sentence-boundaries": ("Sentence boundary positions", D + "lookup table over the vocabulary, positions after sentence-ending tokens."),
    "lm-padded-batch": ("Right-pad sequences and pick per-row rows", D + "pad, mask, per-row start rows."),
    "lm-prompt-score": ("Maximum over rollouts and positions", D + "pad ragged traces with -inf, reduce with max, first k rollouts."),
    "lm-auroc": ("AUROC from pairwise comparisons", D + "wins plus half ties over all pairs."),
    "lm-bootstrap-ci": ("Bootstrap interval for a statistic", D + "resample with replacement, quantiles of the recomputed statistic."),
    "lm-verbalisation-baseline": ("Fraction of rollouts a judge flagged", D + "mean of flags over the first k rollouts."),
    "lm-control-confound": ("Control phrase and prompt-length checks", D + "primary minus control, AUROC of prompt length."),
}
EDGES = [  # prerequisite, dependent, hard, rationale
    ("index-by-tensor", "lm-next-token-row", True, "[builds-on] the next token's score is read with an index tensor"),
    ("lm-next-token-row", "lm-forced-logprob", True, "[builds-on] the phrase is scored on the rows that predict it"),
    ("log-sum-exp-stability", "lm-forced-logprob", False, "[uses] log_softmax is scores minus their log-sum-exp"),
    ("einops-reduce", "lm-forced-logprob", False, "[uses] the phrase score is a sum reduction over the token axis"),
    ("lm-next-token-row", "lm-forced-sequence", True, "[builds-on] the phrase start decides which rows are read"),
    ("stack-vs-cat", "lm-forced-sequence", False, "[uses] the sequence is a concatenation"),
    ("index-by-tensor", "lm-reasoning-cut", False, "[uses] positions are compared and selected with tensors"),
    ("lm-reasoning-cut", "lm-sentence-boundaries", True, "[builds-on] boundaries are searched inside the cut reasoning"),
    ("index-by-tensor", "lm-sentence-boundaries", False, "[uses] the lookup table is indexed by token ids"),
    ("lm-forced-logprob", "lm-padded-batch", True, "[builds-on] each padded row is scored as a forced phrase"),
    ("lm-forced-sequence", "lm-padded-batch", True, "[builds-on] the batch rows are forced sequences"),
    ("broadcasting-rules", "lm-padded-batch", False, "[uses] the mask and the row table are broadcast comparisons and sums"),
    ("lm-forced-logprob", "lm-prompt-score", True, "[builds-on] the table entries are forced-phrase scores"),
    ("lm-sentence-boundaries", "lm-prompt-score", True, "[builds-on] one score per boundary"),
    ("einops-reduce", "lm-prompt-score", False, "[uses] the prompt score is a max reduction"),
    ("lm-prompt-score", "lm-auroc", True, "[builds-on] AUROC compares prompt scores"),
    ("broadcasting-rules", "lm-auroc", False, "[uses] all pairs are compared by broadcasting"),
    ("lm-auroc", "lm-bootstrap-ci", True, "[builds-on] the interval recomputes AUROC on resamples"),
    ("lm-auroc", "lm-verbalisation-baseline", True, "[builds-on] the baseline is judged by the same AUROC"),
    ("lm-prompt-score", "lm-control-confound", True, "[builds-on] the control is subtracted before the maximum"),
    ("lm-auroc", "lm-control-confound", True, "[builds-on] prompt length is tested with the same AUROC"),
]


def upstream(kc):
    """Every registry concept before `kc`, through this unit's rows and the registry's own."""
    reg = {k["id"]: k.get("prereqs") or [] for k in json.loads(F.REG_PATH.read_text())["kcs"]}
    for k, v in KCS.items():
        reg[k] = v["prereqs"]
    seen, todo = set(), list(reg[kc])
    while todo:
        k = todo.pop()
        if k not in seen:
            seen.add(k)
            todo += reg.get(k, [])
    return seen


def _dump(path, obj, **kw):
    path.write_text(json.dumps(obj, **kw) + "\n")


def apply_registry(kc_ids):
    reg = json.loads(F.REG_PATH.read_text())
    if not any(l["id"] == F.LESSON_ID for l in reg["lessons"]):
        reg["lessons"].append(LESSON_ROW)
    have = {k["id"]: i for i, k in enumerate(reg["kcs"])}
    for kc in kc_ids:
        s = KCS[kc]
        row = {"id": kc, "lesson": F.LESSON_ID, "title": s["title"], "syntax": False, "prereqs": s["prereqs"],
               "encompassing": s["encompassing"]}
        if kc in have:
            reg["kcs"][have[kc]] = row
        else:
            reg["kcs"].append(row)
    _dump(F.REG_PATH, reg, indent=1)


def apply_graph(atom_ids, links):
    g = json.loads(F.GRAPH_PATH.read_text())
    have = {c["id"] for c in g["concepts"]}
    for a in sorted(atom_ids):
        if a not in have:
            title, desc = ATOMS[a]
            g["concepts"].append({"id": a, "title": title, "topic": F.TOPIC, "subtopic": SUB, "node_type": "concept",
                                  "kind": "skill", "tier": "core", "description": desc, "tags": ["drill-atom", "logit-monitor"]})
    have = {c["id"] for c in g["concepts"]}
    pairs = {(e["prerequisite_id"], e["dependent_id"]) for e in g["prerequisite_edges"]}
    for pre, dep, hard, why in EDGES:
        assert pre in have, f"edge from unknown atom {pre}"
        if dep in atom_ids and (pre, dep) not in pairs:
            g["prerequisite_edges"].append({"prerequisite_id": pre, "dependent_id": dep, "weight": 1.0, "confidence": 0.8,
                                            "is_hard_gate": hard, "is_encompassing": False, "propagation_weight": 0.0,
                                            "rationale": why})
    have_links = {(l["problem_ref"], l["concept_id"]) for l in g["problem_links"]}
    for qid, atom in links:
        if (str(qid), atom) not in have_links:
            g["problem_links"].append({"source_bank": "delta_drills", "problem_ref": str(qid), "concept_id": atom,
                                       "role": "primary", "weight": 1, "notes": "Logit-monitor unit drill."})
    _dump(F.GRAPH_PATH, g, indent=2, ensure_ascii=False)


def apply_caps(kc_ids):
    raw = F.CAPS_PATH.read_text()
    d = json.loads(raw)
    for kc in kc_ids:
        d["kcs"][kc] = KCS[kc]["secs"]
    lines = raw.split("\n")
    indent = None if "\n" not in raw.strip() else (2 if lines[1].startswith("  ") and not lines[1].startswith("   ") else 1)
    F.CAPS_PATH.write_text(json.dumps(d, indent=indent, ensure_ascii=False) + "\n")


def apply_glossary(kc_ids):
    s = F.GLOSS_PATH.read_text()
    m = re.search(r"window\.DD_GLOSSARY\.kcLesson = \{\n(.*?)\n\};", s, re.S)
    body = m.group(1)
    for kc in kc_ids:
        line = f'  "{kc}": [{json.dumps(LESSON_TITLE, ensure_ascii=False)}, {json.dumps(KCS[kc]["title"], ensure_ascii=False)}]'
        if f'"{kc}":' in body:
            body = re.sub(rf'  "{re.escape(kc)}": \[.*?\](,?)$', lambda mm: line + mm.group(1), body, flags=re.M)
        else:
            body = body.rstrip(",") + ",\n" + line
    F.GLOSS_PATH.write_text(s[:m.start(1)] + body + s[m.end(1):])
