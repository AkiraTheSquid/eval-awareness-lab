"""Authoring framework for the logit-monitor unit (lesson lm-1).

The drill machinery (Drill, prepare, CSV block, overrides, tags, page renderer) is the refusal
course's `framework.py`, loaded here as a PRIVATE module instance and pointed at this unit's
ids file and lesson id. Nothing in `scripts/refusal_course/` is edited or shares state with us.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LESSON_ID = "lm-1"
TOPIC = "PyTorch"
FOLDER = "logit-monitor"

_NAME = "logit_monitor_refusal_framework"
_spec = importlib.util.spec_from_file_location(_NAME, HERE.parent / "refusal_course" / "framework.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_NAME] = R            # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(R)
R.HERE = HERE
R.IDS_PATH = HERE / "ids.json"
R.LESSON_ID = LESSON_ID
R.TOPIC = TOPIC

Drill = R.Drill
ROOT, LESSONS = R.ROOT, R.LESSONS
REG_PATH, CAPS_PATH, GLOSS_PATH, GRAPH_PATH = R.REG_PATH, R.CAPS_PATH, R.GLOSS_PATH, R.GRAPH_PATH
IDS_PATH = R.IDS_PATH
FADE_BLANK = R.FADE_BLANK
prepare, load_ids, save_ids = R.prepare, R.load_ids, R.save_ids
upsert_csv, resolve_ids, drop_overrides = R.upsert_csv, R.resolve_ids, R.drop_overrides
write_overrides, write_tags = R.write_overrides, R.write_tags


def render_kp(kc, drills, ids, prepared, mc_prompts=None):
    """The refusal renderer, plus the `new_syntax` this page declares (it hard-codes an empty list)."""
    md = R.render_kp(kc, drills, ids, prepared, mc_prompts)
    new = kc.get("new_syntax") or []
    if new:
        assert md.count("new_syntax: []") == 1
        md = md.replace("new_syntax: []", f"new_syntax: [{', '.join(new)}]")
    return md


def _src(args, body, doc=None):
    """Bank source for one drill. `einops` is imported only by drills whose body calls it."""
    head = "import torch as t\n"
    if "einops." in body:
        head += "import einops\n"
    head += f"\n\ndef solve({args}):\n"
    if doc:
        head += f'    """{doc}"""\n'
    return head + body.rstrip("\n") + "\n"


R.src = _src
src = _src


def maker(prefix, kc):
    """Drill constructor for one concept module: keys get the module's prefix."""
    def make(key, rung, seg, prompt, args, doc, ref, cases, wrong, diff, atoms, scaffold="", show=0):
        return Drill(key=f"{prefix}-{key}", kc=kc, rung=rung, seg=seg, prompt=prompt, args=args, doc=doc, ref=ref,
                     cases=cases, wrong=wrong, diff=diff, atoms=atoms, scaffold=scaffold, show=show)
    return make
