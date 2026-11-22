"""Turn rerun_source.py (cells split on `# %%`) into logit_monitor_rerun.ipynb.

  python make_notebook.py            write the notebook
  python make_notebook.py --test     run the code cells here on CPU with a tiny model and 2 pairs
  python make_notebook.py --script   write rerun_gpu.py: Qwen3-32B full precision, scaled down for a rented GPU
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
TEST = {'MODEL = "auto"': 'MODEL = "Qwen/Qwen3-0.6B"', "N_PAIRS = 150": "N_PAIRS = 2", "R = 6 ": "R = 2 ",
        "T = 512": "T = 48", "MAX_PROMPT = 1500": "MAX_PROMPT = 120", "plt.show()": "pass"}
SCRIPT = {'MODEL = "auto"': 'MODEL = "Qwen/Qwen3-32B"', 'LOAD_4BIT = "auto"': "LOAD_4BIT = False",
          "N_PAIRS = 150": "N_PAIRS = int(os.environ.get('N_PAIRS', 150))", "R = 6 ": "R = 4 ", "T = 512": "T = 384",
          "MINUTES = 600": "MINUTES = float(os.environ.get('MINUTES', 600))",
          "plt.show()": "pass", "import matplotlib.pyplot as plt": "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"}


def cells():
    out = []
    for block in ("\n" + (HERE / "rerun_source.py").read_text()).split("\n# %%")[1:]:
        head, _, body = block.partition("\n")
        if "[markdown]" in head:
            out.append(("markdown", "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#")
                                              for line in body.strip("\n").split("\n"))))
        else:
            out.append(("code", body.strip("\n")))
    return out


if "--test" in sys.argv or "--script" in sys.argv:
    src = "\n\n".join(body for kind, body in cells() if kind == "code")
    for a, b in (TEST if "--test" in sys.argv else SCRIPT).items():
        assert a in src, a
        src = src.replace(a, b)
    if "--test" in sys.argv:
        exec(compile(src, "rerun_source.py", "exec"), {"__name__": "__main__"})
    else:
        (HERE / "rerun_gpu.py").write_text(src + "\n")
        print("wrote", HERE / "rerun_gpu.py")
else:
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "cells": [{"cell_type": kind, "metadata": {}, "source": body.splitlines(keepends=True),
                     **({"outputs": [], "execution_count": None} if kind == "code" else {})}
                    for kind, body in cells()]}
    (HERE / "logit_monitor_rerun.ipynb").write_text(json.dumps(nb, indent=1))
    print("wrote", HERE / "logit_monitor_rerun.ipynb", len(nb["cells"]), "cells")
