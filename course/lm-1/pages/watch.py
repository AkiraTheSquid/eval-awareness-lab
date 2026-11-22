"""watch.py — health checks for logit-monitor

Static: the eight generated pages exist, are registered and carry asserted cells.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = ['next-token-logits', 'forced-logprob', 'forced-sequence', 'reasoning-boundaries',
         'padded-batch', 'prompt-score', 'auroc', 'baselines']


def check_imports():
    reg = json.load(open(os.path.join(HERE, '..', 'kc_registry.json')))
    ids = {k['id'] for k in reg['kcs']}
    for p in PAGES:
        assert f'monitor.{p}' in ids, f'monitor.{p} missing from the registry'


def check_public_api():
    for p in PAGES:
        assert os.path.exists(os.path.join(HERE, f'kp-{p}.md')), f'missing kp-{p}.md'


def check_invariants():
    for p in PAGES:
        text = open(os.path.join(HERE, f'kp-{p}.md')).read()
        assert 'modulario:template' not in text
        assert '# Hidden checks' in text, f'kp-{p}: no asserted cells'
        assert f'kc: monitor.{p}' in text, f'kp-{p}: wrong kc in frontmatter'


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
