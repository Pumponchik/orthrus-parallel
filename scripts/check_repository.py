"""Offline source, asset, link, and published-result checks; no dependencies."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
for path in ROOT.rglob('*.py'):
    if not any(part.startswith('.') or part == '__pycache__' for part in path.relative_to(ROOT).parts):
        ast.parse(path.read_text(), filename=str(path))
for path in ROOT.rglob('*.md'):
    if any(part.startswith('.') for part in path.relative_to(ROOT).parts):
        continue
    text = path.read_text()
    assert text.count('```') % 2 == 0, path
    assert not re.search(r'[\u0400-\u04ff]', text), path
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if '://' not in target and not target.startswith('#'):
            assert (path.parent / target.split('#')[0]).exists(), (path, target)
for name, digest in json.loads((ROOT / 'weights/manifest.json').read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
with (ROOT / 'benchmarks/results/measurements.csv').open() as f:
    rows = list(csv.DictReader(f))
summary = json.loads((ROOT / 'benchmarks/results/summary.json').read_text())
assert len(rows) == summary['generations'] == 120
assert len({x['prompt'] for x in rows}) == 30
for arm in ['baseline', 'fixed']:
    subset = [x for x in rows if x['arm'] == arm]
    speed = sum(int(x['tokens']) for x in subset) / sum(float(x['seconds']) for x in subset)
    assert abs(speed - summary['arms'][arm]['tokens_per_second']) < 1e-10
print('Passed: Python syntax, English Markdown, links, weight hashes, and published throughput.')
