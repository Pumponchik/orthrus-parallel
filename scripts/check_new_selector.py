"""Check the shipped selector, weights and measured aggregates without GPU dependencies."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'learned_selector'
RESULTS = ROOT / 'results/selector_training_20261015'


def main():
    manifest = json.loads((PACKAGE / 'MANIFEST_SHA256.json').read_text())
    for name, expected in manifest.items():
        path = PACKAGE / name
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    for path in PACKAGE.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    proof = json.loads((PACKAGE / 'source_equivalence.json').read_text())
    for name, row in proof['proof'].items():
        assert row['ast_equivalent']
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == row['release_sha256'], name
    config = json.loads((PACKAGE / 'best_selector.json').read_text())
    assert (config['block_size'], config['k'], config['m'], config['adaptive']) == (16, 32, 48, False)
    assert hashlib.sha256((PACKAGE / 'trained_bilinear/hidden_64_ce_20261015.pt').read_bytes()).hexdigest() == '18d6dc12b0bf68416f263bf51f05ffdeeb59b79074ab68efe98fb4684d9bb457'
    summary = json.loads((RESULTS / 'analysis_test.json').read_text())
    with (RESULTS / 'measurements_test.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({(r['prompt'], r['repeat'], r['arm']) for r in rows}) == 240
    assert len({r['prompt'] for r in rows}) == 30
    for arm, result in summary['arms'].items():
        rr = [r for r in rows if r['arm'] == arm]
        rate = sum(int(r['tokens']) for r in rr) / sum(float(r['seconds']) for r in rr)
        assert math.isclose(rate, result['tokens_per_second'], rel_tol=1e-12), arm
    primary = summary['comparisons']['fixed_vs_old8m48']
    assert primary['exact_outputs'] == primary['total_outputs'] == 60
    assert primary['ci95'][0] > 1 and primary['speed_ratio'] < 1.03
    assert not config['target_3_percent_reached']
    audit = json.loads((RESULTS / 'completion_audit.json').read_text())
    assert audit['data']['stored'] <= audit['data']['cap'] == 20000
    print(json.dumps({'state': 'passed', 'scope': 'offline package and published aggregate checks; no new GPU benchmark', 'manifest_files': len(manifest), 'test_measurement_rows': len(rows), 'trained_weight_sha256_exact': True, 'runtime_bytes_match_measured_release': True}, indent=2))


if __name__ == '__main__':
    main()
