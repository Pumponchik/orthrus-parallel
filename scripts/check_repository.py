"""Offline standard-library audit: syntax, links, measured data and preserved history."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_repository import ROOT, repository_files


def main():
    files = list(repository_files())
    for path in files:
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        if path.suffix == '.md':
            text = path.read_text()
            if text.count('```') % 2:
                raise ValueError(f'Unclosed code fence: {path}')
            for target in re.findall(r'\]\(([^\s)]+)\)',text):
                if '://' not in target and not target.startswith('#'):
                    if not (path.parent / target.split('#')[0]).exists():
                        raise FileNotFoundError((path,target))
        if path.suffix in {'.py','.md','.json','.csv'}:
            text = path.read_text()
            for forbidden in ['/Users/phimosis/','/home/phimosis/','zombie.yandex.net','ai.yandex-team.ru']:
                # This audit's own list of strings is intentional.
                if path.name != 'check_repository.py' and forbidden in text:
                    raise ValueError(f'Personal host/path in export: {path}')
    summary = json.loads((ROOT/'results/cv10_20261012/summary.json').read_text())
    with (ROOT/'results/cv10_20261012/measurements.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 288 or len({(x['prompt'],x['repeat'],x['arm']) for x in rows}) != 288:
        raise ValueError('Incomplete historical measurement table')
    for arm, value in summary['arms'].items():
        matching = [r for r in rows if r['arm']==arm]
        rate = sum(int(r['tokens']) for r in matching) / sum(float(r['seconds']) for r in matching)
        if not math.isclose(rate,value['tokens_per_second'],rel_tol=1e-12):
            raise ValueError(f'Inconsistent measured rate: {arm}')
    validation = json.loads((ROOT/'results/refactor_validation.json').read_text())
    assert validation['state']=='passed'
    assert all(row['matches_original'] for row in validation['real_fp16_generations'])
    assert len(validation['real_fp16_generations'])==12
    assert validation['selector_features_and_choices']=='matched original'
    source_receipt = json.loads((ROOT/'results/source_validation.json').read_text())
    assert source_receipt['state']=='passed'
    for rel, digest in source_receipt['source_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest, rel
    training = json.loads((ROOT/'results/training_validation.json').read_text())
    assert training['state']=='passed'
    assert all(error==0 for head in training['maximum_parameter_errors'].values() for error in head.values())
    relocation = ROOT/'archive/relocation.json'
    archived = 0
    if relocation.exists():
        record = json.loads(relocation.read_text())
        for rel,digest in record['file_sha256'].items():
            actual = hashlib.sha256((ROOT/record['new_root']/rel).read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f'Historical artifact changed: {rel}')
            archived += 1
    # Working code must remain independent of the local research archive.
    for path in (ROOT/'orthrus_parallel').glob('*.py'):
        if 'archive/' in path.read_text() or 'sys.path' in path.read_text():
            raise ValueError(f'Runtime depends on local history: {path}')
    print(json.dumps({'state':'passed','repository_files':len(files),'measurement_rows':len(rows),
        'real_model_regressions':12,'preserved_historical_files':archived},indent=2))


if __name__ == '__main__':
    main()
