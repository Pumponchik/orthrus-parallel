"""Paired, warmed end-to-end benchmark. Run from the repository root."""
import argparse
import json
from pathlib import Path
import random

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--prompts', type=Path, default=Path('benchmarks/prompts.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--max-new-tokens', type=int, default=256)
    a = p.parse_args()
    if a.output.exists() or a.max_new_tokens < 1 or (a.limit is not None and a.limit < 1):
        p.error('Choose a new output file and positive limits')
    from orthrus_parallel.session import Session
    prompts = json.loads(a.prompts.read_text())[:a.limit]
    if not prompts:
        p.error('Empty prompt set')
    session = Session(a.checkpoint)
    rng = random.Random(20261015)
    try:
        for mode in ['baseline', 'parallel']:
            session.run(prompts[0]['prompt'], mode, a.max_new_tokens)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('x') as f:
            for prompt in prompts:
                order = ['baseline', 'parallel']
                rng.shuffle(order)
                for repeat in range(2):
                    for mode in order if repeat == 0 else reversed(order):
                        result = session.run(prompt['prompt'], mode, a.max_new_tokens)
                        result.update(prompt_id=prompt['id'], domain=prompt['domain'], repeat=repeat)
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
    finally:
        session.close()

if __name__ == '__main__':
    main()
