"""Generate with the released selector or ordinary Orthrus."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--method', choices=['parallel', 'baseline'], default='parallel')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompt')
    group.add_argument('--prompt-file', type=Path)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.max_new_tokens < 1 or (a.output and a.output.exists()):
        p.error('Use a positive output limit and a new output path')
    from .session import Session
    session = Session(a.checkpoint)
    try:
        prompt = a.prompt if a.prompt is not None else a.prompt_file.read_text()
        result = session.run(prompt, a.method, a.max_new_tokens)
        if a.output:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            with a.output.open('x') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        print(result['text'])
    finally:
        session.close()

if __name__ == '__main__':
    main()
