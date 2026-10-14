"""Run with: ./lab run --locked python -m orthrus_parallel --prompt '...'"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description='Native Orthrus inference with optional branch selection')
    parser.add_argument('--method', choices=['baseline16', 'baseline32', 'fixed48', 'uniform80', 'all128', 'selected48'], default='selected48')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path, help='Existing local pinned checkpoint directory')
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument('--prompt')
    prompt.add_argument('--prompt-file', type=Path)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--output', type=Path, help='Optional new JSON file with tokens, timings and counters')
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    if args.max_new_tokens < 1 or (args.output and args.output.exists()):
        parser.error('Use a positive token limit and a new output path')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from .generation import baseline, generate
    from .loading import ROOT, load_model, encode_prompt
    from .selector import FixedSelector, LinearSelector
    model, tokenizer = load_model(args.checkpoint)
    text = args.prompt if args.prompt is not None else args.prompt_file.read_text()
    ids = encode_prompt(tokenizer, text, 'cuda')
    selector = None
    if not args.method.startswith('baseline'):
        selector = (LinearSelector(ROOT / 'selectors/zarya_06b', 'cuda', model.config.eos_token_id)
                    if args.method == 'selected48' else FixedSelector(args.method))
    torch.cuda.synchronize()
    started = time.perf_counter()
    if selector is None:
        output = baseline(model, ids, int(args.method.removeprefix('baseline')), args.max_new_tokens)
        stats = {}
    else:
        output, counters = generate(model, ids, selector, args.max_new_tokens, trace=args.trace)
        stats = asdict(counters)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    tokens = output[0].tolist()
    answer = tokenizer.decode(tokens, skip_special_tokens=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump({'method': args.method, 'text': answer, 'ids': tokens, 'tokens': len(tokens),
                'seconds': seconds, 'stats': stats, 'timing_scope': 'single cold request, not warmed benchmark'}, stream, ensure_ascii=False, indent=2)
    print(answer)


if __name__ == '__main__':
    main()
