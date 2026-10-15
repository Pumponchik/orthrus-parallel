"""Run the confirmed default selector from best_selector.json, with pinned model loading."""
import argparse, json, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
pg = p.add_mutually_exclusive_group(required=True)
pg.add_argument('--prompt')
pg.add_argument('--prompt-file', type=Path)
p.add_argument('--max-new-tokens', type=int, default=256)
p.add_argument('--output', type=Path)
a = p.parse_args()
import torch
from orthrus_parallel.loading import load_model, encode_prompt
from orthrus_parallel.selector import LinearSelector
from orthrus_parallel.generation import generate as old_generate
from runtime import generate
from selection import Selector
from dataclasses import asdict
ROOT = Path(__file__).resolve().parent
config = json.load(open(ROOT / 'best_selector.json'))
model, tok = load_model(a.checkpoint)
model.config.block_size = 16
prompt = a.prompt if a.prompt is not None else a.prompt_file.read_text()
ids = encode_prompt(tok, prompt, 'cuda')
if config['kind'] == 'old':
    selector = LinearSelector(ROOT / 'selectors/zarya_06b', 'cuda', model.config.eos_token_id)
else:
    selector = Selector(ROOT, config['kind'], config['name'], model, 'cuda', model.config.eos_token_id, config['m'])
    selector.k = config['k']
    selector.adaptive = config.get('adaptive', False)
    if selector.adaptive:
        selector.bank_costs = config['bank_costs']
        selector.draft_cost = config['draft_cost']
        selector.temperature = config['temperature']

class Capture:

    def __init__(self):
        self.last = None

    def hook(self, m, i, o):
        self.last = o
capture = Capture()
handle = model.model.norm.register_forward_hook(capture.hook)
torch.cuda.synchronize()
t = time.perf_counter()
if config['kind'] == 'old':
    tokens, stats = old_generate(model, ids, selector, a.max_new_tokens)
else:
    tokens, stats = generate(model, ids, selector, capture, a.max_new_tokens)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t
result = {'ids': tokens[0].tolist(), 'text': tok.decode(tokens[0], skip_special_tokens=True), 'tokens': tokens.shape[1], 'seconds': elapsed, 'tokens_per_second': tokens.shape[1] / elapsed, 'selector': config, 'stats': asdict(stats)}
if a.output:
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(result['text'])
