"""Shared model session for the CLI and paired benchmark."""
from dataclasses import asdict
import json
import time
import threading
import torch
from .generation import baseline, generate
from .loading import ROOT, load_model, encode_prompt
from .selector import Selector

class Session:
    def __init__(self, checkpoint):
        self._lock = threading.RLock()
        self._closed = False
        self.model, self.tokenizer = load_model(checkpoint)
        self.model.config.block_size = 16
        config = json.loads((ROOT / 'configs/selector.json').read_text())
        self.selector = Selector(ROOT / 'weights', 'cuda', self.model.config.eos_token_id,
                                 m=config['branches'], k=config['top_anchors'])
        self.last = None

    def capture(self, module, inputs, output):
        self.last = output

    def run(self, prompt, method='parallel', max_new_tokens=256):
        """Serialize requests on this batch-one model; close waits for active work."""
        with self._lock:
            if self._closed:
                raise RuntimeError('Session is closed')
            handle = self.model.model.norm.register_forward_hook(self.capture)
            try:
                return self._run(prompt, method, max_new_tokens)
            finally:
                handle.remove()
                self.last = None
                self.selector.state = None

    def _run(self, prompt, method, max_new_tokens):
        ids = encode_prompt(self.tokenizer, prompt, 'cuda')
        if method not in ('baseline', 'parallel'):
            raise ValueError(method)
        torch.cuda.synchronize()
        started = time.perf_counter()
        if method == 'baseline':
            output = baseline(self.model, ids, 16, max_new_tokens)
            counters = {}
        else:
            output, stats = generate(self.model, ids, self.selector, self, max_new_tokens)
            counters = asdict(stats)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        tokens = output[0].tolist()
        return {'method': method, 'ids': tokens,
                'text': self.tokenizer.decode(tokens, skip_special_tokens=True),
                'tokens': len(tokens), 'seconds': seconds,
                'tokens_per_second': len(tokens) / seconds, 'stats': counters}

    def close(self):
        with self._lock:
            self._closed = True
            self.last = None
            self.selector.state = None

