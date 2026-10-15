"""Rank candidate (accepted prefix, next token) pairs before verification."""
from pathlib import Path
import json
from .integrity import checked_path, sha256
import numpy as np
import torch
from .models import BilinearScorer
from .features import build
from .branches import BranchBank

class Selector:
    def __init__(self, root, device, eos, m=48, k=32):
        self.root = Path(root)
        self.device = device
        self.eos = eos if isinstance(eos, list) else [eos]
        self.m, self.k = m, k
        self.needs_hidden, self.needs_context = True, False
        manifest = json.loads((self.root / 'manifest.json').read_text())
        for name in ('selector.pt', 'embedding_projected.npy', 'projection.npy'):
            path = checked_path(self.root, name)
            if sha256(path) != manifest['weights/' + name]:
                raise ValueError(f'Selector hash mismatch: {name}')
        ck = torch.load(self.root / 'selector.pt', map_location=device, weights_only=True)
        self.model = BilinearScorer(ck['mode'], ck['width']).to(device).eval()
        self.model.load_state_dict(ck['state_dict'])
        self.embedding = torch.tensor(np.load(self.root / 'embedding_projected.npy', allow_pickle=False), device=device, dtype=torch.float32)
        self.projection = torch.tensor(np.load(self.root / 'projection.npy', allow_pickle=False), device=device, dtype=torch.float32)

    def set_state(self, hidden, context, source, prefix, cycle):
        self.state = (hidden, context, source, prefix, cycle)

    @torch.inference_mode()
    def score(self, logits):
        hidden, context, source, prefix, cycle = self.state
        pack = build(logits, source, prefix, cycle)
        x = pack['x'][None]
        ids = pack['ids'][None]
        h = (hidden[0].float() @ self.projection)[None] if self.needs_hidden else torch.zeros((1, 16, 64), device=self.device)
        c = (context.float() @ self.projection)[None] if self.needs_context else torch.zeros((1, 64), device=self.device)
        s = self.model(x, h, c, ids, self.embedding)[0]
        return (s, pack['ids'])

    def choose(self, logits, proposals, source):
        scores, candidates = self.score(logits)
        pair = scores[:-1].reshape(16, 64)
        for token in self.eos:
            if token is not None:
                pair = pair.masked_fill(candidates == token, -torch.inf)
        if self.k < 64:
            pair[:, self.k:] = -torch.inf
        order = pair.flatten().sort(descending=True, stable=True).indices
        m = self.m
        if m == 0:
            return None
        keep = order[:m]
        r, k = (keep // 64, keep % 64)
        return BranchBank(r, candidates[r, k])

