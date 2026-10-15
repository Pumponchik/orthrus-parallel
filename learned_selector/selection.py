"""Deployable selectors using only pre-verification state."""
from pathlib import Path
import numpy as np
import torch
from models import Scorer, Analytic, FrozenHazard, BilinearScorer
from features import build
from orthrus_parallel.selector import BranchBank, LinearSelector, draft_features

class Selector:

    def __init__(self, root, kind, name, model, device, eos, m=32):
        self.root = Path(root)
        self.kind = kind
        self.name = name
        self.device = device
        self.eos = eos if isinstance(eos, list) else [eos]
        self.m = m
        self.k = 64
        self.adaptive = False
        self.temperature = 1.0
        self.needs_hidden = False
        self.needs_context = False
        self.embedding = None
        self.projection = None
        if kind in ['neural', 'bilinear']:
            folder = 'trained' if kind == 'neural' else 'trained_bilinear'
            ck = torch.load(self.root / folder / f'{name}.pt', map_location=device, weights_only=False)
            cls = Scorer if kind == 'neural' else BilinearScorer
            self.model = cls(ck['mode'], ck['width']).to(device).eval()
            self.model.load_state_dict(ck['state_dict'])
            self.needs_hidden = ck['mode'] != 'numeric'
            self.needs_context = ck['mode'] == 'context'
        elif kind == 'analytic':
            ck = torch.load(self.root / 'trained/analytic.pt', map_location=device, weights_only=False)
            self.model = Analytic(FrozenHazard(ck['hazard'])).to(device).eval()
            self.model.load_state_dict(ck['state_dict'])
        elif kind == 'factorized':
            self.parts = {}
            for head in ['hazard', 'rank']:
                with np.load(self.root / 'trained' / f'refit_{head}.npz') as d:
                    self.parts[head] = {k: torch.tensor(d[k], device=device, dtype=torch.float32) for k in d.files}
        elif kind == 'catboost':
            from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor
            cls = {'classifier': CatBoostClassifier, 'ranker': CatBoostRanker, 'regressor': CatBoostRegressor}[name]
            self.model = cls()
            self.model.load_model(str(self.root / 'catboost_models' / f'{name}.cbm'))
        else:
            raise ValueError(kind)
        if self.needs_hidden:
            self.embedding = torch.tensor(np.load(self.root / 'artifacts/embedding_projected.npy'), device=device, dtype=torch.float32)
            self.projection = torch.tensor(np.load(self.root / 'artifacts/projection.npy'), device=device, dtype=torch.float32)

    def set_state(self, hidden, context, source, prefix, cycle):
        self.state = (hidden, context, source, prefix, cycle)

    def part(self, name, x):
        d = self.parts[name]
        return ((x - d['mean']) / d['scale'] @ d['p0'] + d['p1']).softmax(-1)

    @torch.inference_mode()
    def score(self, logits):
        hidden, context, source, prefix, cycle = self.state
        pack = build(logits, source, prefix, cycle)
        x = pack['x'][None]
        ids = pack['ids'][None]
        if self.kind in ['neural', 'bilinear']:
            h = (hidden[0].float() @ self.projection)[None] if self.needs_hidden else torch.zeros((1, 16, 64), device=self.device)
            c = (context.float() @ self.projection)[None] if self.needs_context else torch.zeros((1, 64), device=self.device)
            s = self.model(x, h, c, ids, self.embedding)[0]
        elif self.kind == 'analytic':
            s = self.model(x)[0]
        elif self.kind == 'factorized':
            f = pack['rows']
            a = self.part('hazard', f[:-1])[:, 1]
            sur = torch.cat((torch.ones(1, device=self.device), a.cumprod(0)))
            b = torch.cat((sur[:-1] * (1 - a), sur[-1:]))
            rank = self.part('rank', f)
            s = torch.cat(((b[:, None] * rank[:, :64]).flatten(), (b * rank[:, 64]).sum()[None])).clamp_min(1e-12).log()
        else:
            raw = x[0].reshape(1024, 26).cpu().numpy()
            outside = np.r_[raw.mean(0), 1].astype(np.float32)
            rows = np.concatenate((np.pad(raw, ((0, 0), (0, 1))), outside[None]))
            if self.name == 'classifier':
                out = self.model.predict(rows, prediction_type='RawFormulaVal', thread_count=2)
            elif self.name == 'regressor':
                out = np.log(np.clip(self.model.predict(rows, thread_count=2), 1e-09, 1))
            else:
                out = self.model.predict(rows, thread_count=2)
            s = torch.tensor(out, device=self.device, dtype=torch.float32)
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
        if self.adaptive:
            probs = (scores / self.temperature).softmax(0)
            cum = probs[order].cumsum(0)
            budgets = torch.tensor([0, 16, 24, 32, 48], device=self.device)
            mass = torch.cat((torch.zeros(1, device=self.device), cum[budgets[1:] - 1]))
            cost = torch.tensor(self.bank_costs, device=self.device) - mass * self.draft_cost
            m = int(budgets[cost.argmin()])
        if m == 0:
            return None
        keep = order[:m]
        r, k = (keep // 64, keep % 64)
        return BranchBank(r, candidates[r, k])
