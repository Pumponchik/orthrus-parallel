"""Select (accepted-prefix length, anchor-token) pairs BEFORE AR verification."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch

BLOCK_SIZE = 16
TOP_ANCHORS = 8
# Historical schedules. Each row sums to 48; they are not retrained for 0.6B.
FIXED_COUNTS = {
    'fresh': (5, 8, 8, 4, 6, 1, 8, 2, 0, 0, 0, 1, 0, 2, 1, 2),
    'hit': (3, 7, 7, 7, 6, 2, 3, 1, 3, 1, 1, 2, 1, 1, 1, 2),
}


@dataclass
class BranchBank:
    boundaries: torch.Tensor
    anchors: torch.Tensor

    def match(self, boundary, anchor):
        matches = ((self.boundaries == boundary) & (self.anchors == anchor)).nonzero().flatten()
        return int(matches[0]) if len(matches) else None


def candidate_tokens(logits, proposals, count=TOP_ANCHORS):
    """At a rejection the true token cannot equal the rejected greedy proposal.

    The final row predicts a possible bonus anchor, so no token is excluded there.
    """
    scores = logits[0].float().clone()
    scores[torch.arange(len(proposals), device=scores.device), proposals] = -torch.inf
    return scores.topk(count, dim=-1).indices


def draft_features(logits):
    """Return [16, 17] features. The order is part of the saved-weight format.

    [B/32, r/15, last-row flag, top-10 probabilities, top1-top2 gap,
     mean top1 probability, mean gap, last-row gap]. B is fixed to 16.
    Probabilities are normalized over the ORIGINAL whole vocabulary, before
    excluding proposed tokens for anchor candidates. No context encoding pass.
    """
    if logits.ndim != 3 or logits.shape[:2] != (1, BLOCK_SIZE):
        raise ValueError('The released selector expects draft logits [1, 16, vocabulary]')
    scores = logits[0].float()
    values = scores.topk(10, dim=-1).values
    top = (values - scores.logsumexp(dim=-1, keepdim=True)).exp()
    gap = top[:, 0] - top[:, 1]
    positions = torch.arange(BLOCK_SIZE, device=scores.device, dtype=torch.float32)
    positional = torch.stack((torch.full_like(positions, .5), positions / 15, (positions == 15).float()), dim=-1)
    global_features = torch.stack((top[:, 0].mean(), gap.mean(), gap[-1])).expand(BLOCK_SIZE, -1)
    return torch.cat((positional, top, gap[:, None], global_features), dim=-1)


class LinearSelector:
    """Two affine softmax classifiers; their joint posterior ranks 128 pairs.

    The hazard classifier has two classes (stop/accept), the rank classifier
    has nine (top eight alternative tokens, plus OUTSIDE). No MLP or RNN.
    """
    def __init__(self, weights, device, eos_token_id):
        path = Path(weights)
        self.device = torch.device(device)
        selection = json.loads((path / 'selection.json').read_text())
        if selection['dimensions'] != 17:
            raise ValueError('This runtime supports the 17-feature selector only')
        self.temperature = selection['temperature']
        if self.temperature <= 0:
            raise ValueError('Temperature must be positive')
        self.models = {}
        for name, classes in [('hazard', 2), ('rank', 9)]:
            with np.load(path / f'linear_{name}.npz', allow_pickle=False) as data:
                model = {key: torch.tensor(data[key], device=self.device, dtype=torch.float32) for key in data.files}
            if model['p0'].shape != (17, classes) or model['p1'].shape != (classes,):
                raise ValueError(f'Unexpected {name} weight shape')
            if not all(bool(torch.isfinite(x).all()) for x in model.values()) or bool((model['scale'] <= 0).any()):
                raise ValueError(f'Invalid {name} weights or normalization')
            self.models[name] = model
        self.one = torch.ones(1, device=self.device)
        self.eos = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]

    def predict(self, name, features):
        model = self.models[name]
        standardized = (features - model['mean']) / model['scale']
        return (standardized @ model['p0'] + model['p1']).softmax(dim=-1)

    def posterior(self, features):
        acceptance = self.predict('hazard', features[:-1])[:, 1]
        survival = torch.cat((self.one, acceptance.cumprod(dim=0)))
        boundary = torch.cat((survival[:-1] * (1 - acceptance), survival[-1:]))
        joint = boundary[:, None] * self.predict('rank', features)
        # Temperature calibration uses all 16x9 outcomes, including OUTSIDE.
        return (joint.clamp(min=1e-12).log().flatten() / self.temperature).softmax(dim=-1).reshape(16, 9)

    def choose(self, logits, proposals, source=None):
        candidates = candidate_tokens(logits, proposals)
        probabilities = self.posterior(draft_features(logits))[:, :TOP_ANCHORS]
        for token in self.eos:
            if token is not None:
                probabilities = probabilities.masked_fill(candidates == token, 0)
        order = probabilities.flatten().sort(descending=True, stable=True).indices[:48]
        boundaries, ranks = order // TOP_ANCHORS, order % TOP_ANCHORS
        return BranchBank(boundaries, candidates[boundaries, ranks])


class FixedSelector:
    def __init__(self, method):
        if method not in ('fixed48', 'uniform80', 'all128'):
            raise ValueError(method)
        self.method = method

    def choose(self, logits, proposals, source):
        count = 5 if self.method == 'uniform80' else 8
        candidates = candidate_tokens(logits, proposals, count)
        boundaries = torch.arange(BLOCK_SIZE, device=logits.device)[:, None].expand_as(candidates)
        boundaries, anchors = boundaries.reshape(-1), candidates.reshape(-1)
        if self.method == 'fixed48':
            keep = torch.tensor([r * 8 + k for r, n in enumerate(FIXED_COUNTS[source]) for k in range(n)],
                                device=logits.device, dtype=torch.long)
            boundaries, anchors = boundaries[keep], anchors[keep]
        return BranchBank(boundaries, anchors)
