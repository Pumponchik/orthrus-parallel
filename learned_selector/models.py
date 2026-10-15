"""Candidate scorers. Each model produces 16*64 pair scores and OUTSIDE."""
import torch
from torch import nn

class Scorer(nn.Module):

    def __init__(self, mode='numeric', width=0):
        super().__init__()
        self.mode = mode
        self.width = width
        dim = {'numeric': 26, 'hidden': 26 + 64 + 3, 'context': 26 + 128 + 6}[mode]
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        self.net = nn.Sequential(nn.Linear(dim, width), nn.SiLU(), nn.Linear(width, 1)) if width else nn.Linear(dim, 1)
        self.outside = nn.Linear(26, 1)

    def inputs(self, x, h, c, ids, embedding):
        if self.mode == 'numeric':
            return x
        e = embedding[ids].float()
        hp = h[:, :, None, :].expand_as(e)
        prod = hp * e / 8
        norms = torch.stack((hp.norm(dim=-1) / 32, e.norm(dim=-1) / 4, (hp * e).sum(-1) / 32), dim=-1)
        values = [x, prod, norms]
        if self.mode == 'context':
            cp = c[:, None, None, :].expand_as(e)
            values.extend((cp * e / 8, torch.stack((cp.norm(dim=-1) / 32, (cp * e).sum(-1) / 32, (cp * hp).sum(-1) / 256), dim=-1)))
        return torch.cat(values, dim=-1)

    def forward(self, x, h, c, ids, embedding):
        features = self.inputs(x, h, c, ids, embedding)
        scores = self.net((features - self.mean) / self.scale).squeeze(-1).flatten(1)
        outside = self.outside(x.mean((1, 2)))
        return torch.cat((scores, outside), dim=-1)

class Analytic(nn.Module):
    """Calibrate existing hazard and allowable draft-token probability jointly."""

    def __init__(self, hazard):
        super().__init__()
        self.hazard = hazard
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.boundary_bias = nn.Parameter(torch.zeros(16))
        self.outside_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, *unused):
        f = x[:, :, 0, :17]
        a = self.hazard(f[:, :-1])[:, :, 1]
        survival = torch.cat((torch.ones_like(a[:, :1]), a.cumprod(1)), 1)
        boundary = torch.cat((survival[:, :-1] * (1 - a), survival[:, -1:]), 1).clamp_min(1e-09)
        conditional = x[:, :, :, 18] * 20
        raw = boundary.log()[:, :, None] + self.boundary_bias[None, :, None] + conditional / self.log_temperature.exp().clamp(0.3, 3)
        mass = conditional.exp().sum(-1).clamp(max=1 - 1e-06)
        outside = (boundary * (1 - mass)).sum(-1).clamp_min(1e-09).log() + self.outside_bias
        return torch.cat((raw.flatten(1), outside[:, None]), 1)

class FrozenHazard(nn.Module):

    def __init__(self, arrays):
        super().__init__()
        for k, v in arrays.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def forward(self, f):
        return ((f - self.mean) / self.scale @ self.p0 + self.p1).softmax(-1)

class BilinearScorer(nn.Module):
    """Numeric MLP plus learned cross-component compatibility of context and token."""

    def __init__(self, mode='hidden', width=64):
        super().__init__()
        self.mode = mode
        self.width = width
        self.register_buffer('mean', torch.zeros(26))
        self.register_buffer('scale', torch.ones(26))
        self.net = nn.Sequential(nn.Linear(26, width), nn.SiLU(), nn.Linear(width, 1))
        self.query_hidden = nn.Linear(64, 64, bias=False)
        self.query_context = nn.Linear(64, 64, bias=False) if mode == 'context' else None
        self.token = nn.Linear(64, 64, bias=False)
        self.outside = nn.Linear(26, 1)

    def inputs(self, x, h, c, ids, embedding):
        return x

    def forward(self, x, h, c, ids, embedding):
        q = self.query_hidden(torch.nn.functional.layer_norm(h, (64,)))
        if self.query_context is not None:
            q = q + self.query_context(torch.nn.functional.layer_norm(c, (64,)))[:, None, :]
        e = self.token(torch.nn.functional.layer_norm(embedding[ids].float(), (64,)))
        compatibility = (q[:, :, None, :] * e).sum(-1) / 8
        score = self.net((x - self.mean) / self.scale).squeeze(-1) + compatibility
        return torch.cat((score.flatten(1), self.outside(x.mean((1, 2)))), 1)
