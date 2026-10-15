"""Learned pair scorer for the released branch selector."""
import torch
from torch import nn

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
