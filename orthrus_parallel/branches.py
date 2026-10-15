"""Branch identities and candidate construction."""
from dataclasses import dataclass
import torch

BLOCK_SIZE = 16


@dataclass
class BranchBank:
    boundaries: torch.Tensor
    anchors: torch.Tensor

    def match(self, boundary, anchor):
        matches = ((self.boundaries == boundary) & (self.anchors == anchor)).nonzero().flatten()
        return int(matches[0]) if len(matches) else None


def candidate_tokens(logits, proposals, count=64):
    """At a rejection the true token cannot equal the rejected greedy proposal.

    The final row predicts a possible bonus anchor, so no token is excluded there.
    """
    scores = logits[0].float().clone()
    scores[torch.arange(len(proposals), device=scores.device), proposals] = -torch.inf
    return scores.topk(count, dim=-1).indices


