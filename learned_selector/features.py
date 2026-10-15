"""Pre-verify features sharing one vocabulary reduction and one candidate top-k."""
import torch
from orthrus_parallel.selector import candidate_tokens

def build(logits, source, prefix_length, cycle):
    values = logits[0].float()
    proposals = logits[0, :-1].argmax(-1)
    candidates = candidate_tokens(logits, proposals, 64)
    candidate_values = values.gather(1, candidates)
    proposed_values = values.gather(1, values.argmax(-1, keepdim=True))
    top_values = torch.cat((candidate_values, proposed_values), -1).topk(10, dim=-1).values
    top_values[-1] = candidate_values[-1, :10]
    logz = values.logsumexp(-1, keepdim=True)
    top = (top_values - logz).exp()
    clp = candidate_values - logz
    gap = top[:, 0] - top[:, 1]
    pos = torch.arange(16, device=values.device, dtype=torch.float32)
    global_features = torch.stack((top[:, 0].mean(), gap.mean(), gap[-1])).expand(16, -1)
    f = torch.cat((torch.stack((torch.full_like(pos, 0.5), pos / 15, (pos == 15).float()), -1), top, gap[:, None], global_features), -1)
    row = pos[:, None].expand(16, 64)
    rank = torch.arange(64, device=values.device, dtype=torch.float32)[None].expand(16, 64)
    log_reject = torch.log1p(-f[:, 3].clamp(max=1 - 1e-07))
    log_reject[-1] = 0
    conditional = clp - log_reject[:, None]
    extras = torch.stack((torch.full_like(row, float(source == 'hit')), torch.full_like(row, prefix_length / 1024), torch.full_like(row, cycle / 64)), -1)
    candidate = torch.stack((clp / 20, conditional / 20, rank / 63, torch.log1p(rank) / 4.2, clp - clp[:, :1], clp.exp()), -1)
    return {'x': torch.cat((f[:, None].expand(16, 64, 17), candidate, extras), -1), 'rows': f, 'ids': candidates, 'logp': clp, 'proposals': proposals}
