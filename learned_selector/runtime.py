"""Greedy, batch-one decoding with optional precomputed branch continuations."""
from dataclasses import dataclass, field
import torch
from transformers.cache_utils import DynamicCache
from orthrus_parallel.joint import draft_forward, joint_forward
from orthrus_parallel.selector import BLOCK_SIZE

@dataclass
class GenerationStats:
    rounds: int = 0
    hits: int = 0
    misses: int = 0
    fresh_drafts: int = 0
    joint_forwards: int = 0
    ar_forwards: int = 0
    trace: list = field(default_factory=list)
    accepted_draft_tokens: int = 0
    branch_positions: int = 0
    bank_sizes: dict = field(default_factory=dict)
    continuing_eligible_cycles: int = 0

@torch.inference_mode()
def generate(model, input_ids, selector, capture, max_new_tokens=256, *, trace=False):
    """Return NEW token IDs and counters, including the initial AR-chosen token.

    A matching bank entry saves the NEXT separate draft forward. A miss runs
    an ordinary draft next cycle. Branch diffusion KV never enters AR cache.
    The newest committed token remains uncached and anchors the next cycle.
    """
    if max_new_tokens < 1 or input_ids.ndim != 2 or input_ids.shape[0] != 1 or (input_ids.shape[1] < 1):
        raise ValueError('Expected a nonempty batch of one and a positive output limit')
    if model.training:
        raise ValueError('Call model.eval() before generation')
    cache = DynamicCache(config=model.config)
    initial = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    anchor = initial.logits[:, -1].argmax(dim=-1)
    tokens = [int(anchor.item())]
    del initial
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pending = None
    context = capture.last[0, -1].clone() if selector.needs_context else None
    stats = GenerationStats()
    while len(tokens) < max_new_tokens and tokens[-1] not in eos:
        remaining = max_new_tokens - len(tokens)
        prefix_length = cache.get_seq_length()
        source = 'fresh' if pending is None else 'hit'
        if pending is None:
            length = min(BLOCK_SIZE, remaining + 1)
            draft = draft_forward(model, anchor, cache, length)
            hidden = capture.last if selector.needs_hidden else None
            stats.fresh_drafts += 1
        else:
            hidden = pending
            draft = model.lm_head(hidden)
            length = draft.shape[1]
            if length > remaining + 1:
                raise RuntimeError('A prepared block must not be shortened after conditioning')
        pending = None
        proposals = draft[0, :-1].argmax(dim=-1)
        ar_ids = torch.cat((anchor[:, None], proposals[None]), dim=1)
        selector.set_state(hidden, context, source, prefix_length, stats.rounds)
        bank = selector.choose(draft, proposals, source) if remaining >= length + BLOCK_SIZE else None
        if remaining >= length + BLOCK_SIZE:
            budget = str(bank.anchors.numel() if bank is not None else 0)
            stats.bank_sizes[budget] = stats.bank_sizes.get(budget, 0) + 1
        if bank is None:
            ar_logits = model(input_ids=ar_ids, past_key_values=cache, use_cache=True).logits
            stats.ar_forwards += 1
        else:
            ar_logits, branch_hidden = joint_forward(model, ar_ids, cache, bank.boundaries, bank.anchors, BLOCK_SIZE, validate=False)
            stats.branch_positions += bank.anchors.numel() * BLOCK_SIZE
            stats.joint_forwards += 1
        ar_hidden = capture.last[:, :length] if selector.needs_context else None
        targets = ar_logits[0].argmax(dim=-1)
        accepted = int((targets[:-1] == proposals).long().cumprod(dim=0).sum().item())
        correction = targets[accepted]
        issued = (proposals[:accepted].tolist() + [int(correction)])[:remaining]
        before = len(tokens)
        for token in issued:
            tokens.append(token)
            if token in eos:
                break
        cache.crop(prefix_length + len(tokens) - before)
        if selector.needs_context:
            context = ar_hidden[0, len(tokens) - before - 1].clone()
        stats.accepted_draft_tokens += min(accepted, len(tokens) - before)
        anchor = correction.reshape(1)
        hit = False
        if remaining >= length + BLOCK_SIZE and len(tokens) < max_new_tokens and (tokens[-1] not in eos):
            stats.continuing_eligible_cycles += 1
        if bank is not None and len(tokens) < max_new_tokens and (tokens[-1] not in eos):
            selected = bank.match(accepted, correction)
            hit = selected is not None
            if hit:
                pending = branch_hidden[selected][None]
                stats.hits += 1
            else:
                stats.misses += 1
        stats.rounds += 1
        if trace:
            stats.trace.append({'accepted': accepted, 'anchor': int(correction), 'source': source, 'hit': hit, 'bank': bank is not None, 'branches': 0 if bank is None else bank.anchors.numel(), 'emitted': len(tokens) - before, 'cache_length': cache.get_seq_length()})
    if cache.get_seq_length() != input_ids.shape[1] + len(tokens) - 1:
        raise RuntimeError('AR cache length does not match the emitted prefix')
    return (torch.tensor([tokens], device=input_ids.device), stats)
