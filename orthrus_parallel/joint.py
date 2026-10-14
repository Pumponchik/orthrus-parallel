"""One transformer traversal for causal verification and isolated draft branches.

Only causal AR keys/values enter the persistent cache. Branch keys/values
exist within this call; branches cannot attend to each other or affect AR rows.
"""
import sys
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache


def attention_mask(prefix_length, ar_length, boundaries, block_size):
    """Boolean SDPA mask: True means that the query may read that key."""
    device = boundaries.device
    branch_rows = boundaries.numel() * block_size
    total = ar_length + branch_rows
    mask = torch.zeros((total, prefix_length + total), dtype=torch.bool, device=device)
    ar_positions = torch.arange(prefix_length, prefix_length + ar_length, device=device)
    key_positions = torch.arange(prefix_length + ar_length, device=device)
    mask[:ar_length, :prefix_length + ar_length] = key_positions[None] <= ar_positions[:, None]
    branch_id = torch.arange(branch_rows, device=device) // block_size
    # A branch at r sees the current anchor and r proposed tokens, all via AR KV.
    cutoff = prefix_length + boundaries[branch_id] + 1
    mask[ar_length:, :prefix_length + ar_length] = key_positions[None] < cutoff[:, None]
    mask[ar_length:, prefix_length + ar_length:] = branch_id[:, None] == branch_id[None, :]
    return mask[None, None]


def _linear(module, values):
    """Keep the flattened projection order of the measured native implementation."""
    projected = F.linear(values.reshape(-1, values.shape[-1]), module.weight, module.bias)
    return projected.reshape(*values.shape[:-1], module.out_features)


def _project(attention, hidden, suffix):
    shape = (*hidden.shape[:-1], -1, attention.head_dim)
    query = getattr(attention, 'q_norm' + suffix)(
        _linear(getattr(attention, 'q_proj' + suffix), hidden).view(shape)).transpose(1, 2)
    key = getattr(attention, 'k_norm' + suffix)(
        _linear(getattr(attention, 'k_proj' + suffix), hidden).view(shape)).transpose(1, 2)
    value = _linear(getattr(attention, 'v_proj' + suffix), hidden).view(shape).transpose(1, 2)
    return query, key, value


def _sdpa(query, key, value, mask, scale):
    if query.shape[1] != key.shape[1]:
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                          dropout_p=0.0, scale=scale)


@torch.inference_mode()
def joint_forward(model, ar_ids, cache, boundaries, anchors, block_size=16, *, validate=True):
    """Return AR logits and branch hidden states; append only AR KV to cache.

    boundaries[i] = r: branch i starts AFTER ar_ids[:r+1]. Its first token
    is anchors[i], at position old_cache_length + r + 1. Valid r is 0..15
    for a full block. The anchor itself uses diffusion projections in a branch.
    """
    if model.training or ar_ids.ndim != 2 or ar_ids.shape[0] != 1 or ar_ids.shape[1] < 1:
        raise ValueError('Expected an eval model and a nonempty batch of one')
    if not isinstance(cache, DynamicCache) or model.model.has_sliding_layers:
        raise ValueError('Only DynamicCache and full-attention Orthrus layers are supported')
    if block_size < 1 or boundaries.ndim != 1 or anchors.shape != boundaries.shape or not boundaries.numel():
        raise ValueError('Expected a positive block size and nonempty paired branch tensors')
    if any(x.dtype not in (torch.int32, torch.int64) or x.device != ar_ids.device for x in (boundaries, anchors)):
        raise ValueError('Branch tensors must contain integer indices on the input device')
    ar_length = ar_ids.shape[1]
    if validate:
        if bool(((boundaries < 0) | (boundaries >= ar_length)).any()):
            raise ValueError('Boundary outside the current AR block')
        if bool(((anchors < 0) | (anchors >= model.config.vocab_size)).any()):
            raise ValueError('Anchor outside the vocabulary')
    prefix_length = cache.get_seq_length()
    branches = boundaries.numel()
    device = ar_ids.device
    draft_ids = torch.full((branches, block_size), model.config.mask_token_id,
                           dtype=torch.long, device=device)
    draft_ids[:, 0] = anchors
    ar_positions = torch.arange(prefix_length, prefix_length + ar_length, device=device)[None]
    draft_positions = prefix_length + boundaries[:, None] + 1 + torch.arange(block_size, device=device)[None]
    positions = torch.cat((ar_positions, draft_positions.reshape(1, -1)), dim=1)
    hidden = model.model.embed_tokens(torch.cat((ar_ids, draft_ids.reshape(1, -1)), dim=1))
    cos, sin = model.model.rotary_emb(hidden, positions)
    mask = attention_mask(prefix_length, ar_length, boundaries, block_size)
    rotary = sys.modules[type(model).__module__].apply_rotary_pos_emb
    total = ar_length + branches * block_size
    try:
        for index, layer in enumerate(model.model.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attention = layer.self_attn
            aq, ak, av = _project(attention, normalized[:, :ar_length], '')
            dq, dk, dv = _project(attention, normalized[:, ar_length:], '_diff')
            aq, ak = rotary(aq, ak, cos[:, :ar_length], sin[:, :ar_length])
            dq, dk = rotary(dq, dk, cos[:, ar_length:], sin[:, ar_length:])
            ak, av = cache.update(ak, av, index)
            attended = _sdpa(torch.cat((aq, dq), dim=2), torch.cat((ak, dk), dim=2),
                             torch.cat((av, dv), dim=2), mask, attention.scaling)
            attended = attended.transpose(1, 2).reshape(1, total, -1)
            hidden = residual + torch.cat((
                _linear(attention.o_proj, attended[:, :ar_length]),
                _linear(attention.o_proj_diff, attended[:, ar_length:])), dim=1)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = model.model.norm(hidden)
        return model.lm_head(hidden[:, :ar_length]), hidden[:, ar_length:].reshape(branches, block_size, -1)
    except Exception:
        cache.crop(prefix_length)
        raise


@torch.inference_mode()
def draft_forward(model, anchor, cache, block_size):
    """An ordinary diffusion draft, needed at startup and after a bank miss."""
    prefix_length = cache.get_seq_length()
    ids = torch.full((1, block_size), model.config.mask_token_id, device=anchor.device, dtype=torch.long)
    ids[:, 0] = anchor.reshape(-1)
    positions = torch.arange(prefix_length, prefix_length + block_size, device=anchor.device)[None]
    return model(input_ids=ids, position_ids=positions, past_key_values=cache,
                 use_cache=False, is_diffusion_pass=True, ar_seq_len=prefix_length).logits
