"""Check causal cache, branch equivalence, and the released selector."""
import argparse
import copy
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    args = p.parse_args()
    import torch
    from transformers import Qwen3Config
    from transformers.cache_utils import DynamicCache
    from orthrus_parallel.loading import load_model, encode_prompt, ROOT
    from orthrus_parallel.joint import joint_forward, draft_forward, attention_mask
    from orthrus_parallel.branches import candidate_tokens
    from orthrus_parallel.selector import Selector
    from orthrus_parallel.generation import generate
    model, tokenizer = load_model(args.checkpoint)
    cfg = Qwen3Config(vocab_size=31, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, layer_types=['full_attention']*2,
        block_size=4, mask_token_id=30, eos_token_id=None, pad_token_id=0, tie_word_embeddings=False)
    cfg._attn_implementation = 'sdpa'
    tiny = type(model)(cfg).float().eval()
    errors = []
    with torch.inference_mode():
        prefix = DynamicCache(config=cfg)
        tiny(input_ids=torch.tensor([[1, 2, 3]]), past_key_values=prefix, use_cache=True)
        for n, length in [(4, 4), (16, 16)]:
            ids = torch.arange(4, 4+n)[None]
            reference = copy.deepcopy(prefix)
            expected = tiny(input_ids=ids, past_key_values=reference, use_cache=True).logits
            cache = copy.deepcopy(prefix)
            boundaries = torch.arange(n)
            anchors = torch.arange(n) + 2
            actual, hidden = joint_forward(tiny, ids, cache, boundaries, anchors, length)
            ar_error = float((actual-expected).abs().max())
            kv_error = max(float((getattr(x,k)-getattr(y,k)).abs().max())
                for x,y in zip(cache.layers,reference.layers) for k in ['keys','values'])
            branch_error = 0.
            for j in range(n):
                branch_prefix = copy.deepcopy(reference)
                branch_prefix.crop(3+j+1)
                target = draft_forward(tiny, anchors[j:j+1], branch_prefix, length)
                branch_error = max(branch_error,float((tiny.lm_head(hidden[j:j+1])-target).abs().max()))
            assert max(ar_error,kv_error,branch_error)<3e-5
            mask=attention_mask(3,n,boundaries,length)[0,0]
            assert not mask[:n,3+n:].any()  # AR never reads diffusion keys.
            assert not mask[n:n+length,3+n+length:].any()  # First branch cannot read siblings.
            errors.append({'rows':n,'ar_error':ar_error,'kv_error':kv_error,'branch_error':branch_error})

    model.config.block_size = 16
    selector = Selector(ROOT / 'weights', 'cuda', model.config.eos_token_id)
    class Capture:
        last = None
        def hook(self, module, inputs, output):
            self.last = output
    capture = Capture()
    handle = model.model.norm.register_forward_hook(capture.hook)
    try:
        ids = encode_prompt(tokenizer, 'Explain binary search briefly.', 'cuda')
        cache = DynamicCache(config=model.config)
        with torch.inference_mode():
            initial = model(input_ids=ids, past_key_values=cache, use_cache=True)
            anchor = initial.logits[:, -1].argmax(-1)
            logits = draft_forward(model, anchor, cache, 16)
            proposals = logits[0, :-1].argmax(-1)
            wide = candidate_tokens(logits, proposals, 64)
            for count in [1, 4, 8, 16, 32]:
                assert torch.equal(wide[:, :count], candidate_tokens(logits, proposals, count))
            selector.set_state(capture.last, None, 'fresh', ids.shape[1], 0)
            bank = selector.choose(logits, proposals, 'fresh')
            assert bank.anchors.numel() == 48
            assert bool((wide[bank.boundaries, :32] == bank.anchors[:, None]).any(-1).all())
        for cap in [1, 16, 17, 31, 32, 33, 96]:
            tokens, stats = generate(model, ids, selector, capture, cap)
            assert 0 < tokens.shape[1] <= cap
        print({'state': 'passed', 'branch_equivalence': errors})
    finally:
        handle.remove()

if __name__ == '__main__':
    main()
