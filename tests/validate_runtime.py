"""Real checkpoint regression plus an independent tiny-model causal/cache test.

Uses existing dependencies; no downloads. --reference-root optionally points
at the untouched original CV10 experiment to compare all six decoding modes.
"""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file')
    import torch
    from transformers import Qwen3Config
    from transformers.cache_utils import DynamicCache
    from orthrus_parallel.loading import load_model, encode_prompt
    from orthrus_parallel.joint import joint_forward, draft_forward, attention_mask
    from orthrus_parallel.selector import FixedSelector, LinearSelector, draft_features
    from orthrus_parallel.generation import generate, baseline
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
    selectors={name:FixedSelector(name) for name in ['fixed48','uniform80','all128']}
    selectors['selected48']=LinearSelector(ROOT/'selectors/zarya_06b','cuda',model.config.eos_token_id)
    legacy = None
    if args.reference_root:
        reference_root=args.reference_root.resolve()
        sys.path.insert(0,str(reference_root))
        import runtime_support  # Sets the original bundle import paths; does not load another model.
        import frozen_v2_runtime as frozen
        import context_controller as controller
        from native_executor import NativeExecutor
        from optimized_runtime import Reduced
        legacy=(frozen,controller,NativeExecutor(model),Reduced(reference_root/'selector06',model,compact=True),
                json.loads((reference_root/'design.json').read_text()),json.loads((reference_root/'bundle/cost_profile.json').read_text()))
    comparisons=[]
    finite=[0]
    def check_finite(module, inputs, output):
        assert bool(torch.isfinite(output).all())
        finite[0]+=1
    hook=model.lm_head.register_forward_hook(check_finite)
    try:
        for prompt,cap in [('Explain binary search with a short Python example.',96),
                           ('What is 17 times 23? Give the calculation and result.',80)]:
            ids=encode_prompt(tokenizer,prompt,'cuda')
            for name in ['baseline16','baseline32','fixed48','uniform80','all128','selected48']:
                model.config.block_size=32 if name=='baseline32' else 16
                if name.startswith('baseline'):
                    tokens=baseline(model,ids,int(name[-2:]),cap); stats={}
                else:
                    tokens,counters=generate(model,ids,selectors[name],cap,trace=True);stats=asdict(counters)
                assert 0<tokens.shape[1]<=cap
                if legacy:
                    frozen,controller,exe,heads,design,profile=legacy
                    with torch.inference_mode():
                        if name.startswith('baseline'):
                            old=model.generate(ids,max_new_tokens=cap,temperature=0.,use_diffusion_mode=True)[:,ids.shape[1]:]
                        elif name=='selected48':
                            old,_=controller.generate(exe,ids,{'kind':'branch_fixed','B':16,'L':16,'M':48,'K':8,'policy':'fixed'},heads,profile,cap=cap)
                        else:
                            old,_=frozen.generate(model,ids,max_new_tokens=cap,trace=True,**design['options']['sparse48' if name=='fixed48' else name])
                    assert torch.equal(tokens,old), (name,'refactor changed output tokens')
                comparisons.append({'method':name,'prompt':prompt,'tokens':tokens.shape[1],
                    'matches_original':True if legacy else None,'stats':stats})
        # Immediate EOS and short output limits exercise cache/termination boundaries.
        ids=encode_prompt(tokenizer,'Say hello.','cuda')
        original_eos=model.config.eos_token_id
        with torch.inference_mode(): first=int(model(input_ids=ids,logits_to_keep=1).logits[0,-1].argmax())
        try:
            model.config.eos_token_id=first
            values,_=generate(model,ids,selectors['selected48'],64)
            assert values.shape[1]==1
        finally:
            model.config.eos_token_id=original_eos
        for cap in [1,2,16,17,31,32,33]:
            values,_=generate(model,ids,selectors['selected48'],cap)
            assert 0<values.shape[1]<=cap
        ids=encode_prompt(tokenizer,'Explain how a cache works.','cuda')
        with torch.inference_mode():
            cache=DynamicCache(config=model.config)
            initial=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
            draft=draft_forward(model,initial.logits[:,-1].argmax(-1),cache,16)
            features=draft_features(draft)
            posterior=selectors['selected48'].posterior(features)
            assert features.shape==(16,17) and posterior.shape==(16,9)
            torch.testing.assert_close(posterior.sum(),torch.tensor(1.,device='cuda'))
            if legacy:
                torch.testing.assert_close(features,heads.rows(draft),rtol=0,atol=0)
                proposals=draft[0,:-1].argmax(-1)
                old_r,old_x,_=heads.choose(model,None,draft,proposals,{'M':48},None)
                new=selectors['selected48'].choose(draft,proposals)
                assert torch.equal(new.boundaries,old_r) and torch.equal(new.anchors,old_x)
    finally:
        hook.remove()
    result={'state':'passed','tiny_fp32_parity':errors,'real_fp16_generations':comparisons,
        'finite_lm_head_calls':finite[0],'short_caps':[1,2,16,17,31,32,33],
        'immediate_eos':'passed','selector_features_and_choices':'matched original' if legacy else 'shape/normalization passed',
        'scope':'Refactor correctness regression, not a new speed benchmark or a quality evaluation',
        'environment':{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'dtype':'FP16'}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('PASSED:',len(comparisons),'generations;',finite[0],'finite logits checks; tiny FP32 cache/branch parity')


if __name__=='__main__':main()
