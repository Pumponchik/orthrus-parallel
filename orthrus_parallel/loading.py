"""Pinned checkpoint loading. Generation never silently downloads model files."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_config():
    return json.loads((ROOT / 'configs/checkpoint.json').read_text())


def default_checkpoint():
    config = checkpoint_config()
    return ROOT / 'models' / config['repo'].split('/')[-1] / config['revision']


def load_model(checkpoint=None, device='cuda', threads=2):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = Path(checkpoint) if checkpoint else default_checkpoint()
    config = checkpoint_config()
    for name, expected in [('model.safetensors', config['weights_sha256']),
                           ('modeling_orthrus.py', config['code_sha256'])]:
        if not (path / name).is_file():
            raise FileNotFoundError(f'Missing {path / name}; see scripts/download_model.py')
        if sha256(path / name) != expected:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261012)
    torch._dynamo.config.disable = True
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
        local_files_only=True, dtype=torch.float16, attn_implementation='sdpa').to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def encode_prompt(tokenizer, prompt, device):
    result = tokenizer.apply_chat_template([{'role': 'system', 'content': ''},
        {'role': 'user', 'content': prompt}], enable_thinking=False,
        add_generation_prompt=True, return_tensors='pt')
    return (result.input_ids if hasattr(result, 'input_ids') else result).to(device)
