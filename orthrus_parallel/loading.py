"""Pinned checkpoint loading. Generation never silently downloads model files."""
import json
from pathlib import Path
from .integrity import checked_path, sha256, verify_file

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_config():
    return json.loads((ROOT / 'configs/checkpoint.json').read_text())


def default_checkpoint():
    config = checkpoint_config()
    return ROOT / 'models' / config['repo'].split('/')[-1] / config['revision']


def load_model(checkpoint=None, device='cuda'):
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
    for item in config['files']:
        if item['type'] == 'file':
            verify_file(checked_path(path, item['path']), item)
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

