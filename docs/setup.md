# Setup

The supported environment is Linux x86_64 with NVIDIA CUDA: Python 3.11, PyTorch 2.6.0+cu118, Transformers 5.12.1, and NumPy 2.2.6. `uv.lock` pins dependencies; no compilation or optional training packages are needed.

```bash
uv sync --locked
uv run --locked python scripts/download_model.py
uv run --locked python scripts/download_model.py --download
```

The downloader reports the size and destination before downloading, checks available space, and verifies each file against the pinned manifest. Checkpoint files total about 1.56 GB; dependencies require additional disk space. It refuses downloads with less than 20 GiB free.

An existing checkpoint can be passed with `--checkpoint`. Required model: [ai-forever/ZaryaOrthrus-0.6B](https://huggingface.co/ai-forever/ZaryaOrthrus-0.6B), revision `6c93071b27ab0d2061007dda0416f0439ed4d7df`. Loading checks every manifest file (including configuration and tokenizer files), plus model/code SHA-256 hashes and executes the checkpoint's custom model code with `trust_remote_code=True`; generation does not download files.

```bash
uv run --locked python -m orthrus_parallel --checkpoint /path/to/checkpoint \
  --prompt 'Write a short Python binary search.' --max-new-tokens 256 \
  --output runs/answer.json
```

The default is parallel inference with 48 selected branches. `--method baseline` runs the checkpoint author's ordinary sequential draft/verify implementation. Both use block size 16. JSON includes output token IDs, text, total generation time, and available counters. A single cold request is not a warmed benchmark.

## Checks

```bash
python3 -m unittest tests.test_safety -v
python3 scripts/check_repository.py
uv run --locked python -m tests.validate_runtime --checkpoint /path/to/checkpoint
```

The first command checks source syntax, links, asset hashes, and published throughput. The second uses the checkpoint class to test causal AR cache equivalence, independent branch drafts, attention isolation, and candidate nesting on a small FP32 model, then runs the released selector on the real GPU model. It does not assert universal equivalence to ordinary Orthrus.


## Service integration

`Session.run()` serializes calls on the same session. This prevents concurrent
requests from sharing captured hidden states or selector state; it does not add
batching or parallel request execution. `close()` waits for the active request,
is idempotent, and rejects subsequent runs. Hooks exist only during a request,
and captured tensors are cleared even after inference raises an exception.

The loader leaves process-wide thread counts, RNG seed, Dynamo/compilation, and
TF32 settings unchanged. Configure these explicitly in your application entry
point if needed. The historical benchmark used two intra-op threads, one
inter-op thread, seed 20261012, Dynamo disabled, and TF32 matmul disabled;
application defaults can change timing and numerical results. Set inter-op
thread counts before starting PyTorch work.

Selector assets are verified against `weights/manifest.json` before loading.
The selector uses `torch.load(weights_only=True)` and NumPy loading disables
pickle. An incompatible checkpoint fails closed; unsafe pickle fallback is not
supported. Checkpoint custom code still executes through `trust_remote_code=True`
after verification, so only deploy a reviewed, pinned repository and manifest.

Downloads require HTTPS, reject redirects to HTTP, enforce manifest size and
hashes, and publish only verified files using an atomic rename. Unique temporary
files isolate concurrent downloads and are removed on failure. Install model
artifacts into a directory not writable by untrusted processes.

Offline safety tests exercise interrupted/corrupt downloads, path validation,
HTTPS downgrade rejection, request serialization, and exception cleanup. They do
not replace the GPU runtime checks above.
