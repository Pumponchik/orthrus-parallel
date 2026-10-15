# Benchmark

The published comparison uses ZaryaOrthrus-0.6B at the pinned revision in `configs/checkpoint.json`: RTX 2080 Ti, batch 1, greedy decoding, block size 16, and a 256-new-token limit. The parallel method selects 48 branches from top-32 anchors at 16 boundaries.

| Metric | Ordinary Orthrus | Parallel + selector |
|---|---:|---:|
| Tokens/s | 58.28 | 69.57 |
| Generated tokens | 11,782 | 11,776 |
| Total timed seconds | 202.15 | 169.27 |

Measured throughput increase: **19.4%**, paired 95% interval **17.0–22.1%**. Parallel branch hit rate is 84.6% (1,952 hits / 2,306 continuing branch cycles); 552 separate draft passes were needed. The run averaged 4.70 emitted tokens per decoding round.

## Measurement protocol

The author clarified that one historical run used the target in FP16 and the other in BF16; the method-to-precision mapping was not recorded in the published data. These measurements therefore do not isolate the effect of parallel drafting at a fixed precision. Cross-run token equality is not used as a method-equivalence metric. The current runner loads both methods in FP16.

Thirty held-out prompts cover code, math, and text equally, with two repeats per prompt. Both methods run on the same GPU for each prompt, after warmup, with randomized first-repeat order and reversed second-repeat order. CUDA synchronization brackets timing. Time includes prefill, features, selection, drafting, and verification; model loading, tokenization, and text decoding are excluded.

Throughput is total generated tokens divided by total seconds. The paired interval uses 10,000 prompt-level bootstrap samples, retaining repeats and stratifying by domain and physical GPU (seed 20261015). Hardware, prompt length, and output length can change the result; no broader hardware or quality claim is made.

[Measurements](results/measurements.csv) and [summary](results/summary.json) contain only these two methods, with the original timing and token-count values from the completed evaluation. The full original test prompts are not bundled. `prompts.json` provides a separate public example set; rerunning it is a new measurement, not an exact reproduction of the table.

## Run your own comparison

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m benchmarks.run \
  --checkpoint /path/to/checkpoint --limit 10 --output runs/benchmark.jsonl
```

The runner warms both methods and records two paired repeats, token IDs, complete generation time, and counters. Use an idle GPU. Compare token IDs as well as throughput; faster generation alone does not establish equivalent answers.
