# Parallel verification and drafting

A branch is a pair `(r, token)`: accept `r` proposed tokens, then use `token` as the next anchor. A full block has one anchor and 15 mask tokens. The draft produces 15 proposals and a distribution for the bonus-token boundary, giving 16 possible boundaries (`r = 0…15`).

## Where branch context comes from

Suppose the verified cache holds `P`, the current anchor is `a`, and the draft proposes `t1, t2, …, t15`. One candidate is `(3, return)`.

```text
Already cached:    P
AR input:          a  t1  t2  t3  t4 ... t15       causal attention
Candidate input:   return  MASK ... MASK           16 diffusion rows
Candidate reads:   P + AR states of a t1 t2 t3 + its own 16 rows
Candidate cannot read: t4...t15 or any other branch
```

These inputs traverse each transformer layer together. The AR rows use AR projections and causal attention. Each branch uses diffusion projections and bidirectional attention within its own block. The accepted-prefix states are computed **once on the AR path**, shared by all relevant branches at that layer, and are never substituted with diffusion states.

Only AR keys and values enter the persistent cache. After verification, the cache is cropped to the committed prefix; the newest emitted anchor stays uncached for the next cycle. Branch states are temporary. The hypothetical anchor uses diffusion projections now and receives an AR state in the next verification if selected.

This is one transformer traversal, not one CUDA kernel. See [joint.py](../orthrus_parallel/joint.py) for the mask and shared layer loop.

## What the selector sees

For `(3, return)`, the selector receives:

| Input | Meaning |
|---|---|
| Candidate probability and rank | How strongly the draft favors `return` at boundary 3 |
| Position and confidence | Boundary index, top probabilities, probability gaps, and block-wide confidence |
| Draft hidden state | The already-computed representation at boundary 3, projected to 64 dimensions |
| Candidate embedding | A fixed 64-dimensional projection of the embedding of `return` |
| Cycle metadata | Fresh or reused draft, prefix length, and cycle number |

The numeric inputs feed a 64-unit MLP. Separately, learned linear transforms of the projected hidden state and candidate embedding produce a dot-product compatibility score. Their sum is the pair's score. This is a small neural scorer, not a standalone linear classifier or an autoregressive RNN.

Training uses cross-entropy over candidate pairs plus an OUTSIDE outcome. At inference, the scorer evaluates 16 × 64 pairs, excludes ranks above 32 and EOS anchors, and takes the 48 highest scores globally. This exact preparation cost is included in the benchmark. For rejection boundaries, the rejected greedy proposal is excluded from candidate tokens; the final bonus boundary has no such exclusion.

If verification accepts exactly three proposals and emits `return`, that branch wins. Its hidden states go through the language-model head to supply the next draft. Otherwise, another exact pair can win; if none matches, the next cycle runs a separate draft. Startup always needs a draft. Branch preparation is disabled when fewer than two full blocks fit in the remaining output budget.

## Training scope

The released scorer used states from GSM8K, MBPP, and Alpaca prompts: 14,264 usable training states, 1,815 validation states, and 1,817 test states. Prompt families were separated across splits. Features precede the current AR verification; the actual accepted boundary and next token provide labels afterward.

Weights and inference are included. The complete training corpus and training pipeline are not included, so this repository supports inference and evaluation, not full retraining from scratch. See [license and provenance](../README.md#license).

