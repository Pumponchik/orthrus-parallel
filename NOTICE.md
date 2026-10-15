# Licenses and provenance

## Code and documentation

Original source code and documentation in this repository are covered by the root [MIT license](LICENSE). This grant does not relicense third-party models, datasets, or the binary assets listed below.

The runtime implements parallel verification and drafting for [Zarya-Orthrus](https://github.com/ai-forever/Zarya-Orthrus), published by AI Forever under MIT. It loads [ZaryaOrthrus-0.6B](https://huggingface.co/ai-forever/ZaryaOrthrus-0.6B), whose model card declares MIT, and whose base model [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) is Apache-2.0. The base checkpoint is downloaded separately. Dependency licenses remain with their respective authors.

## Bundled weights

| Asset | Provenance and terms |
|---|---|
| `weights/selector.pt` | Trained on model-generated states from GSM8K, MBPP, and Stanford Alpaca prompts. Released for noncommercial research only; not covered by the code's MIT grant. |
| `weights/projection.npy` | Fixed random projection created for this implementation; MIT. |
| `weights/embedding_projected.npy` | Projection of the ZaryaOrthrus-0.6B token embeddings, not independently generated embeddings. Retains applicable Zarya MIT and Qwen Apache-2.0 terms. |

The selector used Alpaca instruction/input text, not Alpaca reference responses. [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) publishes its data under [CC BY-NC 4.0](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE) and asks that models trained with it remain for research use. No unrestricted commercial rights to the supplied selector weights are asserted here. Attribution: Stanford Alpaca authors, [GSM8K authors](https://github.com/openai/grade-school-math), and [MBPP authors](https://github.com/google-research/google-research/tree/master/mbpp).

Permission is granted by this repository's contributors to use, copy, and redistribute their contribution to the selector weights for noncommercial research, with this notice retained, subject to applicable upstream rights. The weights are provided without warranty. This is a separate limited grant, not a declaration that all upstream material has been relicensed.

Upstream model license notices are retained in [licenses/](licenses/). The training datasets and base checkpoint are not bundled. The repository's MIT license cannot grant rights owned by an employer or other third party.
