# FACI: code for anonymous review

Code accompanying *When Is Activation Patching Well-Posed? Failure-Aware Causal Evaluation for Mechanistic Interpretability*.

| Material | Location |
| --- | --- |
| FACI framework, benchmark, and control drivers | `historical/code/` |
| Corrected BERT/RoBERTa NLI experiment | `notebooks/FACI_NLI_Review.ipynb` |
| Frozen NLI examples and checkpoint revisions | `inputs/nli_manifest.json` |
| Token-site experiment | `notebooks/FACI_Token_Sites_v1.ipynb` |
| Paper-layout token-site figure generator | `scripts/plot_token_sites.py` |

Start with [REPRODUCING.md](REPRODUCING.md) for sequential Colab instructions, dependencies, hardware, and output locations. The notebooks download public checkpoints and require a GPU and network access. The token-site notebook obtains its data from pinned sources. The NLI input manifest preserves the exact examples and model revisions used in the correction; it contains no completed measurements.

This release contains code and required inputs. Results, model weights, execution logs, and generated figures are created by running the experiments. Original notebook outputs and interactive session metadata have been removed.

The corrected NLI and token-site experiments are separate from the historical benchmark. The historical source is provided for inspection and new runs; exact reconstruction of every historical paper value is not certified. In particular, historical generative-scoring provenance and missing pair-median profiles remain unresolved. See the reproduction guide for the precise scope and small packaging adaptations.

Third-party sources and terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
