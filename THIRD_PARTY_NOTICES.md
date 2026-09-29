# Third-party sources

## Counterfactually augmented NLI and SNLI

The token-site experiment uses data from Kaushik, Hovy, and Lipton (2020), *Learning the Difference that Makes a Difference with Counterfactually-Augmented Data*, ICLR:

- Paper: https://arxiv.org/abs/1909.12434
- Source: https://github.com/acmi-lab/counterfactually-augmented-data
- Pinned commit: `6f232a1d2a11462a30ce08fb4825b734ab30828e`

The notebook downloads the source README and Apache-2.0 LICENSE with the data and preserves them in the generated export. Its pinned hashes are part of the notebook. The upstream SNLI material has CC BY-SA 4.0 terms; the repository license does not replace dataset-specific terms. See Bowman et al. (2015), *A large annotated corpus for learning natural language inference*, EMNLP, https://aclanthology.org/D15-1075/, and https://nlp.stanford.edu/projects/snli/.

## MNLI inputs

`inputs/nli_manifest.json` contains selected MultiNLI examples and input provenance used by the correction. MultiNLI is described by Williams, Nangia, and Bowman (2018), *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference*, NAACL, https://aclanthology.org/N18-1101/.

- Dataset source and terms: https://cims.nyu.edu/~sbowman/multinli/
- Dataset distribution: https://huggingface.co/datasets/nyu-mll/glue
- Revision: `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`, configuration `mnli`, split `validation_matched`.

## Models and dependencies

The notebooks and framework reference public Hugging Face model checkpoints and datasets. Their identifiers remain intact for download and attribution. The corrected NLI input file and token-site notebook pin their checkpoint revisions. Weights are downloaded from the original providers; none are redistributed here. Each third-party model, dataset, and library remains subject to its own terms.

No new blanket open-source license is asserted for the FACI code in this anonymous review snapshot. Existing third-party terms remain effective.
