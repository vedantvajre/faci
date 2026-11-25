# Running the experiments

## Requirements and scope

Use a fresh Google Colab GPU runtime for either notebook. The completed experiments used a Tesla T4, FP32 inference, and deterministic settings. The recorded runtime used Python 3.13.15 and PyTorch 2.11.0+cu128; Colab supplies PyTorch, while the first cell of each notebook installs its experiment-specific dependencies. These include Transformers 4.51.3, Tokenizers 0.21.4, and Hugging Face Hub 0.34.4. GPU type and package versions are recorded when running. Numerical identity across different hardware and PyTorch builds is not guaranteed.

Both notebooks use your Google Drive for checkpoints and new outputs. Do not mix a new run with an existing experiment directory. Preserve an existing directory before starting independently. No saved experiment output is required as an input.

## 1. Corrected NLI experiment

1. Download this repository as a ZIP and extract a local copy. Keep the ZIP for the upload in step 5. Use the notebook from that same ZIP. To resume an existing run, use the original notebook and archive for that run.
2. Open Colab, choose **File → Upload notebook**, and upload `notebooks/FACI_NLI_Review.ipynb` from the extracted copy.
3. Select **Runtime → Change runtime type → T4 GPU**, or another available GPU.
4. Run **Cell 1**, then choose **Runtime → Restart session**. After restarting, proceed to Cell 2; do not run Cell 1 again in the same session.
5. Run **Cell 2** and authorize your own Drive. Upload the complete code ZIP when prompted. Setup locates `src/mechinterp/` and `inputs/nli_manifest.json`, checks the frozen input file, and creates `MyDrive/FACI_NLI_Review`.
6. Run **Cells 3–6** in order to load the fixed protocol and define the measurements.
7. Run **Cell 7** for calibration. Continue only if its checks pass.
8. Run **Cell 8** for the two NLI classifiers, then **Cell 9** for the pretrained auxiliary measurements.
9. Run **Cell 10** to aggregate the corrected measurements and download the ZIP. Its `summary/` directory contains `per_seed.csv`, `corrected_conditions.csv`, and the corrected layer profiles.

The frozen manifest fixes the selected texts, labels, ten seeds, auxiliary examples, four model revisions, and MNLI dataset revision. It is a required experimental input, not a saved result. Do not regenerate it to select different examples after inspecting outcomes.

Only packaging-dependent parts of the original correction notebook were adapted: loading this code archive and its frozen input manifest; using a separate Drive directory; and omitting the optional comparison with historical saved results. The measurement, calibration, filtering, and corrected-summary calculations are unchanged. The omitted comparison constructed provisional mixtures of historical and corrected results; it is not needed to measure the corrected conditions. The original terminal cell containing only `0` was removed.

## 2. Token-site experiment

1. Upload `notebooks/FACI_Token_Sites_v1.ipynb` through **File → Upload notebook** in Colab.
2. Select a **T4 GPU** runtime or another available GPU.
3. Run **Cell 1**, then choose **Runtime → Restart session**.
4. Run **Cell 2** and authorize your own Drive. For an independent run, `MyDrive/FACI_Token_Sites_v1` must not contain an earlier run. The notebook checks existing state before resuming.
5. Run **Cells 3–8** in order. These freeze the protocol, download verified source data, select the common 100-pair cohort, and define the patching and analysis code.
6. Run **Cell 9** for BERT and **Cell 10** for RoBERTa. Completed pair measurements are saved incrementally.
7. Run **Cell 11** to analyze the completed runs and generate the notebook figure.
8. Run **Cell 12** to download `FACI_Token_Sites_v1_review.zip`. Keep this archive; its `analysis/layer_site_summary.csv` is the input to the paper-layout plot below.

All code cells in this notebook are unchanged. Its pinned source commit, source hashes, checkpoint revisions, cohort rule, analysis settings, and 5,000 bootstrap resamples are specified in the notebook. It downloads and checks the external sources before inference. Selection uses the fixed lexical and tokenization criteria rather than model outcomes. This experiment measures site-dependent recovery; it does not test filtering reliability.

After an interruption, reconnect with the same GPU type and environment and rerun from Cell 2. Reinstall only if necessary, restarting after installation. Guards prevent combining incompatible runs. Preserve any failure report and investigate its cause rather than deleting a guard. A partial export is labelled as partial.

## 3. Generate the paper-layout token-site figure

Extract the new token-site output ZIP. Locate its `analysis/layer_site_summary.csv`. From the code repository root, run the following on a CPU machine with Python; substitute the actual CSV path:

```bash
python -m pip install -r requirements-figure.txt
python scripts/plot_token_sites.py \
  --input /path/to/extracted-run/analysis/layer_site_summary.csv \
  --output reproduced/fig_token_sites.pdf
```

The script writes the PDF and a PNG preview beside it. It uses the notebook's layer statistics without recalculating or changing the estimator. Plot dimensions and styling match the paper figure; installed fonts can affect appearance. The separate plotting requirements match the environment used to prepare the paper layout.

## 4. Run the original benchmark

The retained implementation covers the three tiers, architecture and task wrappers, ten-seed runner, negative controls, gap ablation, and historical analysis/plotting scripts. `configs/full8.yaml` is the shared configuration of the ten archived canonical runs; only their seed and output directory differed.

The following runs the supplied historical implementation. It is not a certification that a new run recovers every published historical value: historical model and dataset revisions are unpinned, generative-scoring provenance remains unresolved, and saved pair-median profiles are incomplete. A dedicated runner for the historical grouping-ablation output was not present in the supplied source. Use the corrected NLI notebook above for the corrected encoder-NLI conditions.

In a separate Python environment with a suitable PyTorch GPU build, start from the repository root:

```bash
python -m pip install -r requirements-benchmark.txt
PYTHONPATH="$PWD/src" python scripts/run_multiseed.py \
  --config configs/full8.yaml \
  --seeds 42 123 456 789 1000 2024 3141 5555 7777 9999 \
  --output-dir reproduced/benchmark
```

The working directory matters: the runner launches `scripts/run_benchmark.py` as a subprocess. The explicit seed list avoids its five-seed default. The supplied requirements are a starting environment based on the corrected experiment's compatible package versions, not a recovered historical environment lockfile. The full benchmark is substantially larger than either bounded notebook, and no runtime estimate is asserted here.

Every retained original Python source file and `full8.yaml` is byte-for-byte unchanged. One docstring-only `src/mechinterp/baselines.py` was added because the original package initializer imports that absent optional module. No retained runner consumes any symbols from it, and it implements no baseline or scientific computation. The correction notebook already used an equivalent import-only shim. Unused convenience drivers, incomplete external-baseline drivers, and superseded configurations were excluded.
