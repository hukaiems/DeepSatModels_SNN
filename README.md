# DeepSatModels_SNN — Spiking Temporo-Spatial Vision Transformer

Compact, production-friendly README for the spiking temporo-spatial vision transformer used for land-cover recognition (graduation thesis project).

Highlights
- SOTA performance on the PASTIS dataset using a spiking TSViT+SVF hybrid.
- Very low spike firing rate (~5.8%) and an estimated ~85x energy efficiency vs equivalent ANN.

Badges
- (optional) License: see `LICENSE.txt`
- (optional) Paper / citation: add link when available

Table of contents
- Quickstart
- Requirements
- Datasets & setup
- Training
- Inference / Evaluation
- Reproducing results
- Project structure
- Citation & license
- Contributing / Contact

Quickstart
1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Prepare datasets (the repo provides helper scripts):

```bash
bash setup_pastis.sh    # downloads & prepares PASTIS dataset (Linux/macOS)
bash setup_France.sh    # downloads & prepares France dataset
```

3. Train or run inference with a config (examples below).

Requirements
- See `requirements.txt` for full pinned dependencies. Use a recent Python 3.8+ interpreter and a PyTorch build compatible with your CUDA driver when training on GPU.

Datasets & setup
- This repository supports PASTIS and France datasets. The provided scripts `setup_pastis.sh` and `setup_France.sh` download and prepare the data automatically (Kaggle credentials may be required for PASTIS). For manual dataset code, see `spike_data/pastis_dataset.py` and `spike_data/france_dataset.py`.

Training (example)
Use the training entrypoint in `train_and_eval/train_stsvit.py` with a configuration file from `configs/`.

```bash
python train_and_eval/train_stsvit.py --cfg configs/PASTIS24/TSViT_fold1.yaml
```

Notes:
- Replace `--cfg` with any config in `configs/PASTIS24` (or other dataset folders).
- Add `--work-dir` or other CLI args as supported by the script; use `--help` to list options.

Inference example

```bash
python train_and_eval/inference_s_tsvit.py \
	--ckpt model_checkpoint/snn_unet_latest_checkpoint_75subset_max_30_deeper.pth \
	--cfg configs/PASTIS24/TSViT_fold1_test_checkpoint.yaml
```

Pretrained models
- Available checkpoints are placed in `model_checkpoint/` in this repo (examples: `snn_unet_latest_*.pth`, `spike_tsvit_151M_pastis_run_latest_latest.pth`). If additional or larger weights are hosted externally, add links and download helpers here.

Reproducing reported results
- Use the provided configs under `configs/PASTIS24` (folds and evaluation configs are included).
- For deterministic runs, set seeds and deterministic flags via the training script's CLI (check `train_stsvit.py --help`).
- Record the config file, checkpoint, and random seed used for any reported experiment.

Project structure (short)
- `models/` — model implementations (see `models/snn/` for spiking models and helpers).
- `spike_data/` — dataset loaders and preprocessing for PASTIS/France.
- `train_and_eval/` — training, evaluation, and inference scripts.
- `configs/` — YAML experiment configs (dataset-specific subfolders and fold definitions).
- `model_checkpoint/` — included checkpoints for quick evaluation.
- `notebook_code/` — Jupyter notebooks for experiments and visualizations.

Development & contributing
- Report issues or feature requests via the repository issue tracker.
- For contributions, fork the repo, create a feature branch, and open a pull request. Please follow code style in existing files and keep changes focused.

Citation & license
- License: `LICENSE.txt` in this repository.
- Paper: please cite the project paper (link / BibTeX to be added here). If you use this code in published work, include a citation to the thesis/paper and indicate which config and checkpoints were used.

Contact
- For questions about experiments or reproducibility, open an issue or contact the authors (add email or contact method here).

Acknowledgements
- See `LICENSE.txt` for license terms and any third-party acknowledgements.

—
If you'd like, I can:
- Add runnable example commands with exact CLI flags after inspecting `train_and_eval/train_stsvit.py` and `inference_s_tsvit.py`.
- Add a small `USAGE.md` with reproducible steps for the PASTIS baseline.

If you want this text committed, tell me and I'll save it to `README.md` and create a commit.