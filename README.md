# S-TSViT: An Energy-Efficient Spiking Transformer for Satellite Image Time Series Analysis

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)

> **Fusion of [TSViT](https://github.com/michaeltrs/DeepSatModels) (Temporal-Spatial Vision Transformer) and [SVF](https://github.com/JimmyZou/SpikeVideoFormer) (SpikeVideoFormer)**  
> Spiking Neural Networks meet Remote Sensing for low energy consumption, high-performance crop segmentation.

---

## 📋 Overview

**S-TSViT** bridges two research frontiers:

| Component | Source | Contribution |
|-----------|--------|------------|
| **TSViT** | DeepSatModels ([Tarasoiu et al., 2023](https://arxiv.org/abs/2301.04940)) | Temporal-spatial attention for satellite time series |
| **SVF** | SpikeVideoFormer | Energy-efficient spiking neurons for video/spatio-temporal data |

**Our Innovation**: Replace TSViT's standard temporal encoder with **Spiking Neural Network (SNN) dynamics**, achieving:
- ⚡ **58× lower energy consumption in theory** (compared to equivalent ANNs)
- 🧠 **Achieved SOTA in PASTIS dataset** (highest with 66.9%)
- 🔽 **Reduced model size by 12%**.

Paper: [PDF version](/docs/paper.pdf)
---

## 🏗️ Architecture

![S-TSViT Architecture](docs/s-tsvit_final_real.png) <!-- Add your illustration here -->

### Two Variants

| Variant | Temporal Collapse | Use Case |
|---------|-------------------|----------|
| `SpikeTSViTMean` | Before spatial encoding | Faster, less memory | lower accuracy |
| `SpikeTSViTNoMean` | After spatial encoding | Richer spatio-temporal features |

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (10GB+ VRAM recommended)

### Quick Start

```bash
# Clone repository
git clone https://github.com/hukaiems/Spiking_Temporo-Spatial_Vision_Transformer.git
cd DeepSatModels_SNN

# One-command setup (downloads PASTIS data automatically)
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
bash setup_pastis.sh
```

The setup script will:
- Install dependencies (`requirements.txt`)
- Download PASTIS dataset from Kaggle (I have upload it myself)
- Prepare checkpoint directory

---

## 🚀 Training
This is my traning script on kaggle using P100.
```
# 1.51 M 
!python /kaggle/working/DeepSatModels_SNN/train_and_eval/train_stsvit.py \
    --csv_path /kaggle/working/DeepSatModels_SNN/configs/PASTIS24/splits/old_split_kaggle/train_exp1_chunks_123.csv.bak \
    --val_csv_path /kaggle/working/DeepSatModels_SNN/configs/PASTIS24/splits/old_split_kaggle/chunk_4_paths.csv.bak \
    --resume /kaggle/input/spike-tsvit/pytorch/default/54/128dim_pastis_ver3/training/s_tsvit_ignore_128dim_21e_pastis_1.pth \
    --model_type no_mean \
    --att_mode 2D_ham \
    --norm_type gn \
    --loss_type focal \
    --focal_a_weight 2 \
    --max_seq_len 49 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --temporal_depth 3 \
    --spatial_depth 2 \
    --embed_dim 128 \
    --heads 8 \
    --lr 5e-4 \
    --epochs 24 \
    --checkpoint_path /kaggle/working/s_tsvit_ignore_128dim_24e_pastis_1.pth \
    --no_progress_bar
```

### Key Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--model_type` | `mean`, `no_mean` | Temporal collapse strategy (use the no_mean for the best result) | 
| `--att_mode` | `2D_dot`, `2D_ham` | Attention mechanism (ham is better) |
| `--loss_type` | `standard`, `weighted`, `focal` | Loss function |
| `--norm_type` | `bn`, `gn` | Batch vs Group normalization |

### Resume Training
```bash
python train_and_eval/train_stsvit.py \
    --resume checkpoints/spiketsvit_best_latest.pth \
    ... # other args
```

---

## 🎯 Inference & Analysis
My inference script in kaggle.
```bash
!python /kaggle/working/DeepSatModels_SNN/train_and_eval/inference_s_tsvit.py \
    --val_csv_path /kaggle/working/DeepSatModels_SNN/configs/PASTIS24/splits/old_split_kaggle/chunk_4_paths.csv.bak \
    --checkpoint_path /kaggle/input/s-tsvit-testing/pytorch/default/12/128dim_pastis_ver3_final/testing/spike_tsvit_151M_pastis_5_best.pth \
    --model_type no_mean \
    --att_mode 2D_ham \
    --norm_type gn \
    --max_seq_len 49 \
    --batch_size 2 \
    --temporal_depth 3 \
    --spatial_depth 2 \
    --embed_dim 128 \
    --heads 8 \
    --inference
```

### Analysis Tools

| Flag | Analysis |
|------|----------|
| `--inference` | Standard accuracy (mIoU, OA) + top-10 best/worst samples |
| `--test_per_class` | Per-class accuracy breakdown |
| `--test_energy` | SNN energy consumption (synaptic operations) |
| `--temporal_importance` | Which time steps matter most per class |
| `--NDVI` | Phenological confusion (crop growth cycles) |
| `--analyze_cloud` | Robustness to cloud cover |
| `--confusion_matrix` | Full classification confusion |
| `--visual_comparison` | Error map visualization |
| `--deploy_inference` | Single-sample latency test |

---

## 📊 Results

### PASTIS Dataset (Crop Segmentation)

| Model | mIoU | Energy* | Params |
|-------|------|---------|--------|
| TSViT (baseline) | ~0.654 | 100% | 1.7M |
| **S-TSViT (Ours)** | **~0.669** | **~6%** | **1.5M** |

*Energy estimated via synaptic operations (SynOps)

> **Key Insight**: S-TSViT achieves comparable accuracy with **~58× energy reduction**, critical for edge deployment on satellites or IoT devices.

---

## 🧪 Reproducibility

### Datasets

| Dataset | Classes | Bands | Resolution | Source |
|---------|---------|-------|------------|--------|
| **PASTIS** | 20 (19 crops + bg) | 10 (S2) | 10m | [GitHub](https://github.com/VSainteuf/pastis-benchmark) |
| **France** | 21 | 13 (S2) | 10m | Custom split |

### Pre-trained Weights

| Model | Checkpoint |
|-------|-----------|
| S-TSViT-NoMean | [Download](https://www.kaggle.com/models/nguyenlecao/s-tsvit-testing) |

---

## 🏛️ Citation

If you use this code, please cite:
Acutally i havent published it anywhere so you can't cite me =))). But you can site other paper that i used to create this.

```bibtex

@inproceedings{tarasoiu2023tsvit,
  title={DeepSatModels: Temporal-Spatial Vision Transformers for Satellite Image Time Series},
  author={Tarasoiu, Michail and others},
  booktitle={ICLR},
  year={2023}
}

@article{zhu2022spikevideoformer,
  title={SpikeVideoFormer: Spiking Neural Networks for Video Understanding},
  author={Zhu, Zhenyu and others},
  journal={arXiv preprint},
  year={2022}
}
```

---

## 📂 Repository Structure

```
S-TSViT/
├── models/
│   └── snn/
│       ├── spike_tsvit.py          # Main architectures
│       ├── snn_transformer.py      # MS_Block, attention layers
│       ├── loss_function.py        # Focal loss
│       └── helper_functions.py     # Analysis tools
├── train_and_eval/
│   ├── train_stsvit.py             # Training script
│   └── inference_s_tsvit.py        # Evaluation & analysis
├── spike_data/
│   ├── pastis_dataset.py           # PASTIS dataloader
│   └── france_dataset.py           # France dataloader
├── setup_pastis.sh                 # One-command setup
├── requirements.txt
└── LICENSE.txt
```

---

## 🤝 Acknowledgments

- [DeepSatModels](https://github.com/michaeltrs/DeepSatModels) — TSViT baseline and PASTIS preprocessing
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) — SNN framework
- [SpikeVideoFormer](https://github.com/JimmyZou/SpikeVideoFormer) — Spiking video transformer inspiration

---

## 📧 Contact

For questions or collaborations: [Gmail](lecaonguyen1524@gmail.com)

---

**License**: Apache 2.0 — see [LICENSE.txt](LICENSE.txt)