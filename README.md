# MR.DEC

Multimodal readmission prediction with CXR (chest X-ray) and EHR. This repo contains training and evaluation code only; large assets (CSV, weights, images) are excluded via `.gitignore`.

---

## Environment (Conda)

Create the env and activate it:

```bash
conda create -n readmit python=3.10 -y
conda activate readmit
```

Then install PyTorch (CUDA 12.1) and dependencies in this order:

```bash
conda install pytorch=2.1.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install torchdata==0.7.1
pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu121/repo.html
pip install numpy pandas scipy matplotlib scikit-learn tqdm
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric rad-dino dotted-dict networkx
```

Install the rest (transformers, timm, wandb, etc.) as needed:

```bash
pip install transformers timm wandb pillow
```

---

## Project structure

| Path | Description |
|------|-------------|
| `main.py` | Main entry: feature extraction, training, evaluation (decoder / encoder / cross-attention / MLP) |
| `main.sh` | Example run script; set data and pretrained paths inside |
| `online_processor.py` | CXR (EVA-X / RAD-DINO) and EHR (Clinical BERT) feature extractors |
| `utils.py` | Logging, checkpointing, seeding |
| `group_balanced_sampler.py` | Group-balanced batch sampler for contrastive learning |
| `model/autoregressive_transformer.py` | ReadmissionEncoder, AutoregressiveTransformer, CrossAttention, MLP |
| `cxr/eva_x.py` | EVA-X CXR encoder (timm) |
| `data/readmission_utils.py` | MIMIC readmission labels and explain helpers |
| `ablation/train_end_to_end.py` | Minimal end-to-end training entry |
| `ablation/train_end_to_end.sh` | Example script for ablation |

---

## Data

Place your data and weights outside the repo (or adjust paths in the scripts). Required inputs:

- **Datasets**:
  - **MIMIC-IV EHR (v3.1)**: [https://physionet.org/content/mimiciv/3.1](https://physionet.org/content/mimiciv/3.1) (CSV: admission demo, ICD, lab, med; paths set in `main.sh`).
  - **MIMIC-CXR (v2.1.0)**: [https://physionet.org/content/mimic-cxr/2.1.0](https://physionet.org/content/mimic-cxr/2.1.0)
- **Pretrained Weights**:
  - **EVA-X**: [https://github.com/hustvl/EVA-X](https://github.com/hustvl/EVA-X) (CXR pretrained weights e.g., `.pt`; path set via `--cxr_pretrained_path`).
  - **ModernBERT**: [https://huggingface.co/answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)
---

## Running

From the repo root (`MR.DEC`):
### 1. Data Preprocessing

First, run the preprocessing script to prepare the cohort data:

```bash
python get_mimic_cohort3.1.py \
  --raw_data_dir "path_to_MIMIC-IV_dir" \
  --cxr_data_dir "path_to_MIMIC-CXR-JPG_dir" \
  --save_dir "path_to_your_dir"
```
2. Run:

```bash
cd MR.DEC
conda activate readmit
bash main.sh
```

Optional: feature caching to avoid re-extracting when encoder config is unchanged:

```bash
python main.py ... --use_feature_cache --feature_cache_dir ./cache/features
```

---

## Main options (summary)

| Option | Description |
|--------|-------------|
| `--do_train` | Run training |
| `--model_name` | `decoder` \| `encoder` \| `cross-attention` \| `mlp` |
| `--modality` | `multimodal` \| `image` \| `text` |
| `--feature_mode` | `cls_only` \| `all_features` |
| `--loss_option` | `bce` \| `contrast` \| `mix` |
| `--balance_data` | Balance train/val by label |
| `--use_feature_cache` | Cache extracted features on disk |
| `--enable_explainability` | Run explainability after training |

Full list: `python main.py --help`.
