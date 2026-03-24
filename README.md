# glow-cocoBrain

Global Workspace (GLoW) aligning vision (COCO images) and language (COCO captions) in a shared latent space. Built as a prototype for the "deep learning module for brain decoding and encoding" internship at ANITI C3PO (PI: Rufin VanRullen).

Next step: plug in a brain module (fMRI, NSD dataset) as a third modality.

## Project structure

```
glow-cocoBrain/
├── config/config.yaml         # All hyperparameters (dims, losses, training, data)
├── main.py                    # Training entry point (Phase 1 + Phase 2 + eval)
├── models/
│   ├── encoders.py            # VisionEncoder (Linear on ResNet features), TextEncoder (Transformer)
│   ├── decoders.py            # VisionDecoder (MLP), TextDecoder (Transformer)
│   ├── domain_module.py       # DomainModule: encoder + decoder + projections
│   ├── projections.py         # MLP projections: latent <-> workspace
│   └── workspace.py           # GlobalWorkspace: fusion, translate, encode_domains
├── losses/
│   ├── contrastive.py         # InfoNCE loss in workspace space
│   ├── translation.py         # Cross-modal reconstruction (MSE / cross-entropy)
│   └── cycle.py               # Cycle consistency (A -> B -> A round-trip)
├── pipeline/
│   ├── train.py               # Training loop (pretrain + alignment)
│   └── eval.py                # Evaluation (retrieval metrics, translation examples)
├── data/
│   ├── dataset.py             # COCODataset (pre-extracted features + tokenized captions)
│   └── tokenizer.py           # SimpleTokenizer (word-level, 10k vocab)
├── scripts/
│   ├── download_coco.sh       # Download COCO 2017 (train + val + annotations)
│   ├── extract_features.py    # Extract ResNet-50 features from images
│   └── build_vocab.py         # Build vocabulary from captions
├── notebooks/
│   └── inference_viz.ipynb    # Visualization: translations, retrieval, losses, telephone arabe
├── DEVLOG.md                  # Development log (what changed, why, results)
└── CLAUDE.md                  # Project context and technical plan
```

## Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Data preparation

Run these 3 scripts in order. Each one is idempotent (skips if data already exists).

```bash
# 1. Download COCO 2017 (~19 GB: train images, val images, annotations)
bash scripts/download_coco.sh

# 2. Extract ResNet-50 features (takes ~20 min with GPU)
#    Saves features_train.npy, features_val.npy, image_ids_*.json in data/coco/
python scripts/extract_features.py

# 3. Build vocabulary from captions (top 10k words)
#    Saves vocab.json in data/coco/
python scripts/build_vocab.py
```

After this, `data/coco/` should contain:
```
data/coco/
├── train2017/              # ~118k images
├── val2017/                # ~5k images
├── annotations/            # captions_train2017.json, captions_val2017.json
├── features_train.npy      # (118287, 2048) ResNet-50 features
├── features_val.npy        # (5000, 2048) ResNet-50 features
├── image_ids_train.json    # image_id ordering matching features
├── image_ids_val.json
└── vocab.json              # SimpleTokenizer vocabulary
```

## Training

Training runs in 2 phases automatically:

```bash
# Full training (Phase 1: pretrain autoencoders + Phase 2: workspace alignment)
python main.py

# With a custom config
python main.py --config path/to/config.yaml

# Skip Phase 1 (resume from pretrain checkpoint)
python main.py --resume_pretrain checkpoint_pretrain.pt
```

**Phase 1 — Pretrain autoencoders** (20 epochs): each domain learns to reconstruct its own input. Projections are frozen.

**Phase 2 — Workspace alignment** (50 epochs): projections + workspace are trained with 3 losses (contrastive + translation + cycle consistency).

Checkpoints saved:
- `checkpoint_pretrain.pt` — after Phase 1
- `checkpoint.pt` — after Phase 2 (final model)

**Tip:** training takes a few hours on a 4070. Use tmux to keep it running:
```bash
tmux new -s train
python main.py
# Ctrl+B then D to detach
# tmux attach -t train to come back
```

## Inference / Visualization

```bash
jupyter notebook notebooks/inference_viz.ipynb
```

The notebook includes:
- Image -> Text translation (with real COCO images)
- Retrieval examples: image -> top-5 captions, text -> top-5 images
- Loss visualization per sample
- "Telephone arabe": multiple passes through the workspace to see signal stability
- t-SNE of workspace representations
- Similarity distribution histograms

## Configuration

All hyperparameters are in `config/config.yaml`. Key settings:

| Parameter | Value | Description |
|-----------|-------|-------------|
| workspace_dim | 384 | Shared workspace dimensionality (bottleneck) |
| latent_dims | 512 | Per-domain latent space dimensionality |
| vision.feature_dim | 2048 | ResNet-50 output dimension |
| text.vocab_size | 10000 | Vocabulary size |
| text.max_seq_len | 32 | Max caption length in tokens |
| training.batch_size | 512 | Batch size |
| losses.contrastive.temperature | 0.07 | InfoNCE temperature |

## Architecture overview

```
Image (pixels)                    Caption (text)
      |                                |
  [ResNet-50 frozen]             [Tokenizer]
      |                                |
  features (2048)              token IDs (32)
      |                                |
  VisionEncoder                 TextEncoder
  (Linear+LN+GELU)          (Embed+Transformer)
      |                                |
   z_vision (512)              z_text (512)
      |                                |
  projection MLP               projection MLP
      \                              /
       \                            /
        -----> WORKSPACE (384) <-----
              (attention fusion)
```

The workspace is the shared bottleneck. Cross-modal translation goes:
`image features -> encode -> z_v -> project -> w -> inv_project -> z_t -> decode -> text logits`
