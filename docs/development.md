# Development and experiments

GLoW cocoBrain is an experimental research project before V1. This guide describes the current source, including the incomplete NSD branch. The image/text workflow needs external datasets and feature extraction; the NSD preprocessing needs fixes before new brain experiments. Historical experiment notes are in [DEVLOG.md](../DEVLOG.md), and the [README](../README.md) introduces the project.

## Current implementation

Each [domain module](../models/domain_module.py) contains an encoder, a decoder and projections to and from the shared workspace. The current encoders and decoders are MLPs operating on features extracted beforehand.

| Domain              | Input features                               | Domain latent | Decoder output                |
| ------------------- | -------------------------------------------- | ------------- | ----------------------------- |
| Vision              | 2,048-dimensional ResNet-50 features         | 64            | Reconstructed ResNet features |
| Text                | 1,024-dimensional E5-large-v2 embeddings     | 64            | Reconstructed text embeddings |
| Brain, experimental | 1,024 PCA components from NSD fMRI responses | 256           | Reconstructed PCA features    |

The dimensions above come from [config/config.yaml](../config/config.yaml) and [config/config_brain.yaml](../config/config_brain.yaml). Both use a 32-dimensional workspace and six-layer projection MLPs with a hidden dimension of 256. Some older comments, experiment entries and constructor defaults describe previous architectures; the selected configuration determines a run.

[GlobalWorkspace.fuse()](../models/workspace.py) averages the available workspace vectors and applies `tanh`. During alignment, the fusion loss mixes each configured pair using a coefficient sampled from `Beta(5, 5)`, then applies `tanh`. The separate `translate()` method goes directly through the source encoder, projections and target decoder.

The model reconstructs feature vectors. The notebook displays captions and images through retrieval from existing COCO examples. It does not generate image pixels or decode text tokens. The older tokenizer and vocabulary script remain in the repository, but the current training path uses E5 embeddings.

## Environment

Run commands from the repository root in a shell with Python 3 and virtual-environment support. The repository does not pin a Python minor version or provide a locked dependency environment. [requirements.txt](../requirements.txt) lists PyTorch, torchvision and the remaining Python dependencies; choose a PyTorch installation compatible with your machine.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The download scripts require Bash and `wget`; COCO extraction also requires `unzip`. Training and feature extraction select CUDA when available, otherwise CPU. There is no MPS selection in the current code. The default training batch size is 1,024, and memory use depends on the data and evaluation stage. The repository does not establish a minimum RAM/VRAM requirement or a portable runtime estimate.

Both supplied configurations set `logging.use_wandb: true`. Alignment initializes Weights & Biases and sends run configuration, losses and diagnostics. Set that field to `false` before training if you want to keep logging local. Local JSON logs are written independently of W&B.

Datasets, pretrained feature extractors and trained project checkpoints are not bundled. Feature extraction can download pretrained weights on first use. Check the upstream dataset and model terms before downloading or redistributing their contents.

## Prepare COCO features

The current COCO workflow has three steps:

```sh
bash scripts/download_coco.sh
python scripts/extract_features.py
python scripts/extract_text_features.py
```

The first script downloads the COCO 2017 training images, validation images and caption annotations to `data/coco/`. Its comments estimate about 19 GB of archives. Allow additional space for extracted images, temporary archives, feature arrays, model caches and checkpoints. This is an estimate from the script, not a measured total storage requirement.

[extract_features.py](../scripts/extract_features.py) loads torchvision's default pretrained ResNet-50 weights, replaces the final classifier with an identity layer, and saves one 2,048-dimensional vector per image. [extract_text_features.py](../scripts/extract_text_features.py) then loads `intfloat/e5-large-v2`, prefixes each caption with `query: ` and saves its raw 1,024-dimensional embedding. It uses the image-ID files from the vision step, so run vision extraction first.

The training dataset requires these files for both `train` and `val`:

```text
data/coco/
├── train2017/
├── val2017/
├── annotations/
│   ├── captions_train2017.json
│   └── captions_val2017.json
├── features_train.npy
├── features_val.npy
├── image_ids_train.json
├── image_ids_val.json
├── text_features_train.npy
├── text_features_val.npy
├── caption_ids_train.json
└── caption_ids_val.json
```

The feature scripts accept `--data_root` and `--batch_size`; the text script also accepts `--model_name`. A different text model must still produce the dimension configured under `text.feature_dim`, or the configuration and trained weights must be changed together. The current code does not record a model revision in the extracted arrays.

The extraction scripts skip a split when its feature array already exists. They do not check that the corresponding ID file exists or matches it. After an interrupted extraction, inspect both files before reusing the output. The download script similarly skips existing image or annotation directories without validating their contents.

`scripts/build_vocab.py` is part of the earlier token-based experiment and is not required by [COCODataset](../data/dataset.py). Running it does not replace the text-feature extraction step.

## Train vision and text

Review [config/config.yaml](../config/config.yaml), including logging and data paths, before starting. The default configuration uses 20 autoencoder epochs followed by 50 alignment epochs.

```sh
# Train both phases, then evaluate.
python main.py --config config/config.yaml --name vision_text

# Train the autoencoders only, then evaluate.
python main.py --config config/config.yaml --phase pretrain --name pretrain

# Reuse a compatible pretraining checkpoint for alignment.
python main.py --config config/config.yaml --phase align \
  --resume_pretrain checkpoints/run_001_pretrain/checkpoint_pretrain.pt \
  --name alignment
```

Replace the example checkpoint path with the directory created by your run. Run numbers increase from the existing `checkpoints/run_*` directories. `--config` accepts a different YAML file, and `--name` adds a suffix to the generated directory name.

Phase 1 trains each domain as an autoencoder with MSE reconstruction loss while freezing its workspace projections. For the vision/text configuration, Phase 2 freezes the encoders and decoders and trains the projections with five losses: contrastive alignment, translation, same-domain reconstruction (demi-cycle), round-trip reconstruction (cycle), and fusion.

The default `paired_fraction: 0.2` enables supervised contrastive and translation losses for images whose ID is divisible by five. The other losses use all samples. This setting is not a general fraction selector: [the dataset implementation](../data/dataset.py) uses the same modulo rule for every value below `1.0`; a value of at least `1.0` marks every sample as paired.

`--resume_pretrain` loads matching tensor names and shapes from either a raw state dictionary or a checkpoint containing `model_state_dict`. It skips incompatible tensors and prints counts. It does not restore optimizer state or resume an interrupted epoch. It also does not skip pretraining when `--phase both` is selected; use `--phase align` when you intend to skip Phase 1. Only load checkpoints you trust, since this loader uses `torch.load(..., weights_only=False)`.

Each run copies its configuration and writes the applicable files below:

| File                     | Written when                     |
| ------------------------ | -------------------------------- |
| `config.yaml`            | The run is created               |
| `log_pretrain.json`      | Each pretraining epoch completes |
| `checkpoint_pretrain.pt` | Pretraining completes            |
| `log_align.json`         | Each alignment epoch completes   |
| `diagnostics.csv`        | Alignment diagnostics run        |
| `checkpoint.pt`          | Final evaluation completes       |
| `results.json`           | Final evaluation completes       |

Every phase choice proceeds to final evaluation. There is no separate evaluation-only CLI or flag to skip evaluation. The final model checkpoint is saved after evaluation, so an evaluation failure can leave an alignment run with logs but without its final checkpoint.

## Evaluation and notebooks

[pipeline/eval.py](../pipeline/eval.py) computes retrieval, feature reconstruction error and workspace statistics. Final retrieval builds a full `N × N` similarity matrix on the selected device. Reducing the training batch size does not remove this allocation. During alignment, diagnostics use a fixed subset of up to 1,000 validation pairs.

The basic retrieval metric treats the exact paired row as the correct answer. COCO has several captions per image, so this metric differs from image-level retrieval where any matching caption can count. Do not compare results across candidate-set sizes or protocols without accounting for those differences.

An optional script downloads Karpathy test IDs:

```sh
python scripts/download_karpathy.py
```

When `data/coco/karpathy_test_ids.json` exists, evaluation runs a separate image-level retrieval calculation. It selects the IDs present in the supplied validation dataset and divides those images into five folds. The code does not verify that all 5,000 expected test images are available, so the printed `Karpathy 1K` label alone does not establish a standard 1,000-image-per-fold benchmark. This optional branch compares vision and text, including when the selected training configuration contains brain pairs.

The [inference notebook](../notebooks/inference_viz.ipynb) contains vision/text retrieval, caption lookup, repeated cross-modal translation, t-SNE and representation diagnostics. Install Jupyter separately if needed:

```sh
python -m pip install notebook
jupyter notebook notebooks/inference_viz.ipynb
```

The notebook uses paths relative to `notebooks/`. Set `RUN_DIR` explicitly to a compatible vision/text run, such as `../checkpoints/run_001_vision_text`, before executing its cells. Its default selects the latest run, but it constructs only vision and text modules and strictly loads their state dictionary. A newer brain checkpoint will not load through that path. Dataset files and the chosen checkpoint must already exist.

## Experimental NSD workflow

The brain branch is implemented, but successful fMRI decoding has not been established. The April 9 entries in [DEVLOG.md](../DEVLOG.md#9-avril-2026--phase-3--module-cerveau-nsd-fmri) report retrieval near chance for the recorded brain runs; the larger 256-dimensional brain model is left as an experiment in progress. The log contains no completed result for that final run.

### Known NSD blockers

**The current NSD extractor has a known trial-to-image mapping error.** It omits the subject-specific `subjectim` lookup before converting trial order to global NSD image IDs. The [official NSD example](https://github.com/cvnlab/nsdexamples/blob/master/matlab/example06_basicbetaloading.m#L104-L184) shows that `masterordering` indexes a subject's image subset and must be mapped through `subjectim`. Using it directly produces incorrect COCO labels with the standard metadata. Correct and validate this path before preparing brain features or starting a new brain experiment. Existing experiment artifacts have not been audited.

Missing sessions are also unsupported. Extraction skips them, but later averaging still expects the full trial count and can index beyond the accumulated response array. A missing session in the middle also shifts later trial associations. Truncating the array length would not restore the missing trial positions.

### Data and training entry points

The scripts target Subject 1 of the [Natural Scenes Dataset](https://naturalscenesdataset.org/), using the `nsdgeneral` ROI and 40 sessions of preprocessed fMRI response estimates. NSD asks users to complete its Data Access Agreement and consult its Data Manual before accessing the data. The script's direct download URLs do not replace that process.

The extraction script imports `pandas`, `scipy` and `nibabel`. They are not explicitly listed in `requirements.txt`, so install them for this branch:

```sh
python -m pip install pandas scipy nibabel
```

After preparing the COCO vision/text features and fixing the mapping issue above, these are the data preparation entry points:

```sh
bash scripts/download_nsd.sh
python scripts/extract_brain_features.py \
  --nsd_root data/nsd_raw \
  --coco_root data/coco \
  --output_root data/nsd \
  --n_components 1024
```

The download script estimates about 20 GB of compressed NSD files. Extraction also loads full NIfTI volumes and accumulates ROI responses in memory. Account for those files and in-memory arrays before running it; no memory budget was verified here.

[extract_brain_features.py](../scripts/extract_brain_features.py) masks the responses, groups trials by image, averages repetitions, and splits by the existing COCO train/validation image IDs. It fits PCA and z-score statistics on the training split, then applies them to validation. It writes brain feature arrays, image-ID lists, `pca_model.pkl`, `zscore_mean.npy`, `zscore_std.npy` and `brain_meta.json` under `data/nsd/`. [NSDDataset](../data/nsd_dataset.py) keeps images represented in all three modalities and associates each with its COCO captions.

This pipeline needs further validation before its outputs can support decoding claims:

- The download script checks file existence and nonzero size, not content integrity. Validate the complete session set before extraction.
- Train/validation membership follows COCO image IDs. The code does not implement a separate NSD benchmark split or a study across subjects.
- Partial checkpoint loading can leave incompatible or absent weights at initialization. Brain training freezes vision and text even if their expected pretrained weights were not loaded.

Once data correspondence and a compatible aligned vision/text checkpoint have been verified, the intended training command is:

```sh
python main.py --config config/config_brain.yaml --phase both \
  --resume_pretrain checkpoints/run_001_vision_text/checkpoint.pt \
  --name brain_experiment
```

Replace the checkpoint path with a completed aligned vision/text run, using the 32-dimensional workspace, 64-dimensional vision/text latents and matching projection structure. Review the printed loaded/skipped tensor counts. The supplied brain configuration freezes vision and text, pretrains the new brain autoencoder for 100 epochs, and then trains the brain encoder, decoder and projections for 200 alignment epochs. Its configured pairs are vision/brain and text/brain.

The current decoder output is a vector in PCA feature space. The saved preprocessing statistics could support later conversion back toward voxel space, but the repository does not provide a complete image reconstruction or raw-fMRI reconstruction application.

## Source map and validation scope

| Path                        | Role                                                               |
| --------------------------- | ------------------------------------------------------------------ |
| [main.py](../main.py)       | Configuration, model construction, run directories and checkpoints |
| [models/](../models/)       | Domain encoders/decoders, projections and shared workspace         |
| [losses/](../losses/)       | Contrastive and reconstruction losses                              |
| [data/](../data/)           | COCO, NSD and dummy dataset loaders; legacy tokenizer              |
| [pipeline/](../pipeline/)   | Training, diagnostics and evaluation                               |
| [scripts/](../scripts/)     | Dataset download and feature preparation                           |
| [config/](../config/)       | Vision/text and experimental brain configurations                  |
| [notebooks/](../notebooks/) | Vision/text inspection notebook                                    |
| [DEVLOG.md](../DEVLOG.md)   | Historical experiments, corrections and reported results           |

The repository currently has no automated test suite or CI workflow. There is a dummy dataset for debugging, but its random features do not establish learning performance or validate the real data pipeline. Documentation review and syntax checks cannot replace environment, data-alignment and experiment reproduction checks.

## Research references

The development log identifies these papers as the architectural background. Their published results are separate from this prototype's experiments.

- Benjamin Devillers, Léopold Maytié and Rufin VanRullen, [Semi-supervised Multimodal Representation Learning through a Global Workspace](https://arxiv.org/abs/2306.15711).
- Léopold Maytié, Roland Bertin Johannet and Rufin VanRullen, [Multimodal Dreaming: A Global Workspace Approach to World Model-Based Reinforcement Learning](https://arxiv.org/abs/2502.21142).
