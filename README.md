# RARV-SMM: Semantic Mismatch as a Novel DeepFake Challenge

Code for *"Are DeepFakes Realistic Enough? Exploring Semantic Mismatch as a Novel
Challenge."* This repository contains only the **new** contributions of the paper:

1. The **RARV-SMM data construction pipeline** — building a fifth class
   (Real Audio-Real Video with Semantic Mismatch) from VoxCeleb2, in three
   variants of increasing audio-visual divergence (V1: same identity/different
   context, V2: different identity/same gender, V3: different identity/different
   gender).
2. The **semantic reinforcement strategy** — a frozen ImageBind audio-visual
   cosine similarity score, concatenated as an extra feature into the final
   classifier of three baseline architectures (FGMDF, FGI, AVDF).

It does **not** vendor the baseline SOTA detectors themselves (FGMDF, FGI, AVDF/MRDF) —
those are third-party repositories cited in the paper. The model files here
*extend* those backbones and expect them to be cloned separately and placed on
`PYTHONPATH` (see Setup below).

## Repository layout

```
data/                   # empty on git; populate locally (raw VoxCeleb2 / FakeAVCeleb / LAV-DF, processed RARV-SMM clips)
models/
  semantic_scorer.py    # frozen ImageBind audio-visual cosine similarity scorer
  fgmdf_semantic.py      # GAT_video_audio_semantic_v3 (extends FGMDF)
  fgi_semantic.py         # My_Network_Semantic (extends FGI)
  avdf_semantic.py        # AVDF_Multiclass_Semantic (extends AVDF/MRDF)
  weights/                # trained checkpoints (.pth/.ckpt), git-ignored
utils/
  dataset.py             # Multimodal_dataset_semantic + collate_fn_semantic (FGMDF/FGI)
  dataset_avdf.py         # FakeavcelebSemantic(DataModule) (AVDF)
  transforms.py           # ffmpeg-based clip standardization used in data construction
  logger.py               # lightweight epoch logger
configs/
  default.yaml            # hyperparameters from the paper's Implementation Details
scripts/
  01_index_voxceleb2.py             # Phase 1: index VoxCeleb2 audio/video files
  02_filter_speakers.py             # Phase 2: select speakers/sessions per variant
  03_generate_pairing_plan.py       # Phase 3: build the audio/video pairing plan
  04_process_clips.py               # Phase 4: standardize + mux into RARV-SMM clips
  05_generate_metadata.py           # Phase 5: per-clip metadata CSV
  06_combine_rarv_smm_datasets.py   # Phase 6: combine V1/V2/V3 into train/test splits
  compute_semantic_scores.py        # pre-compute ImageBind scores for a txt split
  train.py / train_avdf.py          # five-class training (FGMDF/FGI; AVDF separately)
  test.py                           # evaluation + metrics
  inference.py                      # single-clip inference
```

## Setup

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/ImageBind.git

# Clone the baseline backbones extended by models/*_semantic.py and add to PYTHONPATH:
git clone https://github.com/yinqi04/Fine-grained-Multimodal-DeepFake-Classification.git  # FGMDF
git clone https://github.com/mAst97/FGI.git                                                # FGI
git clone https://github.com/Zhixi-Cai/MRDF.git                                            # AVDF
export PYTHONPATH=$PYTHONPATH:/path/to/Fine-grained-Multimodal-DeepFake-Classification:/path/to/FGI:/path/to/MRDF
```

## Building the RARV-SMM dataset

```bash
python scripts/01_index_voxceleb2.py --audio-dir vox2_dev_aac/dev/aac --video-dir vox2_dev_mp4/dev/mp4 --output dataset_index.json
python scripts/02_filter_speakers.py --variant v1 --index dataset_index.json --output selected_speakers.json
python scripts/03_generate_pairing_plan.py --variant v1 --selected selected_speakers.json --index dataset_index.json --output pairing_plan.csv
python scripts/04_process_clips.py --variant v1 --plan pairing_plan.csv --output-dir output_clips
python scripts/05_generate_metadata.py --variant v1 --plan pairing_plan.csv --output-dir output_clips --output output_clips/rarv_smm_v1_metadata.csv --speaker-metadata vox2_meta.csv
```

Repeat for `--variant v2` / `v3` (V2/V3 additionally need `--metadata` and `--exclude-v1`/`--exclude-v2`),
then combine all three into unified train/test splits:

```bash
python scripts/06_combine_rarv_smm_datasets.py --split train \
  --v1-dir output_clips --v1-meta output_clips/rarv_smm_v1_metadata.csv \
  --v2-dir output_clips_v2 --v2-meta output_clips_v2/rarv_smm_v2_metadata.csv \
  --v3-dir output_clips_v3 --v3-meta output_clips_v3/rarv_smm_v3_metadata.csv \
  --output-dir output_clips_rarv_smm_train --output-meta output_clips_rarv_smm_train/rarv_smm_train_metadata.csv
```

## Semantic reinforcement: training and evaluation

```bash
# 1. Pre-compute frozen ImageBind scores for a given variant's train/test split
python scripts/compute_semantic_scores.py --txt_files data_path/train_path_5class_v1.txt data_path/test_path_5class_v1.txt \
  --output semantic_scores_v1.json

# 2. Train (FGMDF or FGI)
python scripts/train.py --model fgmdf --variant v1 \
  --train_txt data_path/train_path_5class_v1.txt --test_txt data_path/test_path_5class_v1.txt \
  --scores_json semantic_scores_v1.json

# AVDF uses PyTorch Lightning conventions:
python scripts/train_avdf.py --data_root data/combined_5class --scores_json semantic_scores_v1.json

# 3. Evaluate
python scripts/test.py --model fgmdf --checkpoint summary/weight/5class_fgmdf_v1_imagebind/12.pth \
  --test_txt data_path/test_path_5class_v1.txt --scores_json semantic_scores_v1.json --output_dir results/fgmdf_v1

# 4. Single-clip inference
python scripts/inference.py --model fgmdf --checkpoint summary/weight/5class_fgmdf_v1_imagebind/12.pth \
  --frame_dir /path/to/frames --audio_path /path/to/audio.wav
```

## Citation

If you use this code, please cite the paper (see the PDF in this repository's
release / arXiv listing for full bibliographic details).
