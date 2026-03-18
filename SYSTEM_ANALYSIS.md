# SAIS System Analysis: Architecture, Components & Execution Guide

## 📋 Executive Summary

**SAIS** (Surgical Activity Intelligence System) is a deep learning system that decodes surgical videos to automatically identify and analyze intraoperative surgical activity. It leverages Vision Transformers (ViT) with supervised contrastive learning to analyze robotic surgical videos and provide insights into:

- **Surgical Steps**: Identifying phases/steps within a procedure
- **Surgical Gestures**: Recognizing specific hand movements and actions
- **Surgical Skills**: Assessing the quality and precision of surgical actions
- **Frame Importance**: Highlighting which video frames are most important for each prediction

The system generalizes across videos, surgeons, hospitals, and surgical procedures.

---

## 🏗️ System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: Surgical Video                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: VIDEO PREPROCESSING                                       │
├─────────────────────────────────────────────────────────────────────┤
│ • Extract frames from video (video_to_frames.sh)                   │
│ • Generate optical flow maps (ptlflow RAFT)                        │
│ • Create path mappings for efficient processing                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │  RGB Frames          │  │  Optical Flow Maps   │
        │  (images/)           │  │  (flows/)            │
        └─────────┬────────────┘  └──────────┬───────────┘
                  │                          │
                  │                          ▼
                  │      ┌─────────────────────────────────────────┐
                  │      │  Stage 2: FEATURE EXTRACTION             │
                  │      ├─────────────────────────────────────────┤
                  │      │ • Use DINO ViT-small (self-supervised)  │
                  │      │ • Extract spatial features per frame    │
                  │      │ • Output: 384-dim embeddings (h5 format)│
                  │      │ • Results stored in results/ directory  │
                  │      └─────────────────────────────────────────┘
                  │                          │
                  └────────────┬─────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │  RGB Features + Flow Features                │
        │  (384-dim vectors per frame)                 │
        └───────────────────┬──────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │  Stage 3: TEMPORAL MODELING & CLASSIFICATION │
        ├──────────────────────────────────────────────┤
        │ • Segment features into snippets (5 frames)  │
        │ • Pass through Transformer Encoder           │
        │ • Apply Prototype-based Learning             │
        │ • Supervised Contrastive Learning Loss       │
        │ • Output: Class predictions per snippet      │
        │ • Extract: Attention maps (frame importance) │
        └───────────────────┬──────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │  PREDICTIONS & INTERPRETABILITY               │
        ├──────────────────────────────────────────────┤
        │ • Gesture classifications                    │
        │ • Surgical step predictions                  │
        │ • Skill assessment scores                    │
        │ • Frame importance scores (attention)        │
        └─────────────────────────────────────────────┘
```

---

## 🔧 Component Description

### 1. **Video Preprocessing Components**

#### `video_to_frames.sh`
- **Purpose**: Convert MP4 video into individual frame images
- **Output**: Frames stored in `images/{videoname}/` directory
- **Dependencies**: ffmpeg via imageio

#### `generate_paths.py`
- **Purpose**: Create CSV files with paths to frames and optical flow files
- **Outputs**: 
  - `paths/{videoname}_FramePaths.csv`
  - `paths/{videoname}_FlowPaths.csv`
- **Key params**: Video name, root path

---

### 2. **Feature Extraction Components**

#### `extract_representations.py` + DINO Model
- **Purpose**: Extract deep features from frames using a pre-trained Vision Transformer
- **Feature Extractor**: DINO (ViT-small-16)
  - Self-supervised pre-training on natural images
  - Provides rich spatial understanding
  - Output: 384-dimensional embeddings per frame
- **Processes**:
  - RGB frame features
  - Optical flow features
- **Output Location**: `results/` directory (HDF5 format)
- **Key params**:
  ```
  --arch vit_small
  --patch_size 16
  --batch_size_per_gpu 1024 (RGB) / 256 (Flow)
  --save_type h5
  --optical_flow / --optical_flow_to_reps
  ```

#### DINO Pre-trained Weights
- **File**: `dino_deitsmall16_pretrain.pth` (~350 MB)
- **Location**: `scripts/dino-main/outputs/`
- **Source**: Facebook Research DINO repository
- **Required**: Must be downloaded manually before feature extraction

---

### 3. **Optical Flow Generation**

#### ptlflow with RAFT Model
- **Purpose**: Generate optical flow (motion) maps between frames
- **Model**: RAFT (Recurrent All-Pairs Field Transforms)
- **Output**: Flow vectors stored alongside frame features
- **Used by**: Temporal modeling to capture motion information

---

### 4. **Model Architecture Components**

#### Core Classes (in `prepare_model.py`)

**Vision Transformer Encoder**
- Dimension: 384 (from DINO)
- Pre-trained: Yes (frozen or fine-tunable)
- Modality fusion: RGB + Optical Flow concatenation

**Temporal Modeling (Transformer Encoder)**
- Number of layers: Configurable
- Self-attention mechanisms
- Frame-level importance weighting via attention maps

**Prototype-based Classification Head**
- Uses supervised contrastive learning
- Learns prototypes (representative exemplars) per class
- Enables interpretable predictions

---

### 5. **Training & Inference Pipeline**

#### `train.py` + `run_experiments.py`
**Main orchestrator** that:
1. Loads data via `prepare_dataset.py`
2. Creates model via `prepare_model.py`
3. Runs training via `perform_training.py`
4. Computes metrics via `prepare_miscellaneous.py`

**Key features**:
- Multi-GPU distributed training (torch.distributed.launch)
- Fold-based cross-validation
- Flexible task support: Classification, Gesture Recognition, Skill Assessment
- Inference mode: Load trained models and predict on new videos

**Parameters**:
- Model: ViT-based temporal model
- Task: Prototypes (contrastive learning)
- Modalities: RGB-Flow (multimodal fusion)
- Batch size: 2 (inference), higher for training
- Learning rate: 0.1 (adaptive)
- Number of classes: Task-specific (2-10)

---

### 6. **Result Processing**

#### `process_inference_results.py`
- **Purpose**: Post-process raw model predictions into valid predictions
- **Converts**: Frame-level predictions into snippet/action-level predictions
- **Output**: Final results ready for interpretation

---

## 📊 Data Flow Diagram

```
Surgical Video (.mp4)
        │
        ├─► video_to_frames.sh ──► images/{video_name}/*.jpg
        │
        └─► generate_paths.py ──► paths/{video_name}_FramePaths.csv
                                  └─► paths/{video_name}_FlowPaths.csv
                                      
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 │
        ┌─────────────────────┐  ┌──────────────────┐  │
        │ Extract RGB         │  │ Extract Optical  │  │
        │ Features via DINO   │  │ Flow via RAFT    │  │
        │ (extract_reps.py)   │  │ (extract_reps.py)│  │
        └─────────┬───────────┘  └────────┬─────────┘  │
                  │                       │            │
                  └───────────┬───────────┘            │
                              ▼                        │
                ┌────────────────────────────┐         │
                │ Results: RGB + Flow        │         │
                │ Features (HDF5)            │         │
                │ results/                   │         │
                └────────────┬───────────────┘         │
                             │                        │
                             │◄───────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │ prepare_dataset.py: Load features   │
        │ Create data loader with snippets    │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │ prepare_model.py: Build model       │
        │ ViT Encoder + RNN + Prototypes     │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │ run_experiments.py / train.py:      │
        │ Training or Inference Mode         │
        │ Load params/prototypes from        │
        │ params/{Fold_X}/                   │
        └────────────┬───────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
    Inference Mode          Training Mode
    (--inference flag)      (accumulate metrics)
        │                         │
        └─────────┬───────────────┘
                  │
                  ▼
    ┌────────────────────────────────────┐
    │ process_inference_results.py        │
    │ Convert frame predictions to        │
    │ action-level predictions           │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌───────────────────────────────────────────┐
    │ FINAL OUTPUT                              │
    │ • Gesture classifications per action      │
    │ • Surgical step segmentation             │
    │ • Skill scores                           │
    │ • Frame importance (attention maps)      │
    └───────────────────────────────────────────┘
```

---

## 🚀 Building & Execution Guide

### Prerequisites

1. **System Requirements**
   - GPU with CUDA support (NVIDIA recommended)
   - At least 16 GB VRAM for feature extraction
   - 8+ GB RAM for model training
   - ~100 GB disk space per surgical video dataset

2. **Software Requirements**
   - Python 3.9.7
   - CUDA 11.x compatible PyTorch 1.8.0
   - ffmpeg (for video processing)

---

### Step 1: Clone & Setup Environment

```bash
# Clone repository
git clone https://github.com/danikiyasseh/SAIS.git
cd SAIS

# Create conda environment
conda create -n SAIS python=3.9.7 -y
conda activate SAIS

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

#### Dependencies Breakdown:
| Package | Purpose |
|---------|---------|
| torch==1.8.0 | Deep learning framework |
| torchvision==0.9.0 | Computer vision utilities |
| opencv-python | Image processing |
| timm==0.6.5 | Vision Transformer models |
| ptlflow==0.2.5 | Optical flow computation (RAFT) |
| h5py | Feature storage/loading |
| imageio[ffmpeg] | Video frame extraction |
| moviepy | Video manipulation |
| scikit-learn, scipy | ML utilities |

---

### Step 2: Modify PyTorch Transformer Module

⚠️ **Critical Step**: SAIS requires attention map outputs which aren't available in standard PyTorch 1.8.0

```bash
# Navigate to your conda environment
# Example path: C:\Users\User\anaconda3\envs\SAIS\lib\python3.9\site-packages\torch\nn\modules\transformer.py

# Edit transformer.py:
# Line 181: Add 'attn' as second output of mod function
# Line 294: Remove [0] indexing and add 'attn' as output

# The transformer's forward method should return (output, attention_weights)
```

**Detailed Instructions**:
- Locate: `{anaconda_env}\lib\python3.9\site-packages\torch\nn\modules\transformer.py`
- Edit MultiheadAttention output to include attention weights
- Edit TransformerEncoderLayer/TransformerEncoder to propagate attention

---

### Step 3: Download Pre-trained DINO Weights

```bash
# Create output directory
mkdir -p SAIS/scripts/dino-main/outputs

# Download DINO pre-trained weights (ViT-small-16)
# Option A: Manual download
cd SAIS/scripts/dino-main/outputs
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
cd ../../..

# Option B: From Python
import urllib.request
url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
path = "SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth"
urllib.request.urlretrieve(url, path)
```

---

### Step 4: Prepare Input Video

```bash
# Create videos directory if not exists
mkdir -p SAIS/videos

# Copy your surgical video(s) into this directory
# Example:
# SAIS/videos/surgery_001.mp4
# SAIS/videos/surgery_002.mp4

# Each video should have unique filename
```

---

### Step 5: Download/Prepare Model Parameters

For inference, you need trained model parameters:

```bash
# Structure required:
# SAIS/
#   scripts/
#   params/
#     Fold_0/
#       params.zip          # Core architecture parameters
#       prototypes.zip      # Prototype parameters for contrastive learning
#     Fold_1/
#       ...

# These are obtained after training on your specific surgical video dataset
# They encode task-specific knowledge (gesture classes, skill patterns, etc.)
```

**If you don't have trained models**:
- Train on your own dataset (see **Step 8**)
- Or obtain pre-trained models from the authors

---

### Step 6: Run Inference Pipeline (End-to-End)

#### Option A: Automated Bash Script

```bash
# Navigate to repository root
cd SAIS

# Run complete pipeline for a video
bash ./SAIS/main.sh -f surgery_001

# What main.sh does (automatically):
# 1. video_to_frames.sh: Extract frames
# 2. generate_paths.py: Create path CSVs
# 3. extract_representations.py (3x calls):
#    - Optical flow extraction
#    - RGB feature extraction  
#    - Flow feature extraction
# 4. run_experiments.py: Perform inference
# 5. process_inference_results.py: Post-process predictions
```

#### Option B: Manual Step-by-Step

```bash
# 1. Extract frames from video
bash ./SAIS/scripts/video_to_frames.sh -f surgery_001

# 2. Generate path mappings
python ./SAIS/scripts/generate_paths.py -f surgery_001 -p ./SAIS/

# 3. Extract optical flow
python ./SAIS/scripts/extract_representations.py \
  --arch vit_small \
  --patch_size 16 \
  --model_type ViT_SelfSupervised_ImageNet \
  --batch_size_per_gpu 2 \
  --data_path ./SAIS/ \
  --data_list Custom \
  --save_type h5 \
  --optical_flow

# 4. Extract RGB features (distributed)
python -m torch.distributed.launch \
  ./SAIS/scripts/extract_representations.py \
  --arch vit_small \
  --patch_size 16 \
  --model_type ViT_SelfSupervised_ImageNet \
  --batch_size_per_gpu 1024 \
  --data_path ./SAIS/ \
  --data_list Custom \
  --save_type h5

# 5. Extract flow features (distributed)
python -m torch.distributed.launch \
  ./SAIS/scripts/extract_representations.py \
  --arch vit_small \
  --patch_size 16 \
  --model_type ViT_SelfSupervised_ImageNet \
  --batch_size_per_gpu 256 \
  --data_path ./SAIS/ \
  --data_list Custom \
  --save_type h5 \
  --optical_flow_to_reps

# 6. Perform inference
python -m torch.distributed.launch \
  ./SAIS/scripts/run_experiments.py \
  -p ./SAIS/ \
  -data Custom_Gestures \
  -d Custom \
  -m ViT \
  -enc ViT_SelfSupervised_ImageNet \
  -t Prototypes \
  -mod RGB-Flow \
  -dim 384 \
  -bs 2 \
  -lr 1e-1 \
  -nc 2 \
  -bc \
  -sa \
  -domains in_vs_out \
  -ph Custom_inference \
  -dt reps \
  -e 1 \
  -f 1 \
  --inference

# 7. Process results
python ./SAIS/scripts/process_inference_results.py -p ./SAIS/
```

---

### Step 7: Output Directory Structure

After processing, your directory will contain:

```
SAIS/
├── videos/                          # Input videos
│   └── surgery_001.mp4
├── images/                          # NEW: Extracted frames
│   └── surgery_001/
│       ├── frame_00000.jpg
│       ├── frame_00001.jpg
│       └── ...
├── flows/                           # NEW: Optical flow maps
│   └── surgery_001/
│       ├── flow_00000.npy
│       └── ...
├── paths/                           # NEW: Path mappings (CSVs)
│   ├── Custom_FramePaths.csv
│   └── Custom_FlowPaths.csv
├── results/                         # NEW: Extracted features (HDF5)
│   ├── rgb_features_custom.h5
│   └── flow_features_custom.h5
└── predictions/                     # NEW: Inference results
    └── surgery_001_predictions.csv
```

---

### Step 8: Training Your Own Models

If you have labeled surgical video data:

```bash
# Example: Train gesture classifier
python -m torch.distributed.launch \
  ./SAIS/scripts/run_experiments.py \
  -p ./SAIS/ \
  -data Custom_Gestures \
  -d Custom \
  -m ViT \
  -enc ViT_SelfSupervised_ImageNet \
  -t Prototypes \
  -mod RGB-Flow \
  -dim 384 \
  -bs 8 \
  -lr 0.1 \
  -nc 10 \
  -bc \
  -sa \
  -domains gesture_classification \
  -ph train_val_test \
  -dt reps \
  -e 50 \
  -f 5 \
  -tf 1.0

# Parameters explained:
# -bs 8: Batch size (adjust for your GPU memory)
# -lr 0.1: Learning rate
# -nc 10: Number of gesture classes
# -e 50: Train for 50 epochs
# -f 5: 5-fold cross-validation
# -tf 1.0: Use 100% of training data
# Removes --inference flag for training mode
```

**Required structure for training**:
```
SAIS/
├── (preprocessing done above)
├── annotations/
│   └── Custom_annotations.csv  # Columns: video_name, frame_num, gesture_label
└── params/
    └── Fold_0/                 # Will be created after training
        ├── params.zip
        └── prototypes.zip
```

---

## 📈 Performance Considerations

| Operation | GPU Memory | Time Est. | Notes |
|-----------|-----------|----------|-------|
| Feature extraction (10 min video @ 20 FPS) | 8-12 GB | 2-5 min | Batch size affects speed |
| Inference per 10 min video | 4 GB | 30 sec | Depends on model complexity |
| Training (full dataset) | 16+ GB | Hours | Distributed across GPUs |
| Optical flow computation | 6 GB | 3-10 min | RAFT model is intensive |

---

## 🔍 Key Configuration Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `snippet_length` | 5 | 1-30 | Temporal context window (frames) |
| `frame_skip` | 1 | 1-10 | Downsampling factor |
| `batch_size` | 8 | 1-256 | Training speed vs GPU memory |
| `learning_rate` | 0.1 | 1e-4 to 1e-1 | Convergence speed & stability |
| `rep_dim` | 384 | - | Feature dimension (fixed for DINO) |
| `n_classes` | - | 2-∞ | Task-specific |

---

## 🐛 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "ModuleNotFoundError: No module named 'dino'" | DINO not in path | Ensure `sys.path.append('./SAIS/scripts/dino-main')` in extract_representations.py |
| CUDA out of memory | Batch size too large | Reduce `--batch_size_per_gpu` parameter |
| "AttributeError: attention" | PyTorch transformer not modified | Follow Step 2: Modify transformer.py |
| Missing DINO weights | File not downloaded | Download from FB Research link (Step 3) |
| No predictions generated | Model params missing | Ensure `params/Fold_X/` exists with params.zip |
| Video not detected | Wrong directory structure | Place in `SAIS/videos/`, not subdirectories |

---

## 📚 Component Dependency Graph

```
run_experiments.py (Orchestrator)
├── prepare_model.py
│   └── Vision Transformer + Prototypes
├── prepare_dataset.py
│   ├── feature files (HDF5)
│   └── annotations (CSV)
├── train.py
│   ├── perform_training.py (single_epoch)
│   ├── prepare_miscellaneous.py (metrics)
│   └── prepare_dataset.py (DataLoader)
├── extract_representations.py
│   ├── DINO (dino-main/main_dino.py)
│   └── ptlflow (RAFT optical flow)
├── generate_paths.py (CSV generation)
│   └── video metadata
└── process_inference_results.py
    └── raw predictions → final outputs
```

---

## 📖 References

- **Paper**: ["A Vision Transformer for Decoding Surgery" - Nature Biomedical Engineering](https://www.nature.com/articles/s41551-023-01010-8)
- **DINO**: [Facebook Research - Emerging Properties in Self-Supervised ViTs](https://github.com/facebookresearch/dino)
- **ptlflow**: [PyTorch Lightning Optical Flow](https://github.com/hmorimitsu/ptlflow)
- **Vision Transformers**: [Original ViT Paper - Google Research](https://arxiv.org/abs/2010.11929)

---

## 📝 License

CC BY-NC 4.0 - See LICENSE file for details
