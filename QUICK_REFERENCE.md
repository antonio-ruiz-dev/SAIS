# SAIS Quick Reference Guide

## 🎯 What is SAIS?

**SAIS** = Surgical Activity Intelligence System

A deep learning system that analyzes surgical videos to automatically identify:
- ✂️ Surgical steps (phases of the operation)
- 🤚 Surgical gestures (specific hand movements)
- ⭐ Surgical skill (quality of actions)
- 👁️ Frame importance (which video frames matter most)

**Technology**: Vision Transformer (ViT) + Supervised Contrastive Learning

**Key Paper**: [Nature Biomedical Engineering - 2023](https://www.nature.com/articles/s41551-023-01010-8)

---

## ⚡ Quick Start (5 Commands)

```bash
# 1. Clone & setup
git clone https://github.com/danikiyasseh/SAIS.git
cd SAIS
conda create -n SAIS python=3.9.7 -y && conda activate SAIS
pip install -r requirements.txt && pip install -e .

# 2. Download DINO weights (CRITICAL)
mkdir -p SAIS/scripts/dino-main/outputs
# Download from: https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
# OR use: wget [URL] -O SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth

# 3. Modify PyTorch transformer (CRITICAL)
# Edit: {CONDA_ENV}/lib/python3.9/site-packages/torch/nn/modules/transformer.py
# Line 181 & 294: Return attention weights alongside output

# 4. Place video in videos/ folder
mkdir -p SAIS/videos
# Copy your_video.mp4 into SAIS/videos/

# 5. Run end-to-end inference
cd SAIS
bash ./main.sh -f your_video
```

**Output**: Predictions in `predictions/` folder with gesture labels & confidence scores

---

## 📂 File Structure

```
SAIS/
├── videos/                  # INPUT: Your surgical videos here
├── SAIS/
│   ├── main.sh             # Entry point - run this
│   ├── scripts/
│   │   ├── run_experiments.py        # Inference orchestrator
│   │   ├── extract_representations.py # Feature extraction
│   │   ├── generate_paths.py         # Path mapping
│   │   ├── train.py / perform_training.py
│   │   ├── prepare_model.py          # Model builder
│   │   ├── prepare_dataset.py        # Dataset loader
│   │   ├── dino-main/               # Feature extractor module
│   │   │   └── outputs/
│   │   │       └── dino_deitsmall16_pretrain.pth  # REQUIRED
│   │   └── video_to_frames.sh
│   └── params/
│       └── Fold_0/
│           ├── params.zip            # Model weights
│           └── prototypes.zip        # Class prototypes
├── images/                 # Generated: Frames extracted from video
├── flows/                  # Generated: Optical flow maps
├── paths/                  # Generated: CSV with frame/flow paths
├── results/                # Generated: Extracted features (HDF5)
└── predictions/            # OUTPUT: Final predictions
```

---

## 🔄 Data Flow Pipeline

```
Surgical Video (MP4)
    ↓
[video_to_frames.sh] → Extract 2000 frames @ 20 FPS
    ↓
[generate_paths.py] → Create path CSVs
    ↓
[RAFT] → Optical flow (motion)     [DINO] → RGB features
    ↓                                  ↓
[HDF5 Storage] ←────────────────────┘
    ↓
[prepare_dataset.py] → Load & batch into 5-frame snippets
    ↓
[ViT Model] → Process through attention layers
    ↓
[Prototypes] → Classify each snippet
    ↓
[process_inference_results.py] → Aggregate to action labels
    ↓
PREDICTIONS: Gesture type, confidence, frame importance
```

---

## 🛠️ Key Configuration Parameters

| Parameter | What it does | Typical Value |
|-----------|-------------|---------------|
| `-f {videoname}` | Which video to process | `surgery_001` |
| `-m ViT` | Model architecture | Always `ViT` |
| `-mod RGB-Flow` | Use RGB + optical flow | Recommended |
| `-nc {num_classes}` | Number of gesture types | 2-10 typically |
| `-bs {batch_size}` | Batch size for processing | 2 (inference) / 8+ (training) |
| `-t Prototypes` | Learning method | Always `Prototypes` |
| `--inference` | Run in inference mode | Add this flag for predictions |
| `-e {epochs}` | Epochs for training | 50 typical |
| `-f {folds}` | K-fold cross-validation | 5 typical |

---

## 📊 Component Relationships

```
                    ORCHESTRATOR
                   run_experiments.py
                    /   |   |   \
                   /    |   |    \
          [MODEL]  [DATASET] [TRAINING] [INFERENCE]
            /           |         |        \
           /            |         |         \
      prepare_model   prepare_  perform_    load params/
         .py           dataset   training    prototypes
                        .py      .py
                        
      ↓ INPUT           ↓         ↓ LOSS      ↓ OUTPUT
      
   HDF5 Features    Labeled      Contrastive  Predictions
   + CSV Paths      snippets     loss          + attention
```

---

## 🎬 Common Workflows

### Workflow 1: Analyze New Surgical Video (Inference Only)

```bash
# Prerequisites:
# ✓ Environment setup (Step 1-2 from Quick Start)
# ✓ DINO weights downloaded
# ✓ Pre-trained model in params/Fold_0/

# Execute:
cp your_surgical_video.mp4 SAIS/videos/
bash SAIS/main.sh -f your_surgical_video

# Results:
# Check: SAIS/predictions/
# Contains: gesture labels, confidence scores, frame importance
```

### Workflow 2: Train Model on Your Dataset

```bash
# 1. Prepare dataset:
#    - Extract videos into images/
#    - Create annotations CSV: annotations/dataset_annotations.csv
#    - Format: video_name, frame_num, gesture_label

# 2. Run feature extraction (same as inference)
bash SAIS/main.sh -f video_1
bash SAIS/main.sh -f video_2
# ... repeat for all videos

# 3. Train model:
python -m torch.distributed.launch \
  SAIS/scripts/run_experiments.py \
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
  -e 50 \
  -f 5

# 4. Model saved to: SAIS/params/Fold_X/
```

### Workflow 3: Evaluate Model Performance

```bash
# Same as training but add more validation data
# Metrics computed: Accuracy, F1-score, Confusion matrix
# Results saved to: ptlflow_logs/

cat ptlflow_logs/log_run.txt
```

---

## ⚙️ Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Video Processing** | ffmpeg, OpenCV | Frame extraction, video I/O |
| **Feature Extraction** | DINO ViT-small | Self-supervised visual representations |
| **Optical Flow** | RAFT (ptlflow) | Motion estimation |
| **Temporal Modeling** | PyTorch Transformer | Temporal sequence processing |
| **Learning Paradigm** | Supervised Contrastive Learning | Metric learning with prototypes |
| **Deep Learning** | PyTorch 1.8.0 | Training & inference |
| **Distributed Training** | torch.distributed.launch | Multi-GPU support |

---

## 🔧 Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: dino` | Path not configured | Check sys.path in extract_representations.py |
| `CUDA out of memory` | Batch size too large | Reduce `--batch_size_per_gpu` |
| `RuntimeError: attention` | PyTorch transformer not modified | Follow Step 3 in Quick Start |
| `FileNotFoundError: dino_weights.pth` | Weights not downloaded | Download & place in correct folder |
| `KeyError` in data loading | Wrong feature format | Ensure HDF5 files created successfully |
| No predictions generated | Missing params.zip/prototypes.zip | Download pre-trained models or train your own |

---

## 💡 Key Insights

1. **Pre-training is Critical**: DINO features are pre-trained on ImageNet. This transfer learning is what makes SAIS work.

2. **Multimodal Fusion**: RGB + Flow features capture both appearance and motion, crucial for surgical understanding.

3. **Interpretability**: Attention maps from transformer show which frames drive each prediction.

4. **Generalization**: Model tested on 3 hospitals, 2 continents, shows strong cross-domain transfer.

5. **Supervised Contrastive Learning**: Learns discriminative prototypes per gesture class instead of just softmax regression.

---

## 📝 Output Interpretation

**Example Prediction Output** (CSV format):

```
video_name,action_id,start_frame,end_frame,gesture_label,confidence,frame_importance
surgery_001,1,0,49,knot_tying,0.92,"[0.1, 0.2, 0.85, 0.9, 0.95]"
surgery_001,2,50,99,tissue_grasping,0.88,"[0.3, 0.7, 0.6, 0.4, 0.2]"
surgery_001,3,100,149,suturing,0.95,"[0.05, 0.15, 0.92, 0.88, 0.91]"
```

- **gesture_label**: Predicted surgical gesture (trained classes)
- **confidence**: Probability score (0-1)
- **frame_importance**: Attention weights per frame in action (shows which frames matter most)

---

## 🎓 Learning Resources

- **[Original SAIS Paper](https://www.nature.com/articles/s41551-023-01010-8)** - Full technical details
- **[Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929)** - Core architecture
- **[DINO: Self-supervised ViTs](https://github.com/facebookresearch/dino)** - Feature extractor
- **[RAFT Optical Flow](https://github.com/princeton-vl/RAFT)** - Motion estimation
- **[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)** - Learning paradigm

---

## 📞 Support & Issues

- **GitHub**: [danikiyasseh/SAIS](https://github.com/danikiyasseh/SAIS)
- **Citation**: See repository README for BibTeX reference
- **License**: CC BY-NC 4.0 (Non-commercial use)

---

## 🚀 Tips for Success

1. **Good GPU is Essential**: Need at least 8 GB VRAM for feature extraction
2. **Preprocess Videos**: Ensure consistent frame rates (20 FPS typical in surgical videos)
3. **Annotation Quality**: For training, accurate labels are critical
4. **Cross-validation**: Use 5-fold CV to get robust performance estimates
5. **Monitor Training**: Watch loss curves in ptlflow_logs for convergence
6. **Freeze Encoder**: Usually best to freeze DINO weights (found during development)

