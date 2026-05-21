# SAIS Agentic AI - Usage Guide

## Overview

The **SAIS Agentic AI** automatically orchestrates the complete surgical video analysis workflow:

```
Surgical Videos → Feature Extraction → Model Training → Inference → Predictions
```

The agent:
- ✅ Validates environment and dependencies
- ✅ Extracts DINO features from surgical videos
- ✅ Trains Vision Transformer model with supervised contrastive learning
- ✅ Performs inference on new videos
- ✅ Generates execution reports and predictions

---

## Prerequisites

### 1. Clone SAIS Repository
```bash
git clone https://github.com/antonio-ruiz-dev/SAIS.git
cd SAIS
```

### 2. Setup Environment
```bash
conda create -n SAIS python=3.9.7 -y
conda activate SAIS
pip install -r requirements.txt
pip install -e .
```

### 3. Download DINO Pre-trained Weights
```bash
mkdir -p SAIS/scripts/dino-main/outputs
cd SAIS/scripts/dino-main/outputs
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
cd ../../..
```

### 4. Modify PyTorch Transformer (Critical!)
Edit `{CONDA_ENV}/lib/python3.9/site-packages/torch/nn/modules/transformer.py`:
- **Line 181**: Add `attn` as second output of `mod` function
- **Line 294**: Remove `[0]` indexing and add `attn` as second output

---

## Installation

### Copy Agent Script
```bash
cp sais_agent.py /path/to/SAIS/
```

---

## Usage

### Basic Command Structure
```bash
python sais_agent.py \
  --sais_root /path/to/SAIS \
  --dino_weights /path/to/SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos video_1 video_2 video_3 \
  --training_dataset_name "SurgicalGestures_v1" \
  --training_dataset_type "Custom" \
  --inference_videos test_video_1 test_video_2 \
  --epochs 50 \
  --batch_size 8 \
  --num_classes 10 \
  --report execution_report.json
```

### Minimal Example
```bash
python sais_agent.py \
  --sais_root . \
  --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos surgery_001 surgery_002 \
  --inference_videos surgery_test_001
```

---

## Parameters

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `--sais_root` | ✅ | Path | Root directory of SAIS repository |
| `--dino_weights` | ✅ | Path | Path to DINO pre-trained weights (`.pth` file) |
| `--training_videos` | ✅ | List | Training video names (without `.mp4`) |
| `--inference_videos` | ✅ | List | Inference video names (without `.mp4`) |
| `--training_dataset_name` | ❌ | String | Dataset name (default: `CustomSurgicalDataset`) |
| `--training_dataset_type` | ❌ | String | Dataset type (default: `Custom`) |
| `--epochs` | ❌ | Int | Training epochs (default: `50`) |
| `--batch_size` | ❌ | Int | Training batch size (default: `8`) |
| `--num_classes` | ❌ | Int | Number of gesture classes (default: `10`) |
| `--report` | ❌ | Path | Output report path (default: `sais_report.json`) |

---

## Workflow Stages

### Stage 1: Environment Validation
- Checks Python packages (torch, transformers, torchvision)
- Validates SAIS root directory structure
- Verifies DINO weights availability

### Stage 2: Feature Extraction (Training Videos)
For each training video:
1. Extract frames from MP4 video (20 FPS)
2. Generate path CSVs mapping frames to storage
3. Extract DINO visual features + RAFT optical flow
4. Save features to HDF5 files

### Stage 3: Model Training
- Load extracted features into batches (5-frame snippets)
- Train Vision Transformer with supervised contrastive learning
- Optimize prototypes for gesture classification
- Save model weights to `params/Fold_0/`

### Stage 4: Feature Extraction (Inference Videos)
Same as Stage 2, but for videos to be analyzed

### Stage 5: Inference
- Load trained model and prototypes
- Process each inference video
- Generate predictions with confidence scores
- Output frame importance (attention weights)
- Save results to `predictions/` folder

---

## Input Data Format

### Video Organization
```
SAIS/
├── videos/
│   ├── surgery_001.mp4
│   ├── surgery_002.mp4
│   ├── surgery_test_001.mp4
│   └── ...
```

**Requirements:**
- MP4 format
- At least 20 FPS recommended
- Deidentified surgical video (no patient identifiers)

### Annotation Format (for labeled training data)
Expected format for dataset annotations:
```
video_name,frame_number,gesture_label,start_frame,end_frame
surgery_001,0,knot_tying,0,49
surgery_001,50,tissue_grasping,50,99
surgery_001,100,suturing,100,149
```

---

## Output Structure

After execution, your SAIS directory will contain:

```
SAIS/
├── videos/                 # Input: Surgical videos
├── images/                 # Generated: Extracted frames
├── flows/                  # Generated: Optical flow maps
├── paths/                  # Generated: CSV with frame/flow paths
├── results/                # Generated: Extracted HDF5 features
├── params/
│   └── Fold_0/
│       ├── params.zip      # Trained model weights
│       └── prototypes.zip  # Learned gesture prototypes
├── predictions/            # Output: Inference predictions
└── execution_report.json   # Detailed execution log
```

---

## Output Format

### Predictions File
Location: `predictions/{video_name}_predictions.json`

```json
{
  "video_name": "surgery_test_001",
  "predictions": [
    {
      "action_id": 1,
      "start_frame": 0,
      "end_frame": 49,
      "gesture_label": "knot_tying",
      "confidence": 0.92,
      "frame_importance": [0.1, 0.2, 0.85, 0.9, 0.95]
    },
    {
      "action_id": 2,
      "start_frame": 50,
      "end_frame": 99,
      "gesture_label": "tissue_grasping",
      "confidence": 0.88,
      "frame_importance": [0.3, 0.7, 0.6, 0.4, 0.2]
    }
  ]
}
```

### Execution Report
Location: `execution_report.json`

Contains detailed log of all agent actions, timestamps, and status codes.

---

## Example Workflows

### Workflow 1: Train on Prostatectomy, Test on New Cases
```bash
python sais_agent.py \
  --sais_root . \
  --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos prostate_case_1 prostate_case_2 prostate_case_3 prostate_case_4 \
  --training_dataset_name "ProstatectomyGestures" \
  --training_dataset_type "ProstateSurgery" \
  --inference_videos prostate_test_1 prostate_test_2 \
  --epochs 50 \
  --batch_size 8 \
  --num_classes 7 \
  --report prostate_report.json
```

### Workflow 2: Multi-Procedure Training
```bash
python sais_agent.py \
  --sais_root . \
  --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos prostate_1 nephrectomy_1 prostate_2 nephrectomy_2 prostate_3 nephrectomy_3 \
  --training_dataset_name "MultiProcedureSurgeries" \
  --training_dataset_type "MixedProcedures" \
  --inference_videos prostate_new nephrectomy_new \
  --epochs 100 \
  --batch_size 16 \
  --num_classes 15 \
  --report multi_procedure_report.json
```

### Workflow 3: High-Precision Training (More Epochs)
```bash
python sais_agent.py \
  --sais_root . \
  --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos case_1 case_2 case_3 case_4 case_5 case_6 case_7 case_8 \
  --training_dataset_name "PrecisionModel" \
  --inference_videos test_1 test_2 test_3 \
  --epochs 200 \
  --batch_size 32 \
  --num_classes 12 \
  --report precision_report.json
```

---

## Troubleshooting

### Issue: `CUDA out of memory`
**Solution**: Reduce batch size
```bash
--batch_size 4  # Default is 8
```

### Issue: `ModuleNotFoundError: dino`
**Solution**: Ensure DINO package is in Python path
```bash
export PYTHONPATH="${PYTHONPATH}:./SAIS/scripts/dino-main"
```

### Issue: `RuntimeError: attention` when training
**Solution**: Verify PyTorch transformer modification was applied correctly
- Re-read the PyTorch transformer modifications section
- Restart Python/Jupyter kernel

### Issue: `FileNotFoundError: dino_weights.pth`
**Solution**: Download DINO weights to correct location
```bash
mkdir -p SAIS/scripts/dino-main/outputs
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth \
  -O SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth
```

### Issue: No predictions generated
**Solution**: Check if model training completed successfully
```bash
ls -la params/Fold_0/params.zip params/Fold_0/prototypes.zip
```
Both files must exist and have non-zero size.

---

## Performance Optimization

### GPU Acceleration
Ensure CUDA-enabled GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Multi-GPU Training
Modify agent to support multiple GPUs:
```bash
# Automatically uses all available GPUs via torch.distributed.launch
```

### Memory Management
- **Reduce batch size** for limited VRAM (e.g., `--batch_size 4`)
- **Reduce feature dimension** (modify agent code: `feature_dim=256`)
- **Process videos in batches** rather than all at once

### Speed Optimization
- **Increase batch size** if GPU VRAM permits (e.g., `--batch_size 32`)
- **More epochs** leads to better accuracy but longer training
- **Parallel processing**: Run inference on multiple videos simultaneously

---

## Integration with Production Systems

### Monitor Training Progress
```bash
tail -f ptlflow_logs/log_run.txt
```

### Automated Scheduling
```bash
# Cron job for daily inference (2 AM)
0 2 * * * cd /path/to/SAIS && python sais_agent.py --sais_root . --dino_weights ... >> sais_cron.log 2>&1
```

### RESTful API Wrapper
```python
from fastapi import FastAPI
from pathlib import Path
from sais_agent import SAISAgent, SAISConfig

app = FastAPI()

@app.post("/train")
async def train(training_videos: list, inference_videos: list):
    config = SAISConfig(...)
    agent = SAISAgent(config)
    result = agent.end_to_end_workflow(
        training_videos=training_videos,
        training_dataset_name="APIDataset",
        training_dataset_type="Custom",
        inference_videos=inference_videos
    )
    return result
```

---

## Citation

If you use this agentic AI with SAIS, please cite:

```bibtex
@article{Kiyasseh2023SAIS,
  title={A vision transformer for decoding surgeon activity from surgical videos},
  author={Kiyasseh, Dani and Ma, Runzhuo and Haque, Taseen F and others},
  journal={Nature Biomedical Engineering},
  year={2023}
}
```

---

## License

This agentic AI system is provided as-is for research use. SAIS model is released under CC-BY-NC-ND 4.0 license.

Non-commercial academic research only. Commercial use prohibited without prior approval.
