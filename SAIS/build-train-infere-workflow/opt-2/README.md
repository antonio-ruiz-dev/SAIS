# SAIS Agentic AI System

## Executive Summary

The **SAIS Agentic AI** is an autonomous orchestration system that simplifies surgical video analysis. It transforms the complex multi-stage SAIS workflow into a single intelligent agent that handles:

✅ **Automated Feature Extraction** - Extract DINO + optical flow features  
✅ **Intelligent Training** - Train Vision Transformer with contrastive learning  
✅ **Seamless Inference** - Generate predictions on new surgical videos  
✅ **Execution Monitoring** - Detailed logging and reporting  

**Problem Solved:** Manual orchestration of 15+ complex commands replaced by **1 simple command**.

---

## 📦 Deliverables

### Core Agent
- **`sais_agent.py`** (467 lines)
  - Main agentic system with state machine
  - Full end-to-end workflow orchestration
  - Comprehensive error handling and logging

### Documentation
- **`SAIS_AGENT_GUIDE.md`** - Complete usage guide with examples
- **`COMMAND_MAPPING.md`** - Traditional commands → agent automation mapping
- **`QUICK_REFERENCE.md`** - Cheat sheet for common workflows
- **`example_usage.py`** - Practical code examples

### Features
- ✨ Single unified command replaces 5+ manual stages
- 📊 Structured JSON execution logs with timestamps
- 🛡️ Automatic error detection and recovery
- 🎯 State machine for reliable workflow execution
- 📈 Performance monitoring and reporting
- 🔧 Fully configurable via Python API or CLI

---

## 🎯 Quick Start

### 1. Setup (One-Time)
```bash
# Clone SAIS
git clone https://github.com/danikiyasseh/SAIS.git
cd SAIS

# Create environment
conda create -n SAIS python=3.9.7 -y
conda activate SAIS
pip install -r requirements.txt && pip install -e .

# Download DINO weights
mkdir -p SAIS/scripts/dino-main/outputs
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth \
  -O SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth

# Modify PyTorch transformer (see SAIS_AGENT_GUIDE.md Step 3)
```

### 2. Install Agent
```bash
cp sais_agent.py /path/to/SAIS/
```

### 3. Run Workflow
```bash
python sais_agent.py \
  --sais_root . \
  --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos surgery_001 surgery_002 surgery_003 \
  --inference_videos surgery_test_001 surgery_test_002 \
  --epochs 50 \
  --batch_size 8 \
  --num_classes 10
```

### 4. Review Results
```bash
cat predictions/surgery_test_001_predictions.json
cat sais_report.json
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              SAIS AGENTIC AI ORCHESTRATOR                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
            ┌─────────────────────────────┐
            │   ENVIRONMENT VALIDATION    │
            │  • Check packages           │
            │  • Verify DINO weights      │
            │  • Setup directories        │
            └─────────────────────────────┘
                          ↓
        ┌──────────────────────────────────────┐
        │  TRAINING PHASE                      │
        ├──────────────────────────────────────┤
        │ 1. Extract frames from videos       │
        │ 2. Generate path mappings           │
        │ 3. Extract DINO features + flow     │
        │ 4. Train Vision Transformer         │
        │ 5. Save model weights & prototypes  │
        └──────────────────────────────────────┘
                          ↓
        ┌──────────────────────────────────────┐
        │  INFERENCE PHASE                     │
        ├──────────────────────────────────────┤
        │ 1. Extract frames from test videos  │
        │ 2. Generate path mappings           │
        │ 3. Extract DINO features + flow     │
        │ 4. Load trained model               │
        │ 5. Run inference on features        │
        │ 6. Aggregate predictions            │
        └──────────────────────────────────────┘
                          ↓
        ┌──────────────────────────────────────┐
        │  REPORTING & MONITORING              │
        ├──────────────────────────────────────┤
        │ • Execution log (JSON)               │
        │ • Predictions (JSON)                 │
        │ • Model artifacts (ZIP)              │
        │ • Feature files (HDF5)               │
        └──────────────────────────────────────┘
```

---

## 🔄 Workflow Stages

### Stage 1: Environment Validation
- Validates Python packages (torch, transformers, torchvision)
- Checks DINO weights availability
- Creates necessary directories
- **Time**: < 1 min

### Stage 2: Feature Extraction (Training Videos)
For each training video:
1. Extract frames @ 20 FPS using ffmpeg
2. Generate CSV paths mapping frames to storage
3. Extract DINO visual features (384-dim)
4. Compute RAFT optical flow
5. Save to HDF5 format
- **Time**: ~5-10 min per video

### Stage 3: Model Training
- Load extracted features in batches (5-frame snippets)
- Train Vision Transformer backbone
- Learn class-specific prototypes
- Use supervised contrastive loss
- Optimize with Adam optimizer
- **Time**: ~30-120 min (depending on epochs & batch size)

### Stage 4: Feature Extraction (Inference Videos)
Same as Stage 2, but for test videos
- **Time**: ~5-10 min per video

### Stage 5: Inference & Aggregation
- Load trained model and prototypes
- Process each inference video
- Generate per-snippet predictions
- Aggregate into action segments
- Extract attention weights (frame importance)
- **Time**: ~2-5 min per video

---

## 💻 System Requirements

### Hardware
- **GPU**: 8GB+ VRAM (12GB+ recommended for batch_size=16)
- **CPU**: 4+ cores
- **RAM**: 16GB+ system memory
- **Storage**: 100GB+ (for intermediate features)

### Software
- Python 3.9.7
- PyTorch 1.8.0+
- CUDA 11.0+ (for GPU acceleration)
- ffmpeg (for video processing)

### Disk Space
- **Per video features**: ~1-2 GB
- **Model weights**: ~500 MB
- **Total for 10 videos + model**: ~15 GB

---

## 🎓 API Reference

### Main Agent Class
```python
from sais_agent import SAISAgent, SAISConfig

# Configure
config = SAISConfig(
    sais_root=Path("."),
    videos_dir=Path("./videos"),
    params_dir=Path("./params"),
    results_dir=Path("./results"),
    predictions_dir=Path("./predictions"),
    dino_weights_path=Path("./SAIS/.../dino_deitsmall16_pretrain.pth"),
    
    # Training hyperparameters
    epochs=50,
    batch_size=8,
    num_classes=10,
    learning_rate=0.1
)

# Create agent
agent = SAISAgent(config)

# Validate environment
valid = agent.validate_environment()

# Extract features
success = agent.extract_features_from_videos(["video_1", "video_2"])

# Train model
success = agent.train_model("MyDataset", "Custom")

# Run inference
results = agent.run_inference(["test_1", "test_2"])

# End-to-end workflow
result = agent.end_to_end_workflow(
    training_videos=["v1", "v2"],
    training_dataset_name="MyDataset",
    training_dataset_type="Custom",
    inference_videos=["test"]
)

# Save execution log
agent.save_execution_report(Path("report.json"))
```

### Configuration Options
```python
config = SAISConfig(
    # Paths (required)
    sais_root: Path,
    videos_dir: Path,
    params_dir: Path,
    results_dir: Path,
    predictions_dir: Path,
    dino_weights_path: Path,
    
    # Model architecture (optional, defaults shown)
    model_type: str = "ViT",
    encoder_type: str = "ViT_SelfSupervised_ImageNet",
    learning_paradigm: str = "Prototypes",
    modality: str = "RGB-Flow",
    
    # Training hyperparameters (optional)
    feature_dim: int = 384,
    batch_size: int = 8,
    learning_rate: float = 0.1,
    num_classes: int = 10,
    epochs: int = 50,
    folds: int = 5,
    
    # Inference parameters (optional)
    inference_batch_size: int = 2
)
```

---

## 📊 Input/Output Specifications

### Input: Surgical Videos
- **Format**: MP4
- **Codec**: H.264 recommended
- **Frame Rate**: 20 FPS standard (other FPS supported)
- **Resolution**: Any (internally standardized)
- **Location**: `SAIS/videos/` folder
- **Naming**: `{video_name}.mp4` (no special chars)

### Output: Predictions
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

### Output: Execution Report
```json
{
  "agent_state": "complete",
  "execution_log": [
    {
      "timestamp": 1716158400.123,
      "action": "Environment validation",
      "status": "success",
      "details": {...}
    },
    ...
  ],
  "config": {
    "sais_root": ".",
    "model_type": "ViT",
    "epochs": 50,
    ...
  }
}
```

---

## 🔍 Monitoring & Debugging

### View Execution Log
```bash
# All actions
cat sais_report.json | jq '.execution_log'

# Only failures
cat sais_report.json | jq '.execution_log[] | select(.status=="failed")'

# Timeline of events
cat sais_report.json | jq '.execution_log[] | {timestamp, action, status}'
```

### Enable Debug Logging
```python
import logging
logging.getLogger('SAISAgent').setLevel(logging.DEBUG)
```

### Check Intermediate Files
```bash
# Extracted frames
ls -lh SAIS/images/

# Frame paths
ls -lh SAIS/paths/

# Extracted features
ls -lh results/

# Model weights
ls -lh params/Fold_0/
```

---

## 🚀 Performance Tuning

### Faster Training
```bash
--epochs 20          # Fewer epochs
--batch_size 32      # Larger batch (if GPU allows)
```

### Better Accuracy
```bash
--epochs 200         # More epochs
--batch_size 32      # Larger batch size
--training_videos ... # More training videos (15-20+)
```

### Lower Memory Usage
```bash
--batch_size 2       # Smaller batch
# Modify code: feature_dim=256
```

### Multi-GPU Training
Automatically used via `torch.distributed.launch` in train_model()

---

## 📚 Documentation Guide

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **QUICK_REFERENCE.md** | Cheat sheet & examples | Quick lookups, copy-paste commands |
| **SAIS_AGENT_GUIDE.md** | Complete usage guide | Setup, detailed instructions, troubleshooting |
| **COMMAND_MAPPING.md** | Traditional → agent mapping | Understanding automation, migration |
| **example_usage.py** | Code examples | Learning API, custom workflows |
| **This README** | System overview | Understanding architecture, getting started |

---

## 🔧 Extending the Agent

### Custom Evaluation Metrics
```python
class ExtendedAgent(SAISAgent):
    def evaluate_predictions(self, predictions: Dict, ground_truth: Dict):
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(ground_truth, predictions)
        f1 = f1_score(ground_truth, predictions)
        return {"accuracy": accuracy, "f1": f1}
```

### MLflow Integration
```python
import mlflow

agent = SAISAgent(config)
with mlflow.start_run():
    result = agent.end_to_end_workflow(...)
    mlflow.log_params({
        "epochs": config.epochs,
        "batch_size": config.batch_size
    })
    mlflow.log_artifact("sais_report.json")
```

### REST API Wrapper
```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/train_and_infer")
async def train_and_infer(training_videos: list, inference_videos: list):
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

## ✅ Verification Checklist

After running the agent, verify:

- [ ] `sais_report.json` created with "status": "success"
- [ ] `params/Fold_0/params.zip` exists (size > 10MB)
- [ ] `params/Fold_0/prototypes.zip` exists (size > 1MB)
- [ ] `predictions/{video}_predictions.json` created for each inference video
- [ ] All predictions have confidence scores between 0-1
- [ ] Execution log shows all 5 stages completed
- [ ] No failed actions in execution log

---

## 🤝 Integration Examples

### Batch Processing Multiple Datasets
```python
datasets = [
    ("Dataset1", ["v1", "v2", "v3"]),
    ("Dataset2", ["v4", "v5", "v6"]),
]

for dataset_name, videos in datasets:
    agent = SAISAgent(config)
    result = agent.end_to_end_workflow(
        training_videos=videos,
        training_dataset_name=dataset_name,
        training_dataset_type="Custom",
        inference_videos=[videos[-1]]
    )
    print(f"{dataset_name}: {result['status']}")
```

### Cross-Validation
```python
from sklearn.model_selection import KFold

videos = ["v1", "v2", "v3", "v4", "v5"]
kf = KFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(kf.split(videos)):
    train_videos = [videos[i] for i in train_idx]
    test_videos = [videos[i] for i in test_idx]
    
    agent = SAISAgent(config)
    result = agent.end_to_end_workflow(
        training_videos=train_videos,
        training_dataset_name=f"Fold_{fold}",
        training_dataset_type="Custom",
        inference_videos=test_videos
    )
```

---

## 📖 Citation

If you use this agentic AI system in your research:

```bibtex
@misc{SAISAgent2024,
  title={SAIS Agentic AI: Autonomous Orchestration for Surgical Video Analysis},
  author={Your Name},
  year={2024}
}

@article{Kiyasseh2023SAIS,
  title={A vision transformer for decoding surgeon activity from surgical videos},
  author={Kiyasseh, Dani and Ma, Runzhuo and Haque, Taseen F and others},
  journal={Nature Biomedical Engineering},
  year={2023}
}
```

---

## 📝 License

This agentic AI system is provided for research use. SAIS model is released under CC-BY-NC-ND 4.0 license.

**Non-commercial academic research only.** Commercial use prohibited without prior approval.

---

## 🆘 Support & Issues

- **Documentation**: See `SAIS_AGENT_GUIDE.md`
- **Quick Fix**: See `QUICK_REFERENCE.md` troubleshooting section
- **Issues**: https://github.com/antonio-ruiz-dev/SAIS/issues
- **Questions**: Review example workflows in `example_usage.py`

---

## 🎉 Key Benefits

| Benefit | Impact |
|---------|--------|
| **Reduced Complexity** | 15+ commands → 1 command |
| **Error Handling** | Automatic recovery + detailed logging |
| **Reproducibility** | Full execution log preserved |
| **Scalability** | Easy to process 100+ videos |
| **Monitoring** | Structured JSON for integration |
| **Flexibility** | Python API for custom workflows |
| **Documentation** | Comprehensive guides + examples |

---

**SAIS Agentic AI** - Making surgical AI analysis simple, reliable, and scalable.

Version: 1.0  
Last Updated: 2024-05-19  
Compatibility: SAIS repository (danikiyasseh/SAIS)
