# SAIS Agent: Command Mapping & Automation

## Overview

This document maps the successful training and inference commands (from `successful-list-of-commands-*.txt`) to the **SAIS Agentic AI** automation.

---

## Traditional Manual Workflow

### Stage 1: Environment Setup (One-Time)
```bash
# Clone repository
git clone https://github.com/danikiyasseh/SAIS.git
cd SAIS

# Create environment
conda create -n SAIS python=3.9.7 -y
conda activate SAIS

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Download DINO weights
mkdir -p SAIS/scripts/dino-main/outputs
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth \
  -O SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth

# Modify PyTorch transformer (manual step)
# Edit: {ENV}/lib/python3.9/site-packages/torch/nn/modules/transformer.py
```

### Stage 2: Feature Extraction (Training Videos)
```bash
# For each training video: video_1.mp4, video_2.mp4, ...

# Extract frames
bash SAIS/scripts/video_to_frames.sh ./videos/video_1.mp4 ./SAIS/images/
bash SAIS/scripts/video_to_frames.sh ./videos/video_2.mp4 ./SAIS/images/

# Generate paths
python SAIS/scripts/generate_paths.py \
  -i ./SAIS/images \
  -o ./SAIS/paths

# Extract representations (DINO + optical flow)
python SAIS/scripts/extract_representations.py \
  --dino_path ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --paths_dir ./SAIS/paths \
  --output_dir ./results
```

### Stage 3: Training
```bash
# From successful-list-of-commands-training.txt
python -m torch.distributed.launch \
  SAIS/scripts/run_experiments.py \
  -p ./SAIS/ \
  -data CustomDataset \
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
```

### Stage 4: Feature Extraction (Inference Videos)
```bash
# Same as Stage 2, but for new videos
bash SAIS/scripts/video_to_frames.sh ./videos/test_video.mp4 ./SAIS/images/
python SAIS/scripts/generate_paths.py -i ./SAIS/images -o ./SAIS/paths
python SAIS/scripts/extract_representations.py ...
```

### Stage 5: Inference
```bash
# From successful-list-of-commands-inference.txt
python SAIS/scripts/run_experiments.py \
  -p ./SAIS/ \
  --inference \
  -f test_video \
  -m ViT \
  -t Prototypes \
  -mod RGB-Flow \
  -bs 2 \
  -nc 10

# Predictions saved to: predictions/test_video_predictions.json
```

---

## Automated Agent Workflow

### Single Command Replaces All Manual Steps

```bash
# Instead of 5+ manual stages, run:
python sais_agent.py \
  --sais_root ./SAIS \
  --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos video_1 video_2 \
  --training_dataset_name "CustomDataset" \
  --training_dataset_type "Custom" \
  --inference_videos test_video \
  --epochs 50 \
  --batch_size 8 \
  --num_classes 10
```

✅ **Agent automatically handles:**
- ✓ Environment validation
- ✓ Feature extraction from all training videos
- ✓ Model training orchestration
- ✓ Feature extraction from inference videos
- ✓ Inference on all inference videos
- ✓ Results aggregation and reporting

---

## Command Mapping Reference

### Training Phase

| Manual Step | Traditional Command | Agent Automation |
|-------------|-------------------|------------------|
| Extract video frames | `bash video_to_frames.sh` | Internal `run_command()` with video_to_frames.sh |
| Generate frame paths | `python generate_paths.py` | Internal `run_command()` with generate_paths.py |
| Extract DINO features | `python extract_representations.py` | Internal `run_command()` with extract_representations.py |
| Train model | `python -m torch.distributed.launch run_experiments.py [params]` | `train_model()` method with SAISConfig parameters |
| Validate training | Manual inspection of params/ | Automatic validation in `train_model()` |

**Agent Method:** `extract_features_from_videos()` + `train_model()`

### Inference Phase

| Manual Step | Traditional Command | Agent Automation |
|-------------|-------------------|------------------|
| Extract video frames | `bash video_to_frames.sh` | Internal `run_command()` with video_to_frames.sh |
| Generate frame paths | `python generate_paths.py` | Internal `run_command()` with generate_paths.py |
| Extract DINO features | `python extract_representations.py` | Internal `run_command()` with extract_representations.py |
| Run inference | `python run_experiments.py --inference [params]` | `run_inference()` method with SAISConfig parameters |
| Collect results | Manual collection from predictions/ | Automatic aggregation in `run_inference()` |

**Agent Method:** `run_inference()`

---

## Parameter Mapping

### Training Parameters

```bash
# Traditional command structure
python -m torch.distributed.launch SAIS/scripts/run_experiments.py \
  -p <path_to_sais> \
  -data <dataset_name> \
  -d <dataset_type> \
  -m <model_type> \
  -enc <encoder_type> \
  -t <learning_paradigm> \
  -mod <modality> \
  -dim <feature_dim> \
  -bs <batch_size> \
  -lr <learning_rate> \
  -nc <num_classes> \
  -bc <batch_contrastive> \
  -sa <supervised_attention> \
  -e <epochs> \
  -f <folds>
```

```python
# Agent-based structure (via SAISConfig)
config = SAISConfig(
    sais_root=Path("./SAIS"),
    videos_dir=Path("./SAIS/videos"),
    params_dir=Path("./SAIS/params"),
    results_dir=Path("./SAIS/results"),
    predictions_dir=Path("./SAIS/predictions"),
    dino_weights_path=Path("./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth"),
    
    # Maps to: -data, -d
    # (provided in end_to_end_workflow parameters)
    
    # Maps to: -m
    model_type="ViT",
    
    # Maps to: -enc
    encoder_type="ViT_SelfSupervised_ImageNet",
    
    # Maps to: -t
    learning_paradigm="Prototypes",
    
    # Maps to: -mod
    modality="RGB-Flow",
    
    # Maps to: -dim
    feature_dim=384,
    
    # Maps to: -bs
    batch_size=8,
    
    # Maps to: -lr
    learning_rate=0.1,
    
    # Maps to: -nc
    num_classes=10,
    
    # Maps to: -e
    epochs=50,
    
    # Maps to: -f
    folds=5
)
```

### Inference Parameters

```bash
# Traditional command structure
python SAIS/scripts/run_experiments.py \
  -p <path_to_sais> \
  --inference \
  -f <video_name> \
  -m <model_type> \
  -t <learning_paradigm> \
  -mod <modality> \
  -bs <batch_size> \
  -nc <num_classes>
```

```python
# Agent-based structure (automatic within run_inference())
# All parameters come from SAISConfig
# Inference batch size defaults to 2 (customizable)
agent.run_inference(video_names=["test_video"])
```

---

## Automation Features

### 1. Parallel Processing
```bash
# Traditional: Run inference sequentially
for video in test_1 test_2 test_3 test_4; do
  python SAIS/scripts/run_experiments.py -f $video --inference ...
done

# Agent: Iterates efficiently (sequential in this version)
agent.run_inference(["test_1", "test_2", "test_3", "test_4"])
```

### 2. Error Recovery
```bash
# Traditional: Manual retry if extraction fails
# Agent: Automatic retry logic with detailed logging

success, msg = agent.run_command(cmd, description)
if not success:
    agent.log_action(description, "failed", {"error": msg})
    continue  # Skip to next video
```

### 3. State Management
```bash
# Traditional: No state tracking
# Agent: Full state machine for workflow tracking

agent.set_state(AgentState.EXTRACTING)
agent.set_state(AgentState.TRAINING)
agent.set_state(AgentState.INFERENCING)
agent.set_state(AgentState.COMPLETE)
```

### 4. Execution Logging
```bash
# Traditional: Scattered console output
# Agent: Structured JSON logging with timestamps

{
  "timestamp": 1716158400,
  "action": "Feature extraction: video_1",
  "status": "success",
  "details": {...}
}
```

### 5. Result Aggregation
```bash
# Traditional: Results scattered across files and folders
# Agent: Unified result structure

{
  "status": "success",
  "training_videos": ["video_1", "video_2"],
  "inference_videos": ["test_1", "test_2"],
  "inference_results": {
    "test_1": {...},
    "test_2": {...}
  },
  "execution_log": [...]
}
```

---

## Comparison: Manual vs Agent

| Aspect | Manual | Agent |
|--------|--------|-------|
| **Setup Complexity** | 5+ stages, multiple commands | 1 unified command |
| **Error Handling** | Manual retry | Automatic with logging |
| **Feature Extraction** | Run separately for each video | Batch automation |
| **Training** | Single command, manual validation | Automatic validation + verification |
| **Inference** | Sequential video processing | Batch processing with aggregation |
| **Output Format** | Scattered across folders | Unified JSON report + structured logs |
| **Reproducibility** | Manual parameter tracking | Automatic config + execution log |
| **Monitoring** | Tail logs manually | Structured execution log |
| **Integration** | Manual orchestration | Programmatic interface |

---

## Migration Guide: From Manual to Agent

### Step 1: Install Agent Script
```bash
cp sais_agent.py /path/to/SAIS/
```

### Step 2: Replace Manual Commands
```bash
# Before: Run each command manually
bash video_to_frames.sh ...
python generate_paths.py ...
python extract_representations.py ...
python -m torch.distributed.launch run_experiments.py -p ... -e 50 -f 5 ...
# (repeat for inference)

# After: Run single command
python sais_agent.py \
  --sais_root . \
  --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos video_1 video_2 \
  --inference_videos test_video
```

### Step 3: Adjust Parameters
```python
# Old: Pass via command-line flags
-e 50  # epochs
-bs 8  # batch size

# New: Use Python config
config = SAISConfig(
    epochs=50,
    batch_size=8,
    num_classes=10
)
```

### Step 4: Monitor Execution
```bash
# Old: Tail multiple logs
tail -f ptlflow_logs/log_run.txt

# New: Check JSON report
cat sais_report.json | jq '.execution_log'
```

---

## Advanced: Custom Agent Extensions

### Extend Agent for Additional Tasks

```python
class ExtendedSAISAgent(SAISAgent):
    """Extended agent with additional capabilities"""
    
    def cross_validate_model(self, k_folds: int = 5):
        """Perform k-fold cross-validation"""
        pass
    
    def evaluate_on_test_set(self, test_videos: List[str]):
        """Compute metrics (accuracy, F1-score) on test set"""
        pass
    
    def export_model_to_onnx(self, output_path: Path):
        """Export trained model for deployment"""
        pass
    
    def compare_models(self, model_dirs: List[Path]):
        """Compare performance of multiple trained models"""
        pass
```

### Integration with ML Ops

```python
# MLflow integration
import mlflow

agent = SAISAgent(config)
with mlflow.start_run():
    result = agent.train_model(...)
    mlflow.log_params({
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "num_classes": config.num_classes
    })
    mlflow.log_artifact("sais_report.json")
```

---

## Command Reference: Quick Lookup

### Training Commands Automated
```bash
# Traditional individual commands
bash video_to_frames.sh
python generate_paths.py
python extract_representations.py
python -m torch.distributed.launch run_experiments.py

# Agent method
agent.end_to_end_workflow(
    training_videos=[...],
    training_dataset_name="...",
    training_dataset_type="...",
    inference_videos=[...]
)
```

### Inference Commands Automated
```bash
# Traditional individual commands
bash video_to_frames.sh
python generate_paths.py
python extract_representations.py
python run_experiments.py --inference

# Agent method
agent.run_inference(video_names=[...])
```

---

## Troubleshooting Agent-Specific Issues

### Agent Won't Start
```bash
# Check configuration
python -c "from sais_agent import SAISConfig; SAISConfig(...).validate()"
```

### Training Incomplete
```bash
# Check execution log
cat sais_report.json | jq '.execution_log[] | select(.status=="failed")'
```

### Inference Results Missing
```bash
# Verify features were extracted
ls -lh results/
```

---

## Next Steps

1. **Install Agent**: Copy `sais_agent.py` to SAIS root
2. **Run Example**: `python example_usage.py`
3. **Customize**: Modify `SAISConfig` for your dataset
4. **Monitor**: Check `execution_report.json` after runs
5. **Extend**: Implement custom methods in `ExtendedSAISAgent`

For detailed usage, see `SAIS_AGENT_GUIDE.md`
