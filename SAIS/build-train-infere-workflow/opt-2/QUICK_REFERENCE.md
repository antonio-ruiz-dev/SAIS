# SAIS Agent - Quick Reference Cheat Sheet

## 🚀 Quick Start (30 seconds)

```bash
# 1. Copy agent to SAIS root
cp sais_agent.py /path/to/SAIS/

# 2. Run complete workflow
cd /path/to/SAIS
python sais_agent.py \
  --sais_root . \
  --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos video_1 video_2 \
  --inference_videos test_video

# 3. Check results
cat predictions/test_video_predictions.json
```

---

## 📋 Parameter Cheat Sheet

### Essential Parameters
| Flag | Type | Example | Required |
|------|------|---------|----------|
| `--sais_root` | Path | `.` or `/data/SAIS` | ✅ |
| `--dino_weights` | Path | `./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth` | ✅ |
| `--training_videos` | List | `video_1 video_2 video_3` | ✅ |
| `--inference_videos` | List | `test_1 test_2` | ✅ |

### Optional Parameters
| Flag | Type | Default | Range |
|------|------|---------|-------|
| `--epochs` | Int | 50 | 10-500 |
| `--batch_size` | Int | 8 | 1-64 |
| `--num_classes` | Int | 10 | 2-100 |
| `--training_dataset_name` | String | `CustomSurgicalDataset` | - |
| `--training_dataset_type` | String | `Custom` | - |
| `--report` | Path | `sais_report.json` | - |

---

## 🎯 Common Workflows

### 1️⃣ Quick Test (Small Dataset)
```bash
python sais_agent.py \
  --sais_root . \
  --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos video_1 video_2 \
  --inference_videos test_video \
  --epochs 10 \
  --batch_size 4 \
  --num_classes 5
```
⏱️ **Time**: ~15-30 min | 💾 **VRAM**: 8GB

---

### 2️⃣ Standard Training (Medium Dataset)
```bash
python sais_agent.py \
  --sais_root . \
  --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos video_1 video_2 video_3 video_4 \
  --inference_videos test_1 test_2 \
  --epochs 50 \
  --batch_size 8 \
  --num_classes 10
```
⏱️ **Time**: ~1-2 hours | 💾 **VRAM**: 12GB

---

### 3️⃣ High-Precision Model (Large Dataset)
```bash
python sais_agent.py \
  --sais_root . \
  --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos $(seq -f "video_%g" 1 10) \
  --inference_videos test_1 test_2 test_3 test_4 test_5 \
  --epochs 200 \
  --batch_size 32 \
  --num_classes 15
```
⏱️ **Time**: ~4-6 hours | 💾 **VRAM**: 24GB

---

### 4️⃣ Multi-Procedure Training
```bash
python sais_agent.py \
  --sais_root . \
  --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth \
  --training_videos prostate_1 prostate_2 nephrectomy_1 nephrectomy_2 hysterectomy_1 hysterectomy_2 \
  --training_dataset_name "MultiProcedure_v1" \
  --training_dataset_type "MixedProcedures" \
  --inference_videos prostate_test nephrectomy_test hysterectomy_test \
  --epochs 100 \
  --batch_size 16 \
  --num_classes 20
```
⏱️ **Time**: ~2-3 hours | 💾 **VRAM**: 16GB

---

## 🔧 Configuration Quick Presets

### Minimal Config (Testing)
```python
from sais_agent import SAISAgent, SAISConfig
from pathlib import Path

config = SAISConfig(
    sais_root=Path("."),
    videos_dir=Path("./videos"),
    params_dir=Path("./params"),
    results_dir=Path("./results"),
    predictions_dir=Path("./predictions"),
    dino_weights_path=Path("./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth"),
    epochs=10, batch_size=2, num_classes=5
)
agent = SAISAgent(config)
```

### Production Config
```python
config = SAISConfig(
    sais_root=Path("/data/SAIS"),
    videos_dir=Path("/data/SAIS/videos"),
    params_dir=Path("/data/SAIS/params"),
    results_dir=Path("/data/SAIS/results"),
    predictions_dir=Path("/data/SAIS/predictions"),
    dino_weights_path=Path("/data/SAIS/SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth"),
    epochs=100, batch_size=16, num_classes=15,
    learning_rate=0.01, folds=10
)
agent = SAISAgent(config)
```

---

## 📊 Output Files

After running the agent:

```
SAIS/
├── params/Fold_0/
│   ├── params.zip          ✅ Trained weights
│   └── prototypes.zip      ✅ Class prototypes
├── predictions/
│   ├── test_1_predictions.json  ✅ Results for test_1
│   ├── test_2_predictions.json  ✅ Results for test_2
│   └── ...
├── results/
│   ├── video_1_features.h5      📦 Extracted features
│   ├── video_2_features.h5      📦 Extracted features
│   └── ...
└── sais_report.json             📋 Execution log
```

---

## 📖 Predicting Gesture from Output

### Output JSON Format
```json
{
  "video_name": "test_1",
  "predictions": [
    {
      "action_id": 1,
      "start_frame": 0,
      "end_frame": 49,
      "gesture_label": "knot_tying",
      "confidence": 0.92,
      "frame_importance": [0.1, 0.2, 0.85, 0.9, 0.95]
    }
  ]
}
```

### Interpreting Results
- **gesture_label**: Predicted surgical gesture
- **confidence**: Probability (0-1), higher is better
- **frame_importance**: Attention weights showing important frames
- **start_frame, end_frame**: Video segment duration

### Quality Metrics
| Confidence | Quality | Action |
|------------|---------|--------|
| > 0.90 | Excellent | ✅ Trust prediction |
| 0.80-0.90 | Good | ✅ Use with confidence |
| 0.70-0.80 | Fair | ⚠️ Manual review |
| < 0.70 | Low | ❌ Retrain with more data |

---

## 🐛 Quick Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
--batch_size 4  # instead of 8
```

### "FileNotFoundError: dino_weights.pth"
```bash
# Download DINO weights
mkdir -p SAIS/scripts/dino-main/outputs
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth \
  -O SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth
```

### "No predictions generated"
```bash
# Check model was trained
ls -la params/Fold_0/
# Both params.zip and prototypes.zip must exist

# Check features were extracted
ls -la results/
```

### "RuntimeError: attention"
```bash
# PyTorch transformer not modified correctly
# Follow Step 3 in installation guide
# Edit: {CONDA_ENV}/lib/python3.9/site-packages/torch/nn/modules/transformer.py
```

---

## ⚡ Performance Tips

### Speed Up Training
```bash
# 1. Increase batch size (if GPU memory allows)
--batch_size 32  # instead of 8

# 2. Use fewer epochs for initial testing
--epochs 20  # instead of 50

# 3. Reduce feature dimension (trade-off: accuracy)
# Modify in code: feature_dim=256 instead of 384
```

### Improve Accuracy
```bash
# 1. More training videos
--training_videos video_1 video_2 ... video_20

# 2. More epochs
--epochs 200  # instead of 50

# 3. Lower learning rate (slower but more stable)
# Modify in code: learning_rate=0.01 instead of 0.1

# 4. Larger batch size
--batch_size 32  # instead of 8
```

---

## 🎓 Learning Resources

| Topic | Resource |
|-------|----------|
| **SAIS Paper** | https://www.nature.com/articles/s41551-023-01010-8 |
| **Vision Transformers** | https://arxiv.org/abs/2010.11929 |
| **DINO Pre-training** | https://github.com/facebookresearch/dino |
| **Optical Flow (RAFT)** | https://github.com/princeton-vl/RAFT |
| **Contrastive Learning** | https://arxiv.org/abs/2004.11362 |

---

## 📞 Getting Help

### Check Execution Log
```bash
# View all actions and status
cat sais_report.json | jq '.execution_log'

# View only failures
cat sais_report.json | jq '.execution_log[] | select(.status=="failed")'
```

### Debug Mode
```python
import logging
logging.getLogger('SAISAgent').setLevel(logging.DEBUG)

# Then run agent - will print detailed debug info
agent = SAISAgent(config)
```

### GitHub Issues
https://github.com/antonio-ruiz-dev/SAIS/issues

---

## 📝 Template: Custom Workflow

### Python API Usage
```python
from sais_agent import SAISAgent, SAISConfig
from pathlib import Path

# 1. Configure
config = SAISConfig(
    sais_root=Path("."),
    videos_dir=Path("./videos"),
    params_dir=Path("./params"),
    results_dir=Path("./results"),
    predictions_dir=Path("./predictions"),
    dino_weights_path=Path("./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth"),
    epochs=50,
    batch_size=8,
    num_classes=10
)

# 2. Create agent
agent = SAISAgent(config)

# 3. Run workflow
result = agent.end_to_end_workflow(
    training_videos=["video_1", "video_2"],
    training_dataset_name="MyDataset",
    training_dataset_type="Custom",
    inference_videos=["test_1"]
)

# 4. Save report
agent.save_execution_report(Path("report.json"))

# 5. Process results
print(f"Status: {result['status']}")
for video, pred in result['inference_results'].items():
    print(f"  {video}: {pred}")
```

---

## ✅ Pre-Flight Checklist

Before running agent, verify:

- [ ] SAIS repository cloned
- [ ] Python 3.9.7 environment created
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] DINO weights downloaded to correct location
- [ ] PyTorch transformer modified (Lines 181, 294)
- [ ] Surgical videos placed in `./videos/` folder
- [ ] GPU available (8GB+ VRAM recommended)
- [ ] Enough disk space for features (100GB+ for large datasets)
- [ ] Write permissions in SAIS directory

---

## 🚀 One-Liner Examples

```bash
# Minimal inference
python sais_agent.py --sais_root . --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth --training_videos v1 v2 --inference_videos test

# With custom parameters
python sais_agent.py --sais_root . --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth --training_videos v1 v2 v3 v4 v5 --inference_videos t1 t2 --epochs 100 --batch_size 16 --num_classes 12

# With custom report output
python sais_agent.py --sais_root . --dino_weights ./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth --training_videos v1 v2 --inference_videos test --report my_results.json
```

---

**Last Updated**: 2024-05-19  
**SAIS Agent Version**: 1.0  
**Compatible with**: SAIS repository (danikiyasseh/SAIS)
