# SAIS Local Setup Checklist & Hands-On Guide

## ✅ Pre-Setup Verification

Before starting, verify you have:

- [ ] **GPU**: NVIDIA GPU with CUDA support (check: `nvidia-smi`)
- [ ] **Storage**: At least 100 GB free space (varies per video size)
- [ ] **RAM**: 16+ GB system RAM
- [ ] **Software**: Git, Python 3.9+, conda/pip
- [ ] **Surgical Video**: MP4 video file ready

---

## 🔧 Step-by-Step Installation

### Phase 1: Environment Setup (30 min)

#### 1.1 Clone Repository
```bash
git clone https://github.com/danikiyasseh/SAIS.git
cd SAIS
# Verify directory structure
ls -la
# Should show: README.md, requirements.txt, setup.py, SAIS/, LICENSE
```

#### 1.2 Create Conda Environment
```bash
# Create with exact Python version
conda create -n SAIS python=3.9.7 -y

# Activate environment
conda activate SAIS

# Verify activation (should show (SAIS) prefix in terminal)
which python
# Output should contain: .../envs/SAIS/...
```

#### 1.3 Install Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Verify key packages installed
pip list | grep -E "torch|torchvision|opencv|h5py|ptlflow|timm"

# Expected output should show:
# torch                    1.8.0
# torchvision              0.9.0
# opencv-python            X.X.X
# h5py                     X.X.X
# ptlflow                  0.2.5
# timm                     0.6.5
```

#### 1.4 Install Package in Development Mode
```bash
pip install -e .

# Verify installation
python -c "import SAIS; print('SAIS package installed successfully')"
```

#### 1.5 Verify PyTorch CUDA Support
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# Output example:
# CUDA available: True
# Device count: 1
```

---

### Phase 2: PyTorch Transformer Modification (20 min) ⚠️ CRITICAL

#### 2.1 Locate transformer.py

```bash
# Find your conda environment
conda info -a | grep "active environment"

# Navigate to transformer module
# Windows example:
# C:\Users\User\anaconda3\envs\SAIS\Lib\site-packages\torch\nn\modules\transformer.py

# Linux example:
# ~/anaconda3/envs/SAIS/lib/python3.9/site-packages/torch/nn/modules/transformer.py

# Get the exact path with Python:
python -c "import torch; import os; path = os.path.dirname(torch.__file__); print(os.path.join(path, 'nn/modules/transformer.py'))"
```

#### 2.2 Backup Original File
```bash
# Windows PowerShell:
$TORCH_PATH = python -c "import torch; import os; print(os.path.dirname(torch.__file__))"
$TRANSFORMER_FILE = "$TORCH_PATH\nn\modules\transformer.py"
Copy-Item $TRANSFORMER_FILE "$TRANSFORMER_FILE.backup"

# Linux:
TORCH_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")
TRANSFORMER_FILE="$TORCH_PATH/nn/modules/transformer.py"
cp $TRANSFORMER_FILE $TRANSFORMER_FILE.backup
```

#### 2.3 Edit Transformer Module

**Find and modify these sections**:

**Location 1 - Around line 181 (forward method of multihead attention):**

Original:
```python
def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
            average_attn_weights=True):
    ...
    return attn_output, attn_output_weights
```

Should also return attention in the calling function.

**Location 2 - Around line 294 (TransformerEncoderLayer forward):**

Original:
```python
def forward(self, src, src_mask=None, src_key_padding_mask=None):
    ...
    return src[0]
```

Should be modified to:
```python
def forward(self, src, src_mask=None, src_key_padding_mask=None):
    ...
    return src[0], src[1]  # Also return attention weights
```

**Recommended**: Use the official SAIS repository as reference for exact modifications.

#### 2.4 Verify Modification
```bash
python -c "
import torch.nn as nn
import inspect

# Check if transformer now returns attention
transformer = nn.TransformerEncoderLayer(d_model=384, nhead=8)
source = inspect.getsource(transformer.forward)
if 'attn' in source:
    print('✓ Transformer modification appears successful')
else:
    print('⚠ Double-check transformer.py modifications')
"
```

---

### Phase 3: Download Pre-trained Weights (15 min)

#### 3.1 Create Output Directory
```bash
mkdir -p SAIS/scripts/dino-main/outputs
cd SAIS/scripts/dino-main/outputs
```

#### 3.2 Download DINO Weights

**Option A: Using wget (Linux/Mac/Windows with Git Bash)**
```bash
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth \
  -O dino_deitsmall16_pretrain.pth

# Verify download
ls -lh dino_deitsmall16_pretrain.pth
# Should show file ~350 MB
```

**Option B: Using PowerShell (Windows Native)**
```powershell
$url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
$output = "dino_deitsmall16_pretrain.pth"
Invoke-WebRequest -Uri $url -OutFile $output

# Verify
dir dino_deitsmall16_pretrain.pth
```

**Option C: Using Python**
```bash
python -c "
import urllib.request
import os

url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth'
output_dir = './SAIS/scripts/dino-main/outputs'
os.makedirs(output_dir, exist_ok=True)

filepath = os.path.join(output_dir, 'dino_deitsmall16_pretrain.pth')

print(f'Downloading DINO weights to {filepath}...')
urllib.request.urlretrieve(url, filepath)
print(f'✓ Downloaded successfully: {os.path.getsize(filepath) / 1e6:.1f} MB')
"
```

#### 3.3 Verify Download
```bash
# Check file exists and size
ls -lh SAIS/scripts/dino-main/outputs/

# Should show:
# -rw-r--r-- 1 user group 350M ... dino_deitsmall16_pretrain.pth
```

---

### Phase 4: Prepare Input Video (10 min)

#### 4.1 Create Videos Directory
```bash
cd SAIS  # Back to repo root
mkdir -p SAIS/videos
```

#### 4.2 Add Your Video
```bash
# Copy your surgical video
# Windows:
Copy-Item "path/to/your/video.mp4" "SAIS/videos/video.mp4"

# Linux/Mac:
cp /path/to/your/video.mp4 SAIS/videos/video.mp4

# Verify
ls -lh SAIS/videos/
```

#### 4.3 Video Requirements
- [ ] Format: MP4 (H.264 codec preferred)
- [ ] Duration: Can handle 5-60 min videos
- [ ] Frame rate: 20-25 FPS typical for surgical videos
- [ ] Resolution: 1080p or higher
- [ ] Size: Expect ~200-500 MB per 10 min

**To check video properties** (using ffmpeg):
```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,duration \
  -of default=noprint_wrappers=1:nokey=1:noescapes=1 \
  SAIS/videos/video.mp4
```

---

### Phase 5: Verify Model Parameters Available (10 min)

#### 5.1 Check for Pre-trained Model Parameters
```bash
# List model directories
ls -la SAIS/params/

# If empty or doesn't exist, you need to either:
# A) Train your own (see Phase 7)
# B) Download from authors
```

#### 5.2 Expected Structure
```bash
# Should eventually look like:
SAIS/params/
├── Fold_0/
│   ├── params.zip          # Architecture weights (~100-200 MB)
│   ├── prototypes.zip      # Prototype vectors (~50-100 MB)
│   └── checkpoint/         # Optional training checkpoints
├── Fold_1/
└── ...
```

#### 5.3 Get Pre-trained Models
```bash
# If not available, train first (Phase 7)
# OR download from paper authors if provided
# Check repository releases/issues for links
```

---

### Phase 6: Quick Test - Run Feature Extraction on Sample Frame (15 min)

#### 6.1 Test DINO Feature Extraction

```bash
python -c "
import torch
import sys
sys.path.append('./SAIS/scripts/dino-main')
from main_dino import load_model

# Load DINO
model_name = 'dino_deitsmall16'
model = load_model(model_name, 'cpu')  # Use CPU for test
print(f'✓ Loaded {model_name} successfully')

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Forward pass
with torch.no_grad():
    output = model(dummy_input)
    print(f'✓ Feature extraction works. Output shape: {output.shape}')
"
```

#### 6.2 Test PyTorch Distributed Launch
```bash
# Test with 1 GPU
python -m torch.distributed.launch --nproc_per_node=1 \
  -c "print('✓ Distributed launch works')"
```

---

## 🎬 Phase 7: Run Full Inference Pipeline

### 7.1 Quick Inference Test

```bash
# Activate environment
conda activate SAIS

# Navigate to repo root
cd SAIS

# Run end-to-end pipeline
bash ./SAIS/main.sh -f video

# Monitor output - should show:
# ✓ Extracting frames...
# ✓ Generating paths...
# ✓ Computing optical flow...
# ✓ Extracting RGB features...
# ✓ Extracting flow features...
# ✓ Running inference...
# ✓ Processing results...
```

### 7.2 Expected Output

```bash
# After completion, check outputs
ls -lh SAIS/

# Should now contain:
# images/          # Extracted frames
# flows/           # Optical flow maps
# paths/           # CSV path files
# results/         # HDF5 feature files
# predictions/     # Final predictions
```

### 7.3 View Results

```bash
# Check predictions
ls SAIS/predictions/

# View CSV results
head -20 SAIS/predictions/video_predictions.csv

# Expected format:
# video_name,action_id,gesture_label,confidence,frame_importance
# video,1,knot_tying,0.92,"[0.1,0.2,0.85,...]"
```

---

## 🏋️ Phase 8: Train on Custom Dataset (Optional - 2+ hours)

### 8.1 Prepare Training Data

```bash
# Structure:
SAIS/
├── annotations/
│   └── train_annotations.csv
├── videos/
│   ├── surgery_001.mp4
│   ├── surgery_002.mp4
│   └── ...
```

### 8.2 Create Annotations CSV

**Format** (train_annotations.csv):
```csv
video_name,frame_number,gesture_label,surgeon_id
surgery_001,0,rest,surgeon_a
surgery_001,1,rest,surgeon_a
surgery_001,25,knot_tying,surgeon_a
surgery_001,50,tissue_grasping,surgeon_a
surgery_001,75,suturing,surgeon_a
surgery_002,0,rest,surgeon_b
...
```

### 8.3 Extract Features for All Videos

```bash
# Process each video (can parallelize)
for video in SAIS/videos/*.mp4; do
    videobase=$(basename "$video" .mp4)
    bash ./SAIS/main.sh -f "$videobase" --skip-inference
done
```

### 8.4 Train Model

```bash
python -m torch.distributed.launch --nproc_per_node=1 \
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
  -domains gesture_classification \
  -ph train_val_test \
  -dt reps \
  -e 50 \
  -f 5 \
  -tf 1.0

# Monitor training log
tail -f ptlflow_logs/log_run.txt
```

### 8.5 Use Trained Model for Inference

```bash
# Models automatically saved to SAIS/params/Fold_X/
# Now inference runs with your trained model

bash ./SAIS/main.sh -f new_video
```

---

## 📋 Verification Checklist

After completing all phases, verify:

- [ ] **Environment**: `(SAIS)` prefix in terminal
- [ ] **CUDA**: GPU available (`nvidia-smi` shows devices)
- [ ] **Packages**: `pip list | grep torch` shows 1.8.0
- [ ] **DINO**: Weights file exists (~350 MB)
- [ ] **Transformer**: Modified to return attention
- [ ] **Video**: Present in `SAIS/videos/`
- [ ] **Params**: Model files in `SAIS/params/Fold_0/` (if not training)
- [ ] **Test run**: Successfully completed Phase 7
- [ ] **Output**: Predictions CSV generated

---

## 🚀 Running Inference on New Videos (Repeat Template)

Once setup complete, running on new videos is simple:

```bash
# 1. Add video
cp new_surgical_video.mp4 SAIS/videos/

# 2. Run pipeline
cd SAIS
bash main.sh -f new_surgical_video

# 3. Check results
cat SAIS/predictions/new_surgical_video_predictions.csv
```

---

## 🐛 Troubleshooting During Setup

### Issue: `ModuleNotFoundError: torch`

```bash
# Solution: Activate conda environment
conda activate SAIS
python -c "import torch; print('OK')"
```

### Issue: `CUDA error: out of memory`

```bash
# During feature extraction, reduce batch size
# In main.sh, modify:
python -m torch.distributed.launch ./SAIS/scripts/extract_representations.py \
  --batch_size_per_gpu 256  # Reduce from 1024 if OOM
```

### Issue: `FileNotFoundError: dino_deitsmall16_pretrain.pth`

```bash
# Check path
ls -la SAIS/scripts/dino-main/outputs/

# Re-download if missing
cd SAIS/scripts/dino-main/outputs
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
```

### Issue: Frames not extracted

```bash
# Check video format
ffprobe -v error -select_streams v:0 SAIS/videos/video.mp4

# Verify ffmpeg installed
ffmpeg -version

# Try manual extraction
bash SAIS/scripts/video_to_frames.sh -f video
```

### Issue: Permission denied on scripts

```bash
# Make scripts executable (Linux/Mac)
chmod +x SAIS/scripts/video_to_frames.sh
chmod +x SAIS/main.sh
```

---

## 📊 System Resource Monitoring

### Monitor During Processing

**Windows PowerShell:**
```powershell
# Monitor GPU
while($true) {
    nvidia-smi
    Start-Sleep -Seconds 1
}
```

**Linux/Mac:**
```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Monitor disk usage
watch -n 1 "du -sh SAIS/*"

# Monitor memory
watch -n 1 free -h
```

### Expected Resource Usage

| Phase | GPU Memory | CPU | Duration |
|-------|-----------|-----|----------|
| Feature extraction | 8-10 GB | 50-80% | 3-10 min / 10 min video |
| Inference | 4 GB | 30-50% | 30-60 sec / 10 min video |
| Training epoch | 12-16 GB | 70-90% | 10-30 min |

---

## ✨ Success Indicators

After running `bash main.sh`:

1. **Frame extraction starts**:
   ```
   Extracting frames from SAIS/videos/video.mp4...
   ```

2. **Optical flow computation**:
   ```
   Loading RAFT model...
   Processing optical flow...
   Saved: SAIS/flows/video/
   ```

3. **Feature extraction**:
   ```
   Loading DINO ViT-small...
   Extracting RGB features...
   Extracting flow features...
   Saved: SAIS/results/
   ```

4. **Inference**:
   ```
   Loading trained model from: SAIS/params/Fold_0/
   Running inference on snippets...
   Predictions saved...
   ```

5. **Results**:
   ```
   Processing results...
   Final predictions: SAIS/predictions/video_predictions.csv
   ```

---

**Estimated Total Setup Time**: 1-2 hours (depending on download speeds and GPU)

**Estimated Per-Video Processing Time**: 5-15 minutes (10 min video on modern GPU)

