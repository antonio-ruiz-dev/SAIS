# SAIS System Documentation Index

## 📚 Documentation Overview

This folder contains comprehensive analysis of the SAIS (Surgical Activity Intelligence System) codebase, including architecture, components, and implementation guides.

---

## 📖 Documents Guide

### 1. **QUICK_REFERENCE.md** ⭐ START HERE
- **Best for**: Getting started quickly, understanding what SAIS does at a glance
- **Contains**: 
  - Executive summary (30 sec read)
  - 5-command quick start
  - Common workflows
  - Quick troubleshooting
- **Read time**: 10-15 minutes
- **Next step**: Use for immediate setup or reference

### 2. **SETUP_CHECKLIST.md** 🔧 DETAILED SETUP GUIDE
- **Best for**: Step-by-step local installation and verification
- **Contains**:
  - Pre-setup verification checklist
  - 8 detailed setup phases with code examples
  - Verification procedures after each step
  - Troubleshooting for common issues
  - Resource monitoring guidance
- **Read time**: 30 minutes overview, 1-2 hours execution
- **Important**: Critical PyTorch transformer modification (Phase 2)
- **Next step**: Follow systematically for setup

### 3. **SYSTEM_ANALYSIS.md** 🏗️ COMPREHENSIVE ARCHITECTURE
- **Best for**: Deep understanding of system design and components
- **Contains**:
  - Executive summary
  - High-level data flow diagrams (ASCII art)
  - Component descriptions (6 major components)
  - Data flow pipeline explanation
  - Full build & execution guide (80+ commands with explanations)
  - Performance considerations
  - Configuration parameters
  - Dependency graphs
  - Common issues & solutions
- **Read time**: 45 minutes (technical)
- **Best chapters**:
  - Section 1: What is SAIS (non-technical)
  - Section 2: System Architecture (visual overview)
  - Section 3: Component Description (reference)
  - Section 6: Building & Execution (implementation)
- **Next step**: Reference for understanding specific components

---

## 🎯 Recommended Reading Path

### Path A: I Want to Run SAIS on My Video (Quickest)
1. Start: **QUICK_REFERENCE.md** → Section "Quick Start"
2. Setup: Follow the 5 commands
3. Issues: Check Troubleshooting section
4. Stuck?: Consult **SETUP_CHECKLIST.md** for detailed steps

### Path B: I Want to Understand the System (Educational)
1. Start: **QUICK_REFERENCE.md** → Sections "What is SAIS" + "Data Flow"
2. Deep dive: **SYSTEM_ANALYSIS.md** → Sections 1-3 (Architecture)
3. Implementation: **SYSTEM_ANALYSIS.md** → Section 6 (Building & Execution)
4. Reference: Check diagrams and component relationships

### Path C: I Want to Set Everything Up Properly (Recommended)
1. Start: **QUICK_REFERENCE.md** → Entire document
2. Setup: **SETUP_CHECKLIST.md** → Follow Phase 1-5 exactly
3. Test: Phase 6 (Quick Test)
4. Run: Phase 7 (Full Inference)
5. Advanced: Phase 8 (Training) if needed
6. Reference: Keep **SYSTEM_ANALYSIS.md** handy for details

### Path D: I'm Having Issues (Debugging)
1. Check: **QUICK_REFERENCE.md** → Troubleshooting section
2. Detailed troubleshooting: **SYSTEM_ANALYSIS.md** → Section 8 (Common Issues)
3. Phase-by-phase: **SETUP_CHECKLIST.md** → Corresponding Phase
4. Fallback: Manual step-by-step execution and verification

---

## 📊 Visual Guides

Three Mermaid diagrams are included in **SYSTEM_ANALYSIS.md**:

1. **Component Architecture Diagram**
   - Shows: Input layer → Preprocessing → Feature extraction → Model → Results
   - Useful: Understanding data flow and component interactions
   - Reference point: Where DINO, RAFT, ViT fit

2. **Execution Pipeline Diagram**
   - Shows: 7-step execution sequence with prerequisites and outputs
   - Useful: Understanding which scripts to run and in what order
   - Reference point: Validating your execution matches expected flow

3. **Model Architecture Diagram**
   - Shows: Input → Encoding → Temporal modeling → Classification → Output
   - Useful: Understanding what happens inside the neural network
   - Reference point: Interpreting predictions and attention maps

---

##  Key Concepts Quick Reference

| Concept | What It Does | Learn More |
|---------|-------------|-----------|
| **DINO** | Self-supervised vision transformer that extracts visual features | SYSTEM_ANALYSIS.md § 4.2 |
| **RAFT** | Optical flow model that captures motion between frames | SYSTEM_ANALYSIS.md § 4.3 |
| **ViT (Vision Transformer)** | Deep learning model for temporal sequence modeling | SYSTEM_ANALYSIS.md § 3 |
| **Supervised Contrastive Learning** | Learning method using prototype-based classification | SYSTEM_ANALYSIS.md § 4.4 |
| **Snippet** | 5-frame temporal window processed as unit | QUICK_REFERENCE.md § Tech Stack |
| **Attention Maps** | Frame importance scores showing which frames matter | SYSTEM_ANALYSIS.md § Interpretability |
| **Multimodal Fusion** | Combining RGB features + optical flow | SYSTEM_ANALYSIS.md § 4.1 |

---

## 🔄 Data Flow Summary

```
Video Input
    ↓ [video_to_frames.sh]
Frames
    ├→ RGB Features via DINO
    └→ Optical Flow via RAFT
         ↓
    Concatenate & Fuse (768-dim)
         ↓ [ViT Encoder + Attention]
    Temporal Embeddings
         ↓ [Prototype Layer]
    Class Predictions
         ↓ [Post-processing]
    Final Predictions + Interpretability
```

**Complete flow diagram**: See SYSTEM_ANALYSIS.md § Data Flow Diagram

---

## ⚙️ Critical Setup Steps

1. **Install dependencies** (SETUP_CHECKLIST.md § Phase 1)
   ```bash
   conda create -n SAIS python=3.9.7
   pip install -r requirements.txt
   ```

2. **Modify PyTorch transformer** ⚠️ (SETUP_CHECKLIST.md § Phase 2)
   - Required for attention map extraction
   - Edit: torch/nn/modules/transformer.py
   - Lines: 181 and 294

3. **Download DINO weights** (SETUP_CHECKLIST.md § Phase 3)
   - File: dino_deitsmall16_pretrain.pth (~350 MB)
   - Location: SAIS/scripts/dino-main/outputs/

4. **Prepare model parameters** (SETUP_CHECKLIST.md § Phase 5)
   - Location: SAIS/params/Fold_0/
   - Contains: params.zip + prototypes.zip

---

## 🚀 Execution Commands

### Minimal (Run Inference)
```bash
# Assumes: environment setup, DINO weights, model params
cd SAIS
bash main.sh -f your_video
```

### Full Manual
```bash
# Individual steps for debugging/learning
bash ./SAIS/scripts/video_to_frames.sh -f video
python ./SAIS/scripts/generate_paths.py -f video -p ./SAIS/
python ./SAIS/scripts/extract_representations.py --optical_flow [params]
python -m torch.distributed.launch ./SAIS/scripts/extract_representations.py [params]
python -m torch.distributed.launch ./SAIS/scripts/run_experiments.py --inference [params]
python ./SAIS/scripts/process_inference_results.py -p ./SAIS/
```

**Full commands**: See SYSTEM_ANALYSIS.md § Step 6

---

## 📁 Repository Structure

```
SAIS/
├── README.md                    # Original project README
├── QUICK_REFERENCE.md           # This documentation
├── SETUP_CHECKLIST.md           # Step-by-step setup
├── SYSTEM_ANALYSIS.md           # Comprehensive analysis
│
├── SAIS/                        # Main package
│   ├── main.sh                  # Entry point (run this!)
│   ├── __init__.py
│   └── scripts/
│       ├── run_experiments.py   # Orchestrator (training/inference)
│       ├── train.py             # Training coordinator
│       ├── extract_representations.py  # Feature extraction
│       ├── generate_paths.py    # Path generation
│       ├── prepare_model.py     # Model builder
│       ├── prepare_dataset.py   # Data loader
│       ├── prepare_miscellaneous.py    # Utilities
│       ├── perform_training.py  # Training loop
│       ├── process_inference_results.py # Post-processing
│       ├── video_to_frames.sh   # Frame extraction
│       └── dino-main/           # Feature extractor (DINO)
│
├── videos/                      # INPUT: Place your videos here
├── params/                      # Model parameters (if available)
│   └── Fold_0/
│       ├── params.zip
│       └── prototypes.zip
│
├── images/                      # Generated: Extracted frames
├── flows/                       # Generated: Optical flow
├── paths/                       # Generated: Path CSV files
├── results/                     # Generated: Extracted features
└── predictions/                 # OUTPUT: Final predictions
```

---

## 🎓 Learning Resources

### Theory
- **Original Paper**: [SAIS - Nature Biomedical Engineering](https://www.nature.com/articles/s41551-023-01010-8)
- **Vision Transformers**: [ViT Paper](https://arxiv.org/abs/2010.11929)
- **DINO**: [Self-supervised ViT](https://arxiv.org/abs/2104.14294)
- **Contrastive Learning**: [SimCLR](https://arxiv.org/abs/2002.05709)

### Code References
- **DINO Repository**: https://github.com/facebookresearch/dino
- **PyTorch Documentation**: https://pytorch.org/docs/
- **RAFT Optical Flow**: https://github.com/princeton-vl/RAFT

---

## 🐛 Common Issues Quick Links

| Issue | Document | Section |
|-------|----------|---------|
| "ModuleNotFoundError" | SYSTEM_ANALYSIS.md | Common Issues |
| CUDA out of memory | QUICK_REFERENCE.md | Troubleshooting |
| PyTorch transformer error | SETUP_CHECKLIST.md | Phase 2 |
| DINO weights missing | SETUP_CHECKLIST.md | Phase 3 |
| No predictions generated | QUICK_REFERENCE.md | Troubleshooting |
| Video not detected | QUICK_REFERENCE.md | Troubleshooting |

---

## ✅ Validation Checklists

### After Installation
- [ ] Conda environment created and activated
- [ ] All packages installed (torch 1.8.0, torchvision 0.9.0, etc.)
- [ ] PyTorch transformer modified (returns attention)
- [ ] CUDA available and accessible
- [ ] DINO weights downloaded (~350 MB)

### Before First Run
- [ ] Video placed in SAIS/videos/
- [ ] Model parameters in SAIS/params/Fold_0/
- [ ] Enough disk space (~100 GB free minimum)
- [ ] GPU available (check: nvidia-smi)

### After First Run
- [ ] images/ folder contains extracted frames
- [ ] flows/ folder contains optical flow
- [ ] results/ folder contains HDF5 feature files
- [ ] predictions/ folder contains CSV results
- [ ] No error messages in console

---

## 📞 Getting Help

1. **In-depth**: Check SYSTEM_ANALYSIS.md § 8 (Common Issues & Solutions)
2. **Quick fix**: Check QUICK_REFERENCE.md § Troubleshooting
3. **Setup issues**: Check SETUP_CHECKLIST.md corresponding Phase
4. **Official**: Visit [SAIS GitHub](https://github.com/danikiyasseh/SAIS)
5. **Citation/Contact**: See repository README

---

## 📝 Document Maintenance

**Last Updated**: 2024
**Documentation Version**: 1.0
**SAIS Version Referenced**: Original paper authors' implementation

If you encounter documentation errors or have improvements:
1. Check against official SAIS repository
2. Report issues to repository maintainers
3. Update local copies with corrections

---

## 🎯 Quick Navigation

**I want to...**
- ✅ Get started immediately → **QUICK_REFERENCE.md**
- ✅ Set up on my machine → **SETUP_CHECKLIST.md**
- ✅ Understand the architecture → **SYSTEM_ANALYSIS.md**
- ✅ Find a specific component → Use table of contents in each document
- ✅ Debug an issue → Search for issue in "Troubleshooting" sections
- ✅ Learn what each file does → **Component Description** in SYSTEM_ANALYSIS.md

---

## 📊 Document Statistics

| Document | Length | Reading Time | Best For |
|----------|--------|--------------|----------|
| QUICK_REFERENCE.md | ~3,000 words | 10-15 min | Quick start, overview |
| SETUP_CHECKLIST.md | ~4,500 words | 30 min read, 1-2 hours execution | Detailed setup |
| SYSTEM_ANALYSIS.md | ~6,500 words | 45 min read | Deep understanding |
| **Total** | **~14,000 words** | **~1.5 hours read** | **Complete coverage** |

---

**Happy analyzing surgical videos! 🎥🔬**

