# JIGSAWS Folder Map for SAIS

SAIS training code expects the following structure rooted at `e:/antonio/SAIS/SAIS`:

```
SAIS/
├── SurgicalPaths/
│   └── JIGSAWS_Suturing_gestures_timestamps.csv
├── JIGSAWS_Suturing/
│   └── Results/
│       ├── ViT_SelfSupervised_ImageNet_RepsAndLabels.h5
│       └── ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5
└── datasets/
    └── jigsaw/
        ├── Suturing/
        │   └── Suturing/
        │       ├── video/
        │       ├── kinematics/
        │       └── transcriptions/
        ├── Knot_Tying/
        │   └── Knot_Tying/
        │       ├── video/
        │       ├── kinematics/
        │       └── transcriptions/
        └── Needle_Passing/
            └── Needle_Passing/
                ├── video/
                ├── kinematics/
                └── transcriptions/
```

## What maps where

- Raw JIGSAWS procedure folders already live under `SAIS/datasets/jigsaw/...`.
- The training loader for suturing classification reads the annotations CSV from `SAIS/SurgicalPaths/JIGSAWS_Suturing_gestures_timestamps.csv`.
- It reads precomputed features from `SAIS/JIGSAWS_Suturing/Results/`.
- The raw `video/`, `kinematics/`, and `transcriptions/` folders are source data; they are not read directly by the current training loader.

## Current workspace state

- Created `SAIS/SurgicalPaths/`
- Created `SAIS/JIGSAWS_Suturing/Results/`
- Created `SAIS/JIGSAWS_Suturing/Raw/`

## Next required inputs for training

- Generate or place `JIGSAWS_Suturing_gestures_timestamps.csv` in `SAIS/SurgicalPaths/`.
- Place the two HDF5 feature files in `SAIS/JIGSAWS_Suturing/Results/`.
