Agentic Workflow Manager

Overview
- This repository now includes an agent-style workflow orchestrator at SAIS/scripts/agentic_workflow_manager.py.
- It executes ordered milestones for training or inference, checkpoints each milestone, and supports rollback to any completed milestone.

Milestone definitions

Training workflow milestones
1. video_image_processing
   - Extracts frames from videos into SAIS/images.
2. annotation
   - Builds SAIS/annotations/Custom_Gestures_Annotations.csv from transcription files.
3. paths_setting
   - Generates SAIS/paths/Custom_Paths.csv and SAIS/paths/Custom_FlowPaths.csv.
4. flow_extraction
   - Computes flow maps into SAIS/flows.
5. rgb_representations
   - Extracts ViT RGB reps into SAIS/results/*_RepsAndLabels.h5.
6. flow_representations
   - Extracts flow reps into SAIS/results/*_FlowRepsAndLabels.h5.
7. training
   - Runs run_experiments.py with Custom_Gestures + Custom_Top15 setup and writes checkpoints/params.

Inference workflow milestones
1. video_image_processing
2. annotation
3. paths_setting
4. flow_extraction
5. rgb_representations
6. flow_representations
7. inference
   - Runs model inference for gesture output.
8. inference_postprocess
   - Runs process_inference_results.py to create final intervals/predictions.

Checkpoint and rollback model
- Each successful milestone updates persisted state in SAIS/workflow_state/<run_id>.json.
- Backup modes:
  - copy: Copies milestone artifacts to SAIS/workflow_state/backups/<run_id>/<milestone>/.
  - manifest: Stores metadata only; no artifact copy.
- Rollback behavior:
  - Deletes downstream artifacts for milestones after the target.
  - Restores target milestone snapshot when backup mode is copy.
  - Marks run state as rolled_back so execution can resume from the rollback point.

Usage

Print workflow plan
python SAIS/scripts/agentic_workflow_manager.py --workflow training plan

Run training workflow
python SAIS/scripts/agentic_workflow_manager.py --workflow training run

Run inference workflow
python SAIS/scripts/agentic_workflow_manager.py --workflow inference run

Stop after a milestone
python SAIS/scripts/agentic_workflow_manager.py --workflow training run --stop-after paths_setting

Rollback to a milestone
python SAIS/scripts/agentic_workflow_manager.py --workflow training --run-id <run_id> rollback --milestone flow_extraction

Inspect run state
python SAIS/scripts/agentic_workflow_manager.py --workflow training --run-id <run_id> status

Notes
- Default setup targets Custom_Gestures with Custom_Top15 domain and 15 classes.
- All major hyperparameters and batch sizes are configurable via CLI flags.
- Use backup mode manifest for faster checkpoints and copy for restorable snapshots.
