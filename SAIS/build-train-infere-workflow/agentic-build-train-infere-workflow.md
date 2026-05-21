I’m going to inspect the existing pipeline entrypoints first, then I’ll implement a new agent-style orchestrator that runs training/inference as step-based workflows with checkpoint milestones and resume/rollback controls.

Read [](file:///e%3A/antonio/SAIS/SAIS/main.sh)

Read [](file:///e%3A/antonio/SAIS/SAIS/scripts)

Search complete: Map workflow steps

Searched codebase for "SAIS repo pipeline scripts preprocessing annotation paths generation flow extraction representation extraction training inference", 23 results

Searched for files matching `**/main.sh`, 1 match

Searched for files matching `**/*.sh`, 2 matches

Searched for files matching `**/*.py`, 24 matches

Searched for regex `main.sh|main.sh:`, 38 results

Searched for regex `preprocessing|annotation|paths.*generation|flow.*extraction|representation.*extraction|train|inference`, 200 results

Read [](file:///e%3A/antonio/SAIS/SAIS/main.sh)

Read [](file:///e%3A/antonio/SAIS/SAIS/scripts/extract_representations.py#1-1), lines 1 to 260

Read [](file:///e%3A/antonio/SAIS/SAIS/scripts/generate_paths.py)

Read [](file:///e%3A/antonio/SAIS/SAIS/scripts/video_to_frames.sh)

Read [](file:///e%3A/antonio/SAIS/SAIS/scripts/run_experiments.py)

I found the concrete pipeline commands; next I’ll implement a new orchestrator script with two workflows (training/inference), milestone checkpoints, and a rollback command that can revert to any defined stage.

Searched for regex `main.sh|run_experiments.py|perform_training|Custom_Top15|Custom_Gestures` (`**/README*.md`), 14 results

Read [](file:///e%3A/antonio/SAIS/SAIS/scripts/perform_training.py)

Read [](file:///e%3A/antonio/SAIS/README.md)

I confirmed where existing commands live; I’m now pulling the annotation and path utilities so the new agent can include those milestones explicitly and run them safely.

Read [](file:///e%3A/antonio/SAIS/SAIS/scripts/write_train_annotations_custom_videos.py)

Read [](file:///e%3A/antonio/SAIS/SAIS/scripts/generate_paths_multiple_or_all_videonames.py)

Read [](file:///e%3A/antonio/SAIS/README_ANALYSIS.md#150-150), lines 150 to 260

Read [](file:///e%3A/antonio/SAIS/SAIS/scripts/video_to_frames.py)

Read [](file:///e%3A/antonio/SAIS/SAIS/scripts/extract_representations.py#360-360), lines 360 to 470

I’m ready to implement: one new workflow manager script plus a short workflow spec doc, including explicit milestone names, commands per stage, checkpoint backups, resume, and rollback-to-milestone logic.

Generating patch (582 lines) in [](file:///e%3A/antonio/SAIS/SAIS/scripts/agentic_workflow_manager.py)


```

```
The orchestrator script is in place with run, plan, status, and rollback actions; I’m now adding a concise workflow spec file so the milestones are explicitly documented for training and inference.

Generating patch (74 lines) in [](file:///e%3A/antonio/SAIS/SAIS/AGENTIC_WORKFLOW.md)


```

```


Checked [](file:///e%3A/antonio/SAIS/SAIS/scripts/agentic_workflow_manager.py), no problems found

Implemented an agentic workflow manager for SAIS with milestone checkpointing, resume, and rollback.

Created:
- agentic_workflow_manager.py
- AGENTIC_WORKFLOW.md

What this new agent does
1. Manages two workflows:
1. training
2. inference

2. Defines explicit milestones for each workflow:
1. video_image_processing
2. annotation
3. paths_setting
4. flow_extraction
5. rgb_representations
6. flow_representations
7. training (training workflow)
8. inference and inference_postprocess (inference workflow)

3. Persists workflow state per run:
1. State saved in SAIS/workflow_state/<run_id>.json
2. Tracks completed milestones, failed milestone, timestamps, and backups

4. Backs up each milestone:
1. copy mode: snapshot artifacts for each milestone
2. manifest mode: metadata only (faster, no file copy)

5. Supports rollback to a milestone:
1. Cleans downstream artifacts after target milestone
2. Restores target milestone snapshot when copy mode is used
3. Marks run as rolled_back so resume continues from that point

6. Supports resume and partial execution:
1. Resume from last completed milestone by default
2. Stop after any milestone with stop-after

How to use it
- Show planned steps and commands:
  python agentic_workflow_manager.py --workflow training plan

- Run full training workflow:
  python agentic_workflow_manager.py --workflow training run

- Run full inference workflow:
  python agentic_workflow_manager.py --workflow inference run

- Stop at a milestone (example):
  python agentic_workflow_manager.py --workflow training run --stop-after paths_setting

- Roll back to a milestone (example):
  python agentic_workflow_manager.py --workflow training --run-id <run_id> rollback --milestone flow_extraction

- Check current run status:
  python agentic_workflow_manager.py --workflow training --run-id <run_id> status

Notes aligned to your request
1. Your requested milestone points are explicitly encoded, including:
1. video/image processing
2. annotation
3. paths setting
4. flow extraction
5. RGB representation extraction
6. flow representation extraction
7. training and inference stages

2. The rollback mechanism is milestone-aware and designed for exactly this kind of stage-based recovery.

If you want, I can next add a thin wrapper in main.sh so you can launch this manager with one short command for either training or inference.

Made changes.