import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class StepSpec:
    name: str
    description: str
    command: List[str]
    artifacts: List[str]


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_run(command: List[str], cwd: Path) -> None:
    printable = " ".join(command)
    print(f"\n[STEP] Running: {printable}")
    result = subprocess.run(command, cwd=str(cwd))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {printable}")


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


class AgenticWorkflowManager:
    """Step-based workflow agent for SAIS training and inference pipelines.

    Features:
    - Deterministic step execution with milestone checkpoints
    - Optional snapshot backups per milestone
    - Rollback to any completed milestone
    - Resume from latest completed milestone
    """

    def __init__(
        self,
        root: Path,
        state_dir: Optional[Path] = None,
        backup_mode: str = "copy",
    ) -> None:
        self.root = root.resolve()
        self.sais_dir = self.root / "SAIS"
        self.scripts_dir = self.sais_dir / "scripts"
        self.state_dir = (state_dir or (self.sais_dir / "workflow_state")).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.backup_mode = backup_mode

    def _state_path(self, run_id: str) -> Path:
        return self.state_dir / f"{run_id}.json"

    def _load_state(self, run_id: str) -> Dict:
        state_path = self._state_path(run_id)
        if not state_path.exists():
            raise FileNotFoundError(f"No workflow state found for run_id={run_id}: {state_path}")
        return json.loads(state_path.read_text(encoding="utf-8"))

    def _save_state(self, run_id: str, state: Dict) -> None:
        state_path = self._state_path(run_id)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _resolve_artifacts(self, artifacts: List[str]) -> List[Path]:
        resolved = []
        for rel in artifacts:
            resolved.append((self.sais_dir / rel).resolve())
        return resolved

    def _backup_step(self, run_id: str, step: StepSpec) -> Dict:
        backup_root = self.state_dir / "backups" / run_id / step.name
        backup_root.mkdir(parents=True, exist_ok=True)

        entry = {
            "mode": self.backup_mode,
            "step": step.name,
            "created_at": _utc_now(),
            "files": [],
        }

        for artifact_path in self._resolve_artifacts(step.artifacts):
            if not artifact_path.exists():
                continue
            rel = artifact_path.relative_to(self.sais_dir)
            backup_target = backup_root / rel
            if self.backup_mode == "copy":
                _copy_path(artifact_path, backup_target)
            entry["files"].append(str(rel).replace("\\", "/"))

        return entry

    def _restore_step_backup(self, run_id: str, milestone: str, backup_entry: Dict) -> None:
        if backup_entry.get("mode") != "copy":
            print(f"[ROLLBACK] No copy backup available for milestone '{milestone}'.")
            return

        backup_root = self.state_dir / "backups" / run_id / milestone
        if not backup_root.exists():
            print(f"[ROLLBACK] Missing backup folder for milestone '{milestone}': {backup_root}")
            return

        for rel in backup_entry.get("files", []):
            src = backup_root / rel
            dst = self.sais_dir / rel
            if src.exists():
                _copy_path(src, dst)

    def _cleanup_artifacts(self, artifacts: List[str]) -> None:
        for rel in artifacts:
            path = self.sais_dir / rel
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def list_workflow_steps(self, workflow: str, args: argparse.Namespace) -> List[StepSpec]:
        py = sys.executable
        model_type = args.model_type
        top15_domain = args.domain

        common_steps = [
            StepSpec(
                name="video_image_processing",
                description="Extract frames from source videos.",
                command=[py, str(self.scripts_dir / "run_video_to_frames_all_no-wsl.py")],
                artifacts=["images"],
            ),
            StepSpec(
                name="annotation",
                description="Generate custom annotation CSV from transcriptions.",
                command=[py, str(self.scripts_dir / "write_train_annotations_custom_videos.py")],
                artifacts=["annotations/Custom_Gestures_Annotations.csv"],
            ),
            StepSpec(
                name="paths_setting",
                description="Build frame and flow CSV path files.",
                command=[
                    py,
                    str(self.scripts_dir / "generate_paths_multiple_or_all_videonames.py"),
                    "-p",
                    "./SAIS/",
                ],
                artifacts=["paths/Custom_Paths.csv", "paths/Custom_FlowPaths.csv"],
            ),
            StepSpec(
                name="flow_extraction",
                description="Generate optical-flow maps.",
                command=[
                    py,
                    str(self.scripts_dir / "extract_representations.py"),
                    "--arch",
                    args.arch,
                    "--patch_size",
                    str(args.patch_size),
                    "--model_type",
                    model_type,
                    "--batch_size_per_gpu",
                    str(args.flow_batch_size),
                    "--data_path",
                    "./SAIS/",
                    "--data_list",
                    "Custom",
                    "--save_type",
                    "h5",
                    "--optical_flow",
                ],
                artifacts=["flows"],
            ),
            StepSpec(
                name="rgb_representations",
                description="Extract ViT embeddings from RGB frames.",
                command=[
                    py,
                    "-m",
                    "torch.distributed.launch",
                    str(self.scripts_dir / "extract_representations.py"),
                    "--arch",
                    args.arch,
                    "--patch_size",
                    str(args.patch_size),
                    "--model_type",
                    model_type,
                    "--batch_size_per_gpu",
                    str(args.rgb_batch_size),
                    "--data_path",
                    "./SAIS/",
                    "--data_list",
                    "Custom",
                    "--save_type",
                    "h5",
                ],
                artifacts=[f"results/{model_type}_RepsAndLabels.h5"],
            ),
            StepSpec(
                name="flow_representations",
                description="Extract ViT embeddings from flow maps.",
                command=[
                    py,
                    "-m",
                    "torch.distributed.launch",
                    str(self.scripts_dir / "extract_representations.py"),
                    "--arch",
                    args.arch,
                    "--patch_size",
                    str(args.patch_size),
                    "--model_type",
                    model_type,
                    "--batch_size_per_gpu",
                    str(args.flow_rep_batch_size),
                    "--data_path",
                    "./SAIS/",
                    "--data_list",
                    "Custom",
                    "--save_type",
                    "h5",
                    "--optical_flow_to_reps",
                ],
                artifacts=[f"results/{model_type}_FlowRepsAndLabels.h5"],
            ),
        ]

        if workflow == "training":
            return common_steps + [
                StepSpec(
                    name="training",
                    description="Train SAIS on prepared representations.",
                    command=[
                        py,
                        str(self.scripts_dir / "run_experiments.py"),
                        "-p",
                        "./SAIS/",
                        "-data",
                        "Custom_Gestures",
                        "-d",
                        "Custom_Top15",
                        "-m",
                        "ViT",
                        "-enc",
                        model_type,
                        "-t",
                        "Prototypes",
                        "-mod",
                        "RGB-Flow",
                        "-dim",
                        str(args.rep_dim),
                        "-bs",
                        str(args.train_batch_size),
                        "-lr",
                        str(args.learning_rate),
                        "-nc",
                        str(args.nclasses),
                        "-bc",
                        "-sa",
                        "-domains",
                        top15_domain,
                        "-ph",
                        "train",
                        "val",
                        "test",
                        "-dt",
                        "reps",
                        "-e",
                        str(args.epochs),
                        "-f",
                        str(args.folds),
                    ],
                    artifacts=["params"],
                )
            ]

        if workflow == "inference":
            return common_steps + [
                StepSpec(
                    name="inference",
                    description="Run SAIS inference using trained params.",
                    command=[
                        py,
                        str(self.scripts_dir / "run_experiments.py"),
                        "-p",
                        "./SAIS/",
                        "-data",
                        "Custom_Gestures",
                        "-d",
                        "Custom_Top15",
                        "-m",
                        "ViT",
                        "-enc",
                        model_type,
                        "-t",
                        "Prototypes",
                        "-mod",
                        "RGB-Flow",
                        "-dim",
                        str(args.rep_dim),
                        "-bs",
                        str(args.infer_batch_size),
                        "-lr",
                        str(args.learning_rate),
                        "-nc",
                        str(args.nclasses),
                        "-bc",
                        "-sa",
                        "-domains",
                        "Gesture",
                        "-ph",
                        "Custom_inference",
                        "-dt",
                        "reps",
                        "-e",
                        "1",
                        "-f",
                        "1",
                        "--inference",
                        "--simulate-single-gesture",
                    ],
                    artifacts=["params"],
                ),
                StepSpec(
                    name="inference_postprocess",
                    description="Convert raw inference outputs into interval predictions.",
                    command=[
                        py,
                        str(self.scripts_dir / "process_inference_results.py"),
                        "-p",
                        "./SAIS/",
                        "--domain",
                        "Gesture",
                        "--nclasses",
                        str(args.nclasses),
                    ],
                    artifacts=["predictions"],
                ),
            ]

        raise ValueError(f"Unsupported workflow: {workflow}")

    def init_run(self, run_id: str, workflow: str, steps: List[StepSpec]) -> Dict:
        state = {
            "run_id": run_id,
            "workflow": workflow,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "steps": [s.name for s in steps],
            "status": "initialized",
            "completed": [],
            "failed_step": None,
            "milestones": {},
        }
        self._save_state(run_id, state)
        return state

    def run(
        self,
        run_id: str,
        workflow: str,
        steps: List[StepSpec],
        resume: bool = True,
        stop_after: Optional[str] = None,
    ) -> None:
        state = None
        if resume and self._state_path(run_id).exists():
            state = self._load_state(run_id)
            if state.get("workflow") != workflow:
                raise ValueError(
                    f"run_id={run_id} belongs to workflow={state.get('workflow')}, not {workflow}."
                )
        if state is None:
            state = self.init_run(run_id, workflow, steps)

        completed = set(state.get("completed", []))
        state["status"] = "running"
        self._save_state(run_id, state)

        for step in steps:
            if step.name in completed:
                print(f"[SKIP] Milestone already completed: {step.name}")
                if stop_after and step.name == stop_after:
                    print(f"[STOP] Requested stop_after milestone reached: {stop_after}")
                    return
                continue

            try:
                print(f"\n=== Milestone: {step.name} ===")
                print(step.description)
                _safe_run(step.command, cwd=self.root)

                milestone_backup = self._backup_step(run_id, step)
                state["milestones"][step.name] = milestone_backup
                state["completed"].append(step.name)
                state["failed_step"] = None
                state["updated_at"] = _utc_now()
                self._save_state(run_id, state)
                print(f"[OK] Milestone completed: {step.name}")

                if stop_after and step.name == stop_after:
                    print(f"[STOP] Requested stop_after milestone reached: {stop_after}")
                    return
            except Exception as exc:
                state["status"] = "failed"
                state["failed_step"] = step.name
                state["updated_at"] = _utc_now()
                self._save_state(run_id, state)
                raise RuntimeError(f"Workflow failed at step '{step.name}': {exc}") from exc

        state["status"] = "completed"
        state["updated_at"] = _utc_now()
        self._save_state(run_id, state)
        print(f"\n[COMPLETE] Workflow '{workflow}' completed for run_id={run_id}")

    def rollback(self, run_id: str, workflow: str, steps: List[StepSpec], milestone: str) -> None:
        state = self._load_state(run_id)
        if state.get("workflow") != workflow:
            raise ValueError(
                f"run_id={run_id} belongs to workflow={state.get('workflow')}, not {workflow}."
            )

        step_names = [s.name for s in steps]
        if milestone not in step_names:
            raise ValueError(f"Milestone '{milestone}' is not in workflow '{workflow}'.")

        completed = state.get("completed", [])
        if milestone not in completed:
            raise ValueError(f"Milestone '{milestone}' has not been completed yet for run_id={run_id}.")

        milestone_index = step_names.index(milestone)
        downstream = steps[milestone_index + 1 :]

        print(f"[ROLLBACK] Target milestone: {milestone}")
        print("[ROLLBACK] Cleaning downstream artifacts...")
        for step in downstream:
            self._cleanup_artifacts(step.artifacts)

        backup_entry = state.get("milestones", {}).get(milestone)
        if backup_entry:
            print(f"[ROLLBACK] Restoring backup snapshot for milestone: {milestone}")
            self._restore_step_backup(run_id, milestone, backup_entry)
        else:
            print(f"[ROLLBACK] No milestone backup metadata found for: {milestone}")

        state["completed"] = [s for s in completed if step_names.index(s) <= milestone_index]
        state["status"] = "rolled_back"
        state["failed_step"] = None
        state["updated_at"] = _utc_now()
        self._save_state(run_id, state)
        print(f"[ROLLBACK] Done. You can now resume from milestone '{milestone}'.")

    def status(self, run_id: str) -> None:
        state = self._load_state(run_id)
        print(json.dumps(state, indent=2))


def _default_run_id(workflow: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{workflow}_{ts}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agentic workflow manager for SAIS training/inference with milestone checkpointing "
            "and rollback."
        )
    )
    parser.add_argument("--root", default=".", help="Repository root containing the SAIS folder.")
    parser.add_argument(
        "--workflow",
        choices=["training", "inference"],
        required=True,
        help="Workflow to run/manage.",
    )
    parser.add_argument("--run-id", default=None, help="Unique run identifier. Auto-generated if omitted.")
    parser.add_argument(
        "--backup-mode",
        choices=["copy", "manifest"],
        default="copy",
        help="copy stores snapshot files at each milestone; manifest stores metadata only.",
    )

    parser.add_argument("--arch", default="vit_small")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--model-type", default="ViT_SelfSupervised_ImageNet")
    parser.add_argument("--domain", default="Custom_Top15")
    parser.add_argument("--nclasses", type=int, default=15)
    parser.add_argument("--rep-dim", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1e-1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--folds", type=int, default=8)

    parser.add_argument("--flow-batch-size", type=int, default=2)
    parser.add_argument("--rgb-batch-size", type=int, default=1024)
    parser.add_argument("--flow-rep-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--infer-batch-size", type=int, default=8)

    subparsers = parser.add_subparsers(dest="action", required=True)

    run_parser = subparsers.add_parser("run", help="Run workflow steps with checkpoints.")
    run_parser.add_argument("--no-resume", action="store_true", help="Start fresh even if state exists.")
    run_parser.add_argument(
        "--stop-after",
        default=None,
        help="Stop once this milestone is completed (e.g., paths_setting, flow_representations).",
    )

    subparsers.add_parser("plan", help="Print ordered workflow milestones and commands.")

    rollback_parser = subparsers.add_parser("rollback", help="Rollback run state to a prior milestone.")
    rollback_parser.add_argument(
        "--milestone",
        required=True,
        help="Target completed milestone to rollback to.",
    )

    subparsers.add_parser("status", help="Print persisted state for run-id.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_id = args.run_id or _default_run_id(args.workflow)
    manager = AgenticWorkflowManager(
        root=Path(args.root),
        backup_mode=args.backup_mode,
    )
    steps = manager.list_workflow_steps(args.workflow, args)

    if args.action == "plan":
        print(f"Run ID: {run_id}")
        print(f"Workflow: {args.workflow}")
        for idx, step in enumerate(steps, start=1):
            print(f"{idx}. {step.name}: {step.description}")
            print(f"   Command: {' '.join(step.command)}")
            print(f"   Artifacts: {', '.join(step.artifacts)}")
        return

    if args.action == "run":
        manager.run(
            run_id=run_id,
            workflow=args.workflow,
            steps=steps,
            resume=(not args.no_resume),
            stop_after=args.stop_after,
        )
        return

    if args.action == "rollback":
        manager.rollback(
            run_id=run_id,
            workflow=args.workflow,
            steps=steps,
            milestone=args.milestone,
        )
        return

    if args.action == "status":
        manager.status(run_id=run_id)
        return


if __name__ == "__main__":
    main()
