#!/usr/bin/env python3
"""
SAIS Agentic AI - Autonomous Model Training & Inference System

Orchestrates the complete SAIS workflow:
1. Environment validation & setup
2. Feature extraction from surgical videos
3. Model training with supervised contrastive learning
4. Inference on new surgical videos
5. Result aggregation and reporting
"""

import os
import sys
import subprocess
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SAISAgent')


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    TRAINING = "training"
    INFERENCING = "inferencing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class SAISConfig:
    """SAIS configuration parameters"""
    sais_root: Path
    videos_dir: Path
    params_dir: Path
    results_dir: Path
    predictions_dir: Path
    dino_weights_path: Path
    
    # Training parameters
    model_type: str = "ViT"
    encoder_type: str = "ViT_SelfSupervised_ImageNet"
    learning_paradigm: str = "Prototypes"
    modality: str = "RGB-Flow"
    feature_dim: int = 384
    batch_size: int = 8
    learning_rate: float = 0.1
    num_classes: int = 10
    epochs: int = 50
    folds: int = 5
    
    # Inference parameters
    inference_batch_size: int = 2
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration"""
        if not self.sais_root.exists():
            return False, f"SAIS root directory not found: {self.sais_root}"
        
        if not self.dino_weights_path.exists():
            return False, f"DINO weights not found: {self.dino_weights_path}"
        
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        return True, "Configuration validated"


class SAISAgent:
    """Agentic AI orchestrating SAIS training and inference"""
    
    def __init__(self, config: SAISConfig):
        self.config = config
        self.state = AgentState.IDLE
        self.execution_log: List[Dict] = []
        self.current_task = None
    
    def log_action(self, action: str, status: str, details: Dict = None):
        """Log agent actions"""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "status": status,
            "details": details or {}
        }
        self.execution_log.append(entry)
        logger.info(f"[{status.upper()}] {action}")
        if details:
            logger.info(f"  Details: {json.dumps(details, indent=2)}")
    
    def set_state(self, state: AgentState):
        """Update agent state"""
        self.state = state
        logger.info(f"Agent state changed to: {state.value}")
    
    def run_command(self, command: List[str], description: str) -> Tuple[bool, str]:
        """Execute shell command with error handling"""
        try:
            logger.info(f"Executing: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                self.log_action(description, "failed", {"error": error_msg[:200]})
                return False, error_msg
            
            self.log_action(description, "success")
            return True, result.stdout
        except subprocess.TimeoutExpired:
            msg = f"Command timed out: {description}"
            self.log_action(description, "timeout", {"timeout": "3600s"})
            return False, msg
        except Exception as e:
            self.log_action(description, "error", {"exception": str(e)})
            return False, str(e)
    
    def validate_environment(self) -> bool:
        """Validate SAIS environment setup"""
        self.set_state(AgentState.VALIDATING)
        self.log_action("Environment validation", "started")
        
        valid, msg = self.config.validate()
        if not valid:
            self.log_action("Environment validation", "failed", {"error": msg})
            self.set_state(AgentState.ERROR)
            return False
        
        # Check Python packages
        required_packages = ["torch", "transformers", "torchvision"]
        for package in required_packages:
            try:
                __import__(package)
                self.log_action(f"Package check: {package}", "success")
            except ImportError:
                self.log_action(f"Package check: {package}", "failed")
                return False
        
        # Check DINO weights
        if not self.config.dino_weights_path.exists():
            self.log_action("DINO weights check", "failed", 
                          {"path": str(self.config.dino_weights_path)})
            return False
        
        self.log_action("Environment validation", "completed")
        return True
    
    def extract_features_from_videos(self, video_names: List[str]) -> bool:
        """Extract features from surgical videos using DINO"""
        self.set_state(AgentState.EXTRACTING)
        self.log_action("Feature extraction", "started", {"videos": video_names})
        
        scripts_dir = self.config.sais_root / "SAIS" / "scripts"
        
        for video_name in video_names:
            video_path = self.config.videos_dir / f"{video_name}.mp4"
            
            if not video_path.exists():
                self.log_action(f"Feature extraction: {video_name}", "skipped", 
                              {"reason": "video not found"})
                continue
            
            # Step 1: Extract frames
            video_to_frames_cmd = [
                "bash",
                str(scripts_dir / "video_to_frames.sh"),
                str(video_path),
                str(self.config.sais_root / "SAIS" / "images")
            ]
            success, msg = self.run_command(
                video_to_frames_cmd,
                f"Extract frames from {video_name}"
            )
            if not success:
                continue
            
            # Step 2: Generate paths
            generate_paths_cmd = [
                "python",
                str(scripts_dir / "generate_paths.py"),
                "-i", str(self.config.sais_root / "SAIS" / "images"),
                "-o", str(self.config.sais_root / "SAIS" / "paths")
            ]
            success, msg = self.run_command(
                generate_paths_cmd,
                f"Generate paths for {video_name}"
            )
            if not success:
                continue
            
            # Step 3: Extract representations (DINO + Optical Flow)
            extract_repr_cmd = [
                "python",
                str(scripts_dir / "extract_representations.py"),
                "--dino_path", str(self.config.dino_weights_path),
                "--paths_dir", str(self.config.sais_root / "SAIS" / "paths"),
                "--output_dir", str(self.config.results_dir)
            ]
            success, msg = self.run_command(
                extract_repr_cmd,
                f"Extract representations for {video_name}"
            )
            if not success:
                continue
            
            self.log_action(f"Feature extraction: {video_name}", "completed")
        
        return True
    
    def train_model(self, dataset_name: str, dataset_type: str) -> bool:
        """Train SAIS model on labeled dataset"""
        self.set_state(AgentState.TRAINING)
        self.log_action("Model training", "started", 
                       {"dataset": dataset_name, "type": dataset_type})
        
        scripts_dir = self.config.sais_root / "SAIS" / "scripts"
        
        train_cmd = [
            "python",
            "-m", "torch.distributed.launch",
            str(scripts_dir / "run_experiments.py"),
            "-p", str(self.config.sais_root),
            "-data", dataset_name,
            "-d", dataset_type,
            "-m", self.config.model_type,
            "-enc", self.config.encoder_type,
            "-t", self.config.learning_paradigm,
            "-mod", self.config.modality,
            "-dim", str(self.config.feature_dim),
            "-bs", str(self.config.batch_size),
            "-lr", str(self.config.learning_rate),
            "-nc", str(self.config.num_classes),
            "-bc",  # Use batch contrastive loss
            "-sa",  # Use supervised attention
            "-e", str(self.config.epochs),
            "-f", str(self.config.folds)
        ]
        
        success, msg = self.run_command(
            train_cmd,
            f"Train model on {dataset_name}"
        )
        
        if not success:
            self.set_state(AgentState.ERROR)
            return False
        
        # Verify model saved
        fold_0_path = self.config.params_dir / "Fold_0"
        if (fold_0_path / "params.zip").exists() and (fold_0_path / "prototypes.zip").exists():
            self.log_action("Model training", "completed", 
                          {"saved_to": str(fold_0_path)})
            return True
        else:
            self.log_action("Model training", "failed", 
                          {"reason": "model files not found after training"})
            return False
    
    def run_inference(self, video_names: List[str]) -> Dict:
        """Run inference on surgical videos"""
        self.set_state(AgentState.INFERENCING)
        self.log_action("Inference", "started", {"videos": video_names})
        
        results = {}
        scripts_dir = self.config.sais_root / "SAIS" / "scripts"
        
        for video_name in video_names:
            # Check if features already extracted
            feature_file = self.config.results_dir / f"{video_name}_features.h5"
            if not feature_file.exists():
                logger.warning(f"Features not found for {video_name}, extracting now...")
                self.extract_features_from_videos([video_name])
            
            # Run inference
            inference_cmd = [
                "python",
                str(scripts_dir / "run_experiments.py"),
                "-p", str(self.config.sais_root),
                "--inference",
                "-f", video_name,
                "-m", self.config.model_type,
                "-t", self.config.learning_paradigm,
                "-mod", self.config.modality,
                "-bs", str(self.config.inference_batch_size),
                "-nc", str(self.config.num_classes)
            ]
            
            success, output = self.run_command(
                inference_cmd,
                f"Inference on {video_name}"
            )
            
            if success:
                # Parse predictions
                predictions_file = self.config.predictions_dir / f"{video_name}_predictions.json"
                if predictions_file.exists():
                    with open(predictions_file, 'r') as f:
                        results[video_name] = json.load(f)
                    self.log_action(f"Inference: {video_name}", "completed")
                else:
                    results[video_name] = {"status": "processed", "output": output}
            else:
                results[video_name] = {"status": "failed", "error": output[:200]}
        
        self.set_state(AgentState.COMPLETE)
        return results
    
    def end_to_end_workflow(self, 
                           training_videos: List[str],
                           training_dataset_name: str,
                           training_dataset_type: str,
                           inference_videos: List[str]) -> Dict:
        """Complete workflow: extract features → train → infer"""
        self.log_action("End-to-end workflow", "started")
        
        # Validate environment
        if not self.validate_environment():
            self.log_action("End-to-end workflow", "failed", 
                          {"stage": "environment validation"})
            return {"status": "failed", "reason": "Environment validation failed"}
        
        # Extract features from training videos
        logger.info("=" * 60)
        logger.info("STAGE 1: FEATURE EXTRACTION (Training Videos)")
        logger.info("=" * 60)
        if not self.extract_features_from_videos(training_videos):
            self.log_action("End-to-end workflow", "failed", 
                          {"stage": "feature extraction"})
            return {"status": "failed", "reason": "Feature extraction failed"}
        
        # Train model
        logger.info("=" * 60)
        logger.info("STAGE 2: MODEL TRAINING")
        logger.info("=" * 60)
        if not self.train_model(training_dataset_name, training_dataset_type):
            self.log_action("End-to-end workflow", "failed", 
                          {"stage": "training"})
            return {"status": "failed", "reason": "Training failed"}
        
        # Extract features from inference videos
        logger.info("=" * 60)
        logger.info("STAGE 3: FEATURE EXTRACTION (Inference Videos)")
        logger.info("=" * 60)
        if not self.extract_features_from_videos(inference_videos):
            logger.warning("Feature extraction for inference videos partially failed")
        
        # Run inference
        logger.info("=" * 60)
        logger.info("STAGE 4: INFERENCE")
        logger.info("=" * 60)
        inference_results = self.run_inference(inference_videos)
        
        # Summary
        self.log_action("End-to-end workflow", "completed")
        
        return {
            "status": "success",
            "training_videos": training_videos,
            "inference_videos": inference_videos,
            "inference_results": inference_results,
            "execution_log": self.execution_log
        }
    
    def save_execution_report(self, output_path: Path):
        """Save detailed execution report"""
        report = {
            "agent_state": self.state.value,
            "execution_log": self.execution_log,
            "config": {
                "sais_root": str(self.config.sais_root),
                "model_type": self.config.model_type,
                "learning_paradigm": self.config.learning_paradigm,
                "modality": self.config.modality,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Execution report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAIS Agentic AI - Autonomous Model Training & Inference"
    )
    parser.add_argument("--sais_root", type=Path, required=True,
                       help="Path to SAIS repository root")
    parser.add_argument("--dino_weights", type=Path, required=True,
                       help="Path to DINO pre-trained weights")
    parser.add_argument("--training_videos", nargs="+", required=True,
                       help="Training video names (without .mp4 extension)")
    parser.add_argument("--training_dataset_name", default="CustomSurgicalDataset",
                       help="Name of training dataset")
    parser.add_argument("--training_dataset_type", default="Custom",
                       help="Type of training dataset")
    parser.add_argument("--inference_videos", nargs="+", required=True,
                       help="Inference video names (without .mp4 extension)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--num_classes", type=int, default=10,
                       help="Number of gesture classes")
    parser.add_argument("--report", type=Path, default=Path("sais_report.json"),
                       help="Path to save execution report")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SAISConfig(
        sais_root=args.sais_root,
        videos_dir=args.sais_root / "videos",
        params_dir=args.sais_root / "params",
        results_dir=args.sais_root / "results",
        predictions_dir=args.sais_root / "predictions",
        dino_weights_path=args.dino_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_classes=args.num_classes
    )
    
    # Create and run agent
    agent = SAISAgent(config)
    
    result = agent.end_to_end_workflow(
        training_videos=args.training_videos,
        training_dataset_name=args.training_dataset_name,
        training_dataset_type=args.training_dataset_type,
        inference_videos=args.inference_videos
    )
    
    # Save report
    agent.save_execution_report(args.report)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(json.dumps(result, indent=2, default=str))
    
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
