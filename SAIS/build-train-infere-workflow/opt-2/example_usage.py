#!/usr/bin/env python3
"""
Example: SAIS Agentic AI Usage

Demonstrates practical workflow for:
1. Training on surgical video dataset
2. Running inference on new cases
3. Analyzing predictions

Run from SAIS root directory:
    python example_usage.py
"""

import sys
from pathlib import Path
from sais_agent import SAISAgent, SAISConfig, AgentState
import json

def example_1_single_procedure_training():
    """
    Example 1: Train on single surgical procedure (Prostatectomy)
    and test on new cases
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Procedure Training (Prostatectomy)")
    print("="*70)
    
    # Configuration
    sais_root = Path(".")
    dino_weights = Path("./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth")
    
    config = SAISConfig(
        sais_root=sais_root,
        videos_dir=sais_root / "videos",
        params_dir=sais_root / "params",
        results_dir=sais_root / "results",
        predictions_dir=sais_root / "predictions",
        dino_weights_path=dino_weights,
        
        # Training configuration
        model_type="ViT",
        encoder_type="ViT_SelfSupervised_ImageNet",
        learning_paradigm="Prototypes",
        modality="RGB-Flow",
        feature_dim=384,
        batch_size=8,
        learning_rate=0.1,
        num_classes=7,  # 7 surgical gestures in prostatectomy
        epochs=50,
        folds=5
    )
    
    # Create agent
    agent = SAISAgent(config)
    
    # Run workflow
    result = agent.end_to_end_workflow(
        training_videos=[
            "prostate_case_001",
            "prostate_case_002",
            "prostate_case_003",
            "prostate_case_004"
        ],
        training_dataset_name="ProstatectomyGestures_v1",
        training_dataset_type="ProstateSurgery",
        inference_videos=[
            "prostate_test_001",
            "prostate_test_002"
        ]
    )
    
    # Save report
    report_path = Path("example1_report.json")
    agent.save_execution_report(report_path)
    
    print(f"\nReport saved to: {report_path}")
    print(f"Status: {result['status']}")
    
    return result


def example_2_multi_procedure_training():
    """
    Example 2: Train on multiple surgical procedures
    Demonstrates generalization across different surgery types
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multi-Procedure Training")
    print("="*70)
    
    sais_root = Path(".")
    dino_weights = Path("./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth")
    
    config = SAISConfig(
        sais_root=sais_root,
        videos_dir=sais_root / "videos",
        params_dir=sais_root / "params",
        results_dir=sais_root / "results",
        predictions_dir=sais_root / "predictions",
        dino_weights_path=dino_weights,
        
        # For multi-procedure: more classes, more training
        num_classes=15,  # Gestures from multiple procedures
        epochs=100,
        batch_size=16,
        folds=5
    )
    
    agent = SAISAgent(config)
    
    # Mix videos from different procedures
    training_videos = [
        "prostate_001", "prostate_002", "prostate_003",
        "nephrectomy_001", "nephrectomy_002", "nephrectomy_003",
        "hysterectomy_001", "hysterectomy_002"
    ]
    
    inference_videos = [
        "prostate_new",
        "nephrectomy_new",
        "hysterectomy_new"
    ]
    
    result = agent.end_to_end_workflow(
        training_videos=training_videos,
        training_dataset_name="MultiProcedureSurgeries_v1",
        training_dataset_type="MixedProcedures",
        inference_videos=inference_videos
    )
    
    agent.save_execution_report(Path("example2_report.json"))
    
    return result


def example_3_high_precision_model():
    """
    Example 3: Train high-precision model for accurate gesture detection
    Uses more training data and epochs
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: High-Precision Model Training")
    print("="*70)
    
    sais_root = Path(".")
    dino_weights = Path("./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth")
    
    config = SAISConfig(
        sais_root=sais_root,
        videos_dir=sais_root / "videos",
        params_dir=sais_root / "params",
        results_dir=sais_root / "results",
        predictions_dir=sais_root / "predictions",
        dino_weights_path=dino_weights,
        
        # High precision settings
        num_classes=10,
        epochs=200,          # 4x more epochs for better convergence
        batch_size=32,       # Larger batch size (requires 24+ GB VRAM)
        learning_rate=0.01,  # Lower learning rate for fine-tuning
        folds=10             # More folds for robust evaluation
    )
    
    agent = SAISAgent(config)
    
    # Extensive training set
    training_videos = [f"surgery_train_{i:03d}" for i in range(1, 21)]
    inference_videos = [f"surgery_test_{i:03d}" for i in range(1, 6)]
    
    result = agent.end_to_end_workflow(
        training_videos=training_videos,
        training_dataset_name="PrecisionModel_v1",
        training_dataset_type="HighPrecision",
        inference_videos=inference_videos
    )
    
    agent.save_execution_report(Path("example3_report.json"))
    
    return result


def analyze_predictions(predictions_file: Path):
    """
    Analyze inference predictions
    """
    print("\n" + "="*70)
    print("ANALYZING PREDICTIONS")
    print("="*70)
    
    if not predictions_file.exists():
        print(f"Predictions file not found: {predictions_file}")
        return
    
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nVideo: {data.get('video_name', 'Unknown')}")
    predictions = data.get('predictions', [])
    
    print(f"Total predictions: {len(predictions)}")
    
    if predictions:
        print("\nPredictions Summary:")
        print("-" * 70)
        print(f"{'Action':<6} {'Gesture':<20} {'Confidence':<12} {'Frames':<15}")
        print("-" * 70)
        
        for pred in predictions[:10]:  # Show first 10
            action_id = pred.get('action_id', '?')
            gesture = pred.get('gesture_label', 'Unknown')
            confidence = pred.get('confidence', 0)
            start = pred.get('start_frame', 0)
            end = pred.get('end_frame', 0)
            
            print(f"{action_id:<6} {gesture:<20} {confidence:<12.4f} {start}-{end:<8}")
        
        # Confidence statistics
        confidences = [p.get('confidence', 0) for p in predictions]
        print("\n" + "-" * 70)
        print(f"Average Confidence: {sum(confidences)/len(confidences):.4f}")
        print(f"Max Confidence:     {max(confidences):.4f}")
        print(f"Min Confidence:     {min(confidences):.4f}")
        
        # Gesture distribution
        gesture_counts = {}
        for pred in predictions:
            gesture = pred.get('gesture_label', 'Unknown')
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        print("\nGesture Distribution:")
        for gesture, count in sorted(gesture_counts.items()):
            print(f"  {gesture}: {count} predictions")


def print_execution_summary(result: dict):
    """
    Print execution summary from workflow result
    """
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    
    print(f"Status: {result.get('status', 'unknown').upper()}")
    print(f"Training videos: {len(result.get('training_videos', []))}")
    print(f"Inference videos: {len(result.get('inference_videos', []))}")
    
    if result.get('inference_results'):
        print("\nInference Results:")
        for video_name, res in result['inference_results'].items():
            status = res.get('status', 'unknown')
            print(f"  {video_name}: {status}")


def main():
    """
    Main entry point - run examples
    """
    
    print("\n" + "#"*70)
    print("# SAIS AGENTIC AI - USAGE EXAMPLES")
    print("#"*70)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    
    sais_root = Path(".")
    dino_weights = Path("./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth")
    
    if not sais_root.exists():
        print("❌ Error: SAIS root directory not found")
        print("   Please run this script from SAIS repository root")
        return 1
    
    if not dino_weights.exists():
        print("❌ Error: DINO weights not found")
        print(f"   Expected: {dino_weights}")
        print("   Run: wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth \\")
        print(f"        -O {dino_weights}")
        return 1
    
    print("✅ Prerequisites validated")
    
    # Run examples
    examples = [
        ("Example 1: Single Procedure", example_1_single_procedure_training),
        ("Example 2: Multi-Procedure", example_2_multi_procedure_training),
        ("Example 3: High-Precision", example_3_high_precision_model),
    ]
    
    for i, (name, example_func) in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"Running {name}...")
        print(f"{'='*70}")
        
        try:
            result = example_func()
            print_execution_summary(result)
            
            # Try to analyze predictions if available
            pred_dir = Path("predictions")
            if pred_dir.exists():
                pred_files = list(pred_dir.glob("*_predictions.json"))
                if pred_files:
                    analyze_predictions(pred_files[0])
        
        except Exception as e:
            print(f"⚠️  Example {i} encountered error: {e}")
            print("   (This is expected if input videos don't exist)")
            continue
        
        # Ask if user wants to continue
        if i < len(examples):
            response = input(f"\nContinue to next example? (y/n): ").strip().lower()
            if response != 'y':
                break
    
    print("\n" + "#"*70)
    print("# ALL EXAMPLES COMPLETED")
    print("#"*70)
    print("\nNext steps:")
    print("1. Place your surgical videos in ./videos/")
    print("2. Update video names in the example functions")
    print("3. Adjust hyperparameters (epochs, batch_size, num_classes)")
    print("4. Run: python example_usage.py")
    print("\nFor more details, see SAIS_AGENT_GUIDE.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
