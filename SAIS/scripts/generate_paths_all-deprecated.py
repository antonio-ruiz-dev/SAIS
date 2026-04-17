import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


# VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg", ".m4v"}
VIDEO_EXTENSIONS = {".mp4"}  # Restrict to .mp4 as per current dataset format and script compatibility.

def collect_video_names(videos_dir: Path) -> list[str]:
    """Return sorted unique video names (stem only) from the videos directory."""
    names = {
        path.stem
        for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    }
    return sorted(names)


def build_frame_paths(project_path: Path, video_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Create one combined frame paths dataframe for all videos."""
    images_dir = project_path / "images"
    rows = []
    skipped = []

    for video_name in video_names:
        case_dir = images_dir / video_name
        if not case_dir.exists() or not case_dir.is_dir():
            skipped.append(video_name)
            continue

        files = sorted(os.listdir(case_dir))
        for file in files:
            rows.append(
                {
                    "path": os.path.join("images", video_name, file),
                    "category": video_name,
                    "label": video_name,
                }
            )

    return pd.DataFrame(rows, columns=["path", "category", "label"]), skipped


def build_flow_paths(project_path: Path, video_names: list[str], jump_frames: int = 15) -> tuple[pd.DataFrame, list[str]]:
    """Create one combined flow paths dataframe for all videos."""
    images_dir = project_path / "images"
    rows = []
    skipped = []

    for video_name in video_names:
        case_dir = images_dir / video_name
        if not case_dir.exists() or not case_dir.is_dir():
            skipped.append(video_name)
            continue

        files = sorted(os.listdir(case_dir))
        if len(files) <= jump_frames:
            # Not enough frames to create path pairs for this case.
            continue

        indices = np.arange(0, len(files) - jump_frames, jump_frames)
        sampled_files = [files[idx] for idx in indices]

        for file in sampled_files:
            frame_num = int(file.split("_")[-1].strip(".jpg"))
            next_frame = frame_num + jump_frames
            next_file = f"frames_{next_frame:08d}.jpg"
            rows.append(
                {
                    "path1": os.path.join("images", video_name, file),
                    "path2": os.path.join("images", video_name, next_file),
                    "category": video_name,
                    "label": video_name,
                }
            )

    flow_df = pd.DataFrame(rows, columns=["path1", "path2", "category", "label"])
    if not flow_df.empty:
        flow_df["nflow"] = flow_df[["path1", "label"]].apply(
            lambda row: int(row["path1"].split("frames_")[-1].strip(".jpg")) // jump_frames,
            axis=1,
        )
        flow_df["flowpath"] = flow_df[["label", "nflow"]].apply(
            lambda row: os.path.join("flows", row["label"], f"flows_{int(row['nflow']):08d}.jpg"),
            axis=1,
        )
        flow_df.drop(columns=["nflow"], inplace=True)

    return flow_df, skipped


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_project_path = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Generate combined Custom_Paths.csv and Custom_FlowPaths.csv for all videos."
    )
    parser.add_argument(
        "--videos-dir",
        default=str(default_project_path / "videos"),
        help="Videos directory (default: ./SAIS/SAIS/videos)",
    )
    parser.add_argument(
        "--project-path",
        default=str(default_project_path),
        help="Path passed to generate_paths.py with -p (default: ./SAIS/SAIS)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, without executing.",
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    project_path = Path(args.project_path)
    save_dir = project_path / "paths"
    save_paths_file = save_dir / "Custom_Paths.csv"
    save_flow_file = save_dir / "Custom_FlowPaths.csv"

    if not videos_dir.exists() or not videos_dir.is_dir():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    video_names = collect_video_names(videos_dir)
    if not video_names:
        print(f"No video files found in {videos_dir}")
        return

    if args.dry_run:
        print(f"[DRY-RUN] Build combined paths for {len(video_names)} videos from {videos_dir}")
        print(f"[DRY-RUN] Would write: {save_paths_file}")
        print(f"[DRY-RUN] Would write: {save_flow_file}")
        print(f"Dry-run complete. Planned videos: {len(video_names)}")
        return

    frame_df, skipped_frame = build_frame_paths(project_path, video_names)
    flow_df, skipped_flow = build_flow_paths(project_path, video_names, jump_frames=15)

    frame_df.to_csv(save_paths_file)
    flow_df.to_csv(save_flow_file)

    skipped = sorted(set(skipped_frame).union(set(skipped_flow)))
    if skipped:
        print("Skipped videos without extracted frames directory:")
        for name in skipped:
            print(f" - {name}")

    total = len(video_names)
    if args.dry_run:
        print(f"Dry-run complete. Planned videos: {total}")
        return

    print(f"Completed successfully: {total} video names processed.")
    print(f"Saved frame paths: {save_paths_file} ({len(frame_df)} rows)")
    print(f"Saved flow paths: {save_flow_file} ({len(flow_df)} rows)")


if __name__ == "__main__":
    main()
