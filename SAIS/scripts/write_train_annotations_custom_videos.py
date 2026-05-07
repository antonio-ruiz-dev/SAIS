import argparse
from pathlib import Path

import pandas as pd


TASK_NAMES = ("Knot_Tying", "Needle_Passing", "Suturing")
OUTPUT_COLUMNS = ["id", "Video", "Gesture", "StartFrame", "EndFrame"]


def find_transcription_dir(dataset_root: Path, task_name: str) -> Path:
    candidate_dirs = [
        dataset_root / task_name / "transcriptions",
        dataset_root / task_name / task_name / "transcriptions",
    ]
    for candidate_dir in candidate_dirs:
        if candidate_dir.is_dir():
            return candidate_dir
    raise FileNotFoundError(
        "Could not find transcriptions directory for %s. Checked: %s"
        % (task_name, ", ".join(str(path) for path in candidate_dirs))
    )


def collect_available_videos(videos_dir: Path):
    if not videos_dir.exists() or not videos_dir.is_dir():
        raise FileNotFoundError("Videos directory not found: %s" % videos_dir)

    videos_by_base_name = {}
    for video_path in sorted(videos_dir.glob("*.mp4")):
        stem = video_path.stem
        if stem.endswith("_capture1"):
            base_name = stem[: -len("_capture1")]
        elif stem.endswith("_capture2"):
            base_name = stem[: -len("_capture2")]
        else:
            base_name = stem
        videos_by_base_name.setdefault(base_name, []).append(stem)

    if not videos_by_base_name:
        raise FileNotFoundError("No .mp4 files found in %s" % videos_dir)

    return videos_by_base_name


def parse_transcription_file(file_path: Path, row_id_start: int, video_names: list[str]):
    rows = []
    row_id = row_id_start
    
    for video_name in video_names:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 3:
                    raise ValueError(
                        "Malformed transcription row in %s:%i. Expected 'StartFrame EndFrame Gesture'."
                        % (file_path, line_number)
                    )

                try:
                    start_frame = int(parts[0])
                    end_frame = int(parts[1])
                except ValueError as exc:
                    raise ValueError(
                        "Invalid frame range in %s:%i. StartFrame and EndFrame must be integers."
                        % (file_path, line_number)
                    ) from exc

                row_id += 1
                rows.append(
                    {
                        "id": row_id,
                        "Video": video_name,
                        "Gesture": parts[2].strip(),
                        "StartFrame": start_frame,
                        "EndFrame": end_frame,
                    }
                )

    return rows, row_id


def build_annotations_dataframe(dataset_root: Path, videos_dir: Path) -> pd.DataFrame:
    rows = []
    row_id = 0
    available_videos = collect_available_videos(videos_dir)

    for task_name in TASK_NAMES:
        transcription_dir = find_transcription_dir(dataset_root, task_name)
        transcription_files = sorted(
            file_path
            for file_path in transcription_dir.glob("*.txt")
            if file_path.stem in available_videos
        )
        if not transcription_files:
            continue

        for file_path in transcription_files:
            file_rows, row_id = parse_transcription_file(
                file_path,
                row_id,
                available_videos[file_path.stem],
            )
            rows.extend(file_rows)

    if not rows:
        raise ValueError(
            "No annotation rows were generated from %s using videos from %s"
            % (dataset_root, videos_dir)
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def resolve_default_paths(script_path: Path):
    project_root = script_path.parents[1]
    dataset_root = project_root / "datasets" / "jigsaw"
    videos_dir = project_root / "videos"
    output_path = project_root / "annotations" / "Custom_Gestures_Annotations.csv"
    return dataset_root, videos_dir, output_path


def main() -> None:
    script_path = Path(__file__).resolve()
    default_dataset_root, default_videos_dir, default_output_path = resolve_default_paths(script_path)

    parser = argparse.ArgumentParser(
        description="Generate Custom_Gestures_Annotations.csv from JIGSAWS transcription TXT files."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(default_dataset_root),
        help="Path to the jigsaw dataset root containing Knot_Tying, Needle_Passing and Suturing.",
    )
    parser.add_argument(
        "--output",
        default=str(default_output_path),
        help="Path for the generated annotation CSV.",
    )
    parser.add_argument(
        "--videos-dir",
        default=str(default_videos_dir),
        help="Path to the directory containing the custom .mp4 videos used to select transcription files.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    videos_dir = Path(args.videos_dir).resolve()
    output_path = Path(args.output).resolve()

    annotations_df = build_annotations_dataframe(dataset_root, videos_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_df.to_csv(output_path, index=False)

    print("Wrote %i annotation rows to %s" % (len(annotations_df), output_path))
    print("Videos: %i" % annotations_df["Video"].nunique())
    print("Gestures: %s" % ", ".join(sorted(annotations_df["Gesture"].unique().tolist())))


if __name__ == "__main__":
    main()