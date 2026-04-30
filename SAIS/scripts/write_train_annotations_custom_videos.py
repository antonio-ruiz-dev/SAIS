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


def parse_transcription_file(file_path: Path, row_id_start: int):
    rows = []
    row_id = row_id_start
    base_video_name = file_path.stem
    
    # Generate annotations for both _capture1 and _capture2
    for capture_suffix in ("_capture1", "_capture2"):
        video_name = base_video_name + capture_suffix
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


def build_annotations_dataframe(dataset_root: Path) -> pd.DataFrame:
    rows = []
    row_id = 0

    for task_name in TASK_NAMES:
        transcription_dir = find_transcription_dir(dataset_root, task_name)
        transcription_files = sorted(transcription_dir.glob("*.txt"))
        if not transcription_files:
            raise FileNotFoundError("No transcription files found in %s" % transcription_dir)

        for file_path in transcription_files:
            file_rows, row_id = parse_transcription_file(file_path, row_id)
            rows.extend(file_rows)

    if not rows:
        raise ValueError("No annotation rows were generated from %s" % dataset_root)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def resolve_default_paths(script_path: Path):
    project_root = script_path.parents[1]
    dataset_root = project_root / "datasets" / "jigsaw"
    output_path = project_root / "annotations" / "Custom_Gestures_Annotations.csv"
    return dataset_root, output_path


def main() -> None:
    script_path = Path(__file__).resolve()
    default_dataset_root, default_output_path = resolve_default_paths(script_path)

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
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_path = Path(args.output).resolve()

    annotations_df = build_annotations_dataframe(dataset_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_df.to_csv(output_path, index=False)

    print("Wrote %i annotation rows to %s" % (len(annotations_df), output_path))
    print("Videos: %i" % annotations_df["Video"].nunique())
    print("Gestures: %s" % ", ".join(sorted(annotations_df["Gesture"].unique().tolist())))


if __name__ == "__main__":
    main()