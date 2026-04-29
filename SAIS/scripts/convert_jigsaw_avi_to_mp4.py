import argparse
import subprocess
from pathlib import Path


def build_input_dirs(dataset_root: Path):
    return [
        dataset_root / "Knot_Tying" / "Knot_Tying" / "video",
        dataset_root / "Needle_Passing" / "Needle_Passing" / "video",
        dataset_root / "Suturing" / "Suturing" / "video",
    ]


def find_capture_avi_files(input_dirs):
    avi_files = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"Skipping missing directory: {input_dir}")
            continue

        for avi_path in sorted(input_dir.glob("*.avi")):
            if "_capture1" in avi_path.stem or "_capture2" in avi_path.stem:
                avi_files.append(avi_path)
    return avi_files


def output_name_from_input(avi_path: Path):
    # Keep original name (including _captureX suffix), change extension to .mp4.
    return f"{avi_path.stem}.mp4"


def convert_with_ffmpeg(input_file: Path, output_file: Path, overwrite: bool):
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_file),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_file),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JIGSAWS _capture1/_capture2 AVI files to MP4 under ./SAIS/videos"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./SAIS/datasets/jigsaw"),
        help="Root directory containing JIGSAWS folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./SAIS/videos"),
        help="Directory where converted MP4 files are written",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist",
    )
    args = parser.parse_args()

    input_dirs = build_input_dirs(args.dataset_root)
    avi_files = find_capture_avi_files(input_dirs)

    if not avi_files:
        print("No AVI files with '_capture1' or '_capture2' were found.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for input_file in avi_files:
        output_file = args.output_dir / output_name_from_input(input_file)

        if output_file.exists() and not args.overwrite:
            print(f"Skipping existing file: {output_file}")
            skipped += 1
            continue

        print(f"Converting: {input_file.name} -> {output_file.name}")
        try:
            convert_with_ffmpeg(input_file, output_file, overwrite=args.overwrite)
            converted += 1
        except subprocess.CalledProcessError as exc:
            print(f"Failed to convert {input_file}: {exc}")

    print(f"Done. Converted: {converted}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
