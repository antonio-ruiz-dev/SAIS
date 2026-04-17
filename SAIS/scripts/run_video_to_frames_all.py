import argparse
import subprocess
from pathlib import Path


def run_video_to_frames(script_path: Path, video_name: str) -> int:
    """Run the shell script for a single video name (without extension)."""

    # Fixed — convert all Path objects to str
    cmd = ["wsl.exe", "--cd", ".", "bash", str(script_path).replace("\\", "/"), "-f", str(video_name).replace("\\", "/")]

    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run video_to_frames.sh for all MP4 files in a directory."
    )
    parser.add_argument(
        "--videos-dir",
        default="./SAIS/videos",
        help="Directory with input .mp4 files (default: ./SAIS/videos)",
    )
    parser.add_argument(
        "--script-path",
        default="./SAIS/scripts/video_to_frames.sh",
        help="Path to video_to_frames.sh (default: ./SAIS/scripts/video_to_frames.sh)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing remaining videos even if one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    script_path = Path(args.script_path)

    if not videos_dir.exists() or not videos_dir.is_dir():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    if not script_path.exists() or not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Process only MP4 videos and ignore any other format (e.g., AVI, MOV, MKV).
    mp4_files = sorted(
        path for path in videos_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not mp4_files:
        print(f"No .mp4 files found in {videos_dir}")
        return

    failures = []
    for mp4_file in mp4_files:
        video_name = mp4_file
        cmd_preview = f"wsl.exe --cd . bash {script_path} -f {video_name}"

        if args.dry_run:
            print("[DRY-RUN]", cmd_preview)
            continue

        code = run_video_to_frames(script_path, video_name)
        if code != 0:
            failures.append((video_name, code))
            print(f"Failed: {video_name} (exit code: {code})")
            if not args.continue_on_error:
                break

    total = len(mp4_files)
    if args.dry_run:
        print(f"Dry-run complete. Planned commands: {total}")
        return

    if failures:
        print(f"Completed with failures: {len(failures)} of {total}")
        for video_name, code in failures:
            print(f" - {video_name}: {code}")
    else:
        print(f"Completed successfully: {total} videos processed.")


if __name__ == "__main__":
    main()

