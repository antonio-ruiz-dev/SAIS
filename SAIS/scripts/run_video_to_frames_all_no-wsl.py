import argparse
import subprocess
import time
from pathlib import Path


def run_video_to_frames(video_path: Path, output_dir: Path, frame_pattern: str = "frame_%08d.jpg") -> int:
    """Convert a single video to frames using FFmpeg."""
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build FFmpeg command
    output_pattern = output_dir / frame_pattern
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-q:v", "2",  # Quality (1=best, 31=worst); 2 is good quality
        str(output_pattern)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def format_elapsed_time(total_seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS.ss."""
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert all MP4 videos in a directory to frame sequences using FFmpeg."
    )
    parser.add_argument(
        "--videos-dir",
        default="./SAIS/videos",
        help="Directory with input .mp4 files (default: ./SAIS/videos)",
    )
    parser.add_argument(
        "--output-dir",
        default="./SAIS/images",
        help="Base directory for output frames (default: ./SAIS/images)",
    )
    parser.add_argument(
        "--frame-pattern",
        default="frame_%08d.jpg",
        help="Output frame filename pattern (default: frame_%%08d.jpg). Use %%08d for zero-padded counter.",
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
    output_base_dir = Path(args.output_dir)

    if not videos_dir.exists() or not videos_dir.is_dir():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    # Process only MP4 videos
    mp4_files = sorted(
        path for path in videos_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"
    )
    
    if not mp4_files:
        print(f"No .mp4 files found in {videos_dir}")
        return

    failures = []
    start_time = time.perf_counter()
    for idx, mp4_file in enumerate(mp4_files, 1):
        # Create a subdirectory for each video's frames (optional)
        video_output_dir = output_base_dir / mp4_file.stem
        
        cmd_preview = f"ffmpeg -i {mp4_file} -q:v 2 {video_output_dir}/{args.frame_pattern}"
        print(f"\n[{idx}/{len(mp4_files)}] Processing: {mp4_file.name}")

        if args.dry_run:
            print(f"  [DRY-RUN] {cmd_preview}")
            continue

        code = run_video_to_frames(mp4_file, video_output_dir, args.frame_pattern)
        if code != 0:
            failures.append((mp4_file, code))
            print(f"  ✗ Failed: {mp4_file.name} (exit code: {code})")
            if not args.continue_on_error:
                break
        else:
            print(f"  ✓ Success: {mp4_file.name}")

    elapsed_time = time.perf_counter() - start_time

    total = len(mp4_files)
    
    if args.dry_run:
        print(f"\nDry-run complete. Planned to process: {total} video(s)")
        print(f"Total time taken: {format_elapsed_time(elapsed_time)}")
        return

    print("\n" + "="*60)
    if failures:
        print(f"Completed with failures: {len(failures)} of {total}")
        for video_name, code in failures:
            print(f"  - {video_name.name}: exit code {code}")
    else:
        print(f"✓ Completed successfully: {total} video(s) processed")
    print(f"Total time taken: {format_elapsed_time(elapsed_time)}")
    print("="*60)


if __name__ == "__main__":
    main()
