import argparse
import subprocess
from pathlib import Path


def resolve_video_path(videos_dir: Path, videoname: str) -> Path:
    """Resolve a video name to a file inside videos_dir, appending .mp4 if needed."""
    candidate = Path(videoname).name
    video_path = videos_dir / candidate
    if video_path.suffix == "":
        video_path = video_path.with_suffix(".mp4")
    return video_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames from a video in ./SAIS/videos using ffmpeg."
    )
    parser.add_argument("videoname", help="Video name (with or without .mp4 extension).")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Base output directory. Defaults to ./SAIS/images.",
    )
    args = parser.parse_args()

    sais_root = Path(__file__).resolve().parents[1]
    videos_dir = sais_root / "videos"

    output_base = Path(args.output_dir) if args.output_dir else (sais_root / "images")

    video_path = resolve_video_path(videos_dir, args.videoname)
    video_stem = Path(args.videoname).stem
    output_dir = output_base / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_pattern = output_dir / "frames_%8d.jpg"
    cmd = ["ffmpeg", "-i", str(video_path), str(output_pattern)]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install ffmpeg or add it to PATH."
        ) from exc


if __name__ == "__main__":
    main()
