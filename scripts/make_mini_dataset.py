import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceVideo:
    label: str
    url: str
    filename: str


SOURCES: list[SourceVideo] = [
    SourceVideo(
        label="bunny",
        url="https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
        filename="bigbuckbunny_320x180.mp4",
    ),
    SourceVideo(
        label="sintel",
        # Small extract (CC BY 3.0). Direct download via Wikimedia.
        url="https://commons.wikimedia.org/wiki/Special:FilePath/Sintel_webm_extract.webm",
        filename="sintel_extract.webm",
    ),
    SourceVideo(
        label="tears",
        # One-minute battle scene clip (CC BY 3.0).
        url="https://commons.wikimedia.org/wiki/Special:FilePath/Tears_of_Steel_clip.ogv",
        filename="tearsofsteel_clip.ogv",
    ),
    SourceVideo(
        label="dream",
        # Low-bitrate full movie (CC BY).
        url="https://commons.wikimedia.org/wiki/Special:FilePath/Elephants_Dream_1024.avi.w400vbr180abr48c2two-pass.ogv",
        filename="elephantsdream_ogv.ogv",
    ),
]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _probe_duration(path: str) -> float:
    # Use ffprobe (part of ffmpeg)
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        text=True,
    ).strip()
    return float(out)


def _download(url: str, dst: str) -> None:
    Path(os.path.dirname(dst) or ".").mkdir(parents=True, exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        try:
            _ = _probe_duration(dst)
            return
        except Exception:
            os.remove(dst)

    tmp = dst + ".partial"
    if os.path.exists(tmp):
        os.remove(tmp)

    _run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "3",
            "--retry-delay",
            "1",
            "-A",
            "Mozilla/5.0",
            "-o",
            tmp,
            url,
        ]
    )
    os.replace(tmp, dst)
    # Validate
    _ = _probe_duration(dst)


def _make_clips(src_path: str, out_dir: str, clips_per_class: int, clip_seconds: float, resize: int) -> None:
    dur = _probe_duration(src_path)
    if dur <= clip_seconds + 1.0:
        raise RuntimeError(f"Source too short ({dur}s): {src_path}")

    # Evenly spaced start times, leaving a margin at start/end
    margin = min(5.0, max(1.0, dur * 0.02))
    span = dur - 2 * margin - clip_seconds
    if span <= 0:
        span = max(0.0, dur - clip_seconds - 0.5)
        margin = 0.0

    for i in range(clips_per_class):
        start = margin + (span * i / max(clips_per_class - 1, 1))
        out_path = os.path.join(out_dir, f"clip_{i:02d}.mp4")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue

        # Re-encode to ensure seekable, consistent fps, and small size.
        scale = f"scale={resize}:{resize}:force_original_aspect_ratio=decrease,pad={resize}:{resize}:(ow-iw)/2:(oh-ih)/2"
        _run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{clip_seconds:.3f}",
                "-i",
                src_path,
                "-vf",
                scale,
                "-r",
                "24",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-an",
                out_path,
            ]
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a tiny 4-class x N-clips dataset from public Blender films.")
    ap.add_argument("--out_dir", type=str, default="data/mini_4x10")
    ap.add_argument("--clips_per_class", type=int, default=10)
    ap.add_argument("--clip_seconds", type=float, default=2.5)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--sources_dir", type=str, default="data/_sources")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.sources_dir).mkdir(parents=True, exist_ok=True)

    for src in SOURCES:
        src_path = os.path.join(args.sources_dir, src.filename)
        print(f"Downloading {src.label} -> {src_path}")
        _download(src.url, src_path)

        class_dir = os.path.join(args.out_dir, src.label)
        Path(class_dir).mkdir(parents=True, exist_ok=True)
        print(f"Creating clips for {src.label} in {class_dir}")
        _make_clips(src_path, class_dir, args.clips_per_class, args.clip_seconds, args.resize)

    print(f"Done. Dataset at: {args.out_dir}")
    print("Classes:", ", ".join([s.label for s in SOURCES]))


if __name__ == "__main__":
    main()
