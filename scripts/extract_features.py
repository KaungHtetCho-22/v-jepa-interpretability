import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.embeddings import extract_frame_embeddings
from src.model import load_encoder, preprocess_video

logger = logging.getLogger("vjepa.extract_features")


def _iter_class_videos(video_dir: str, classes: list[str]) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for ci, cname in enumerate(classes):
        class_dir = os.path.join(video_dir, cname)
        if not os.path.isdir(class_dir):
            logger.warning("Missing class directory: %s", class_dir)
            continue
        for p in sorted(Path(class_dir).glob("*")):
            if p.is_file() and p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv", ".webm"}:
                items.append((str(p), ci))
    return items


def _read_video_frames(path: str, max_frames: int) -> list[np.ndarray]:
    try:
        import decord  # type: ignore

        vr = decord.VideoReader(path)
        n = len(vr)
        if n == 0:
            return []
        idxs = np.linspace(0, n - 1, num=min(max_frames, n)).round().astype(int).tolist()
        frames = vr.get_batch(idxs).asnumpy()
        return [frames[i] for i in range(frames.shape[0])]
    except Exception:
        try:
            import torchvision

            v, _, _ = torchvision.io.read_video(path, pts_unit="sec")
            t = v.shape[0]
            if t == 0:
                return []
            idxs = np.linspace(0, t - 1, num=min(max_frames, t)).round().astype(int)
            frames = v[idxs].numpy()
            return [frames[i] for i in range(frames.shape[0])]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read video frames from {path}: {exc}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch feature extraction for linear probing.")
    ap.add_argument("--video_dir", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--model_size", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--classes", nargs="+", required=True, help="Class folder names (UCF-101 subset)")
    ap.add_argument("--num_clips_per_class", type=int, default=20)
    ap.add_argument("--frames_per_clip", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--resume", action="store_true", help="Skip clips already present in output .npz")
    ap.add_argument("--dry_run", action="store_true", help="List files and exit")
    args = ap.parse_args()

    items = _iter_class_videos(args.video_dir, args.classes)
    if len(items) == 0:
        raise SystemExit("No videos found. Check --video_dir and --classes.")

    # Cap per class
    per_class: dict[int, list[str]] = {i: [] for i in range(len(args.classes))}
    for p, ci in items:
        if len(per_class[ci]) < args.num_clips_per_class:
            per_class[ci].append(p)

    selected: list[tuple[str, int]] = []
    for ci in range(len(args.classes)):
        for p in per_class[ci]:
            selected.append((p, ci))

    if args.dry_run:
        print(f"Selected {len(selected)} clips:")
        for p, ci in selected[:20]:
            print(f"  [{args.classes[ci]}] {p}")
        return

    done_paths: set[str] = set()
    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []

    if args.resume and os.path.exists(args.output):
        data = np.load(args.output, allow_pickle=True)
        existing_paths = data.get("paths")
        if existing_paths is not None:
            done_paths = set(existing_paths.tolist())
        existing_features = data.get("features")
        existing_labels = data.get("labels")
        if existing_features is not None and existing_labels is not None and existing_paths is not None:
            features = [row.astype(np.float32, copy=False) for row in existing_features]
            labels = existing_labels.tolist()
            paths = existing_paths.tolist()
        logger.info("Resume: loaded %d existing features from %s", len(paths), args.output)

    encoder, config = load_encoder(model_size=args.model_size)  # type: ignore[arg-type]
    device = str(config["device"])

    for path, ci in tqdm(selected, desc="Extracting"):
        if path in done_paths:
            continue
        frames = _read_video_frames(path, max_frames=args.frames_per_clip)
        if not frames:
            continue
        video_tensor = preprocess_video(frames, image_size=args.image_size, num_frames=args.frames_per_clip, device=device)
        frame_emb = extract_frame_embeddings(encoder, video_tensor, pooling="mean")
        clip_emb = frame_emb.mean(axis=0).astype(np.float32)
        features.append(clip_emb)
        labels.append(int(ci))
        paths.append(path)

        # Save periodically (every 10 clips)
        if len(features) % 10 == 0:
            _save(args.output, features, labels, paths, args.classes)

    _save(args.output, features, labels, paths, args.classes)
    print(f"Saved {len(features)} features -> {args.output}")


def _save(output: str, features: list[np.ndarray], labels: list[int], paths: list[str], class_names: list[str]) -> None:
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(features, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        paths=np.asarray(paths, dtype=object),
        class_names=np.asarray(class_names, dtype=object),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()

