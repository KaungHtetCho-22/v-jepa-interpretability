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

from src.model import load_encoder, preprocess_video
from src.retrieval import save_index
from src.embeddings import extract_frame_embeddings

logger = logging.getLogger("vjepa.build_index")


def _list_videos(video_dir: str) -> list[str]:
    exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    paths: list[str] = []
    for p in sorted(Path(video_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(str(p))
    return paths


def _read_video_frames(path: str, max_frames: int = 16) -> list[np.ndarray]:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", type=str, required=True, help="Directory containing reference clips")
    ap.add_argument("--save_path", type=str, default="data/reference_index.npz")
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls", "max"])
    ap.add_argument("--model_size", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    videos = _list_videos(args.video_dir)
    if args.dry_run:
        print(f"Found {len(videos)} video files under {args.video_dir}")
        for p in videos[:10]:
            print(" ", p)
        return

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    encoder, config = load_encoder(model_size=args.model_size)  # type: ignore[arg-type]

    embs: list[np.ndarray] = []
    paths_out: list[str] = []

    for p in tqdm(videos, desc="Indexing clips"):
        frames = _read_video_frames(p)
        if not frames:
            continue
        video_tensor = preprocess_video(frames, device=str(config["device"]))
        frame_emb = extract_frame_embeddings(encoder, video_tensor, pooling=args.pooling)  # (T,D)
        clip_emb = frame_emb.mean(axis=0).astype(np.float32)
        embs.append(clip_emb)
        paths_out.append(p)

    if not embs:
        raise SystemExit("No embeddings computed (no readable videos?)")

    index = {
        "embeddings": np.stack(embs, axis=0),
        "paths": paths_out,
        "labels": None,
        "metadata": {"model_size": args.model_size, "pooling": args.pooling},
    }
    save_index(index, args.save_path)
    print(f"Saved index: {args.save_path} (N={len(paths_out)}, D={index['embeddings'].shape[1]})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()
