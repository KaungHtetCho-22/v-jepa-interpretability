import argparse
import json
import logging
import os
import sys
from typing import Any, Literal

import numpy as np

# Allow running as `python src/retrieval.py` (or `uv run src/retrieval.py`)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is not installed. Install it with one of:\n"
        "  - `uv sync --extra torch` (default wheels)\n"
        "  - `uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision` (CPU-only)\n"
    ) from exc

from src.embeddings import extract_frame_embeddings

logger = logging.getLogger("vjepa.retrieval")

Metric = Literal["cosine", "l2"]


def save_index(index: dict[str, Any], path: str) -> None:
    """Save an index dict to a .npz file."""
    embeddings = np.asarray(index["embeddings"], dtype=np.float32)
    paths = np.asarray(index["paths"], dtype=object)
    labels = index.get("labels")
    labels_arr = np.asarray(labels, dtype=object) if labels is not None else None
    metadata = index.get("metadata", {})

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embeddings,
        paths=paths,
        labels=labels_arr,
        metadata=json.dumps(metadata),
    )


def load_index(path: str) -> dict[str, Any]:
    """Load a pre-computed index from a .npz file."""
    with np.load(path, allow_pickle=True) as data:
        embeddings = data["embeddings"].astype(np.float32, copy=False)
        paths = data["paths"].tolist()
        labels = data.get("labels")
        labels_list = labels.tolist() if labels is not None else None
        metadata_raw = data.get("metadata")
        metadata: dict[str, Any] = {}
        if metadata_raw is not None:
            try:
                metadata = json.loads(str(metadata_raw))
            except Exception:
                metadata = {}

    return {"embeddings": embeddings, "paths": paths, "labels": labels_list, "metadata": metadata}


def _video_to_clip_embedding(
    encoder: nn.Module,
    video_tensor: torch.Tensor,
    pooling: str = "mean",
) -> np.ndarray:
    """Extract a single (D,) embedding for an entire clip by pooling per-frame embeddings."""
    frame_emb = extract_frame_embeddings(encoder, video_tensor, pooling=pooling)  # (T, D)
    clip_emb = frame_emb.mean(axis=0)
    return clip_emb.astype(np.float32, copy=False)


def build_reference_index(
    encoder: nn.Module,
    video_paths: list[str],
    pooling: str = "mean",
    save_path: str | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Extract embeddings for reference videos and build a simple numpy index.

    Notes:
      - This function expects the caller to handle decoding and preprocessing if needed.
      - For now, this builds an index from paths only if you pass already-preprocessed tensors
        via an external script; see `scripts/build_index.py`.
    """
    if batch_size != 1:
        logger.warning("batch_size>1 is not supported in this lightweight builder; processing sequentially.")

    embeddings: list[np.ndarray] = []
    paths_out: list[str] = []

    from src.model import preprocess_video

    for p in video_paths:
        frames = _read_video_frames(p)
        video_tensor = preprocess_video(frames, device="auto")
        emb = _video_to_clip_embedding(encoder, video_tensor, pooling=pooling)
        embeddings.append(emb)
        paths_out.append(p)

    mat = np.stack(embeddings, axis=0).astype(np.float32)
    index: dict[str, Any] = {
        "embeddings": mat,
        "paths": paths_out,
        "labels": None,
        "metadata": {"pooling": pooling},
    }
    if save_path is not None:
        save_index(index, save_path)
    return index


def find_nearest_neighbors(
    query_embedding: np.ndarray,
    index: dict[str, Any],
    k: int = 5,
    metric: Metric = "cosine",
) -> list[dict[str, Any]]:
    """Return top-k neighbors for query_embedding against index."""
    emb = np.asarray(index["embeddings"], dtype=np.float32)
    paths: list[str] = list(index["paths"])
    labels = index.get("labels")

    q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if q.shape[0] != emb.shape[1]:
        raise ValueError(f"Query dim {q.shape[0]} != index dim {emb.shape[1]}")

    if metric == "cosine":
        qn = q / (np.linalg.norm(q) + 1e-8)
        en = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        sims = en @ qn
    elif metric == "l2":
        sims = -np.linalg.norm(emb - q[None, :], axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    k = min(k, int(emb.shape[0]))
    top_idx = np.argsort(-sims)[:k]

    results: list[dict[str, Any]] = []
    for rank, i in enumerate(top_idx, start=1):
        results.append(
            {
                "path": paths[int(i)],
                "similarity": float(sims[int(i)]),
                "rank": int(rank),
                "label": (labels[int(i)] if labels is not None else None),
            }
        )
    return results


def query_from_video(
    encoder: nn.Module,
    video_tensor: torch.Tensor,
    index: dict[str, Any],
    k: int = 5,
    pooling: str = "mean",
) -> list[dict[str, Any]]:
    """Extract embedding from video tensor and query index."""
    q = _video_to_clip_embedding(encoder, video_tensor, pooling=pooling)
    return find_nearest_neighbors(q, index=index, k=k, metric="cosine")


def compute_retrieval_precision(
    index: dict[str, Any],
    k: int = 5,
    require_labels: bool = True,
) -> float:
    """Compute leave-one-out Precision@K based on label agreement."""
    labels = index.get("labels")
    if labels is None:
        if require_labels:
            raise ValueError("index['labels'] is required to compute precision.")
        return float("nan")

    emb = np.asarray(index["embeddings"], dtype=np.float32)
    labels = list(labels)
    n = emb.shape[0]
    if n <= 1:
        return float("nan")

    correct = 0
    total = 0
    for i in range(n):
        q = emb[i]
        # Build a view without the query element
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sub = {"embeddings": emb[mask], "paths": [p for j, p in enumerate(index["paths"]) if j != i], "labels": [l for j, l in enumerate(labels) if j != i]}
        res = find_nearest_neighbors(q, sub, k=k, metric="cosine")
        gt = labels[i]
        hits = sum(1 for r in res if r.get("label") == gt)
        correct += hits
        total += len(res)

    return float(correct) / float(max(total, 1))


def render_retrieval_results(
    query_frames: list[np.ndarray],
    results: list[dict[str, Any]],
    num_preview_frames: int = 4,
) -> np.ndarray:
    """Render a simple grid image for query + k results."""
    if len(query_frames) == 0:
        raise ValueError("query_frames must be non-empty")

    # Take first frame as thumbnail
    query_thumb = query_frames[0]
    h, w, _ = query_thumb.shape

    def label_bar(text: str) -> np.ndarray:
        bar_h = 28
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        try:
            import cv2

            cv2.putText(bar, text[:32], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass
        return bar

    tiles: list[np.ndarray] = []
    tiles.append(np.concatenate([label_bar("query"), query_thumb], axis=0))

    for r in results:
        path = str(r["path"])
        sim = float(r["similarity"])
        label = r.get("label")
        title = f"{os.path.basename(path)}  sim={sim:.3f}"
        if label is not None:
            title += f"  [{label}]"

        try:
            frames = _read_video_frames(path, max_frames=num_preview_frames)
            thumb = frames[0] if frames else np.zeros_like(query_thumb)
        except Exception:
            thumb = np.zeros_like(query_thumb)

        if thumb.shape[:2] != (h, w):
            thumb = _resize_rgb(thumb, (w, h))

        tiles.append(np.concatenate([label_bar(title), thumb], axis=0))

    # Stack vertically
    grid = np.concatenate(tiles, axis=0)
    return grid


def _resize_rgb(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    try:
        import cv2

        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        # Nearest neighbor fallback
        return img[:h, :w]


def _read_video_frames(path: str, max_frames: int = 16) -> list[np.ndarray]:
    """Read up to max_frames RGB frames from a video file."""
    try:
        import decord  # type: ignore

        vr = decord.VideoReader(path)
        n = len(vr)
        if n == 0:
            return []
        idxs = np.linspace(0, n - 1, num=min(max_frames, n)).round().astype(int).tolist()
        frames = vr.get_batch(idxs).asnumpy()  # (T,H,W,3) RGB
        return [frames[i] for i in range(frames.shape[0])]
    except Exception:
        try:
            import torchvision

            v, _, _ = torchvision.io.read_video(path, pts_unit="sec")
            # v: (T, H, W, C) uint8
            t = v.shape[0]
            if t == 0:
                return []
            idxs = np.linspace(0, t - 1, num=min(max_frames, t)).round().astype(int)
            frames = v[idxs].numpy()
            return [frames[i] for i in range(frames.shape[0])]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read video frames from {path}: {exc}") from exc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build or query a simple V-JEPA retrieval index.")
    sub = p.add_subparsers(dest="cmd", required=False)

    b = sub.add_parser("build", help="Build index from a directory of clips")
    b.add_argument("--video_dir", type=str, required=True)
    b.add_argument("--save_path", type=str, default="data/reference_index.npz")
    b.add_argument("--dry_run", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Basic mechanics test with random embeddings
    fake_embeddings = np.random.randn(20, 768).astype(np.float32)
    fake_paths = [f"clip_{i:02d}.mp4" for i in range(20)]
    fake_labels = ["running"] * 5 + ["guitar"] * 5 + ["cooking"] * 5 + ["typing"] * 5

    index = {"embeddings": fake_embeddings, "paths": fake_paths, "labels": fake_labels, "metadata": {"model_size": "vit_b", "pooling": "mean"}}

    query = np.random.randn(768).astype(np.float32)
    results = find_nearest_neighbors(query, index, k=5)
    for r in results:
        print(f"Rank {r['rank']}: {r['path']} (sim={r['similarity']:.3f}, label={r['label']})")
