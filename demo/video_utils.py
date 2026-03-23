import os
import sys
from collections import OrderedDict
from typing import Any

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


_FRAME_CACHE: "OrderedDict[str, list[np.ndarray]]" = OrderedDict()
_FRAME_CACHE_MAX = 4


def _uniform_indices(n: int, max_frames: int) -> list[int]:
    if n <= 0:
        return []
    if n <= max_frames:
        return list(range(n))
    idxs = np.linspace(0, n - 1, num=max_frames).round().astype(int).tolist()
    return idxs


def _resize_rgb(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    try:
        import cv2

        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        return frame[:size, :size]


def load_video_frames(
    video_path: str,
    max_frames: int = 16,
    resize: int = 224,
) -> list[np.ndarray]:
    """Load a video and return list of (H,W,3) RGB uint8 frames."""
    if not video_path:
        raise ValueError("video_path is empty")

    # Simple cache by absolute path
    key = os.path.abspath(video_path)
    cached = _FRAME_CACHE.get(key)
    if cached is not None:
        _FRAME_CACHE.move_to_end(key)
        return cached

    frames: list[np.ndarray]
    try:
        import decord  # type: ignore

        vr = decord.VideoReader(video_path)
        n = len(vr)
        idxs = _uniform_indices(n, max_frames)
        if not idxs:
            frames = []
        else:
            batch = vr.get_batch(idxs).asnumpy()  # (T,H,W,3) RGB
            frames = [batch[i] for i in range(batch.shape[0])]
    except Exception:
        try:
            import torchvision

            v, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
            n = int(v.shape[0])
            idxs = _uniform_indices(n, max_frames)
            frames = [v[i].numpy() for i in idxs]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to decode video: {exc}") from exc

    frames = [_resize_rgb(f.astype(np.uint8, copy=False), resize) for f in frames]

    _FRAME_CACHE[key] = frames
    _FRAME_CACHE.move_to_end(key)
    while len(_FRAME_CACHE) > _FRAME_CACHE_MAX:
        _FRAME_CACHE.popitem(last=False)

    return frames


def coerce_video_path(video_input: Any) -> str:
    """Gradio video can come through as a path string or a dict-like object."""
    if video_input is None:
        return ""
    if isinstance(video_input, str):
        return video_input
    # Gradio >=5 may pass a FileData-like object
    for attr in ("path", "name"):
        try:
            v = getattr(video_input, attr)
        except Exception:
            v = None
        if isinstance(v, str) and v:
            return v
    if isinstance(video_input, dict):
        for key in ("path", "name", "video", "file"):
            if key in video_input and video_input[key]:
                return str(video_input[key])
    if isinstance(video_input, (tuple, list)) and video_input:
        # Sometimes (path, metadata)
        if isinstance(video_input[0], str):
            return video_input[0]
    # Last resort: try to stringify, but avoid returning "FileData(...)" for empty objects.
    try:
        s = str(video_input)
    except Exception:
        s = ""
    return s if os.path.exists(s) else ""
