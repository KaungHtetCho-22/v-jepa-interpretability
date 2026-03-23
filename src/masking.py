import logging
import os
import sys
from typing import Any

import numpy as np

# Allow running as `python src/masking.py` (or `uv run src/masking.py`)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger("vjepa.masking")


def generate_tube_mask(
    num_frames: int,
    height_patches: int,
    width_patches: int,
    mask_ratio: float = 0.9,
    seed: int | None = None,
) -> np.ndarray:
    """Generate tube mask (same spatial positions masked across all frames)."""
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)

    n = height_patches * width_patches
    num_mask = int(round(mask_ratio * n))
    idx = np.arange(n)
    rng.shuffle(idx)
    masked = idx[:num_mask]

    spatial = np.zeros((height_patches, width_patches), dtype=bool)
    spatial.flat[masked] = True
    # Ensure exact spatial mask ratio (temporal_consistency equals spatial_coverage for tube masking).
    if num_mask < n:
        spatial.flat[idx[num_mask:]] = False
    mask = np.repeat(spatial[None, :, :], repeats=num_frames, axis=0)
    return mask


def generate_random_mask(
    num_frames: int,
    height_patches: int,
    width_patches: int,
    mask_ratio: float = 0.9,
    seed: int | None = None,
) -> np.ndarray:
    """Generate random per-frame masking (independent across time)."""
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)

    n = height_patches * width_patches
    num_mask = int(round(mask_ratio * n))
    mask = np.zeros((num_frames, height_patches, width_patches), dtype=bool)
    for t in range(num_frames):
        idx = np.arange(n)
        rng.shuffle(idx)
        masked = idx[:num_mask]
        mask[t].flat[masked] = True
    return mask


def generate_block_mask(
    num_frames: int,
    height_patches: int,
    width_patches: int,
    block_scale: tuple[float, float] = (0.15, 0.45),
    aspect_ratio: tuple[float, float] = (0.75, 1.5),
    num_blocks: int = 4,
    seed: int | None = None,
) -> np.ndarray:
    """Generate MAE-style block masking per frame (contiguous rectangles)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((num_frames, height_patches, width_patches), dtype=bool)

    h = height_patches
    w = width_patches
    area = h * w

    for t in range(num_frames):
        for _ in range(num_blocks):
            scale = float(rng.uniform(block_scale[0], block_scale[1]))
            target_area = max(1, int(round(scale * area)))
            ar = float(rng.uniform(aspect_ratio[0], aspect_ratio[1]))

            block_h = int(round(np.sqrt(target_area * ar)))
            block_w = int(round(np.sqrt(target_area / ar)))
            block_h = max(1, min(block_h, h))
            block_w = max(1, min(block_w, w))

            top = int(rng.integers(0, max(1, h - block_h + 1)))
            left = int(rng.integers(0, max(1, w - block_w + 1)))
            mask[t, top : top + block_h, left : left + block_w] = True

    return mask


def visualize_masking_on_frames(
    frames: list[np.ndarray],
    mask: np.ndarray,
    patch_size: int = 16,
    mask_color: tuple[int, int, int] = (128, 128, 128),
    alpha: float = 0.85,
) -> list[np.ndarray]:
    """Apply a patch mask to frames and return masked-frame list."""
    if len(frames) == 0:
        raise ValueError("frames must be non-empty")
    if mask.ndim != 3:
        raise ValueError("mask must have shape (T, H_patches, W_patches)")

    t = min(len(frames), int(mask.shape[0]))
    out: list[np.ndarray] = []

    for i in range(t):
        frame = frames[i].copy()
        h, w, c = frame.shape
        hp, wp = int(mask.shape[1]), int(mask.shape[2])
        expected_h = hp * patch_size
        expected_w = wp * patch_size
        if h != expected_h or w != expected_w:
            # Simple resize fallback to match patch grid
            frame = _resize_rgb(frame, (expected_w, expected_h))
            h, w, _ = frame.shape

        overlay = frame.astype(np.float32)
        color = np.array(mask_color, dtype=np.float32).reshape(1, 1, 3)

        for y in range(hp):
            for x in range(wp):
                if not bool(mask[i, y, x]):
                    continue
                y0, y1 = y * patch_size, (y + 1) * patch_size
                x0, x1 = x * patch_size, (x + 1) * patch_size
                overlay[y0:y1, x0:x1] = (1.0 - alpha) * overlay[y0:y1, x0:x1] + alpha * color

        out.append(np.clip(overlay, 0, 255).astype(np.uint8))

    return out


def compare_masking_strategies(
    frames: list[np.ndarray],
    patch_size: int = 16,
    mask_ratio: float = 0.9,
    seed: int = 42,
) -> np.ndarray:
    """Create a 3-row grid: original, tube, random (4 frames each)."""
    if len(frames) == 0:
        raise ValueError("frames must be non-empty")

    t = len(frames)
    # Sample 4 evenly spaced frames
    idxs = np.linspace(0, t - 1, num=min(4, t)).round().astype(int).tolist()
    sampled = [frames[i] for i in idxs]

    h, w, _ = sampled[0].shape
    hp = h // patch_size
    wp = w // patch_size

    tube = generate_tube_mask(t, hp, wp, mask_ratio=mask_ratio, seed=seed)
    rand = generate_random_mask(t, hp, wp, mask_ratio=mask_ratio, seed=seed)

    tube_vis = visualize_masking_on_frames(frames, tube, patch_size=patch_size)
    rand_vis = visualize_masking_on_frames(frames, rand, patch_size=patch_size)

    tube_sampled = [tube_vis[i] for i in idxs]
    rand_sampled = [rand_vis[i] for i in idxs]

    thumb_w = min(112, w)
    thumb_h = min(112, h)

    def thumbs(row: list[np.ndarray]) -> np.ndarray:
        ts = [_resize_rgb(im, (thumb_w, thumb_h)) for im in row]
        return np.concatenate(ts, axis=1)

    row0 = thumbs(sampled)
    row1 = thumbs(tube_sampled)
    row2 = thumbs(rand_sampled)

    # Add small labels on the left
    row0 = _add_row_label(row0, "original")
    row1 = _add_row_label(row1, "tube")
    row2 = _add_row_label(row2, "random")

    grid = np.concatenate([row0, row1, row2], axis=0)
    return grid


def compute_mask_stats(mask: np.ndarray) -> dict[str, float]:
    """Compute mask statistics described in the spec."""
    if mask.ndim != 3:
        raise ValueError("mask must have shape (T, H_patches, W_patches)")

    t, hp, wp = mask.shape
    total = float(t * hp * wp)
    mask_ratio = float(mask.sum()) / max(total, 1.0)

    # Temporal consistency: fraction of spatial positions masked in ALL frames
    all_masked = mask.all(axis=0)
    temporal_consistency = float(all_masked.mean())

    # Spatial coverage: fraction of unique spatial positions masked at least once
    any_masked = mask.any(axis=0)
    spatial_coverage = float(any_masked.mean())

    return {
        "mask_ratio": mask_ratio,
        "temporal_consistency": temporal_consistency,
        "spatial_coverage": spatial_coverage,
    }


def _resize_rgb(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    try:
        import cv2

        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        return img[:h, :w]


def _add_row_label(row: np.ndarray, label: str) -> np.ndarray:
    """Prepend a label bar to a row image."""
    h, w, _ = row.shape
    bar_w = 92
    bar = np.zeros((h, bar_w, 3), dtype=np.uint8)
    try:
        import cv2

        cv2.putText(bar, label, (6, min(24, h - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    except Exception:
        pass
    return np.concatenate([bar, row], axis=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
    tube = generate_tube_mask(16, 14, 14, mask_ratio=0.9, seed=42)
    rand = generate_random_mask(16, 14, 14, mask_ratio=0.9, seed=42)

    tube_stats = compute_mask_stats(tube)
    rand_stats = compute_mask_stats(rand)

    logger.info("Tube mask  — temporal_consistency: %.3f", tube_stats["temporal_consistency"])
    logger.info("Random mask — temporal_consistency: %.3f", rand_stats["temporal_consistency"])

    comparison = compare_masking_strategies(frames, patch_size=16, mask_ratio=0.9, seed=42)
    try:
        from PIL import Image

        Image.fromarray(comparison).save("assets/masking_comparison.png")
        logger.info("Saved assets/masking_comparison.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save assets/masking_comparison.png: %s", exc)
