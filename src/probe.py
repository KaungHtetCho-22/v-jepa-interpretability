import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

# Allow running as `python src/probe.py` (or `uv run src/probe.py`)
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
from src.model import preprocess_video

logger = logging.getLogger("vjepa.probe")


def _list_ucf_videos(video_dir: str, class_names: list[str]) -> list[tuple[str, int]]:
    """Return list of (video_path, class_idx) for a UCF-style directory layout."""
    items: list[tuple[str, int]] = []
    for ci, cname in enumerate(class_names):
        class_dir = os.path.join(video_dir, cname)
        if not os.path.isdir(class_dir):
            logger.warning("Missing class directory: %s", class_dir)
            continue
        for fn in sorted(os.listdir(class_dir)):
            if not fn.lower().endswith((".avi", ".mp4", ".mov", ".mkv", ".webm")):
                continue
            items.append((os.path.join(class_dir, fn), ci))
    return items


def _read_video_frames(path: str, max_frames: int) -> list[np.ndarray]:
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
            t = v.shape[0]
            if t == 0:
                return []
            idxs = np.linspace(0, t - 1, num=min(max_frames, t)).round().astype(int)
            frames = v[idxs].numpy()
            return [frames[i] for i in range(frames.shape[0])]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read video frames from {path}: {exc}") from exc


def extract_features_for_probe(
    encoder: nn.Module,
    video_dir: str,
    class_names: list[str],
    output_path: str,
    num_clips_per_class: int = 20,
    frames_per_clip: int = 16,
    image_size: int = 224,
    device: str = "auto",
) -> None:
    """Extract and save (N,D) clip features + (N,) labels to a .npz file."""
    from tqdm import tqdm

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    items = _list_ucf_videos(video_dir, class_names)
    if len(items) == 0:
        raise RuntimeError(f"No videos found under {video_dir} for classes {class_names}")

    # Cap per class
    per_class: dict[int, list[str]] = {i: [] for i in range(len(class_names))}
    for p, ci in items:
        if len(per_class[ci]) < num_clips_per_class:
            per_class[ci].append(p)

    selected: list[tuple[str, int]] = []
    for ci in range(len(class_names)):
        for p in per_class[ci]:
            selected.append((p, ci))

    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []

    encoder.eval()
    for path, ci in tqdm(selected, desc="Extracting features"):
        frames = _read_video_frames(path, max_frames=frames_per_clip)
        if not frames:
            continue
        video_tensor = preprocess_video(frames, image_size=image_size, num_frames=frames_per_clip, device=device)
        frame_emb = extract_frame_embeddings(encoder, video_tensor, pooling="mean")  # (T,D)
        clip_emb = frame_emb.mean(axis=0).astype(np.float32)
        features.append(clip_emb)
        labels.append(ci)
        paths.append(path)

    if not features:
        raise RuntimeError("No features extracted (no readable videos?).")

    np.savez_compressed(
        output_path,
        features=np.stack(features, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=np.asarray(class_names, dtype=object),
        paths=np.asarray(paths, dtype=object),
    )
    logger.info("Saved features: %s (N=%d, D=%d)", output_path, len(features), int(features[0].shape[0]))


def train_linear_probe(
    features_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
) -> dict[str, Any]:
    """Train a logistic regression probe from saved features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    data = np.load(features_path, allow_pickle=True)
    x = data["features"].astype(np.float32, copy=False)
    y = data["labels"].astype(np.int64, copy=False)
    class_names = data.get("class_names")
    class_names_list = class_names.tolist() if class_names is not None else [str(i) for i in sorted(set(y.tolist()))]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    clf = LogisticRegression(
        max_iter=max_iter,
        C=C,
    )
    clf.fit(x_train, y_train)
    pred_train = clf.predict(x_train)
    pred_test = clf.predict(x_test)

    train_acc = float(accuracy_score(y_train, pred_train))
    test_acc = float(accuracy_score(y_test, pred_test))

    cm = confusion_matrix(y_test, pred_test, labels=list(range(len(class_names_list))))
    per_class_acc: dict[str, float] = {}
    for ci, cname in enumerate(class_names_list):
        denom = cm[ci].sum()
        per_class_acc[str(cname)] = float(cm[ci, ci] / denom) if denom > 0 else float("nan")

    return {
        "model": clf,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "test_acc_per_class": per_class_acc,
        "confusion_matrix": cm.astype(np.int64),
        "class_names": class_names_list,
    }


def train_pixel_baseline(
    video_dir: str,
    class_names: list[str],
    num_clips_per_class: int = 20,
    resize_to: int = 32,
    frames_per_clip: int = 8,
    random_state: int = 42,
    test_size: float = 0.2,
    max_iter: int = 2000,
    C: float = 1.0,
) -> dict[str, Any]:
    """Train the same probe on tiny pixel features (flattened mean frame)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    items = _list_ucf_videos(video_dir, class_names)
    if len(items) == 0:
        raise RuntimeError(f"No videos found under {video_dir} for classes {class_names}")

    # Cap per class
    per_class: dict[int, list[str]] = {i: [] for i in range(len(class_names))}
    for p, ci in items:
        if len(per_class[ci]) < num_clips_per_class:
            per_class[ci].append(p)

    x_list: list[np.ndarray] = []
    y_list: list[int] = []

    for ci, paths in per_class.items():
        for path in paths:
            frames = _read_video_frames(path, max_frames=frames_per_clip)
            if not frames:
                continue
            # Mean frame, resized
            arr = np.stack(frames, axis=0).astype(np.float32).mean(axis=0)  # (H,W,3)
            arr = _resize_rgb(arr, resize_to, resize_to)
            x_list.append(arr.reshape(-1).astype(np.float32))
            y_list.append(ci)

    x = np.stack(x_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # PCA for tractable linear model
    pca = PCA(n_components=min(128, x_train.shape[1], x_train.shape[0] - 1), random_state=random_state)
    x_train_p = pca.fit_transform(x_train)
    x_test_p = pca.transform(x_test)

    clf = LogisticRegression(max_iter=max_iter, C=C)
    clf.fit(x_train_p, y_train)
    pred_train = clf.predict(x_train_p)
    pred_test = clf.predict(x_test_p)

    train_acc = float(accuracy_score(y_train, pred_train))
    test_acc = float(accuracy_score(y_test, pred_test))

    cm = confusion_matrix(y_test, pred_test, labels=list(range(len(class_names))))
    per_class_acc: dict[str, float] = {}
    for ci, cname in enumerate(class_names):
        denom = cm[ci].sum()
        per_class_acc[str(cname)] = float(cm[ci, ci] / denom) if denom > 0 else float("nan")

    return {
        "model": clf,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "test_acc_per_class": per_class_acc,
        "confusion_matrix": cm.astype(np.int64),
        "class_names": list(class_names),
    }


def evaluate_probe(results: dict[str, Any]) -> str:
    """Format a short results summary string."""
    class_names = results.get("class_names", [])
    test_acc = float(results.get("test_acc", float("nan")))
    train_acc = float(results.get("train_acc", float("nan")))
    per_class: dict[str, float] = results.get("test_acc_per_class", {})

    lines: list[str] = []
    lines.append("V-JEPA Linear Probe Results")
    lines.append("─────────────────────────")
    lines.append(f"Train accuracy: {train_acc*100:.1f}%")
    lines.append(f"Test accuracy:  {test_acc*100:.1f}%")
    lines.append("")
    lines.append("Per-class accuracy:")
    for cname in class_names:
        v = per_class.get(str(cname), float("nan"))
        lines.append(f"  {str(cname):<16} {v*100:5.1f}%")

    # Worst class
    if per_class:
        worst = min(per_class.items(), key=lambda kv: (np.nan_to_num(kv[1], nan=1e9)))
        lines.append("")
        lines.append(f"Worst class: {worst[0]} ({worst[1]*100:.1f}%)")

    return "\n".join(lines)


def plot_confusion_matrix(results: dict[str, Any]) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    cm = np.asarray(results["confusion_matrix"], dtype=np.float32)
    class_names = [str(x) for x in results["class_names"]]

    # Normalize rows
    denom = cm.sum(axis=1, keepdims=True) + 1e-8
    cmn = cm / denom

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cmn, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Normalized confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def save_probe(results: dict[str, Any], path: str) -> None:
    """Save a trained probe (sklearn model + metadata) with joblib."""
    import joblib

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": results["model"],
        "class_names": results.get("class_names"),
        "train_acc": results.get("train_acc"),
        "test_acc": results.get("test_acc"),
    }
    joblib.dump(payload, path)


def load_probe(path: str) -> dict[str, Any]:
    """Load a trained probe with joblib."""
    import joblib

    return joblib.load(path)


def _resize_rgb(img: np.ndarray, w: int, h: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        return img[:h, :w]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Testing without real dataset: random features should yield ~random accuracy.
    os.makedirs("data", exist_ok=True)
    test_path = "data/test_features.npz"
    if not os.path.exists(test_path):
        np.random.seed(42)
        n_per_class, n_classes, d = 20, 10, 768
        features = np.random.randn(n_per_class * n_classes, d).astype(np.float32)
        labels = np.repeat(np.arange(n_classes), n_per_class).astype(np.int64)
        np.savez(
            test_path,
            features=features,
            labels=labels,
            class_names=[f"class_{i}" for i in range(n_classes)],
        )
        logger.info("Wrote %s", test_path)

    results = train_linear_probe(test_path)
    print(evaluate_probe(results))

    try:
        fig = plot_confusion_matrix(results)
        os.makedirs("assets", exist_ok=True)
        fig.savefig("assets/probe_confusion_matrix.png", dpi=120, bbox_inches="tight")
        logger.info("Saved assets/probe_confusion_matrix.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save confusion matrix figure: %s", exc)
