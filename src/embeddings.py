import logging
import math
import os
import sys
from typing import Any, Literal

import numpy as np

# Allow running as `python src/embeddings.py` (or `uv run src/embeddings.py`)
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

logger = logging.getLogger("vjepa.embeddings")

Pooling = Literal["mean", "cls", "max"]
Metric = Literal["cosine", "l2", "dot"]


def _coerce_output_to_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported model output type: {type(output)}")


def _split_tokens_by_frame(
    tokens: torch.Tensor,  # (B, seq, D)
    t: int,
) -> tuple[torch.Tensor, bool]:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens shape (B, seq, D); got {tuple(tokens.shape)}")

    seq_len = int(tokens.shape[1])
    if seq_len == t:
        # Already frame-wise: (B, T, D). Treat as one token per frame.
        return tokens.unsqueeze(2), False

    has_global_cls = False
    if seq_len % t == 0:
        per_frame = seq_len // t
        start = 0
    elif (seq_len - 1) % t == 0:
        has_global_cls = True
        per_frame = (seq_len - 1) // t
        start = 1
    else:
        raise ValueError(f"Cannot split seq_len={seq_len} tokens into T={t} frames (with/without global CLS).")

    frame_tokens = tokens[:, start : start + t * per_frame, :]
    frame_tokens = frame_tokens.reshape(tokens.shape[0], t, per_frame, tokens.shape[2])
    return frame_tokens, has_global_cls


def _pool_frame_tokens(frame_tokens: torch.Tensor, pooling: Pooling) -> torch.Tensor:
    # frame_tokens: (B, T, N, D)
    if frame_tokens.ndim != 4:
        raise ValueError(f"Expected frame_tokens (B,T,N,D); got {tuple(frame_tokens.shape)}")

    b, t, n, d = frame_tokens.shape

    # Detect per-frame CLS token: N = 1 + square(P)
    per_frame_has_cls = False
    if n > 1:
        side = int(math.isqrt(n - 1))
        per_frame_has_cls = side * side == (n - 1)

    if pooling == "cls":
        if per_frame_has_cls:
            return frame_tokens[:, :, 0, :]
        logger.warning("Pooling='cls' requested, but no per-frame CLS token detected; falling back to mean.")
        pooling = "mean"

    if per_frame_has_cls:
        patch_tokens = frame_tokens[:, :, 1:, :]
    else:
        patch_tokens = frame_tokens

    if pooling == "mean":
        return patch_tokens.mean(dim=2)
    if pooling == "max":
        return patch_tokens.max(dim=2).values
    raise ValueError(f"Unknown pooling strategy: {pooling}")


def extract_frame_embeddings(
    encoder: nn.Module,
    video_tensor: torch.Tensor,
    layer_idx: int = -1,
    pooling: Pooling = "mean",
) -> np.ndarray:
    """Extract per-frame embeddings as a (T, D) numpy array."""
    encoder.eval()
    t = int(video_tensor.shape[2])

    with torch.no_grad():
        if layer_idx == -1:
            out = encoder(video_tensor)
            tokens = _coerce_output_to_tensor(out)
        else:
            # Prefer HF-style hidden states if available via the adapter (Session 01)
            try:
                out = encoder(video_tensor, output_hidden_states=True, return_dict=True)  # type: ignore[call-arg]
            except TypeError as exc:
                raise RuntimeError(
                    "This encoder does not support `output_hidden_states=True`; "
                    "layer selection is only supported for HuggingFace-loaded models."
                ) from exc
            if not hasattr(out, "hidden_states") or out.hidden_states is None:
                raise RuntimeError("Model did not return hidden_states; cannot select a non-final layer output.")
            hidden_states = list(out.hidden_states)
            n_layers = len(hidden_states)
            idx = layer_idx if layer_idx >= 0 else (n_layers + layer_idx)
            if idx < 0:
                idx = 0
            if idx >= n_layers:
                idx = n_layers - 1
            selected = hidden_states[idx]
            if not isinstance(selected, torch.Tensor):
                raise RuntimeError("Selected hidden state is not a Tensor.")
            tokens = selected

    # tokens: (B, seq, D)
    frame_tokens, _has_global_cls = _split_tokens_by_frame(tokens, t=t)
    pooled = _pool_frame_tokens(frame_tokens, pooling=pooling)  # (B, T, D)
    pooled = pooled.squeeze(0)
    return pooled.detach().float().cpu().numpy()


def compute_temporal_similarity(embeddings: np.ndarray, metric: Metric = "cosine") -> np.ndarray:
    """Compute a (T,T) similarity matrix from (T,D) embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (T, D)")

    x = embeddings.astype(np.float32, copy=False)
    if metric == "dot":
        return (x @ x.T).astype(np.float32)

    if metric == "l2":
        # Similarity as negative Euclidean distance (higher is more similar).
        sq = np.sum(x**2, axis=1, keepdims=True)
        dist2 = sq + sq.T - 2.0 * (x @ x.T)
        dist2 = np.maximum(dist2, 0.0)
        dist = np.sqrt(dist2, dtype=np.float32)
        return (-dist).astype(np.float32)

    if metric == "cosine":
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        xn = x / denom
        return (xn @ xn.T).astype(np.float32)

    raise ValueError(f"Unknown metric: {metric}")


def compute_consecutive_drift(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine distance between consecutive frames (T-1,)."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (T, D)")
    x = embeddings.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    xn = x / denom
    sim = np.sum(xn[:-1] * xn[1:], axis=1)
    drift = 1.0 - sim
    return drift.astype(np.float32)


def extract_layerwise_embeddings(
    encoder: nn.Module,
    video_tensor: torch.Tensor,
    layers: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Extract per-frame embeddings for multiple layers."""
    encoder.eval()
    t = int(video_tensor.shape[2])

    with torch.no_grad():
        try:
            out = encoder(video_tensor, output_hidden_states=True, return_dict=True)  # type: ignore[call-arg]
        except TypeError as exc:
            raise RuntimeError(
                "This encoder does not support `output_hidden_states=True`; "
                "layerwise embeddings are only supported for HuggingFace-loaded models."
            ) from exc
        if not hasattr(out, "hidden_states") or out.hidden_states is None:
            raise RuntimeError("Model did not return hidden_states; cannot extract layerwise embeddings.")
        hidden_states = list(out.hidden_states)

    if layers is None:
        layers = list(range(len(hidden_states)))

    result: dict[int, np.ndarray] = {}
    for li in layers:
        tokens = hidden_states[li]
        if not isinstance(tokens, torch.Tensor):
            continue
        frame_tokens, _ = _split_tokens_by_frame(tokens, t=t)
        pooled = _pool_frame_tokens(frame_tokens, pooling="mean").squeeze(0)  # (T, D)
        result[int(li)] = pooled.detach().float().cpu().numpy()
    return result


def compute_pca_trajectory(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute PCA projection (T, n_components) from (T, D) embeddings."""
    from sklearn.decomposition import PCA

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (T, D)")

    pca = PCA(n_components=n_components)
    traj = pca.fit_transform(embeddings.astype(np.float32, copy=False))
    return traj.astype(np.float32)


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    title: str = "Frame-to-frame representation similarity",
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(similarity_matrix, cmap="viridis", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Frame index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_drift_curve(
    drift: np.ndarray,
    frame_indices: list[int] | None = None,
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(drift)) if frame_indices is None else np.asarray(frame_indices)
    ax.plot(x, drift, linewidth=2)
    ax.set_xlabel("Frame index (t -> t+1)")
    ax.set_ylabel("Cosine distance")
    ax.set_title("Consecutive-frame embedding drift")
    fig.tight_layout()
    return fig


def plot_pca_trajectory(
    trajectory: np.ndarray,
    color_by_time: bool = True,
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 6))

    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("trajectory must have shape (T, 2)")

    if color_by_time:
        t = trajectory.shape[0]
        colors = np.linspace(0.0, 1.0, t)
        sc = ax.scatter(trajectory[:, 0], trajectory[:, 1], c=colors, cmap="coolwarm", s=40)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Time")
    else:
        ax.scatter(trajectory[:, 0], trajectory[:, 1], s=40)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA trajectory through representation space")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    from src.model import load_encoder, preprocess_video

    encoder, config = load_encoder("vit_b")
    dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
    video_tensor = preprocess_video(dummy_frames, device=str(config["device"]))

    emb = extract_frame_embeddings(encoder, video_tensor, pooling="mean")
    logger.info("Embeddings shape: %s", emb.shape)

    sim = compute_temporal_similarity(emb, metric="cosine")
    logger.info("Similarity matrix: min=%.3f max=%.3f", float(sim.min()), float(sim.max()))

    drift = compute_consecutive_drift(emb)
    logger.info("Drift curve (first 8): %s", np.round(drift[:8], 3))

    traj = compute_pca_trajectory(emb, n_components=2)
    logger.info("PCA trajectory shape: %s", traj.shape)

    fig = plot_similarity_heatmap(sim)
    fig.savefig("assets/similarity_heatmap.png", dpi=100, bbox_inches="tight")
    logger.info("Saved assets/similarity_heatmap.png")
