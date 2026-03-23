import logging
import os
import sys
from typing import Any

# Allow running as `python demo/app.py` (or `uv run demo/app.py`)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gradio as gr
import numpy as np

from demo.video_utils import coerce_video_path, load_video_frames
from src.attention import compute_attention_rollout, extract_attention_maps, overlay_attention_on_frame
from src.embeddings import (
    compute_consecutive_drift,
    compute_pca_trajectory,
    compute_temporal_similarity,
    extract_frame_embeddings,
    plot_drift_curve,
    plot_pca_trajectory,
    plot_similarity_heatmap,
)
from src.masking import (
    compute_mask_stats,
    generate_block_mask,
    generate_random_mask,
    generate_tube_mask,
    visualize_masking_on_frames,
)
from src.model import load_encoder, preprocess_video, register_attention_hooks
from src.retrieval import find_nearest_neighbors, load_index

logger = logging.getLogger("vjepa.demo")


ENCODER: Any = None
ENCODER_CONFIG: dict[str, Any] | None = None
HOOK_STORAGE: dict[str, Any] | None = None

REFERENCE_INDEX: dict[str, Any] | None = None
REFERENCE_INDEX_PATH = os.environ.get("VJEPA_REFERENCE_INDEX", "data/reference_index.npz")


def get_encoder() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    global ENCODER, ENCODER_CONFIG, HOOK_STORAGE
    if ENCODER is None:
        ENCODER, ENCODER_CONFIG = load_encoder(model_size="vit_b", device="auto", dtype="float16")
        HOOK_STORAGE = register_attention_hooks(ENCODER)
    assert ENCODER_CONFIG is not None
    assert HOOK_STORAGE is not None
    return ENCODER, ENCODER_CONFIG, HOOK_STORAGE


def get_reference_index() -> dict[str, Any] | None:
    global REFERENCE_INDEX
    if REFERENCE_INDEX is not None:
        return REFERENCE_INDEX
    if os.path.exists(REFERENCE_INDEX_PATH):
        REFERENCE_INDEX = load_index(REFERENCE_INDEX_PATH)
        return REFERENCE_INDEX
    return None


def _head_options() -> list[str]:
    return ["Average heads"] + [f"Head {i}" for i in range(8)]


def _safe_warning(msg: str) -> None:
    try:
        gr.Warning(msg)
    except Exception:
        logger.warning(msg)

def _layer_slider_update(layer_idx: int, cfg: dict[str, Any]) -> tuple[int, Any]:
    """Clamp layer_idx to model depth and return a gr.update() for the slider."""
    n = cfg.get("num_layers")
    if isinstance(n, int) and n > 0:
        idx = layer_idx if layer_idx >= 0 else layer_idx
        if idx >= n:
            idx = n - 1
        if idx < 0 and (-idx) > n:
            idx = -1
        # Always keep the slider's max in sync once we know the true depth.
        return idx, gr.update(maximum=n - 1, value=idx if idx >= 0 else (n - 1))
    return layer_idx, gr.update()


def run_attention_tab(
    video_input: Any,
    layer_idx: int,
    frame_idx: int,
    head_selection: str,
    use_rollout: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    try:
        path = coerce_video_path(video_input)
        if not path:
            _safe_warning("Please upload a video.")
            return None, None, None

        enc, cfg, storage = get_encoder()
        layer_idx, layer_update = _layer_slider_update(int(layer_idx), cfg)
        frames = load_video_frames(path, max_frames=16, resize=224)
        if len(frames) == 0:
            _safe_warning("Could not decode video.")
            return None, None, None

        frame_idx = int(np.clip(frame_idx, 0, len(frames) - 1))
        video = preprocess_video(frames, image_size=224, num_frames=16, device=str(cfg["device"]))

        if use_rollout:
            with np.errstate(all="ignore"):
                _ = extract_attention_maps(enc, storage, video, layer_idx=layer_idx, head_idx=None, target_frame=None)
            maps = compute_attention_rollout(storage)
        else:
            head_idx = None
            if head_selection != "Average heads":
                head_idx = int(head_selection.split()[-1])
            maps = extract_attention_maps(enc, storage, video, layer_idx=layer_idx, head_idx=head_idx, target_frame=None)

        frame = frames[frame_idx]
        overlay = overlay_attention_on_frame(frame, maps[frame_idx], colormap="viridis", alpha=0.5)

        # Head grid: compute first 8 heads if possible
        head_imgs: list[np.ndarray] = []
        for h in range(8):
            try:
                m = extract_attention_maps(enc, storage, video, layer_idx=layer_idx, head_idx=h, target_frame=frame_idx)
                head_imgs.append(overlay_attention_on_frame(frame, m, colormap="viridis", alpha=0.5))
            except Exception:
                break

        if head_imgs:
            cols = 4
            rows = int(np.ceil(len(head_imgs) / cols))
            h0, w0, _ = head_imgs[0].shape
            grid = np.zeros((rows * h0, cols * w0, 3), dtype=np.uint8)
            for i, img in enumerate(head_imgs):
                r = i // cols
                c = i % cols
                grid[r * h0 : (r + 1) * h0, c * w0 : (c + 1) * w0] = img
        else:
            grid = None

        return frame, overlay, grid, layer_update
    except Exception as exc:  # noqa: BLE001
        logger.exception("Attention tab failed")
        _safe_warning(f"Error: {type(exc).__name__}: {exc}")
        return None, None, None, gr.update()


def run_temporal_tab(
    video_input: Any,
    pooling: str,
    layer_idx: int,
) -> tuple[Any, Any, Any]:
    try:
        path = coerce_video_path(video_input)
        if not path:
            _safe_warning("Please upload a video.")
            return None, None, None

        enc, cfg, _ = get_encoder()
        layer_idx, layer_update = _layer_slider_update(int(layer_idx), cfg)
        frames = load_video_frames(path, max_frames=16, resize=224)
        if len(frames) == 0:
            _safe_warning("Could not decode video.")
            return None, None, None

        video = preprocess_video(frames, image_size=224, num_frames=16, device=str(cfg["device"]))
        pool = "mean" if pooling.startswith("Mean") else "max"
        emb = extract_frame_embeddings(enc, video, layer_idx=int(layer_idx), pooling=pool)  # (T,D)

        sim = compute_temporal_similarity(emb, metric="cosine")
        drift = compute_consecutive_drift(emb)
        traj = compute_pca_trajectory(emb, n_components=2)

        return plot_similarity_heatmap(sim), plot_drift_curve(drift), plot_pca_trajectory(traj), layer_update
    except Exception as exc:  # noqa: BLE001
        logger.exception("Temporal tab failed")
        _safe_warning(f"Error: {type(exc).__name__}: {exc}")
        return None, None, None, gr.update()


def run_retrieval_tab(video_input: Any, k: int) -> tuple[np.ndarray | None, list[list[Any]] | None]:
    try:
        path = coerce_video_path(video_input)
        if not path:
            _safe_warning("Please upload a video.")
            return None, None

        index = get_reference_index()
        if index is None:
            _safe_warning("Reference index not found. Run `python scripts/build_index.py` to create data/reference_index.npz.")
            return None, None

        enc, cfg, _ = get_encoder()
        frames = load_video_frames(path, max_frames=16, resize=224)
        video = preprocess_video(frames, image_size=224, num_frames=16, device=str(cfg["device"]))

        pool = str(index.get("metadata", {}).get("pooling", "mean"))
        q_frames = extract_frame_embeddings(enc, video, pooling=pool)  # (T,D)
        q = q_frames.mean(axis=0).astype(np.float32)
        results = find_nearest_neighbors(q, index=index, k=int(k), metric="cosine")

        # Simple render: first frame query + neighbor thumbs
        from src.retrieval import render_retrieval_results

        grid = render_retrieval_results(frames, results, num_preview_frames=4)
        table = [[r["rank"], os.path.basename(r["path"]), r.get("label"), float(r["similarity"])] for r in results]
        return grid, table
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval tab failed")
        _safe_warning(f"Error: {type(exc).__name__}: {exc}")
        return None, None


def run_masking_tab(
    video_input: Any,
    mask_ratio: float,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray | None, str]:
    try:
        path = coerce_video_path(video_input)
        if not path:
            _safe_warning("Please upload a video.")
            return None, ""

        frames = load_video_frames(path, max_frames=16, resize=224)
        if len(frames) == 0:
            _safe_warning("Could not decode video.")
            return None, ""

        patch_size = 16
        hp = 224 // patch_size
        wp = 224 // patch_size
        t = len(frames)

        if strategy == "Tube masking":
            mask = generate_tube_mask(t, hp, wp, mask_ratio=float(mask_ratio), seed=int(seed))
        elif strategy == "Random masking":
            mask = generate_random_mask(t, hp, wp, mask_ratio=float(mask_ratio), seed=int(seed))
        else:
            mask = generate_block_mask(t, hp, wp, seed=int(seed))

        masked = visualize_masking_on_frames(frames, mask, patch_size=patch_size, alpha=0.85)

        # 2x4 grid: originals on top, masked on bottom
        idxs = np.linspace(0, t - 1, num=min(4, t)).round().astype(int).tolist()
        top = np.concatenate([frames[i] for i in idxs], axis=1)
        bot = np.concatenate([masked[i] for i in idxs], axis=1)
        grid = np.concatenate([top, bot], axis=0)

        stats = compute_mask_stats(mask)
        md = (
            f"**Mask ratio:** {stats['mask_ratio']:.1%}  \n"
            f"**Temporal consistency:** {stats['temporal_consistency']:.1%}  \n"
            f"**Spatial coverage:** {stats['spatial_coverage']:.1%}"
        )
        return grid, md
    except Exception as exc:  # noqa: BLE001
        logger.exception("Masking tab failed")
        _safe_warning(f"Error: {type(exc).__name__}: {exc}")
        return None, ""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="What Does a Video World Model Actually Learn?",
    ) as demo:
        gr.Markdown(
            """
# What Does a Video World Model Actually Learn?
Probing **V-JEPA 2** internal representations through attention maps, embedding trajectories, and latent-space retrieval.

**Model:** V-JEPA 2 ViT-B (frozen) · **Paper:** arXiv:2404.08471 · **Demo:** 4 tabs below
""".strip()
        )

        with gr.Tabs():
            with gr.Tab("Attention Explorer"):
                with gr.Row():
                    video = gr.Video(label="Upload video clip", sources=["upload"])
                with gr.Row():
                    layer = gr.Slider(0, 31, value=31, step=1, label="Encoder layer")
                    frame = gr.Slider(0, 15, value=0, step=1, label="Frame index")
                with gr.Row():
                    head = gr.Dropdown(_head_options(), value="Average heads", label="Attention head")
                    rollout = gr.Checkbox(label="Use attention rollout", value=False)
                    btn = gr.Button("Extract attention", variant="primary")
                with gr.Row():
                    out_frame = gr.Image(label="Original frame")
                    out_overlay = gr.Image(label="Attention overlay")
                out_grid = gr.Image(label="Head grid (first 8 heads)")
                btn.click(
                    run_attention_tab,
                    inputs=[video, layer, frame, head, rollout],
                    outputs=[out_frame, out_overlay, out_grid, layer],
                )

            with gr.Tab("Temporal Drift"):
                video = gr.Video(label="Upload video clip", sources=["upload"])
                with gr.Row():
                    pooling = gr.Radio(["Mean pooling", "Max pooling"], value="Mean pooling", label="Frame pooling")
                    layer = gr.Slider(0, 31, value=31, step=1, label="Encoder layer")
                    btn = gr.Button("Analyze", variant="primary")
                with gr.Row():
                    out_sim = gr.Plot(label="Frame similarity heatmap")
                    out_drift = gr.Plot(label="Consecutive drift")
                out_pca = gr.Plot(label="PCA trajectory")
                btn.click(
                    run_temporal_tab,
                    inputs=[video, pooling, layer],
                    outputs=[out_sim, out_drift, out_pca, layer],
                )

            with gr.Tab("Nearest Neighbors"):
                video = gr.Video(label="Upload query video", sources=["upload"])
                with gr.Row():
                    k = gr.Slider(1, 10, value=5, step=1, label="Number of results")
                    btn = gr.Button("Find similar clips", variant="primary")
                out_grid = gr.Image(label="Query + top-K results")
                out_table = gr.Dataframe(label="Similarity scores", headers=["Rank", "Clip", "Label", "Similarity"], interactive=False)
                btn.click(run_retrieval_tab, inputs=[video, k], outputs=[out_grid, out_table])

                idx = get_reference_index()
                if idx is None:
                    gr.Markdown(
                        f"Reference index not found at `{REFERENCE_INDEX_PATH}`. "
                        "Build one with `python scripts/build_index.py`."
                    )

            with gr.Tab("Masking Strategies"):
                video = gr.Video(label="Upload video clip", sources=["upload"])
                with gr.Row():
                    ratio = gr.Slider(0.5, 0.95, value=0.9, step=0.05, label="Mask ratio")
                    strat = gr.Radio(["Tube masking", "Random masking", "Block masking"], value="Tube masking", label="Strategy")
                    seed = gr.Slider(0, 100, value=42, step=1, label="Random seed")
                    btn = gr.Button("Visualize", variant="primary")
                out_img = gr.Image(label="Visualization (original on top, masked on bottom)")
                out_md = gr.Markdown()
                btn.click(run_masking_tab, inputs=[video, ratio, strat, seed], outputs=[out_img, out_md])

        gr.Markdown("Tip: On CPU-only machines, use short clips; feature extraction can be slow.")

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    app = build_demo()
    app.launch(share=False, theme=gr.themes.Soft(), show_error=True)
