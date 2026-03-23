import logging
import math
import os
import sys
from typing import Any, Literal

import numpy as np

# Allow running as `python src/attention.py` (or `uv run src/attention.py`)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is not installed. Install it with one of:\n"
        "  - `uv sync --extra torch` (default wheels)\n"
        "  - `uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision` (CPU-only)\n"
    ) from exc

logger = logging.getLogger("vjepa.attention")

META_KEY = "__vjepa_meta__"
CLS_ATTN_KEY = "__vjepa_cls_attn__"


def _resolve_layer_idx(layer_idx: int, n: int) -> int:
    idx = layer_idx if layer_idx >= 0 else (n + layer_idx)
    if idx < 0 or idx >= n:
        raise IndexError(f"layer_idx {layer_idx} out of range for {n} layers")
    return idx


def _find_qkv_modules(encoder: nn.Module) -> list[tuple[str, nn.Module, nn.Module | None]]:
    modules = dict(encoder.named_modules())
    found: list[tuple[str, nn.Module, nn.Module | None]] = []
    for name, module in modules.items():
        if name.endswith(".qkv"):
            parent = modules.get(name.rsplit(".", 1)[0])
            found.append((name, module, parent))
    return found


def _infer_num_heads(parent_attn: nn.Module | None, embed_dim: int) -> int:
    if parent_attn is not None:
        for attr in ("num_heads", "heads", "n_heads"):
            if hasattr(parent_attn, attr):
                try:
                    return int(getattr(parent_attn, attr))
                except Exception:
                    pass

    # Fallback heuristic: common head dims
    for head_dim in (64, 80, 40, 32):
        if embed_dim % head_dim == 0:
            return embed_dim // head_dim
    raise RuntimeError("Unable to infer num_heads for attention module.")


def _attn_vec_from_qkv(
    qkv: torch.Tensor,  # (B, N, 3*D)
    parent_attn: nn.Module | None,
    global_query: Literal["token0", "mean"],
    head_idx: int | None,
) -> torch.Tensor:
    """Compute a (N,) attention vector from qkv for a single global query."""
    if qkv.ndim != 3:
        raise ValueError(f"Expected qkv output (B, N, 3*D); got {tuple(qkv.shape)}")
    b, n, three_d = qkv.shape
    if b != 1:
        raise ValueError("Only batch size 1 is supported for attention visualization.")
    if three_d % 3 != 0:
        raise ValueError("qkv last dim must be divisible by 3")

    d = three_d // 3
    num_heads = _infer_num_heads(parent_attn, embed_dim=d)
    head_dim = d // num_heads
    if head_dim * num_heads != d:
        raise RuntimeError("embed_dim is not divisible by num_heads")

    q = qkv[:, :, :d].reshape(1, n, num_heads, head_dim).permute(0, 2, 1, 3)  # (1, heads, N, hd)
    k = qkv[:, :, d : 2 * d].reshape(1, n, num_heads, head_dim).permute(0, 2, 1, 3)

    if global_query == "token0":
        qg = q[:, :, 0, :]  # (1, heads, hd)
    else:
        qg = q.mean(dim=2)  # (1, heads, hd)

    scale = None
    if parent_attn is not None and hasattr(parent_attn, "scale"):
        try:
            scale = float(getattr(parent_attn, "scale"))
        except Exception:
            scale = None
    if scale is None:
        scale = 1.0 / math.sqrt(float(head_dim))

    logits = (k * qg.unsqueeze(2)).sum(dim=-1) * scale  # (1, heads, N)
    attn = torch.softmax(logits, dim=-1).squeeze(0)  # (heads, N)

    if head_idx is None:
        return attn.mean(dim=0)
    if head_idx < 0 or head_idx >= int(attn.shape[0]):
        raise IndexError(f"head_idx {head_idx} out of range for {attn.shape[0]} heads")
    return attn[head_idx]


def _storage_keys_in_order(hook_storage: dict[str, Any]) -> list[str]:
    # Preserve insertion order (matches forward execution order better than sorting).
    return [k for k in hook_storage.keys() if k != META_KEY]


def _get_layer_attention(
    hook_storage: dict[str, Any],
    layer_idx: int,
) -> torch.Tensor:
    keys = _storage_keys_in_order(hook_storage)
    if not keys:
        raise RuntimeError("hook_storage is empty; did you register hooks and run a forward pass?")

    resolved_idx = layer_idx if layer_idx >= 0 else (len(keys) + layer_idx)
    if resolved_idx < 0 or resolved_idx >= len(keys):
        raise IndexError(f"layer_idx {layer_idx} out of range for {len(keys)} hooked modules")

    key = keys[resolved_idx]
    values = hook_storage.get(key, [])
    if not values:
        raise RuntimeError(f"No attention tensors captured for layer key '{key}'.")

    # Use the last captured tensor for the most recent forward call
    attn = values[-1]
    return attn


def _has_cls_token(encoder: nn.Module | None, seq_len: int, tokens_per_frame: int | None) -> bool:
    if encoder is not None and hasattr(encoder, "cls_token"):
        return True
    # Heuristic: if seq_len-1 divides nicely but seq_len doesn't, assume CLS.
    if tokens_per_frame is None:
        return False
    return (seq_len - 1) % tokens_per_frame == 0 and seq_len % tokens_per_frame != 0


def _infer_patch_grid(
    video_tensor: torch.Tensor,
    seq_len: int,
    has_cls: bool,
) -> tuple[int, int, int]:
    """Infer (T, H_patches, W_patches) for a given attention seq_len."""
    if video_tensor.ndim != 5:
        raise ValueError("video_tensor must have shape (1, C, T, H, W)")

    t = int(video_tensor.shape[2])
    h = int(video_tensor.shape[3])
    w = int(video_tensor.shape[4])

    tokens_total = seq_len - (1 if has_cls else 0)
    if tokens_total <= 0:
        raise ValueError(f"Invalid seq_len={seq_len} for has_cls={has_cls}")

    # Heuristic: try common ViT patch sizes first
    for patch_size in (16, 14, 8):
        if h % patch_size != 0 or w % patch_size != 0:
            continue
        hp = h // patch_size
        wp = w // patch_size
        if tokens_total == t * (hp * wp):
            return t, hp, wp

    # Fallback: infer from tokens_total and T
    if tokens_total % t != 0:
        raise ValueError(f"Cannot infer per-frame tokens: tokens_total={tokens_total} not divisible by T={t}")

    per_frame = tokens_total // t
    side = int(math.isqrt(per_frame))
    if side * side != per_frame:
        raise ValueError(f"Cannot infer patch grid: per_frame_tokens={per_frame} is not a square")

    return t, side, side


def _attention_to_map(
    attn: torch.Tensor,
    encoder: nn.Module | None,
    video_tensor: torch.Tensor,
    head_idx: int | None,
    target_frame: int | None,
) -> np.ndarray:
    """Convert an attention tensor into a normalized (T,H,W) or (H,W) map."""
    # Expected shape: (B, heads, seq, seq) or (heads, seq, seq)
    if attn.dim() == 4:
        attn = attn.squeeze(0)
    if attn.dim() != 3:
        raise ValueError(f"Expected attention tensor with 3 dims after squeeze; got shape {tuple(attn.shape)}")

    heads, seq_len_q, seq_len_k = attn.shape
    if seq_len_q != seq_len_k:
        logger.warning("Attention matrix is not square: (%d, %d). Proceeding with key length.", seq_len_q, seq_len_k)

    if head_idx is None:
        attn_agg = attn.mean(dim=0)  # (seq, seq)
    else:
        if head_idx < 0 or head_idx >= heads:
            raise IndexError(f"head_idx {head_idx} out of range for {heads} heads")
        attn_agg = attn[head_idx]

    # Determine CLS handling
    # Try to infer tokens_per_frame using divisibility heuristics
    t = int(video_tensor.shape[2])
    tokens_total_if_no_cls = int(seq_len_k)
    if tokens_total_if_no_cls % t == 0:
        tokens_per_frame_guess = tokens_total_if_no_cls // t
    elif (tokens_total_if_no_cls - 1) % t == 0:
        tokens_per_frame_guess = (tokens_total_if_no_cls - 1) // t
    else:
        tokens_per_frame_guess = None
    has_cls = _has_cls_token(encoder, seq_len_k, tokens_per_frame_guess)

    start = 1 if has_cls else 0
    tokens_total = seq_len_k - start

    # Compute a single importance vector over tokens (length tokens_total)
    if has_cls:
        token_importance = attn_agg[0, start:]  # CLS query attends to all patch tokens
    else:
        # No CLS: average attention received by each token (mean over queries)
        token_importance = attn_agg[:, :].mean(dim=0)

    if token_importance.numel() != tokens_total:
        token_importance = token_importance[:tokens_total]

    t, hp, wp = _infer_patch_grid(video_tensor, seq_len_k, has_cls)
    per_frame = hp * wp
    if tokens_total != t * per_frame:
        raise ValueError(f"Token count mismatch: tokens_total={tokens_total} vs T*hp*wp={t*per_frame}")

    token_importance = token_importance.reshape(t, hp, wp)  # (T, hp, wp)

    # Upsample to original frame resolution
    h = int(video_tensor.shape[3])
    w = int(video_tensor.shape[4])
    up = F.interpolate(token_importance.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)

    # Normalize to [0,1] per-map (global across time)
    up = up - up.min()
    denom = up.max().clamp_min(1e-8)
    up = up / denom

    up_np = up.detach().float().cpu().numpy()
    if target_frame is None:
        return up_np

    if target_frame < 0 or target_frame >= up_np.shape[0]:
        raise IndexError(f"target_frame {target_frame} out of range for T={up_np.shape[0]}")
    return up_np[target_frame]


def extract_attention_maps(
    encoder: nn.Module,
    hook_storage: dict[str, Any],
    video_tensor: torch.Tensor,
    layer_idx: int = -1,
    head_idx: int | None = None,
    target_frame: int | None = None,
) -> np.ndarray:
    """Run a forward pass and extract attention maps as numpy arrays.

    Returns:
        If target_frame is None: (T, H, W) float32 in [0,1]
        Else: (H, W) float32 in [0,1]
    """
    # Clear storage for a clean capture (caller is still responsible for lifecycle)
    for k, v in list(hook_storage.items()):
        if k in (META_KEY, CLS_ATTN_KEY):
            continue
        if isinstance(v, list):
            v.clear()

    encoder.eval()

    # Preferred path: compute attention map from qkv projections (doesn't require model to return attention weights).
    qkvs = _find_qkv_modules(encoder)
    if qkvs:
        idx = _resolve_layer_idx(layer_idx, len(qkvs))
        qkv_name, qkv_mod, qkv_parent = qkvs[idx]

        captured: dict[str, torch.Tensor] = {}

        def _hook(_m: nn.Module, _inp: tuple[Any, ...], out: Any) -> None:
            if isinstance(out, torch.Tensor):
                captured["qkv"] = out.detach()

        handle = qkv_mod.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                _ = encoder(video_tensor)
        finally:
            handle.remove()

        if "qkv" not in captured:
            raise RuntimeError("Failed to capture qkv output; attention visualization is unavailable for this model.")

        qkv = captured["qkv"]
        seq_len = int(qkv.shape[1])
        t = int(video_tensor.shape[2])

        # Infer whether a global CLS exists by checking which option yields a valid patch grid.
        has_cls = False
        try:
            _t, hp, wp = _infer_patch_grid(video_tensor, seq_len, has_cls=True)
            has_cls = True
        except Exception:
            _t, hp, wp = _infer_patch_grid(video_tensor, seq_len, has_cls=False)
            has_cls = False

        # Use token0 as global query when CLS exists; otherwise use mean query token.
        global_query: Literal["token0", "mean"] = "token0" if has_cls else "mean"
        attn_vec = _attn_vec_from_qkv(qkv, qkv_parent, global_query=global_query, head_idx=head_idx)  # (N,)

        hook_storage[META_KEY] = {
            "t": _t,
            "h": int(video_tensor.shape[3]),
            "w": int(video_tensor.shape[4]),
            "hp": hp,
            "wp": wp,
            "has_cls": has_cls,
            "source": f"qkv:{qkv_name}",
        }
        hook_storage[CLS_ATTN_KEY] = [attn_vec.detach().cpu()]

        start = 1 if has_cls else 0
        token_importance = attn_vec[start:]
        token_importance = token_importance.reshape(_t, hp, wp)
        up = F.interpolate(
            token_importance.unsqueeze(1),  # treat T as batch
            size=(int(video_tensor.shape[3]), int(video_tensor.shape[4])),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        up = up - up.min()
        up = up / up.max().clamp_min(1e-8)
        up_np = up.detach().float().cpu().numpy()
        if target_frame is None:
            return up_np.astype(np.float32, copy=False)
        return up_np[int(target_frame)].astype(np.float32, copy=False)

    # Fallback path: rely on whatever hooks captured.
    with torch.no_grad():
        _ = encoder(video_tensor)

    attn = _get_layer_attention(hook_storage, layer_idx)
    attn_map = _attention_to_map(attn, encoder, video_tensor, head_idx=head_idx, target_frame=target_frame)
    return attn_map.astype(np.float32, copy=False)


def compute_attention_rollout(
    hook_storage: dict[str, Any],
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """Compute attention rollout (Abnar & Zuidema, 2020) from stored attentions.

    Note: This operates on the last captured attention tensor per hooked module.
    """
    meta = hook_storage.get(META_KEY)
    if not isinstance(meta, dict):
        raise RuntimeError(
            f"Missing {META_KEY} in hook_storage. Call extract_attention_maps(...) first so rollout can infer T/H/W."
        )

    keys = _storage_keys_in_order(hook_storage)
    if not keys:
        raise RuntimeError("hook_storage is empty; run a forward pass first.")

    attn_mats: list[torch.Tensor] = []
    for key in keys:
        vals = hook_storage.get(key, [])
        if not vals:
            continue
        attn = vals[-1]
        if attn.dim() == 4:
            attn = attn.squeeze(0)  # (heads, seq, seq)
        if attn.dim() != 3:
            continue
        attn = attn.mean(dim=0)  # (seq, seq)
        attn_mats.append(attn)

    if not attn_mats:
        raise RuntimeError("No usable attention matrices found in hook_storage.")

    # Use CPU float32 for stability + memory
    attn_mats = [a.detach().float().cpu() for a in attn_mats]
    seq_len = int(attn_mats[0].shape[-1])

    # Initialize with identity (residual connection)
    result = torch.eye(seq_len, dtype=torch.float32)

    for attn in attn_mats:
        if attn.shape[-1] != seq_len or attn.shape[-2] != seq_len:
            raise ValueError("Attention rollout requires square matrices of consistent size across layers.")

        a = attn.clone()
        if discard_ratio > 0:
            flat = a.view(-1)
            k = int(flat.numel() * discard_ratio)
            if 0 < k < flat.numel():
                threshold = torch.kthvalue(flat, k).values
                a[a < threshold] = 0.0

        a = a + torch.eye(seq_len, dtype=a.dtype)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        result = a @ result

    has_cls = bool(meta.get("has_cls", False))
    start = 1 if has_cls else 0
    t = int(meta["t"])
    hp = int(meta["hp"])
    wp = int(meta["wp"])
    h = int(meta["h"])
    w = int(meta["w"])

    token_vec = result[0, start:] if has_cls else result.mean(dim=0)
    if token_vec.numel() != t * hp * wp:
        raise ValueError("Rollout token length mismatch; ensure hooks captured the expected attention tensor.")

    token_map = token_vec.reshape(t, hp, wp)
    up = F.interpolate(token_map.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
    up = up - up.min()
    up = up / up.max().clamp_min(1e-8)
    return up.numpy()


def overlay_attention_on_frame(
    frame: np.ndarray,
    attention_map: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a heatmap (H,W) on a uint8 RGB frame (H,W,3)."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (H, W, 3)")

    attn = attention_map.astype(np.float32)
    attn = np.clip(attn, 0.0, 1.0)

    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)
    heat = cmap(attn)[..., :3]  # (H, W, 3) float
    heat_u8 = (heat * 255.0).astype(np.uint8)

    base = frame.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * heat_u8.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def visualize_attention_heads(
    hook_storage: dict[str, Any],
    video_tensor: torch.Tensor,
    frame_idx: int = 0,
    layer_idx: int = -1,
    num_heads_to_show: int = 8,
) -> np.ndarray:
    """Return a grid visualization of per-head attention maps for one frame."""
    attn = _get_layer_attention(hook_storage, layer_idx)
    if attn.dim() == 4:
        attn = attn.squeeze(0)
    if attn.dim() != 3:
        raise ValueError(f"Unexpected attention tensor shape: {tuple(attn.shape)}")

    heads = int(attn.shape[0])
    show = min(num_heads_to_show, heads)
    maps: list[np.ndarray] = []
    for h in range(show):
        m = _attention_to_map(attn, encoder=None, video_tensor=video_tensor, head_idx=h, target_frame=frame_idx)
        maps.append(m)

    # Colorize each map and pack into a grid
    colored = [overlay_attention_on_frame(np.zeros((*maps[0].shape, 3), dtype=np.uint8), m, alpha=1.0) for m in maps]
    cols = int(math.ceil(math.sqrt(show)))
    rows = int(math.ceil(show / cols))
    h, w, _ = colored[0].shape
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(colored):
        r = i // cols
        c = i % cols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    return grid


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    from src.model import load_encoder, register_attention_hooks, preprocess_video

    try:
        encoder, config = load_encoder(model_size="vit_b", device="auto", dtype="float16")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load encoder (%s: %s)", type(exc).__name__, exc)
        raise

    storage = register_attention_hooks(encoder)

    dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
    video = preprocess_video(dummy_frames, device=str(config["device"]))

    maps = extract_attention_maps(encoder, storage, video, layer_idx=-1, head_idx=None, target_frame=None)
    logger.info("Attention map shape: %s", maps.shape)
    logger.info("Value range: [%.3f, %.3f]", float(maps.min()), float(maps.max()))

    rollout = compute_attention_rollout(storage)
    logger.info("Rollout shape: %s", rollout.shape)

    # Overlay and save one example
    example = overlay_attention_on_frame(dummy_frames[0], maps[0], alpha=0.5)
    try:
        import cv2

        cv2.imwrite("assets/attention_example.png", cv2.cvtColor(example, cv2.COLOR_RGB2BGR))
        logger.info("Saved assets/attention_example.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save overlay image (cv2 missing?): %s", exc)
