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
SALIENCY_KEY = "__vjepa_saliency__"


def _resolve_layer_idx(layer_idx: int, n: int) -> int:
    idx = layer_idx if layer_idx >= 0 else (n + layer_idx)
    if idx < 0 or idx >= n:
        raise IndexError(f"layer_idx {layer_idx} out of range for {n} layers")
    return idx


def _find_qkv_modules(encoder: nn.Module) -> list[tuple[str, nn.Module, nn.Module | None]]:
    modules = dict(encoder.named_modules())
    found: list[tuple[str, nn.Module, nn.Module | None]] = []
    for name, module in modules.items():
        # timm-style ViT uses `...attn.qkv`; other variants may use plain `qkv`
        if name.endswith(".qkv") or name.endswith("qkv"):
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


def _coerce_attention_tensor(attn_layer: Any) -> torch.Tensor | None:
    """Best-effort extraction of an attention tensor from a layer attention object."""
    if isinstance(attn_layer, torch.Tensor):
        return attn_layer
    if isinstance(attn_layer, (tuple, list)):
        for item in attn_layer:
            if isinstance(item, torch.Tensor):
                return item
            if isinstance(item, (tuple, list, dict)):
                coerced = _coerce_attention_tensor(item)
                if coerced is not None:
                    return coerced
        return None
    if isinstance(attn_layer, dict):
        for key in ("attn", "attn_weights", "attention", "attentions"):
            val = attn_layer.get(key)
            if isinstance(val, torch.Tensor):
                return val
        return None
    return None


def _coerce_output_to_tokens(out: Any) -> torch.Tensor | None:
    """Extract (B, seq, D) tokens from a HF output object/dict/tuple."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "last_hidden_state") and isinstance(out.last_hidden_state, torch.Tensor):
        return out.last_hidden_state
    if isinstance(out, dict):
        v = out.get("last_hidden_state")
        if isinstance(v, torch.Tensor):
            return v
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    return None


def _tokens_to_saliency_map(video_tensor: torch.Tensor, tokens: torch.Tensor) -> np.ndarray:
    """Fallback: derive a per-patch saliency map from token embedding norms.

    Returns (T, H, W) map in [0,1].
    """
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens shape (B, seq, D); got {tuple(tokens.shape)}")
    if tokens.shape[0] != 1:
        raise ValueError("Only batch size 1 supported.")

    seq_len = int(tokens.shape[1])
    t_in = int(video_tensor.shape[2])

    # Infer CLS by whichever option yields a valid patch grid.
    has_cls = False
    try:
        _t, hp, wp = _infer_patch_grid(video_tensor, seq_len, has_cls=True)
        has_cls = True
    except Exception:
        _t, hp, wp = _infer_patch_grid(video_tensor, seq_len, has_cls=False)
        has_cls = False

    start = 1 if has_cls else 0
    patch_tokens = tokens[0, start:, :]  # (T*hp*wp, D)
    sal = patch_tokens.float().pow(2).sum(dim=-1).sqrt()  # L2 norm (T*hp*wp,)
    sal = sal.reshape(_t, hp, wp)

    h = int(video_tensor.shape[3])
    w = int(video_tensor.shape[4])
    up = F.interpolate(sal.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
    up = _time_resample(up, t_in)
    up = up - up.min()
    up = up / up.max().clamp_min(1e-8)
    return up.detach().float().cpu().numpy()


def _storage_keys_in_order(hook_storage: dict[str, Any]) -> list[str]:
    # Preserve insertion order (matches forward execution order better than sorting).
    return [k for k in hook_storage.keys() if k not in (META_KEY, CLS_ATTN_KEY, SALIENCY_KEY)]


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
    # Some video ViTs use a temporal tubelet size > 1, so the token sequence may
    # correspond to an effective T' < T. Search divisors of T for a square grid.
    candidates: list[int] = []
    for d in range(1, t + 1):
        if t % d == 0:
            candidates.append(d)
    candidates = sorted(candidates, reverse=True)  # prefer higher temporal resolution

    for t_eff in candidates:
        if tokens_total % t_eff != 0:
            continue
        per_frame = tokens_total // t_eff
        side = int(math.isqrt(per_frame))
        if side * side == per_frame:
            return t_eff, side, side

        # Allow rectangular grids for some models (e.g., 7x14 = 98).
        for hp_try in range(1, per_frame + 1):
            if per_frame % hp_try != 0:
                continue
            wp_try = per_frame // hp_try
            # Prefer near-square
            if abs(hp_try - wp_try) <= 2:
                return t_eff, hp_try, wp_try

    raise ValueError(
        f"Cannot infer patch grid: tokens_total={tokens_total}, T_in={t}. "
        "T' search did not yield a square/near-square per-frame token grid."
    )


def _time_resample(maps: torch.Tensor, t_in: int) -> torch.Tensor:
    """Resample maps from (T',H,W) to (T_in,H,W) by nearest index selection."""
    if maps.ndim != 3:
        raise ValueError("maps must have shape (T,H,W)")
    t_eff = int(maps.shape[0])
    if t_eff == t_in:
        return maps
    idx = torch.linspace(0, t_eff - 1, steps=t_in, device=maps.device).round().to(torch.int64)
    return maps.index_select(0, idx)


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

    t_in = int(video_tensor.shape[2])
    t, hp, wp = _infer_patch_grid(video_tensor, seq_len_k, has_cls)
    per_frame = hp * wp
    if tokens_total != t * per_frame:
        raise ValueError(f"Token count mismatch: tokens_total={tokens_total} vs T*hp*wp={t*per_frame}")

    token_importance = token_importance.reshape(t, hp, wp)  # (T, hp, wp)

    # Upsample to original frame resolution
    h = int(video_tensor.shape[3])
    w = int(video_tensor.shape[4])
    up = F.interpolate(token_importance.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
    up = _time_resample(up, t_in)

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
        t_in = int(video_tensor.shape[2])
        up = F.interpolate(
            token_importance.unsqueeze(1),  # treat T as batch
            size=(int(video_tensor.shape[3]), int(video_tensor.shape[4])),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        up = _time_resample(up, t_in)
        up = up - up.min()
        up = up / up.max().clamp_min(1e-8)
        up_np = up.detach().float().cpu().numpy()
        if target_frame is None:
            return up_np.astype(np.float32, copy=False)
        return up_np[int(target_frame)].astype(np.float32, copy=False)

    # Second-best path (HuggingFace models): request attentions explicitly.
    with torch.no_grad():
        try:
            out = encoder(video_tensor, output_attentions=True, return_dict=True)  # type: ignore[call-arg]
        except TypeError:
            out = None

    if out is not None:
        attentions = getattr(out, "attentions", None)
        if attentions is None and isinstance(out, dict):
            attentions = out.get("attentions")

        if isinstance(attentions, (tuple, list)) and len(attentions) > 0:
            # Populate storage so rollout/head viz can reuse
            for i, a in enumerate(attentions):
                a_t = _coerce_attention_tensor(a)
                if a_t is not None:
                    hook_storage[f"attentions.{i}"] = [a_t.detach()]

            attn_raw = attentions[_resolve_layer_idx(layer_idx, len(attentions))]
            attn = _coerce_attention_tensor(attn_raw)
            if attn is None:
                # Some models return `None` attentions even when requested (e.g., flash attention).
                # Fall back to a token-norm saliency map rather than failing hard.
                try:
                    tokens = _coerce_output_to_tokens(out)
                except Exception:
                    tokens = None
                if tokens is None:
                    raise RuntimeError("Model returned attentions but could not coerce selected layer to a Tensor.")

                sal_map = _tokens_to_saliency_map(video_tensor, tokens)
                hook_storage[SALIENCY_KEY] = [torch.from_numpy(sal_map)]
                # Best-effort meta
                seq_len = int(tokens.shape[1])
                try:
                    _t, hp, wp = _infer_patch_grid(video_tensor, seq_len, has_cls=True)
                    has_cls = True
                except Exception:
                    _t, hp, wp = _infer_patch_grid(video_tensor, seq_len, has_cls=False)
                    has_cls = False
                hook_storage[META_KEY] = {
                    "t": _t,
                    "h": int(video_tensor.shape[3]),
                    "w": int(video_tensor.shape[4]),
                    "hp": hp,
                    "wp": wp,
                    "has_cls": has_cls,
                    "source": "fallback:token_norm",
                }
                if target_frame is None:
                    return sal_map.astype(np.float32, copy=False)
                return sal_map[int(target_frame)].astype(np.float32, copy=False)

            # Save meta
            if attn.dim() == 4:
                seq_len = int(attn.shape[-1])
            else:
                seq_len = int(attn.shape[-1])
            t = int(video_tensor.shape[2])
            if seq_len % t == 0:
                tokens_per_frame_guess = seq_len // t
            elif (seq_len - 1) % t == 0:
                tokens_per_frame_guess = (seq_len - 1) // t
            else:
                tokens_per_frame_guess = None
            has_cls = _has_cls_token(encoder, seq_len, tokens_per_frame_guess)
            _t, hp, wp = _infer_patch_grid(video_tensor, seq_len, has_cls)
            hook_storage[META_KEY] = {
                "t": _t,
                "h": int(video_tensor.shape[3]),
                "w": int(video_tensor.shape[4]),
                "hp": hp,
                "wp": wp,
                "has_cls": has_cls,
                "source": "hf:attentions",
            }

            attn_map = _attention_to_map(attn, encoder, video_tensor, head_idx=head_idx, target_frame=target_frame)
            return attn_map.astype(np.float32, copy=False)

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
        # Fallback: if we computed a saliency map (token-norm), reuse it.
        sal = hook_storage.get(SALIENCY_KEY)
        if isinstance(sal, list) and sal and isinstance(sal[-1], torch.Tensor):
            arr = sal[-1].detach().float().cpu().numpy()
            # Ensure [0,1]
            arr = arr - arr.min()
            arr = arr / (arr.max() + 1e-8)
            return arr.astype(np.float32)

        # Fallback: if we have a CLS attention vector (from qkv path), reshape it.
        cls_vec_list = hook_storage.get(CLS_ATTN_KEY)
        if isinstance(cls_vec_list, list) and cls_vec_list and isinstance(cls_vec_list[-1], torch.Tensor):
            vec = cls_vec_list[-1].detach().float().cpu()
            has_cls = bool(meta.get("has_cls", False))
            start = 1 if has_cls else 0
            t = int(meta["t"])
            hp = int(meta["hp"])
            wp = int(meta["wp"])
            h = int(meta["h"])
            w = int(meta["w"])

            tok = vec[start:]
            if tok.numel() != t * hp * wp:
                raise RuntimeError("CLS attention vector length mismatch; cannot build rollout map.")
            token_map = tok.reshape(t, hp, wp)
            up = F.interpolate(token_map.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
            up = up - up.min()
            up = up / up.max().clamp_min(1e-8)
            return up.numpy().astype(np.float32)

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

    try:
        rollout = compute_attention_rollout(storage)
        logger.info("Rollout shape: %s", rollout.shape)
    except Exception as exc:  # noqa: BLE001
        rollout = None
        logger.warning("Attention rollout unavailable for this model output (%s: %s)", type(exc).__name__, exc)

    # Overlay and save one example
    example = overlay_attention_on_frame(dummy_frames[0], maps[0], alpha=0.5)
    try:
        import cv2

        cv2.imwrite("assets/attention_example.png", cv2.cvtColor(example, cv2.COLOR_RGB2BGR))
        logger.info("Saved assets/attention_example.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save overlay image (cv2 missing?): %s", exc)
