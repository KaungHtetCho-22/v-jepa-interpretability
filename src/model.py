import logging
import os
import sys
from contextlib import contextmanager
from collections import defaultdict
from typing import Any, Literal

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is not installed. Install it with one of:\n"
        "  - `uv sync --extra torch` (default wheels)\n"
        "  - `uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision` (CPU-only)\n"
    ) from exc

logger = logging.getLogger("vjepa.model")

ModelSize = Literal["vit_h", "vit_b", "vit_l"]
DeviceSpec = Literal["auto", "cuda", "cpu"]
DTypeSpec = Literal["float16", "float32"]


def _resolve_device(device: DeviceSpec) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: DTypeSpec) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    return torch.float32


def check_vram_budget(model_size: ModelSize) -> bool:
    """Return True if the model likely fits in available VRAM.

    Estimates are rough and assume inference + attention extraction overhead.
    """
    budgets_gb: dict[ModelSize, float] = {"vit_b": 1.5, "vit_l": 3.5, "vit_h": 5.8}
    required_gb = budgets_gb[model_size]

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; VRAM budget check skipped (CPU mode).")
        return False

    props = torch.cuda.get_device_properties(0)
    total_gb = float(props.total_memory) / (1024**3)
    ok = total_gb >= required_gb

    if not ok:
        logger.warning(
            "VRAM likely insufficient for %s (need ~%.1f GB, have %.1f GB).",
            model_size,
            required_gb,
            total_gb,
        )
    elif total_gb - required_gb < 0.5:
        logger.warning(
            "VRAM is tight for %s (budget ~%.1f GB vs %.1f GB total). Consider vit_b.",
            model_size,
            required_gb,
            total_gb,
        )

    return ok


def _infer_config(model: nn.Module, model_size: ModelSize, device: str) -> dict[str, Any]:
    cfg: Any = getattr(model, "config", None)

    def pick_attr(*names: str, default: Any = None) -> Any:
        for name in names:
            if cfg is not None and hasattr(cfg, name):
                return getattr(cfg, name)
            if hasattr(model, name):
                return getattr(model, name)
        return default

    embed_dim = pick_attr("hidden_size", "embed_dim", "dim", default=None)
    num_heads = pick_attr("num_attention_heads", "num_heads", default=None)
    num_layers = pick_attr("num_hidden_layers", "num_layers", "depth", default=None)
    patch_size = pick_attr("patch_size", default=None)

    if isinstance(patch_size, (tuple, list)) and len(patch_size) > 0:
        patch_size = patch_size[0]

    return {
        "model_size": model_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "patch_size": patch_size,
        "device": device,
    }


def _try_load_hf(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
) -> nn.Module | None:
    try:
        from transformers import AutoModel  # local import to keep module import light

        logger.info("Loading encoder from HuggingFace: %s", model_id)
        model_raw = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        class _HFEncoderAdapter(nn.Module):
            def __init__(self, inner: nn.Module) -> None:
                super().__init__()
                self.inner = inner

            @property
            def config(self) -> Any:  # passthrough for _infer_config
                return getattr(self.inner, "config", None)

            def forward(self, video: torch.Tensor) -> Any:
                # Our convention: (B, C, T, H, W). HF V-JEPA2 expects pixel_values_videos
                # shaped (B, T, C, H, W).
                if video.ndim != 5:
                    raise ValueError("Expected video tensor shape (B, C, T, H, W)")
                video_btchw = video.permute(0, 2, 1, 3, 4).contiguous()
                out = self.inner(pixel_values_videos=video_btchw)
                # Prefer returning last_hidden_state if present, for downstream interpretability.
                if hasattr(out, "last_hidden_state"):
                    return out.last_hidden_state
                return out

        model = _HFEncoderAdapter(model_raw)
        model.eval()
        model.to(device)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF load failed for %s; falling back. (%s: %s)", model_id, type(exc).__name__, exc)
        return None


def _try_load_torchhub(model_size: ModelSize, device: str, torch_dtype: torch.dtype) -> nn.Module | None:
    # TorchHub entrypoints documented in facebookresearch/vjepa2 README:
    #   vjepa2_vit_large / vjepa2_vit_huge / vjepa2_vit_giant / vjepa2_vit_giant_384
    entrypoints: dict[ModelSize, str] = {
        "vit_b": "vjepa2_vit_large",
        "vit_l": "vjepa2_vit_huge",
        "vit_h": "vjepa2_vit_giant",
    }
    entrypoint = entrypoints[model_size]

    try:
        logger.info("Loading encoder from TorchHub: %s", entrypoint)

        @contextmanager
        def _hide_local_src_package() -> Any:
            # TorchHub repo imports `src.*`. Our project also has `src/`, which can shadow it.
            # Temporarily remove the repo root from sys.path and clear any already-imported `src`.
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
            removed: list[str] = []
            for p in list(sys.path):
                if p in ("", ".") or os.path.abspath(p) == repo_root:
                    if p in sys.path:
                        sys.path.remove(p)
                        removed.append(p)
            sys.modules.pop("src", None)
            try:
                yield
            finally:
                # Restore in front to keep original import behavior
                for p in reversed(removed):
                    sys.path.insert(0, p)

        with _hide_local_src_package():
            result = torch.hub.load("facebookresearch/vjepa2", entrypoint)

        if isinstance(result, tuple):
            model, _transforms = result
        else:
            model = result

        if not isinstance(model, nn.Module):
            raise TypeError(f"TorchHub returned unexpected type: {type(model)}")

        model.eval()
        if device == "cpu":
            model.to(device=device, dtype=torch.float32)
        else:
            model.to(device=device, dtype=torch_dtype)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning("TorchHub load failed for %s. (%s: %s)", entrypoint, type(exc).__name__, exc)
        return None


def load_encoder(
    model_size: ModelSize = "vit_h",
    device: DeviceSpec = "auto",
    dtype: DTypeSpec = "float16",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a V-JEPA 2 encoder and return (encoder, config).

    The loader tries:
      1) HuggingFace (primary)
      2) TorchHub (fallback; fixes tuple return)

    Args:
        model_size: One of "vit_h" | "vit_l" | "vit_b".
        device: "auto" | "cuda" | "cpu".
        dtype: "float16" | "float32".
    """
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)

    if resolved_device == "cpu":
        logger.warning("Loading on CPU; demo will be slow.")
        resolved_dtype = torch.float32

    if resolved_device == "cuda":
        check_vram_budget(model_size)

    # HuggingFace model ids are lowercased (vitl/vith/vitg) and commonly published at 256 resolution.
    # We map requested sizes to the closest available checkpoints.
    hf_ids: dict[ModelSize, str] = {
        "vit_b": "facebook/vjepa2-vitl-fpc64-256",
        "vit_l": "facebook/vjepa2-vith-fpc64-256",
        "vit_h": "facebook/vjepa2-vitg-fpc64-256",
    }

    encoder: nn.Module | None = _try_load_hf(hf_ids[model_size], resolved_device, resolved_dtype)
    if encoder is None:
        encoder = _try_load_torchhub(model_size, resolved_device, resolved_dtype)

    if encoder is None:
        raise RuntimeError(
            "Failed to load V-JEPA 2 encoder via HuggingFace and TorchHub. "
            "As a last resort, download a checkpoint locally and load it manually "
            "(see comments in this file)."
        )

    config = _infer_config(encoder, model_size, resolved_device)
    return encoder, config


def register_attention_hooks(
    encoder: nn.Module,
    layer_indices: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Register forward hooks on attention modules and store attention tensors.

    Notes:
    - Architectures vary; this tries to capture attention weights if they appear
      in module outputs.
    - For HuggingFace models, you may also get attentions by passing
      `output_attentions=True` during forward. Hooks remain useful for non-HF models.
    """
    storage: dict[str, list[torch.Tensor]] = defaultdict(list)

    candidates: list[tuple[str, nn.Module]] = []
    for name, module in encoder.named_modules():
        lname = name.lower()
        if isinstance(module, nn.MultiheadAttention) or lname.endswith("attn") or "attention" in lname or "self_attn" in lname:
            candidates.append((name, module))

    selected: list[tuple[int, str, nn.Module]] = [(i, n, m) for i, (n, m) in enumerate(candidates)]
    if layer_indices is not None:
        requested = set(layer_indices)
        selected = [triple for triple in selected if triple[0] in requested]

    handles: list[Any] = []

    def hook_fn(mod_name: str):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            attn: torch.Tensor | None = None
            if isinstance(output, (tuple, list)):
                # Common patterns: (attn_output, attn_weights) or (..., attn)
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dim() >= 3:
                        # Heuristic: attention weights often have shape (B, heads, q, k) or (heads, q, k)
                        if item.shape[-1] == item.shape[-2] or item.dim() == 4:
                            attn = item
                            break
            elif isinstance(output, dict):
                for key in ("attn", "attn_weights", "attention", "attentions"):
                    val = output.get(key)
                    if isinstance(val, torch.Tensor):
                        attn = val
                        break

            if attn is not None:
                storage[mod_name].append(attn.detach())

        return _hook

    for _idx, name, module in selected:
        try:
            handles.append(module.register_forward_hook(hook_fn(name)))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to hook %s (%s: %s)", name, type(exc).__name__, exc)

    # Store handles on the module so callers can remove later if desired.
    setattr(encoder, "_vjepa_attn_hook_handles", handles)
    logger.info("Registered %d attention hooks.", len(handles))
    # Return the live storage object (do not copy), so hooks keep filling it.
    return storage


def preprocess_video(
    frames: list[Any],
    image_size: int = 224,
    num_frames: int = 16,
    device: str = "cuda",
) -> torch.Tensor:
    """Preprocess a list of frames into a (1, C, T, H, W) float tensor.

    Accepts frames as PIL Images or numpy arrays (H, W, C) in uint8/rgb-ish.
    Performs uniform sampling if len(frames) != num_frames.
    """
    if len(frames) == 0:
        raise ValueError("frames must be a non-empty list")

    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Pillow is required for preprocess_video") from exc

    # Uniform sampling (with repetition if needed)
    if len(frames) != num_frames:
        if len(frames) == 1:
            idxs = [0] * num_frames
        else:
            idxs = torch.linspace(0, len(frames) - 1, steps=num_frames).round().to(torch.int64).tolist()
        frames = [frames[i] for i in idxs]

    pil_frames: list[Image.Image] = []
    for fr in frames:
        if isinstance(fr, Image.Image):
            pil_frames.append(fr.convert("RGB"))
        else:
            pil_frames.append(Image.fromarray(fr).convert("RGB"))

    # Minimal torchvision-free pipeline (keeps preprocessing available even when torchvision isn't installed yet).
    import numpy as np

    resized = [im.resize((image_size, image_size), resample=Image.BICUBIC) for im in pil_frames]
    arr = np.stack([np.asarray(im, dtype=np.float32) for im in resized], axis=0) / 255.0  # (T, H, W, C)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    resolved_device = _resolve_device(device if device in ("auto", "cuda", "cpu") else "auto")  # type: ignore[arg-type]
    if resolved_device == "cuda" and not torch.cuda.is_available():
        logger.warning("Requested device='cuda' but CUDA is not available; using CPU.")
        resolved_device = "cpu"
    return tensor.to(resolved_device)


# Manual checkpoint loading (last resort):
# - Download a checkpoint locally
# - Instantiate the matching architecture
# - Load weights via `state_dict` and `load_state_dict`
# This is intentionally not implemented until we know the exact checkpoint format.


def _extract_tensor_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported model output type: {type(output)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    initial_size: ModelSize = "vit_h"
    resolved_device = _resolve_device("auto")
    if resolved_device == "cpu":
        # Avoid pulling the largest checkpoint when CUDA isn't available.
        initial_size = "vit_b"

    enc, cfg = load_encoder(model_size=initial_size, device="auto", dtype="float16")
    logger.info("Loaded V-JEPA 2 encoder")
    logger.info("  model_size: %s", cfg.get("model_size"))
    logger.info("  embed_dim: %s", cfg.get("embed_dim"))
    logger.info("  num_heads: %s", cfg.get("num_heads"))
    logger.info("  num_layers: %s", cfg.get("num_layers"))
    logger.info("  device: %s", cfg.get("device"))

    device_used = str(cfg.get("device"))
    dtype_used = torch.float16 if device_used == "cuda" else torch.float32
    dummy = torch.randn(1, 3, 16, 224, 224, device=device_used, dtype=dtype_used)

    if torch.cuda.is_available() and str(cfg.get("device")) == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        with torch.no_grad():
            try:
                out = enc(dummy)
            except Exception:
                # Some checkpoints expect a fixed number of frames (commonly 64).
                dummy64 = torch.randn(1, 3, 64, 224, 224, device=device_used, dtype=dtype_used)
                out = enc(dummy64)
    except torch.cuda.OutOfMemoryError as exc:
        logger.error("CUDA OOM during forward; try model_size='vit_b'.")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Forward pass failed (%s: %s)", type(exc).__name__, exc)
        raise

    out_tensor = _extract_tensor_from_output(out)
    logger.info("Dummy forward pass: OK -> output shape %s", tuple(out_tensor.shape))

    if torch.cuda.is_available() and str(cfg.get("device")) == "cuda":
        peak_gb = float(torch.cuda.max_memory_allocated()) / (1024**3)
        logger.info("Peak VRAM: ~%.2f GB", peak_gb)
