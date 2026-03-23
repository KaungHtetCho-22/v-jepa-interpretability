# Setup

This repo is reproducible with `uv`.

## Requirements

- Python 3.10+
- `uv`
- (Recommended) CUDA-capable GPU for interactive attention/embedding extraction

## Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell so `uv` is on your `PATH`.

## Install dependencies

From the repo root:

```bash
uv sync
uv sync --extra torch
```

If `decord` fails to install on your machine/OS, the project falls back to `torchvision.io.read_video`.

## (Optional) Build a tiny demo dataset

This creates a small **4 classes × 10 clips** dataset from public-domain / CC videos (requires `curl` + `ffmpeg`):

```bash
uv run python scripts/make_mini_dataset.py --out_dir data/mini_4x10
```

## Run the demo

```bash
bash scripts/run_demo.sh
```

The Nearest Neighbors tab needs a reference index. Build one (for example from the toy dataset):

```bash
uv run python scripts/build_index.py --video_dir data/mini_4x10 --save_path data/reference_index.npz
```

## Run experiments (CLI)

```bash
uv run python src/model.py
uv run python src/attention.py
uv run python src/embeddings.py
uv run python src/masking.py
```

Linear probe:

```bash
uv run python scripts/extract_features.py --video_dir data/mini_4x10 --output data/features_mini.npz --classes bunny sintel tears dream --num_clips_per_class 10
uv run python src/probe.py --features data/features_mini.npz --pixel_baseline_dir data/mini_4x10 --out_dir outputs/probe_mini
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'`**: run `uv sync --extra torch`.
- **Matplotlib cache warnings**: set `MPLCONFIGDIR=/tmp/mpl` if your home dir is read-only.
- **Retrieval says “Reference index not found”**: run `scripts/build_index.py` or use the “Build index” button in the demo.
