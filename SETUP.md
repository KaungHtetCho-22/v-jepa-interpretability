# Setup (Reproducible)

This repo is set up to be reproducible with `uv`.

## Requirements

- Python 3.10+
- `uv` installed

## Install `uv`

If you don't already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell (or open a new terminal) so `uv` is on your `PATH`.

## Create environment + install deps

From the repo root:

```bash
uv sync
```

This installs the base dependencies. PyTorch is intentionally left as an optional extra (CUDA setup varies by machine); it gets installed in the next session via:

```bash
uv sync --extra torch
```

If `decord` fails to install on your machine/OS, you can still proceed; the project will use `torchvision.io.read_video` as a fallback later in development.

## Add a sample video

Place a sample video at `data/sample.mp4`. For convenience, you can download a small public-domain sample:

```bash
mkdir -p data
curl -L -o data/sample.mp4 https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4
```

## Launch the demo

```bash
bash scripts/run_demo.sh
```

At this stage (Session 00), the demo is a stub and will exit with a clear message indicating it is not implemented yet.
