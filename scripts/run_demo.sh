#!/bin/bash
set -e
echo "Launching V-JEPA Interpretability Demo..."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
uv run python demo/app.py
