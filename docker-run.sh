#!/usr/bin/env bash
# Build and run the VRAM Calculator in Docker.
# LM Studio must be running on the host for benchmark features.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolve LM Studio paths
LMSTUDIO_DIR="${LMSTUDIO_DIR:-$HOME/.lmstudio}"
if [ ! -d "$LMSTUDIO_DIR" ]; then
    echo "Warning: LM Studio directory not found at $LMSTUDIO_DIR"
    echo "Model scanning and config reading will be limited."
    echo "Set LMSTUDIO_DIR to override."
fi

export LMSTUDIO_MODELS_DIR="${LMSTUDIO_MODELS_DIR:-$LMSTUDIO_DIR/models}"
export LMSTUDIO_CONFIG_DIR="${LMSTUDIO_CONFIG_DIR:-$LMSTUDIO_DIR}"

echo "Building Docker image..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" build

echo "Starting container..."
echo "  Dashboard: http://localhost:8501"
echo "  LM Studio models: $LMSTUDIO_MODELS_DIR"
echo "  LM Studio config: $LMSTUDIO_CONFIG_DIR"
echo ""
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d

echo "Container running. Use 'docker compose logs -f' to follow logs."
echo "Run ./docker-cleanup.sh to stop and remove."
