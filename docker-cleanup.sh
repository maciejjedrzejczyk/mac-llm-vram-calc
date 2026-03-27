#!/usr/bin/env bash
# Stop and remove the VRAM Calculator container, image, and build cache.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Stopping container..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" down --remove-orphans 2>/dev/null || true

echo "Removing image..."
docker rmi vram-calculator-vram-calculator 2>/dev/null || true

echo "Pruning build cache..."
docker builder prune -f --filter "label=com.docker.compose.project=vram-calculator" 2>/dev/null || true

echo "Cleanup complete."
