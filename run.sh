#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[ai] Stopping existing containers..."
docker compose down

echo "[ai] Rebuilding images..."
docker compose build

echo "[ai] Starting containers in detached mode..."
docker compose up -d
