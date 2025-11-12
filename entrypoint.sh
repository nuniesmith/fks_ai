#!/bin/bash
# Entrypoint script for fks_ai

set -e

# Default values
SERVICE_NAME=${SERVICE_NAME:-fks_ai}
SERVICE_PORT=${SERVICE_PORT:-8007}
HOST=${HOST:-0.0.0.0}

echo "Starting ${SERVICE_NAME} on ${HOST}:${SERVICE_PORT}"

# Run the service
exec python -m uvicorn src.main:app \
    --host "${HOST}" \
    --port "${SERVICE_PORT}" \
    --no-access-log \
    --log-level info
