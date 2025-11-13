# Multi-stage build for fks_ai Python service
# Uses ML base image with LangChain, ChromaDB, and sentence-transformers pre-installed
FROM nuniesmith/fks:docker-ml AS builder

WORKDIR /app

# ML packages (langchain, chromadb, sentence-transformers, ollama, TA-Lib) are already installed in base
# Just install service-specific packages
COPY requirements.txt ./

# Install Python dependencies with BuildKit cache mount
# Install system-wide (not --user) so we can copy to runtime stage easily
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-warn-script-location --no-cache-dir -r requirements.txt \
    && python -m pip cache purge || true \
    && rm -rf /root/.cache/pip/* /tmp/pip-* 2>/dev/null || true

# Runtime stage
FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVICE_NAME=fks_ai \
    SERVICE_PORT=8007 \
    PYTHONPATH=/app/src:/app \
    PATH=/usr/local/bin:$PATH

WORKDIR /app

# Install runtime dependencies only
# Copy TA-Lib libraries from builder (needed at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user first (before copying files)
RUN useradd -u 1000 -m -s /bin/bash appuser

# Copy TA-Lib libraries from builder (needed at runtime)
# Note: Files may not exist if base image doesn't have TA-Lib installed
RUN --mount=type=bind,from=builder,source=/usr/lib,target=/tmp/ta-lib \
    sh -c 'if ls /tmp/ta-lib/libta_lib.so* 1> /dev/null 2>&1; then cp /tmp/ta-lib/libta_lib.so* /usr/lib/; fi' || true

# Copy Python packages from builder (system-wide installation)
# Create directory first, then copy
RUN mkdir -p /usr/local/lib/python3.12/site-packages
COPY --from=builder --chown=appuser:appuser /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy Python executables (uvicorn, etc.) if they exist
RUN --mount=type=bind,from=builder,source=/usr/local/bin,target=/tmp/builder-bin \
    if [ -d /tmp/builder-bin ]; then \
        cp -r /tmp/builder-bin/* /usr/local/bin/ 2>/dev/null || true; \
    fi || true

# Verify uvicorn is accessible
RUN python3 -c "import uvicorn; print(f'✅ uvicorn found: {uvicorn.__file__}')" || echo "⚠️  uvicorn verification failed"

# Copy application source with correct ownership
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser entrypoint.sh ./

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request,sys;port=os.getenv('SERVICE_PORT','8007');u=f'http://localhost:{port}/health';\
import urllib.error;\
try: urllib.request.urlopen(u,timeout=3);\
except Exception: sys.exit(1)" || exit 1

# Expose the service port
EXPOSE 8007

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
