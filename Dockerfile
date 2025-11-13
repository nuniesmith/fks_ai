# Optimized Dockerfile for fks_ai - Uses base image directly to reduce size
# Uses ML base image with all dependencies pre-installed
FROM nuniesmith/fks:docker-ml-latest

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVICE_NAME=fks_ai \
    SERVICE_PORT=8007 \
    PYTHONPATH=/app/src:/app \
    PATH=/usr/local/bin:$PATH

# Install only service-specific packages (uvicorn, fastapi, etc. are already in base or need to be added)
COPY requirements.txt ./

# Install service-specific packages only
# Base image already has: langchain, chromadb, sentence-transformers, ollama, TA-Lib, numpy, pandas, httpx
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip cache purge || true \
    && rm -rf /root/.cache/pip/* /tmp/pip-* 2>/dev/null || true

# Create non-root user if it doesn't exist
RUN id -u appuser 2>/dev/null || useradd -u 1000 -m -s /bin/bash appuser

# Copy application source
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser entrypoint.sh ./

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Verify uvicorn is accessible (before switching user)
RUN python3 -c "import uvicorn; print(f'✅ uvicorn found: {uvicorn.__file__}')" || echo "⚠️  uvicorn verification failed"

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
