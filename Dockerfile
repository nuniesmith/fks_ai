# Multi-stage build for fks_ai Python service
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies needed for TA-Lib, chromadb, sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    wget \
    cmake \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    build-essential \
    libc-bin \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel (better caching with BuildKit)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel

# Install TA-Lib C library (required before installing Python TA-Lib package)
# Note: TA-Lib configure script needs updated config files for modern systems
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    wget -q -O config.guess 'http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD' && \
    wget -q -O config.sub 'http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD' && \
    ./configure --prefix=/usr || (echo "Configure failed, checking error..." && cat config.log && exit 1) && \
    make -j$(nproc) || (echo "Make failed" && exit 1) && \
    make install || (echo "Make install failed" && exit 1) && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    ldconfig

# Copy dependency files (for better layer caching)
COPY requirements.txt ./

# Install Python dependencies with BuildKit cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --user --no-warn-script-location -r requirements.txt

# Runtime stage
FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVICE_NAME=fks_ai \
    SERVICE_PORT=8007 \
    PYTHONPATH=/app/src:/app \
    PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app

# Install runtime dependencies only
# Copy TA-Lib libraries from builder (needed at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user first (before copying files)
RUN useradd -u 1000 -m -s /bin/bash appuser

# Copy TA-Lib libraries from builder (needed at runtime)
COPY --from=builder /usr/lib/libta_lib.so* /usr/lib/

# Copy Python packages from builder with correct ownership
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

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
