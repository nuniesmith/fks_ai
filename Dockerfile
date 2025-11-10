# Multi-stage build for fks_ai Python service
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies needed for TA-Lib, chromadb, sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    wget \
    curl \
    cmake \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    build-essential \
    autoconf \
    automake \
    libtool \
    libc-bin \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel (better caching with BuildKit)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel

# Install TA-Lib C library (required before installing Python TA-Lib package)
# Use GitHub mirror as primary source (more reliable than SourceForge)
RUN set -e; \
    echo "=== Downloading TA-Lib ==="; \
    TA_LIB_DOWNLOADED=0; \
    # Try GitHub mirror first (most reliable)
    if wget -q --timeout=30 --tries=2 -O /tmp/ta-lib.tar.gz https://github.com/TA-Lib/ta-lib/archive/refs/tags/v0.4.0.tar.gz 2>&1; then \
        echo "Downloaded from GitHub mirror"; \
        TA_LIB_DOWNLOADED=1; \
        TA_LIB_DIR="ta-lib-0.4.0"; \
    elif wget -q --timeout=30 --tries=2 -O /tmp/ta-lib.tar.gz http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz 2>&1; then \
        echo "Downloaded from SourceForge primary"; \
        TA_LIB_DOWNLOADED=1; \
        TA_LIB_DIR="ta-lib"; \
    elif curl -f -L --connect-timeout 30 --max-time 120 -o /tmp/ta-lib.tar.gz https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz/download 2>&1; then \
        echo "Downloaded from SourceForge via curl"; \
        TA_LIB_DOWNLOADED=1; \
        TA_LIB_DIR="ta-lib"; \
    fi; \
    if [ "$TA_LIB_DOWNLOADED" -eq 0 ]; then \
        echo "ERROR: All download methods failed"; \
        echo "Attempted: GitHub mirror, SourceForge primary, SourceForge curl"; \
        exit 1; \
    fi; \
    echo "=== Extracting TA-Lib ==="; \
    tar -xzf /tmp/ta-lib.tar.gz -C /tmp; \
    cd /tmp/$TA_LIB_DIR; \
    echo "=== Listing TA-Lib source files ==="; \
    ls -la; \
    echo "=== Checking for configure script ==="; \
    if [ -f configure ]; then \
        echo "Configure script found"; \
        chmod +x configure; \
    elif [ -f configure.ac ] || [ -f configure.in ]; then \
        echo "Configure.ac/in found, generating configure..."; \
        autoreconf -fvi 2>&1 || (echo "autoreconf failed, trying autogen.sh..." && [ -f autogen.sh ] && chmod +x autogen.sh && ./autogen.sh 2>&1 || true); \
        if [ -f configure ]; then \
            chmod +x configure; \
            echo "Configure script generated successfully"; \
        else \
            echo "ERROR: Configure script generation failed"; \
            exit 1; \
        fi; \
    else \
        echo "ERROR: No configure script or configure.ac/in found"; \
        echo "Listing relevant files:"; \
        find . -maxdepth 2 -type f \( -name "configure*" -o -name "Makefile*" -o -name "*.ac" -o -name "*.in" \) 2>/dev/null | head -20; \
        exit 1; \
    fi; \
    echo "=== Running configure ==="; \
    if ! ./configure --prefix=/usr > /tmp/configure.log 2>&1; then \
        echo "=== Configure failed ==="; \
        echo "Configure output:"; \
        cat /tmp/configure.log; \
        if [ -f config.log ]; then \
            echo "=== config.log ==="; \
            tail -100 config.log; \
        fi; \
        exit 1; \
    fi; \
    echo "=== Building TA-Lib ==="; \
    if ! make -j$(nproc) > /tmp/make.log 2>&1; then \
        echo "=== Make failed ==="; \
        echo "Last 100 lines of make.log:"; \
        tail -100 /tmp/make.log; \
        exit 1; \
    fi; \
    echo "=== Installing TA-Lib ==="; \
    if ! make install > /tmp/install.log 2>&1; then \
        echo "=== Make install failed ==="; \
        echo "Install output:"; \
        cat /tmp/install.log; \
        exit 1; \
    fi; \
    cd /; \
    rm -rf /tmp/$TA_LIB_DIR /tmp/ta-lib.tar.gz; \
    ldconfig; \
    echo "=== TA-Lib installation complete ==="

# Copy dependency files (for better layer caching)
COPY requirements.txt ./

# Install Python dependencies with BuildKit cache mount
# Use --no-cache-dir to reduce disk usage in CI
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --user --no-warn-script-location --no-cache-dir -r requirements.txt

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
