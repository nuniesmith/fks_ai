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
    autoconf \
    automake \
    libtool \
    libc-bin \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel (better caching with BuildKit)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel

# Install TA-Lib C library (required before installing Python TA-Lib package)
# TA-Lib source includes configure script, but it may need updated config files for modern systems
RUN set -eux; \
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tar.gz || \
    wget -q https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz/download -O /tmp/ta-lib.tar.gz; \
    tar -xzf /tmp/ta-lib.tar.gz -C /tmp; \
    cd /tmp/ta-lib; \
    echo "=== Listing TA-Lib source files ==="; \
    ls -la; \
    echo "=== Checking for configure script ==="; \
    if [ -f configure ]; then \
        echo "Configure script found"; \
        chmod +x configure; \
    elif [ -f configure.ac ] || [ -f configure.in ]; then \
        echo "Configure.ac/in found, generating configure..."; \
        autoreconf -fvi || (echo "autoreconf failed, trying autogen.sh..." && [ -f autogen.sh ] && chmod +x autogen.sh && ./autogen.sh || true); \
        chmod +x configure 2>/dev/null || true; \
    else \
        echo "No configure or configure.ac/in found, listing all files:"; \
        find . -type f -name "configure*" -o -name "Makefile*" -o -name "*.ac" -o -name "*.in" | head -20; \
        echo "Trying to build with existing Makefile..."; \
        if [ -f Makefile ]; then \
            make -j$(nproc) && make install; \
        else \
            echo "ERROR: No configure script or Makefile found"; \
            exit 1; \
        fi; \
    fi; \
    if [ -f configure ]; then \
        echo "=== Running configure ==="; \
        ./configure --prefix=/usr 2>&1 | tee /tmp/configure.log || \
        (echo "=== Configure failed, showing log ===" && cat /tmp/configure.log && \
         echo "=== Directory contents ===" && ls -la && \
         echo "=== Checking for config.log ===" && [ -f config.log ] && cat config.log || echo "No config.log found" && \
         exit 1); \
        echo "=== Building TA-Lib ==="; \
        make -j$(nproc) 2>&1 | tee /tmp/make.log || \
        (echo "=== Make failed, showing last 50 lines ===" && tail -50 /tmp/make.log && exit 1); \
        echo "=== Installing TA-Lib ==="; \
        make install 2>&1 || (echo "=== Make install failed ===" && exit 1); \
    fi; \
    cd /; \
    rm -rf /tmp/ta-lib /tmp/ta-lib.tar.gz; \
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
