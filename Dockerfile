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
# Try multiple download sources and methods
RUN echo "=== Downloading TA-Lib ===" && \
    (wget -q --timeout=30 --tries=3 http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tar.gz 2>&1 || \
     wget -q --timeout=30 --tries=3 https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz/download -O /tmp/ta-lib.tar.gz 2>&1 || \
     curl -L -o /tmp/ta-lib.tar.gz https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz/download 2>&1 || \
     (echo "ERROR: All download methods failed" && exit 1)) && \
    echo "=== Extracting TA-Lib ===" && \
    tar -xzf /tmp/ta-lib.tar.gz -C /tmp && \
    cd /tmp/ta-lib && \
    echo "=== Listing TA-Lib source files ===" && \
    ls -la && \
    echo "=== Checking for configure script ===" && \
    if [ -f configure ]; then \
        echo "Configure script found" && \
        chmod +x configure; \
    elif [ -f configure.ac ] || [ -f configure.in ]; then \
        echo "Configure.ac/in found, generating configure..." && \
        autoreconf -fvi 2>&1 || (echo "autoreconf failed, trying autogen.sh..." && [ -f autogen.sh ] && chmod +x autogen.sh && ./autogen.sh 2>&1 || echo "autogen.sh also failed"); \
        [ -f configure ] && chmod +x configure || echo "Configure script still not found after autoreconf"; \
    else \
        echo "No configure or configure.ac/in found, listing relevant files:" && \
        find . -maxdepth 2 -type f \( -name "configure*" -o -name "Makefile*" -o -name "*.ac" -o -name "*.in" \) 2>/dev/null | head -20; \
    fi && \
    if [ ! -f configure ]; then \
        echo "ERROR: Configure script not found and could not be generated" && \
        echo "Directory contents:" && ls -la && \
        exit 1; \
    fi && \
    echo "=== Running configure ===" && \
    ./configure --prefix=/usr 2>&1 | tee /tmp/configure.log || \
    (echo "=== Configure failed ===" && \
     echo "Configure log:" && cat /tmp/configure.log 2>/dev/null || echo "No configure.log" && \
     echo "=== Directory contents ===" && ls -la && \
     [ -f config.log ] && (echo "=== config.log ===" && cat config.log) || echo "No config.log found" && \
     exit 1) && \
    echo "=== Building TA-Lib ===" && \
    make -j$(nproc) 2>&1 | tee /tmp/make.log || \
    (echo "=== Make failed ===" && \
     echo "Last 50 lines of make.log:" && tail -50 /tmp/make.log 2>/dev/null || echo "No make.log" && \
     exit 1) && \
    echo "=== Installing TA-Lib ===" && \
    make install 2>&1 || \
    (echo "=== Make install failed ===" && exit 1) && \
    cd / && \
    rm -rf /tmp/ta-lib /tmp/ta-lib.tar.gz && \
    ldconfig && \
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
