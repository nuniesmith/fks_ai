# FKS AI - GPU ML/RAG Service

**Port**: 8007  
**Framework**: Python 3.13 + FastAPI + PyTorch + CUDA  
**Role**: GPU-accelerated machine learning, regime detection, local LLM inference, RAG system

## Overview

FKS AI is the **intelligent ML/RAG engine** of the FKS Trading Platform. It provides:

- **Local LLM Inference**: Zero-cost strategy generation using Ollama (Llama-3)
- **Regime Detection**: VAE + Transformer models for market regime classification
- **RAG System**: Semantic search with pgvector for trading insights
- **Embeddings**: sentence-transformers for local GPU inference
- **Forecasting**: Time-series predictions for price movements

**GPU Requirements**:
- NVIDIA GPU with CUDA 12.2+ support
- At least 8GB VRAM for LLM models
- nvidia-docker2 runtime installed

## Architecture Principles

### What FKS AI DOES

✅ Train and serve VAE models for regime detection  
✅ Run Transformer sequence models for temporal patterns  
✅ Generate trading strategies via local LLM (Ollama)  
✅ Create embeddings for RAG semantic search  
✅ Query pgvector for document retrieval  
✅ Forecast price movements with ML models  
✅ Provide GPU-accelerated inference  

### What FKS AI DOES NOT DO

❌ NO business logic or trading decisions (use fks_app)  
❌ NO data collection (use fks_data service)  
❌ NO order execution (use fks_execution service)  
❌ NO direct database writes (returns predictions to fks_app)  

## Tech Stack

- **Language**: Python 3.13
- **Framework**: FastAPI + uvicorn
- **ML**: PyTorch 2.0+, scikit-learn
- **LLM**: Ollama + llama.cpp (CUDA acceleration)
- **Embeddings**: sentence-transformers, OpenAI (fallback)
- **Vector Store**: pgvector extension in PostgreSQL
- **GPU**: CUDA 12.2+, cuDNN
- **Monitoring**: Prometheus metrics for GPU utilization

## API Endpoints

### Regime Detection

- `GET /ai/regime?symbol={symbol}` - Detect current market regime
- `POST /ai/train-regime-model` - Train regime detection model
- `GET /ai/regime/history` - Historical regime classifications

### LLM Strategy Generation

- `POST /ai/generate-strategy` - Generate strategy via LLM
- `POST /ai/validate-strategy` - Validate LLM-generated strategy
- `GET /ai/strategy-prompt` - Get prompt template

### Embeddings & RAG

- `POST /ai/embed` - Generate embeddings for text
- `POST /ai/rag/query` - Semantic search for trading insights
- `POST /ai/rag/ingest` - Ingest documents for RAG

### Forecasting

- `POST /ai/predict` - Forecast price movements
- `GET /ai/models` - List available models
- `POST /ai/train` - Train forecasting model

### Health & Metrics

- `GET /health` - Service health check
- `GET /metrics` - Prometheus metrics (GPU utilization, inference latency)

## Directory Structure

```
repo/ai/
├── src/
│   ├── main.py              # FastAPI application
│   ├── models/              # ML models
│   │   ├── vae.py          # Variational Autoencoder
│   │   ├── transformer.py  # Sequence model
│   │   ├── gmm.py          # Gaussian Mixture Model
│   │   └── ensemble.py     # Ensemble methods
│   ├── training/            # Training pipelines
│   │   ├── regime.py       # Regime detection trainer
│   │   ├── forecast.py     # Forecasting trainer
│   │   └── metrics.py      # Training metrics
│   ├── inference/           # Inference engines
│   │   ├── regime.py       # Real-time regime detection
│   │   └── forecast.py     # Price forecasting
│   ├── llm/                 # LLM integration
│   │   ├── ollama.py       # Ollama client
│   │   ├── prompts.py      # Prompt engineering
│   │   └── parser.py       # Strategy parsing
│   ├── rag/                 # RAG system
│   │   ├── embeddings.py   # sentence-transformers
│   │   ├── vectorstore.py  # pgvector client
│   │   └── retrieval.py    # Semantic search
│   └── utils/
│       ├── gpu.py          # GPU utilities
│       └── cache.py        # Model caching
├── models/                  # Saved models
│   ├── vae_btc.pt
│   ├── transformer_btc.pt
│   └── ensemble_btc.pkl
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── Dockerfile.gpu          # GPU-enabled container
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Development Setup

### Prerequisites

- Python 3.13
- NVIDIA GPU with CUDA 12.2+
- nvidia-docker2 runtime
- At least 8GB VRAM
- Access to Ollama service (port 11434)
- Access to PostgreSQL with pgvector

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

# Run tests
pytest tests/unit/ -v

# Run locally (requires GPU)
CUDA_VISIBLE_DEVICES=0 uvicorn src.main:app --reload --port 8007

# Run in Docker with GPU
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up fks_ai
```

### Environment Variables

```bash
# Service configuration
FKS_AI_PORT=8007
FKS_AI_HOST=0.0.0.0

# GPU configuration
CUDA_VISIBLE_DEVICES=0
TRANSFORMERS_CACHE=/root/.cache/huggingface

# Ollama (local LLM)
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=llama3:8b

# Database (pgvector)
DATABASE_URL=postgresql://fks_user:password@db:5432/trading_db

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cuda

# Feature flags
ENABLE_GPU_ACCELERATION=true
ENABLE_LLM_GENERATION=true
ENABLE_RAG_SYSTEM=true

# Logging
LOG_LEVEL=INFO
```

## Training Models

### Regime Detection (Phase 2)

```bash
# Train VAE model
curl -X POST http://localhost:8007/ai/train-regime-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "vae",
    "symbol": "BTCUSDT",
    "start_date": "2013-01-01",
    "end_date": "2022-12-31",
    "hyperparameters": {
      "latent_dim": 2,
      "hidden_dim": 64,
      "epochs": 100
    }
  }'

# Train Transformer model
curl -X POST http://localhost:8007/ai/train-regime-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "transformer",
    "symbol": "BTCUSDT",
    "sequence_length": 16,
    "epochs": 50
  }'
```

### LLM Strategy Generation (Phase 3)

```bash
# Generate strategy
curl -X POST http://localhost:8007/ai/generate-strategy \
  -H "Content-Type: application/json" \
  -d '{
    "regime": "calm",
    "risk_tolerance": "moderate",
    "capital": 10000,
    "constraints": "no leverage, max 5 positions"
  }'
```

## Performance Benchmarks

### Regime Detection (from CRYPTO_REGIME_BACKTESTING.md)

| Model | Window | PNL (Backtest) | PNL (Forward) | Sharpe (BT) | Sharpe (FT) |
|-------|--------|----------------|---------------|-------------|-------------|
| GMM | 21d | 80-100% | 40-50% | 4.5-5.5 | 2.5-3.5 |
| VAE | 16d | 90-110% | 45-60% | 5.5-6.5 | 3.0-4.5 |
| Transformer | 16d | 95-115% | 50-65% | 6.0-7.0 | 3.5-5.0 |
| Ensemble | 28d | 100-120% | 50-70% | 6.5-7.5 | 4.0-8.0 |

**Realistic Expectations**: 50-70% degradation from backtest to forward test.

### LLM Strategy Generation (from AI_STRATEGY_INTEGRATION.md)

- **Success Rate**: 60%+ profitable strategies (research-backed)
- **Latency**: <5s per strategy (local Ollama)
- **Cost**: $0 (zero LLM API fees)
- **Quality**: Validated via backtesting in fks_app

## Integration with Ollama

FKS AI uses Ollama for zero-cost local LLM inference:

```yaml
# docker-compose.gpu.yml
ollama:
  image: ollama/ollama:latest
  ports: ["11434:11434"]
  volumes: [ollama_models:/root/.ollama]
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**Available Models**:
- `llama3:8b` - Default model for strategy generation
- `mistral:7b` - Alternative lightweight model
- `codellama:13b` - For code-heavy strategies

## Testing

```bash
# Unit tests (no GPU required, uses mocks)
pytest tests/unit/ -v

# Integration tests (requires GPU + Ollama)
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d ollama
pytest tests/integration/ -v

# Model validation tests
pytest tests/unit/test_models/ -v --run-slow

# Coverage
pytest tests/ --cov=src --cov-report=html

# Lint
ruff check src/
mypy src/
```

## Deployment

### Docker Build (GPU)

```bash
docker build -f Dockerfile.gpu -t fks_ai:latest .
```

### Health Checks

- **Endpoint**: `GET /health`
- **Expected**: `{"status": "healthy", "service": "fks_ai", "gpu": "available"}`
- **Dependencies**: Ollama, PostgreSQL (pgvector), CUDA runtime

## Performance Considerations

- **GPU Memory**: Monitor VRAM usage with `nvidia-smi`
- **Batch Inference**: Process multiple predictions in batches
- **Model Caching**: Cache loaded models in GPU memory
- **Async Endpoints**: Use FastAPI async for I/O-bound operations
- **Rate Limiting**: Prevent GPU overload with request throttling

## Common Issues

**GPU not detected**:
- Check `nvidia-smi` output
- Verify CUDA_VISIBLE_DEVICES is set
- Ensure nvidia-docker2 is installed

**Ollama connection fails**:
- Verify Ollama service is running: `curl http://localhost:11434/api/tags`
- Check docker network configuration
- Ensure models are pulled: `docker exec ollama ollama pull llama3:8b`

**OOM (Out of Memory)**:
- Reduce batch size for inference
- Use smaller models (llama3:8b instead of 13b)
- Clear GPU cache between requests: `torch.cuda.empty_cache()`

**Slow inference**:
- Enable GPU acceleration: `ENABLE_GPU_ACCELERATION=true`
- Use FP16 precision for faster inference
- Batch multiple requests together

## Contributing

1. Write tests for new models
2. Document hyperparameters and expected performance
3. Validate on historical data (2013-2025)
4. Report backtest vs. forward test degradation
5. Use Optuna for hyperparameter tuning

## References

- **AI Strategy**: `docs/AI_STRATEGY_INTEGRATION.md`
- **Regime Backtesting**: `docs/CRYPTO_REGIME_BACKTESTING.md`
- **Transformer Research**: `docs/TRANSFORMER_TIME_SERIES_ANALYSIS.md`

## License

MIT License - See LICENSE file for details

---

**Status**: Active Development (Phase 2 - Regime Detection)  
**Maintainer**: FKS Trading Platform Team  
**Last Updated**: October 2025
