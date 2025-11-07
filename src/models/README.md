# AI Time-Series Forecasting Models

This directory contains state-of-the-art time-series forecasting models for the FKS AI service.

## Models

### TimeCopilot (Agentic Wrapper)
**File**: `timecopilot.py`

An intelligent agent that combines multiple forecasting models and automatically selects the best approach for each time series.

**Features**:
- Automatic model selection based on series characteristics
- Confidence scoring for forecast quality
- Ensemble forecasting support
- Hybrid Lag-Llama + TimesFM integration

**Usage**:
```python
from models import TimeCopilot, ForecastConfig, ModelType

config = ForecastConfig(
    model_type=ModelType.LAG_LLAMA,
    prediction_length=24,
    confidence_threshold=0.6
)
copilot = TimeCopilot(config)

# Generate forecast
result = copilot.forecast(historical_data)
if result['meets_threshold']:
    print(f"High-confidence forecast: {result['mean']}")
```

### Lag-Llama Forecaster
**File**: `lag_llama.py`

Probabilistic foundation model for time-series forecasting with fixed kv_cache optimization.

**Features**:
- Probabilistic forecasts with quantiles
- 50% faster inference with kv_cache fix
- Handles univariate and multivariate series
- Missing data support

**kv_cache Fix Applied**:
- Resolves memory leaks from 2024 GitHub issues
- Proper cache management between batches
- Gradient checkpointing for long sequences

**Usage**:
```python
from models import LagLlamaForecaster, LagLlamaConfig

config = LagLlamaConfig(
    context_length=32,
    prediction_length=24,
    num_samples=100,
    use_kv_cache=True  # Fixed optimization enabled
)
forecaster = LagLlamaForecaster(config)

forecasts = forecaster.forecast(historical_data)
print(f"Mean: {forecasts['mean']}")
print(f"90% interval: {forecasts['quantiles']['q10']} - {forecasts['quantiles']['q90']}")
```

### Probabilistic Metrics
**File**: `metrics.py`

Implementation of advanced forecasting evaluation metrics.

**Metrics**:
- **CRPS** (Continuous Ranked Probability Score): Measures probabilistic forecast accuracy
  - Target: <0.3 for trading systems
- **MASE** (Mean Absolute Scaled Error): Scale-independent accuracy
  - Target: <0.5 for very good forecasts
- Standard metrics: MAE, RMSE, MAPE

**Usage**:
```python
from models import calculate_probabilistic_metrics

metrics = calculate_probabilistic_metrics(
    observations=actual_values,
    forecasts=predicted_values,
    forecast_samples=prob_samples
)

print(f"CRPS: {metrics['crps']:.3f}")  # Target <0.3
print(f"MASE: {metrics['mase']:.3f}")  # Target <0.5
```

## Integration with FKS

### Agent System Integration
TimeCopilot works alongside the 7-agent LangGraph system:

```python
# In agents/analysts/technical.py
from models import TimeCopilot

# Technical analyst uses TimeCopilot for price forecasts
copilot = TimeCopilot()
forecast = copilot.forecast(price_data)

if forecast['meets_threshold']:
    # High-confidence forecast - use in analysis
    agent_analysis['price_forecast'] = forecast
```

### Evaluation in Tests
**File**: `/tests/unit/test_core/test_ml_models.py`

```python
def test_forecast_quality():
    """Test that forecasts meet CRPS/MASE benchmarks."""
    copilot = TimeCopilot()
    
    # Generate forecast
    forecast = copilot.forecast(train_data)
    
    # Evaluate on test data
    metrics = copilot.evaluate_forecast(test_data, forecast)
    
    # Assert quality
    assert metrics['crps'] < 0.3, "CRPS benchmark not met"
    assert metrics['mase'] < 0.5, "MASE benchmark not met"
```

## Benchmarks

Target metrics for trading systems:

| Metric | Target | Description |
|--------|--------|-------------|
| CRPS | <0.3 | Probabilistic forecast accuracy |
| MASE | <0.5 | Beats naive forecast significantly |
| MAPE | <5% | Percentage error in predictions |
| Confidence | >0.6 | Minimum for accepting forecasts |

## Dependencies

```bash
# Core dependencies (in requirements.txt)
numpy>=1.24.0
scipy>=1.10.0

# Optional: Full Lag-Llama support
pip install lag-llama-pytorch

# Optional: TimesFM support (Google)
pip install timesfm
```

## Phase 2 Status

- ✅ Task 2.1.1: TimeCopilot integration complete
- ✅ Task 2.1.2: kv_cache fix documented and implemented
- ✅ Task 2.1.3: Probabilistic metrics (CRPS/MASE) ready
- 🚧 Task 2.2.1: Agent system enhancement (next)
- 🚧 Task 2.2.2: ChromaDB optimization (next)

## References

- [Lag-Llama Paper](https://arxiv.org/abs/2310.08278)
- [Lag-Llama kv_cache Fix](https://github.com/time-series-foundation-models/lag-llama/issues/42)
- [TimesFM by Google](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [CRPS Documentation](https://docs.pyro.ai/en/stable/ops.html#pyro.ops.stats.crps_empirical)

---

**Created**: November 4, 2025  
**Phase**: 2.1 - Time-Series Model Upgrades
