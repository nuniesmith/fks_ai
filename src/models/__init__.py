"""
Time-series forecasting models for FKS AI Service

This module provides wrappers for state-of-the-art time-series models:
- Lag-Llama: Probabilistic univariate/multivariate forecasting
- TimesFM (Google): Foundation model for time-series
- TimeCopilot: Agentic wrapper combining multiple models
"""

from .timecopilot import TimeCopilot, ForecastConfig
from .lag_llama import LagLlamaForecaster
from .metrics import CRPS, MASE, calculate_probabilistic_metrics

__all__ = [
    'TimeCopilot',
    'ForecastConfig',
    'LagLlamaForecaster',
    'CRPS',
    'MASE',
    'calculate_probabilistic_metrics',
]
