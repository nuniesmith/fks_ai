"""
Probabilistic metrics for time-series forecasting evaluation

Implements:
- CRPS (Continuous Ranked Probability Score): Measures probabilistic forecast accuracy
- MASE (Mean Absolute Scaled Error): Scale-independent accuracy metric
"""

import numpy as np
from typing import Union, Optional
import warnings


def CRPS(
    observations: np.ndarray,
    forecasts: np.ndarray,
    quantiles: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Continuous Ranked Probability Score (CRPS).
    
    CRPS measures the difference between the forecast distribution and the observation.
    Lower values indicate better forecast accuracy. Benchmark: <0.3 is good for trading.
    
    Args:
        observations: Actual observed values, shape (n_samples,)
        forecasts: Forecast samples or quantiles, shape (n_samples, n_quantiles)
        quantiles: Optional quantile levels, shape (n_quantiles,)
        
    Returns:
        CRPS score (lower is better, target <0.3)
        
    Example:
        >>> obs = np.array([100, 102, 105])
        >>> forecasts = np.random.normal(100, 5, (3, 100))  # 100 samples per forecast
        >>> crps = CRPS(obs, forecasts)
        >>> print(f"CRPS: {crps:.4f}")
    """
    if forecasts.ndim == 1:
        # Point forecasts - convert to simple CRPS
        return np.mean(np.abs(observations - forecasts))
    
    if forecasts.shape[0] != observations.shape[0]:
        raise ValueError(
            f"Shape mismatch: observations {observations.shape[0]} "
            f"vs forecasts {forecasts.shape[0]}"
        )
    
    # Calculate CRPS for each observation
    crps_values = []
    for i, obs in enumerate(observations):
        forecast_samples = forecasts[i]
        
        # CRPS = E[|F - obs|] - 0.5 * E[|F - F'|]
        # where F and F' are independent samples from forecast distribution
        term1 = np.mean(np.abs(forecast_samples - obs))
        
        # Efficient pairwise difference calculation
        n_samples = len(forecast_samples)
        if n_samples > 1:
            # Broadcast for pairwise differences
            diffs = np.abs(
                forecast_samples[:, np.newaxis] - forecast_samples[np.newaxis, :]
            )
            term2 = 0.5 * np.mean(diffs)
        else:
            term2 = 0.0
        
        crps_values.append(term1 - term2)
    
    return float(np.mean(crps_values))


def MASE(
    observations: np.ndarray,
    forecasts: np.ndarray,
    training_series: Optional[np.ndarray] = None,
    seasonality: int = 1
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    MASE is a scale-independent metric that compares forecast error to naive forecast.
    Values <1 indicate better than naive, <0.5 is very good for trading.
    
    Args:
        observations: Actual observed values, shape (n_samples,)
        forecasts: Forecast values, shape (n_samples,)
        training_series: Historical training data for scaling (optional)
        seasonality: Seasonal period (1 for non-seasonal, 7 for weekly, etc.)
        
    Returns:
        MASE score (lower is better, <1 beats naive forecast)
        
    Example:
        >>> obs = np.array([100, 102, 105, 103])
        >>> preds = np.array([101, 103, 104, 104])
        >>> mase = MASE(obs, preds)
        >>> print(f"MASE: {mase:.4f}")
    """
    if forecasts.shape != observations.shape:
        raise ValueError(
            f"Shape mismatch: observations {observations.shape} "
            f"vs forecasts {forecasts.shape}"
        )
    
    # Mean absolute error of forecasts
    mae = np.mean(np.abs(observations - forecasts))
    
    # Scale by naive forecast error
    if training_series is not None and len(training_series) > seasonality:
        # Use training data for scaling
        naive_errors = np.abs(np.diff(training_series, n=seasonality))
        scale = np.mean(naive_errors)
    elif len(observations) > seasonality:
        # Use in-sample naive forecast
        naive_errors = np.abs(np.diff(observations, n=seasonality))
        scale = np.mean(naive_errors)
    else:
        warnings.warn(
            "Insufficient data for MASE scaling, using MAE instead",
            UserWarning
        )
        return mae
    
    if scale == 0:
        warnings.warn("Zero scale in MASE, returning MAE", UserWarning)
        return mae
    
    return mae / scale


def calculate_probabilistic_metrics(
    observations: np.ndarray,
    forecasts: np.ndarray,
    forecast_samples: Optional[np.ndarray] = None,
    training_series: Optional[np.ndarray] = None
) -> dict:
    """
    Calculate comprehensive probabilistic forecast metrics.
    
    Args:
        observations: Actual observed values
        forecasts: Point forecasts (mean/median)
        forecast_samples: Full distribution samples for CRPS (optional)
        training_series: Historical data for MASE scaling (optional)
        
    Returns:
        Dictionary with metrics:
        - crps: Continuous Ranked Probability Score
        - mase: Mean Absolute Scaled Error
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - mape: Mean Absolute Percentage Error
        
    Example:
        >>> obs = np.array([100, 102, 105])
        >>> preds = np.array([101, 103, 104])
        >>> samples = np.random.normal(preds[:, None], 2, (3, 100))
        >>> metrics = calculate_probabilistic_metrics(obs, preds, samples)
        >>> print(f"CRPS: {metrics['crps']:.3f}, MASE: {metrics['mase']:.3f}")
    """
    metrics = {}
    
    # CRPS (if samples provided)
    if forecast_samples is not None:
        try:
            metrics['crps'] = CRPS(observations, forecast_samples)
        except Exception as e:
            warnings.warn(f"CRPS calculation failed: {e}", UserWarning)
            metrics['crps'] = None
    
    # MASE
    try:
        metrics['mase'] = MASE(observations, forecasts, training_series)
    except Exception as e:
        warnings.warn(f"MASE calculation failed: {e}", UserWarning)
        metrics['mase'] = None
    
    # Standard metrics
    metrics['mae'] = float(np.mean(np.abs(observations - forecasts)))
    metrics['rmse'] = float(np.sqrt(np.mean((observations - forecasts) ** 2)))
    
    # MAPE (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((observations - forecasts) / observations)) * 100
        metrics['mape'] = float(mape) if np.isfinite(mape) else None
    
    return metrics


# Benchmark targets for trading systems
BENCHMARK_TARGETS = {
    'crps': 0.3,  # Target CRPS <0.3 for good probabilistic forecasts
    'mase': 0.5,  # Target MASE <0.5 for very good forecasts
    'mae': None,  # Asset-dependent
    'rmse': None,  # Asset-dependent
    'mape': 5.0,  # Target MAPE <5% for trading
}
