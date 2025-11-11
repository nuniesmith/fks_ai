"""
TimeCopilot: Agentic Time-Series Forecasting Wrapper

Combines multiple forecasting models (Lag-Llama, TimesFM) with agent-based
decision making to select the best model and parameters for each forecast.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass
from enum import Enum

from .lag_llama import LagLlamaForecaster, LagLlamaConfig
from .metrics import calculate_probabilistic_metrics, BENCHMARK_TARGETS


class ModelType(Enum):
    """Available forecasting models."""
    LAG_LLAMA = "lag_llama"
    TIMESFM = "timesfm"  # Google's foundation model
    HYBRID = "hybrid"  # Ensemble of multiple models


@dataclass
class ForecastConfig:
    """Configuration for TimeCopilot forecasting."""
    model_type: ModelType = ModelType.LAG_LLAMA
    context_length: int = 32
    prediction_length: int = 24
    num_samples: int = 100
    confidence_threshold: float = 0.6  # Minimum confidence for accepting forecast
    use_ensemble: bool = False  # Combine multiple models
    device: str = "cpu"


class TimeCopilot:
    """
    Agentic wrapper for time-series forecasting models.
    
    TimeCopilot acts as an intelligent agent that:
    1. Analyzes input time series characteristics
    2. Selects the best model(s) for the data
    3. Generates probabilistic forecasts
    4. Evaluates forecast quality
    5. Provides confidence scores and recommendations
    
    This implements the "agentic" approach where the system makes intelligent
    decisions about model selection and parameters rather than blindly applying
    a single model.
    
    Example:
        >>> config = ForecastConfig(
        ...     model_type=ModelType.LAG_LLAMA,
        ...     prediction_length=24,
        ...     confidence_threshold=0.6
        ... )
        >>> copilot = TimeCopilot(config)
        >>> data = np.random.randn(100)
        >>> result = copilot.forecast(data)
        >>> if result['confidence'] > 0.6:
        ...     print(f"High-confidence forecast: {result['mean']}")
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        """
        Initialize TimeCopilot.
        
        Args:
            config: Forecast configuration (uses defaults if None)
        """
        self.config = config or ForecastConfig()
        self.models = self._initialize_models()
        self.forecast_history: List[Dict[str, Any]] = []
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all available forecasting models."""
        models = {}
        
        # Lag-Llama model
        lag_llama_config = LagLlamaConfig(
            context_length=self.config.context_length,
            prediction_length=self.config.prediction_length,
            num_samples=self.config.num_samples,
            device=self.config.device,
        )
        models['lag_llama'] = LagLlamaForecaster(lag_llama_config)
        
        # TODO: Add TimesFM when available
        # models['timesfm'] = TimesFMForecaster(...)
        
        return models
    
    def analyze_series(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze time series characteristics to guide model selection.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with series characteristics:
            - trend: "up", "down", or "stable"
            - volatility: float (std dev of returns)
            - stationarity: bool
            - seasonality_detected: bool
            - recommended_model: str
        """
        analysis = {}
        
        # Trend detection
        if len(data) > 2:
            slope = np.polyfit(np.arange(len(data)), data, 1)[0]
            if abs(slope) < np.std(data) * 0.1:
                analysis['trend'] = "stable"
            elif slope > 0:
                analysis['trend'] = "up"
            else:
                analysis['trend'] = "down"
        else:
            analysis['trend'] = "unknown"
        
        # Volatility
        if len(data) > 1:
            returns = np.diff(data) / (data[:-1] + 1e-10)
            analysis['volatility'] = float(np.std(returns))
        else:
            analysis['volatility'] = 0.0
        
        # Simple stationarity check (augmented Dickey-Fuller would be better)
        analysis['stationarity'] = analysis['volatility'] < 1.0
        
        # TODO: Proper seasonality detection
        analysis['seasonality_detected'] = False
        
        # Recommend model based on characteristics
        if analysis['volatility'] > 0.5:
            # High volatility - Lag-Llama handles uncertainty better
            analysis['recommended_model'] = ModelType.LAG_LLAMA.value
        else:
            # Lower volatility - could use either
            analysis['recommended_model'] = ModelType.LAG_LLAMA.value
        
        return analysis
    
    def forecast(
        self,
        historical_data: np.ndarray,
        model_type: Optional[ModelType] = None,
        return_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Generate intelligent forecasts with confidence scoring.
        
        Args:
            historical_data: Historical time series data
            model_type: Override model selection (uses auto-selection if None)
            return_analysis: Include series analysis in results
            
        Returns:
            Dictionary containing:
            - mean: Mean forecast
            - median: Median forecast
            - quantiles: Forecast quantiles
            - samples: All probabilistic samples
            - confidence: Confidence score (0-1)
            - model_used: Which model was used
            - analysis: Series analysis (if return_analysis=True)
            - meets_threshold: Bool indicating if confidence > threshold
        """
        # Analyze the time series
        analysis = self.analyze_series(historical_data)
        
        # Select model
        if model_type is None:
            # Agentic model selection based on analysis
            selected_model = ModelType(analysis['recommended_model'])
        else:
            selected_model = model_type or self.config.model_type
        
        # Generate forecast
        if selected_model == ModelType.LAG_LLAMA:
            forecast = self.models['lag_llama'].forecast(
                historical_data,
                prediction_length=self.config.prediction_length,
                num_samples=self.config.num_samples,
            )
        # TODO: Add other models
        # elif selected_model == ModelType.TIMESFM:
        #     forecast = self.models['timesfm'].forecast(...)
        else:
            raise ValueError(f"Model {selected_model} not implemented")
        
        # Calculate confidence score
        confidence = self._calculate_confidence(forecast, analysis)
        
        # Compile results
        result = {
            **forecast,
            'confidence': confidence,
            'model_used': selected_model.value,
            'meets_threshold': confidence >= self.config.confidence_threshold,
        }
        
        if return_analysis:
            result['analysis'] = analysis
        
        # Store in history
        self.forecast_history.append({
            'timestamp': np.datetime64('now'),
            'model': selected_model.value,
            'confidence': confidence,
        })
        
        return result
    
    def _calculate_confidence(
        self,
        forecast: Dict[str, np.ndarray],
        analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence score for forecast.
        
        Confidence is based on:
        1. Forecast uncertainty (wider intervals = lower confidence)
        2. Series characteristics (high volatility = lower confidence)
        3. Historical accuracy (if available)
        
        Args:
            forecast: Forecast results with quantiles
            analysis: Series analysis
            
        Returns:
            Confidence score between 0 and 1
        """
        scores = []
        
        # 1. Uncertainty-based confidence
        if 'quantiles' in forecast:
            # Narrow prediction intervals = higher confidence
            q10 = forecast['quantiles']['q10']
            q90 = forecast['quantiles']['q90']
            mean = forecast['mean']
            
            # Normalized interval width (as fraction of mean)
            interval_width = np.mean((q90 - q10) / (np.abs(mean) + 1e-10))
            uncertainty_score = 1.0 / (1.0 + interval_width)
            scores.append(uncertainty_score)
        
        # 2. Volatility-based confidence
        volatility = analysis.get('volatility', 0.5)
        volatility_score = 1.0 / (1.0 + volatility)
        scores.append(volatility_score)
        
        # 3. Stationarity bonus
        if analysis.get('stationarity', False):
            scores.append(0.8)
        
        # Combine scores (weighted average)
        if scores:
            confidence = np.mean(scores)
        else:
            confidence = 0.5  # Neutral default
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def evaluate_forecast(
        self,
        observations: np.ndarray,
        forecast: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate forecast quality against actual observations.
        
        Args:
            observations: Actual observed values
            forecast: Forecast dictionary from forecast()
            
        Returns:
            Dictionary with evaluation metrics (CRPS, MASE, etc.)
        """
        metrics = calculate_probabilistic_metrics(
            observations=observations,
            forecasts=forecast['mean'],
            forecast_samples=forecast.get('samples'),
        )
        
        # Add benchmark comparisons
        metrics['passes_crps_benchmark'] = (
            metrics.get('crps', float('inf')) < BENCHMARK_TARGETS['crps']
        )
        metrics['passes_mase_benchmark'] = (
            metrics.get('mase', float('inf')) < BENCHMARK_TARGETS['mase']
        )
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of forecast performance over history.
        
        Returns:
            Dictionary with:
            - total_forecasts: Number of forecasts made
            - avg_confidence: Average confidence score
            - high_confidence_rate: % of forecasts meeting threshold
        """
        if not self.forecast_history:
            return {
                'total_forecasts': 0,
                'avg_confidence': 0.0,
                'high_confidence_rate': 0.0,
            }
        
        confidences = [f['confidence'] for f in self.forecast_history]
        high_conf = sum(
            1 for c in confidences 
            if c >= self.config.confidence_threshold
        )
        
        return {
            'total_forecasts': len(self.forecast_history),
            'avg_confidence': float(np.mean(confidences)),
            'high_confidence_rate': high_conf / len(self.forecast_history),
        }
