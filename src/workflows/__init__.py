"""
FKS Market Screening Workflows

Market-specific screening and signal generation workflows.
Each workflow combines data fetching, rule validation, and signal generation
for its specific asset class.
"""

from .base import BaseWorkflow, WorkflowResult, SignalQuality, SignalCandidate, SignalDirection
from .bitcoin import BitcoinWorkflow, DynamicRisk, ConfidenceLevel, TimingMetrics
from .crypto_spot import CryptoSpotWorkflow
from .crypto_perps import CryptoPerpsWorkflow
from .forex import ForexWorkflow
from .futures import FuturesWorkflow
from .stocks import StocksWorkflow

# Workflow registry for easy access
WORKFLOW_REGISTRY = {
    "bitcoin": BitcoinWorkflow,  # Primary focus
    "crypto_spot": CryptoSpotWorkflow,
    "crypto_perps": CryptoPerpsWorkflow,
    "forex": ForexWorkflow,
    "futures": FuturesWorkflow,
    "stocks": StocksWorkflow,
}


def get_workflow(market_type: str) -> BaseWorkflow:
    """
    Factory function to get appropriate workflow for market type.
    
    Args:
        market_type: One of "bitcoin", "crypto_spot", "crypto_perps", "forex", "futures", "stocks"
        
    Returns:
        Instantiated workflow for the market type
        
    Raises:
        ValueError: If market type is not supported
    """
    workflow_class = WORKFLOW_REGISTRY.get(market_type.lower())
    if not workflow_class:
        raise ValueError(
            f"Unknown market type: {market_type}. "
            f"Supported: {list(WORKFLOW_REGISTRY.keys())}"
        )
    return workflow_class()


__all__ = [
    # Base classes
    "BaseWorkflow",
    "WorkflowResult",
    "SignalQuality",
    "SignalCandidate",
    "SignalDirection",
    # Bitcoin specific
    "BitcoinWorkflow",
    "DynamicRisk",
    "ConfidenceLevel",
    "TimingMetrics",
    # Workflows
    "CryptoSpotWorkflow",
    "CryptoPerpsWorkflow",
    "ForexWorkflow",
    "FuturesWorkflow",
    "StocksWorkflow",
    # Factory
    "get_workflow",
    "WORKFLOW_REGISTRY",
]
