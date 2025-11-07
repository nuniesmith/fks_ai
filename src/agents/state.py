"""
Multi-Agent Trading System - Agent State Schema

Defines the shared state structure for all agents in the LangGraph pipeline.
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """
    Shared state passed between agents in the trading graph.

    Attributes:
        messages: Conversation history with automated message management
        market_data: Current market data (OHLCV, indicators, features)
        signals: Generated trading signals from various agents
        debates: Bull/Bear adversarial arguments
        memory: Retrieved context from ChromaDB
        regime: Current market regime (bull/bear/sideways)
        confidence: Overall confidence score (0-1)
        final_decision: Manager's synthesized decision
        timestamp: Processing timestamp
        symbol: Trading pair being analyzed
    """
    messages: Annotated[list[dict[str, Any]], "add_messages"]
    market_data: dict[str, Any]
    signals: list[dict[str, Any]]
    debates: list[str]
    memory: list[str]
    regime: Optional[str]
    confidence: float
    final_decision: Optional[dict[str, Any]]
    timestamp: str
    symbol: str


def create_initial_state(symbol: str, market_data: dict[str, Any]) -> AgentState:
    """
    Create initial state for graph execution.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        market_data: Market data dictionary with OHLCV and indicators

    Returns:
        Initialized AgentState
    """
    return AgentState(
        messages=[],
        market_data=market_data,
        signals=[],
        debates=[],
        memory=[],
        regime=None,
        confidence=0.0,
        final_decision=None,
        timestamp=datetime.now().isoformat(),
        symbol=symbol
    )
