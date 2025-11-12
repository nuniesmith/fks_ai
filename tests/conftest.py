"""
Pytest fixtures for AI service tests

Provides reusable test data and mocked components
"""
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def sample_market_data() -> dict[str, Any]:
    """Sample OHLCV market data for testing"""
    return {
        'symbol': 'BTCUSDT',
        'timeframe': '1h',
        'timestamp': datetime(2024, 10, 30, 12, 0, 0),
        'open': 68000.0,
        'high': 68500.0,
        'low': 67800.0,
        'close': 68200.0,
        'volume': 1500.5,
        'technical': {
            'rsi': 58.5,
            'macd': 150.2,
            'macd_signal': 120.5,
            'bb_upper': 69000.0,
            'bb_middle': 68000.0,
            'bb_lower': 67000.0,
            'atr': 400.0,
            'adx': 25.3
        },
        'regime': 'bull'
    }


@pytest.fixture
def sample_analyst_insights() -> list[str]:
    """Sample analyst responses for testing"""
    return [
        "Technical: RSI at 58.5 shows neutral momentum. MACD bullish crossover. Price above BB middle. Recommendation: BUY with 70% confidence.",
        "Sentiment: Fear & Greed Index at 65 (greed). Social volume increasing. Whale accumulation detected. Recommendation: BUY with 65% confidence.",
        "Macro: Fed maintains rates. USD weakening. Correlation with gold strengthening. Recommendation: BUY with 60% confidence.",
        "Risk: VaR at 2.1%, well within limits. Sharpe ratio 1.8. Max position size: 8%. Recommendation: APPROVE with conservative sizing."
    ]


@pytest.fixture
def sample_bull_argument() -> str:
    """Sample bull agent output"""
    return """
    BULL CASE for BTCUSDT:

    1. Technical momentum strong: MACD crossover, RSI neutral, price above key moving averages
    2. Sentiment turning positive: Fear & Greed at 65, institutional accumulation
    3. Macro tailwinds: Weakening USD, Fed pause supportive for risk assets
    4. Risk-reward favorable: Low volatility, tight stops possible

    RECOMMENDATION: BUY
    CONFIDENCE: 75%
    ENTRY: $68,200
    STOP: $67,400 (-1.17%)
    TARGET: $70,000 (+2.64%)
    """


@pytest.fixture
def sample_bear_argument() -> str:
    """Sample bear agent output"""
    return """
    BEAR CASE for BTCUSDT:

    1. Overbought conditions developing: RSI approaching 60, greed levels elevated
    2. Resistance at $69,000: Multiple rejections at this level historically
    3. Volume declining: Suggests weakening buying pressure
    4. Macro uncertainty: Upcoming economic data could shift sentiment

    RECOMMENDATION: HOLD
    CONFIDENCE: 45%
    REASONING: Wait for pullback to better entry levels around $66,500
    """


@pytest.fixture
def sample_manager_decision() -> str:
    """Sample manager synthesis output"""
    return """
    MANAGER DECISION for BTCUSDT:

    After reviewing both Bull and Bear arguments:

    Bull strengths: Technical momentum confirmed, macro supportive
    Bear concerns: Valid regarding overbought and resistance levels

    SYNTHESIS: Cautious bullish stance
    - Technical setup is constructive but not compelling
    - Risk/reward acceptable with tight stops
    - Position size should be reduced due to mixed signals

    FINAL DECISION: BUY
    CONFIDENCE: 62%
    ENTRY: $68,200
    STOP: $67,400
    TARGET: $69,800
    POSITION SIZE: 5% (reduced from standard 8%)
    """


@pytest.fixture
def sample_agent_state(sample_market_data, sample_analyst_insights) -> dict[str, Any]:
    """Complete AgentState for testing"""
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(
        symbol=sample_market_data['symbol'],
        market_data=sample_market_data
    )

    # Add analyst messages
    for insight in sample_analyst_insights:
        state['messages'].append({
            'role': 'assistant',
            'content': insight
        })

    return state


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama LLM response"""
    async def _mock_response(prompt: str, **kwargs) -> str:
        """Generate contextual mock response based on prompt"""
        if 'technical' in prompt.lower():
            return "Technical analysis: BUY signal with 70% confidence. RSI neutral, MACD bullish."
        elif 'sentiment' in prompt.lower():
            return "Sentiment analysis: BUY signal with 65% confidence. Greed levels moderate."
        elif 'macro' in prompt.lower():
            return "Macro analysis: BUY signal with 60% confidence. Fed supportive."
        elif 'risk' in prompt.lower():
            return "Risk analysis: APPROVE with 8% position size. VaR acceptable."
        elif 'bull' in prompt.lower():
            return "BULL CASE: Strong momentum, BUY with 75% confidence."
        elif 'bear' in prompt.lower():
            return "BEAR CASE: Overbought, HOLD with 45% confidence."
        elif 'manager' in prompt.lower() or 'synthesis' in prompt.lower():
            return "MANAGER DECISION: BUY with 62% confidence, reduced position size."
        else:
            return "Generic LLM response for testing."

    return _mock_response


@pytest.fixture
def mock_chromadb_client():
    """Mock ChromaDB client"""
    mock_client = MagicMock()
    mock_collection = MagicMock()

    # Mock collection methods
    mock_collection.add = MagicMock()
    mock_collection.query = MagicMock(return_value={
        'ids': [['insight_1', 'insight_2']],
        'documents': [['Past decision 1', 'Past decision 2']],
        'metadatas': [[
            {'symbol': 'BTCUSDT', 'confidence': 0.65, 'timestamp': '2024-10-29T12:00:00'},
            {'symbol': 'BTCUSDT', 'confidence': 0.70, 'timestamp': '2024-10-28T12:00:00'}
        ]],
        'distances': [[0.15, 0.22]]
    })
    mock_collection.get = MagicMock(return_value={
        'ids': ['insight_1'],
        'documents': ['Past decision'],
        'metadatas': [{'symbol': 'BTCUSDT', 'confidence': 0.65}]
    })

    mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)

    return mock_client


@pytest.fixture
def mock_agent():
    """Mock LangChain agent"""
    agent = AsyncMock()
    agent.ainvoke = AsyncMock(return_value="Mock agent response")
    return agent


@pytest.fixture
def sample_signal_dict() -> dict[str, Any]:
    """Sample trading signal for testing"""
    return {
        'symbol': 'BTCUSDT',
        'action': 'BUY',
        'confidence': 0.62,
        'entry_price': 68200.0,
        'stop_loss': 67400.0,
        'take_profit': 69800.0,
        'position_size': 5.0,
        'risk_reward_ratio': 2.28,
        'timestamp': datetime(2024, 10, 30, 12, 0, 0),
        'reasoning': 'Cautious bullish stance with reduced position size'
    }


@pytest.fixture
def mock_trading_graph():
    """Mock LangGraph StateGraph"""
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(return_value={
        'messages': [],
        'final_decision': 'BUY with 62% confidence',
        'confidence': 0.62,
        'regime': 'bull'
    })
    return graph
