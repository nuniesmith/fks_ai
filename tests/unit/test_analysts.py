"""
Unit tests for analyst agents

Tests technical, sentiment, macro, and risk analysts
"""
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_technical_analyst_basic(sample_market_data, mock_ollama_response):
    """Test technical analyst with basic market data"""
    from src.services.ai.src.agents.analysts.technical import analyze_technical
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(
        symbol='BTCUSDT',
        market_data=sample_market_data
    )

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke = mock_ollama_response
        mock_create.return_value = mock_agent

        result = await analyze_technical(state)

        assert 'BUY' in result or 'SELL' in result or 'HOLD' in result
        assert 'confidence' in result.lower() or '%' in result


@pytest.mark.asyncio
async def test_sentiment_analyst_basic(sample_market_data, mock_ollama_response):
    """Test sentiment analyst"""
    from src.services.ai.src.agents.analysts.sentiment import analyze_sentiment
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(
        symbol='BTCUSDT',
        market_data=sample_market_data
    )

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke = mock_ollama_response
        mock_create.return_value = mock_agent

        result = await analyze_sentiment(state)

        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.asyncio
async def test_macro_analyst_basic(sample_market_data, mock_ollama_response):
    """Test macro analyst"""
    from src.services.ai.src.agents.analysts.macro import analyze_macro
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(
        symbol='BTCUSDT',
        market_data=sample_market_data
    )

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke = mock_ollama_response
        mock_create.return_value = mock_agent

        result = await analyze_macro(state)

        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.asyncio
async def test_risk_analyst_basic(sample_market_data, mock_ollama_response):
    """Test risk analyst"""
    from src.services.ai.src.agents.analysts.risk import analyze_risk
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(
        symbol='BTCUSDT',
        market_data=sample_market_data
    )

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke = mock_ollama_response
        mock_create.return_value = mock_agent

        result = await analyze_risk(state)

        assert isinstance(result, str)
        assert 'position' in result.lower() or 'risk' in result.lower()


@pytest.mark.asyncio
async def test_all_analysts_produce_output(sample_market_data, mock_ollama_response):
    """Test that all analysts produce valid output"""
    from src.services.ai.src.agents.analysts.macro import analyze_macro
    from src.services.ai.src.agents.analysts.risk import analyze_risk
    from src.services.ai.src.agents.analysts.sentiment import analyze_sentiment
    from src.services.ai.src.agents.analysts.technical import analyze_technical
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(
        symbol='BTCUSDT',
        market_data=sample_market_data
    )

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke = mock_ollama_response
        mock_create.return_value = mock_agent

        analysts = [
            analyze_technical,
            analyze_sentiment,
            analyze_macro,
            analyze_risk
        ]

        for analyst in analysts:
            result = await analyst(state)
            assert isinstance(result, str)
            assert len(result) > 10  # Meaningful output
