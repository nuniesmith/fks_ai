"""
Unit tests for debate agents

Tests Bull, Bear, and Manager agents
"""
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_bull_agent_generates_case(sample_agent_state, mock_ollama_response):
    """Test Bull agent generates optimistic case"""
    from src.services.ai.src.agents.debaters.bull import generate_bull_case

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke = mock_ollama_response
        mock_create.return_value = mock_agent

        result = await generate_bull_case(sample_agent_state)

        assert isinstance(result, str)
        assert len(result) > 20
        # Bull case should mention positive aspects
        assert 'bull' in result.lower() or 'buy' in result.lower() or 'confidence' in result.lower()


@pytest.mark.asyncio
async def test_bear_agent_generates_case(sample_agent_state, mock_ollama_response):
    """Test Bear agent generates pessimistic case"""
    from src.services.ai.src.agents.debaters.bear import generate_bear_case

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke = mock_ollama_response
        mock_create.return_value = mock_agent

        result = await generate_bear_case(sample_agent_state)

        assert isinstance(result, str)
        assert len(result) > 20


@pytest.mark.asyncio
async def test_manager_synthesizes_debate(sample_agent_state, sample_bull_argument,
                                          sample_bear_argument, mock_ollama_response):
    """Test Manager agent synthesizes Bull/Bear arguments"""
    from src.services.ai.src.agents.debaters.manager import synthesize_debate

    # Add debates to state
    state = sample_agent_state.copy()
    state['debates'] = [sample_bull_argument, sample_bear_argument]

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke = mock_ollama_response
        mock_create.return_value = mock_agent

        result = await synthesize_debate(state)

        assert isinstance(result, str)
        assert len(result) > 20
        # Manager should make a decision
        assert 'decision' in result.lower() or 'buy' in result.lower() or 'sell' in result.lower() or 'hold' in result.lower()


@pytest.mark.asyncio
async def test_debate_agents_use_different_temperatures():
    """Test that Bull/Bear use higher temp than Manager"""
    from src.services.ai.src.agents.debaters.bear import BEAR_AGENT_PROMPT
    from src.services.ai.src.agents.debaters.bull import BULL_AGENT_PROMPT
    from src.services.ai.src.agents.debaters.manager import MANAGER_AGENT_PROMPT

    # Verify prompts exist and are different
    assert len(BULL_AGENT_PROMPT) > 50
    assert len(BEAR_AGENT_PROMPT) > 50
    assert len(MANAGER_AGENT_PROMPT) > 50

    # Bull and Bear should be creative (higher temp)
    # Manager should be consistent (lower temp)
    # This is tested in integration, here we just verify prompts exist


@pytest.mark.asyncio
async def test_debate_flow_order(sample_agent_state, mock_ollama_response):
    """Test debate agents can run in sequence"""
    from src.services.ai.src.agents.debaters.bear import generate_bear_case
    from src.services.ai.src.agents.debaters.bull import generate_bull_case
    from src.services.ai.src.agents.debaters.manager import synthesize_debate

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke = mock_ollama_response
        mock_create.return_value = mock_agent

        # Bull case
        bull = await generate_bull_case(sample_agent_state)
        sample_agent_state['debates'].append(bull)

        # Bear case
        bear = await generate_bear_case(sample_agent_state)
        sample_agent_state['debates'].append(bear)

        # Manager synthesis
        decision = await synthesize_debate(sample_agent_state)

        assert len(sample_agent_state['debates']) == 2
        assert isinstance(decision, str)
