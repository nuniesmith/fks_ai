"""
Unit tests for graph nodes

Tests analyst runner, debate, manager, reflection, and conditional logic
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_run_analysts_executes_all(sample_agent_state, mock_ollama_response):
    """Test run_analysts executes all 4 analysts"""
    from src.services.ai.src.graph.nodes import run_analysts

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke = mock_ollama_response
        mock_create.return_value = mock_agent

        result_state = await run_analysts(sample_agent_state)

        # Should have added 4 analyst messages
        assert len(result_state['messages']) > len(sample_agent_state['messages'])


@pytest.mark.asyncio
async def test_run_analysts_parallel_execution(sample_agent_state, mock_ollama_response):
    """Test analysts run in parallel via asyncio.gather"""
    from src.services.ai.src.graph.nodes import run_analysts

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()

        # Track invocations
        call_count = 0
        async def counting_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return f"Analyst {call_count} response"

        mock_agent.ainvoke = counting_response
        mock_create.return_value = mock_agent

        await run_analysts(sample_agent_state)

        # All 4 analysts should have been called
        assert call_count == 4


@pytest.mark.asyncio
async def test_debate_node_runs_bull_and_bear(sample_agent_state, mock_ollama_response):
    """Test debate_node executes Bull and Bear agents"""
    from src.services.ai.src.graph.nodes import debate_node

    # Add analyst insights to state
    sample_agent_state['messages'] = [
        {'role': 'assistant', 'content': 'Technical: BUY signal'},
        {'role': 'assistant', 'content': 'Sentiment: Positive'},
        {'role': 'assistant', 'content': 'Macro: Supportive'},
        {'role': 'assistant', 'content': 'Risk: Acceptable'}
    ]

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke = mock_ollama_response
        mock_create.return_value = mock_agent

        result_state = await debate_node(sample_agent_state)

        # Should have 2 debates (bull + bear)
        assert len(result_state['debates']) == 2


@pytest.mark.asyncio
async def test_manager_decision_node(sample_agent_state, sample_bull_argument,
                                      sample_bear_argument, mock_ollama_response):
    """Test manager_decision_node synthesizes debate"""
    from src.services.ai.src.graph.nodes import manager_decision_node

    # Setup state with debates
    sample_agent_state['debates'] = [sample_bull_argument, sample_bear_argument]

    with patch('src.services.ai.src.agents.base.create_agent') as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke = mock_ollama_response
        mock_create.return_value = mock_agent

        result_state = await manager_decision_node(sample_agent_state)

        # Should have final decision
        assert result_state['final_decision'] is not None
        assert isinstance(result_state['final_decision'], str)


@pytest.mark.asyncio
async def test_reflection_node_stores_memory(sample_agent_state, mock_chromadb_client):
    """Test reflection_node stores decision in ChromaDB"""
    from src.services.ai.src.graph.nodes import reflection_node

    # Setup state with decision
    sample_agent_state['final_decision'] = "BUY BTCUSDT with 65% confidence"
    sample_agent_state['confidence'] = 0.65

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        await reflection_node(sample_agent_state)

        # Memory should have been called to add insight
        mock_chromadb_client.get_or_create_collection().add.assert_called()


@pytest.mark.asyncio
async def test_reflection_node_queries_similar(sample_agent_state, mock_chromadb_client):
    """Test reflection_node queries similar past decisions"""
    from src.services.ai.src.graph.nodes import reflection_node

    sample_agent_state['final_decision'] = "BUY BTCUSDT with 65% confidence"
    sample_agent_state['confidence'] = 0.65

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        result_state = await reflection_node(sample_agent_state)

        # Should have queried similar decisions
        mock_chromadb_client.get_or_create_collection().query.assert_called()

        # Memory should be populated
        assert 'memory' in result_state
        assert isinstance(result_state['memory'], list)


def test_should_execute_trade_high_confidence():
    """Test conditional edge returns 'execute' for high confidence BUY"""
    from src.services.ai.src.agents.state import create_initial_state
    from src.services.ai.src.graph.nodes import should_execute_trade

    state = create_initial_state(symbol='BTCUSDT')
    state['final_decision'] = "BUY with 75% confidence"
    state['confidence'] = 0.75

    result = should_execute_trade(state)

    assert result == 'execute'


def test_should_execute_trade_low_confidence():
    """Test conditional edge returns 'skip' for low confidence"""
    from src.services.ai.src.agents.state import create_initial_state
    from src.services.ai.src.graph.nodes import should_execute_trade

    state = create_initial_state(symbol='BTCUSDT')
    state['final_decision'] = "BUY with 45% confidence"
    state['confidence'] = 0.45

    result = should_execute_trade(state)

    assert result == 'skip'


def test_should_execute_trade_hold():
    """Test conditional edge returns 'skip' for HOLD"""
    from src.services.ai.src.agents.state import create_initial_state
    from src.services.ai.src.graph.nodes import should_execute_trade

    state = create_initial_state(symbol='BTCUSDT')
    state['final_decision'] = "HOLD, wait for clarity"
    state['confidence'] = 0.80  # High confidence, but HOLD

    result = should_execute_trade(state)

    assert result == 'skip'


def test_should_execute_trade_sell_high_confidence():
    """Test conditional edge returns 'execute' for high confidence SELL"""
    from src.services.ai.src.agents.state import create_initial_state
    from src.services.ai.src.graph.nodes import should_execute_trade

    state = create_initial_state(symbol='BTCUSDT')
    state['final_decision'] = "SELL with 70% confidence"
    state['confidence'] = 0.70

    result = should_execute_trade(state)

    assert result == 'execute'
