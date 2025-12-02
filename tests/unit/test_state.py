"""
Unit tests for AgentState schema and initialization

Tests state creation, validation, and message handling
"""
from datetime import datetime
from typing import Any, Dict

import pytest


def test_create_initial_state_basic():
    """Test basic state creation with minimal params"""
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(symbol='BTCUSDT')

    assert state['messages'] == []
    assert state['market_data'] == {}
    assert state['signals'] == []
    assert state['debates'] == []
    assert state['memory'] == []
    assert state['regime'] == 'unknown'
    assert state['confidence'] == 0.0
    assert state['final_decision'] is None


def test_create_initial_state_with_market_data(sample_market_data):
    """Test state creation with full market data"""
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(
        symbol='BTCUSDT',
        market_data=sample_market_data
    )

    assert state['market_data'] == sample_market_data
    assert state['regime'] == 'bull'  # From market_data
    assert len(state['messages']) == 0


def test_state_message_appending(sample_agent_state):
    """Test adding messages to state"""
    state = sample_agent_state

    initial_count = len(state['messages'])

    state['messages'].append({
        'role': 'user',
        'content': 'Analyze BTCUSDT'
    })

    assert len(state['messages']) == initial_count + 1
    assert state['messages'][-1]['role'] == 'user'


def test_state_debate_storage():
    """Test storing debate arguments in state"""
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(symbol='BTCUSDT')

    bull_arg = "Bull argument: Strong momentum"
    bear_arg = "Bear argument: Overbought conditions"

    state['debates'].append(bull_arg)
    state['debates'].append(bear_arg)

    assert len(state['debates']) == 2
    assert 'momentum' in state['debates'][0]
    assert 'Overbought' in state['debates'][1]


def test_state_confidence_update():
    """Test updating confidence in state"""
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(symbol='BTCUSDT')

    state['confidence'] = 0.75

    assert state['confidence'] == 0.75
    assert 0.0 <= state['confidence'] <= 1.0


def test_state_final_decision_storage(sample_manager_decision):
    """Test storing manager's final decision"""
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(symbol='BTCUSDT')

    state['final_decision'] = sample_manager_decision

    assert state['final_decision'] is not None
    assert 'BUY' in state['final_decision']
    assert 'CONFIDENCE: 62%' in state['final_decision']


def test_state_memory_retrieval():
    """Test storing ChromaDB memory results"""
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(symbol='BTCUSDT')

    memory_results = [
        "Past decision 1: BUY with 70% confidence",
        "Past decision 2: SELL with 60% confidence"
    ]

    state['memory'] = memory_results

    assert len(state['memory']) == 2
    assert 'BUY' in state['memory'][0]


def test_state_regime_classification():
    """Test regime field accepts valid values"""
    from src.services.ai.src.agents.state import create_initial_state

    regimes = ['bull', 'bear', 'sideways', 'volatile', 'unknown']

    for regime in regimes:
        state = create_initial_state(symbol='BTCUSDT')
        state['regime'] = regime
        assert state['regime'] == regime


def test_state_immutability_protection():
    """Test that state dict can be modified (TypedDict is not runtime enforced)"""
    from src.services.ai.src.agents.state import create_initial_state

    state = create_initial_state(symbol='BTCUSDT')

    # TypedDict is compile-time only, runtime allows modification
    # This test verifies we can modify state as needed
    state['custom_field'] = 'test'

    assert state['custom_field'] == 'test'
