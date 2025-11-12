"""
Integration tests for Ground Truth Validator.

Tests the complete ground truth validation pipeline:
1. Historical prediction collection from ChromaDB
2. Optimal trade calculation from TimescaleDB
3. Full validation workflow
4. Multi-agent comparison

Author: FKS AI Team
Created: Oct 31, 2025
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
from evaluators.ground_truth import (
    AgentPrediction,
    GroundTruthValidator,
    OptimalTrade,
    PredictionType,
    TradeOutcome,
    ValidationResult,
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def validator():
    """Create GroundTruthValidator instance"""
    return GroundTruthValidator(
        min_confidence=0.6,
        profit_threshold=2.0,
        slippage_percent=0.1,
        fee_percent=0.1
    )


@pytest.fixture
def sample_predictions() -> list[AgentPrediction]:
    """Create sample agent predictions"""
    base_time = datetime(2024, 10, 1, 12, 0)

    return [
        AgentPrediction(
            timestamp=base_time,
            agent_name="technical_analyst",
            symbol="BTCUSDT",
            prediction=PredictionType.BULLISH,
            confidence=0.85,
            reasoning="RSI oversold, MACD bullish crossover",
            timeframe="1h",
            price_at_prediction=65000.0,
            metadata={"rsi": 25, "macd": "bullish"}
        ),
        AgentPrediction(
            timestamp=base_time + timedelta(hours=2),
            agent_name="technical_analyst",
            symbol="BTCUSDT",
            prediction=PredictionType.BEARISH,
            confidence=0.75,
            reasoning="Resistance level reached",
            timeframe="1h",
            price_at_prediction=66500.0,
            metadata={"resistance": 66500}
        ),
        AgentPrediction(
            timestamp=base_time + timedelta(hours=4),
            agent_name="technical_analyst",
            symbol="BTCUSDT",
            prediction=PredictionType.NEUTRAL,
            confidence=0.65,
            reasoning="Sideways market, no clear trend",
            timeframe="1h",
            price_at_prediction=66000.0,
            metadata={"trend": "sideways"}
        ),
    ]


@pytest.fixture
def sample_optimal_trades() -> list[OptimalTrade]:
    """Create sample optimal trades"""
    base_time = datetime(2024, 10, 1, 12, 0)

    return [
        OptimalTrade(
            entry_time=base_time + timedelta(minutes=5),
            exit_time=base_time + timedelta(hours=1),
            direction="long",
            entry_price=65000.0,
            exit_price=66500.0,
            profit_percent=2.31,
            max_profit_percent=2.31,
            slippage_percent=0.1,
            fee_percent=0.1
        ),
        OptimalTrade(
            entry_time=base_time + timedelta(hours=2, minutes=10),
            exit_time=base_time + timedelta(hours=3),
            direction="short",
            entry_price=66500.0,
            exit_price=65000.0,
            profit_percent=2.26,
            max_profit_percent=2.26,
            slippage_percent=0.1,
            fee_percent=0.1
        ),
    ]


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB memory"""
    mock = MagicMock()

    # Mock query_similar response
    mock.query_similar.return_value = {
        "ids": [["pred1", "pred2", "pred3"]],
        "documents": [["doc1", "doc2", "doc3"]],
        "metadatas": [[
            {
                "timestamp": "2024-10-01T12:00:00",
                "agent_name": "technical_analyst",
                "symbol": "BTCUSDT",
                "decision": "bullish",
                "confidence": 0.85,
                "price": 65000.0,
                "timeframe": "1h"
            },
            {
                "timestamp": "2024-10-01T14:00:00",
                "agent_name": "technical_analyst",
                "symbol": "BTCUSDT",
                "decision": "bearish",
                "confidence": 0.75,
                "price": 66500.0,
                "timeframe": "1h"
            },
            {
                "timestamp": "2024-10-01T16:00:00",
                "agent_name": "technical_analyst",
                "symbol": "BTCUSDT",
                "decision": "neutral",
                "confidence": 0.65,
                "price": 66000.0,
                "timeframe": "1h"
            },
        ]]
    }

    return mock


@pytest.fixture
def mock_db_connection():
    """Mock PostgreSQL connection"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Mock OHLCV data
    base_time = datetime(2024, 10, 1, 12, 0)
    mock_cursor.fetchall.return_value = [
        (base_time, 65000, 65100, 64900, 65050, 1000),
        (base_time + timedelta(hours=1), 65050, 66600, 65000, 66500, 1200),
        (base_time + timedelta(hours=2), 66500, 66700, 65800, 66000, 1100),
        (base_time + timedelta(hours=3), 66000, 66200, 64900, 65000, 900),
    ]

    mock_conn.cursor.return_value = mock_cursor

    return mock_conn


# ============================================================================
# Unit Tests for Core Methods
# ============================================================================

@pytest.mark.asyncio
async def test_collect_historical_predictions_success(validator, mock_chromadb):
    """Test successful prediction collection from ChromaDB"""
    validator.memory = mock_chromadb

    predictions = await validator._collect_historical_predictions(
        agent_name="technical_analyst",
        symbol="BTCUSDT",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 2),
        timeframe="1h"
    )

    assert len(predictions) == 3
    assert all(isinstance(p, AgentPrediction) for p in predictions)
    assert predictions[0].prediction == PredictionType.BULLISH
    assert predictions[1].prediction == PredictionType.BEARISH
    assert predictions[2].prediction == PredictionType.NEUTRAL
    assert all(p.confidence >= 0.6 for p in predictions)


@pytest.mark.asyncio
async def test_collect_predictions_filters_low_confidence(validator, mock_chromadb):
    """Test that low-confidence predictions are filtered"""
    # Modify mock to include low-confidence prediction
    mock_chromadb.query_similar.return_value["metadatas"][0].append({
        "timestamp": "2024-10-01T18:00:00",
        "agent_name": "technical_analyst",
        "symbol": "BTCUSDT",
        "decision": "bullish",
        "confidence": 0.3,  # Below threshold
        "price": 66000.0,
        "timeframe": "1h"
    })

    validator.memory = mock_chromadb

    predictions = await validator._collect_historical_predictions(
        agent_name="technical_analyst",
        symbol="BTCUSDT",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 2),
        timeframe="1h"
    )

    # Should still be 3 (low-confidence one filtered)
    assert len(predictions) == 3
    assert all(p.confidence >= 0.6 for p in predictions)


@pytest.mark.asyncio
async def test_calculate_optimal_trades_finds_profitable_moves(validator, mock_db_connection):
    """Test optimal trade calculation from TimescaleDB"""
    validator.db_connection = mock_db_connection

    trades = await validator._calculate_optimal_trades(
        symbol="BTCUSDT",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 2),
        timeframe="1h"
    )

    assert len(trades) > 0
    assert all(isinstance(t, OptimalTrade) for t in trades)

    # Check that trades are profitable after costs
    for trade in trades:
        net_profit = trade.profit_percent - trade.slippage_percent - (2 * trade.fee_percent)
        assert net_profit > validator.profit_threshold


@pytest.mark.asyncio
async def test_calculate_optimal_trades_handles_empty_data(validator):
    """Test optimal trade calculation with no price data"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value = mock_cursor

    validator.db_connection = mock_conn

    trades = await validator._calculate_optimal_trades(
        symbol="BTCUSDT",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 2),
        timeframe="1h"
    )

    assert trades == []


@pytest.mark.asyncio
async def test_compare_predictions_to_optimal_calculates_metrics(
    validator,
    sample_predictions,
    sample_optimal_trades
):
    """Test prediction comparison engine"""
    comparison = await validator._compare_predictions_to_optimal(
        predictions=sample_predictions,
        optimal_trades=sample_optimal_trades,
        timeframe="1h"
    )

    assert "true_positives" in comparison
    assert "false_positives" in comparison
    assert "true_negatives" in comparison
    assert "false_negatives" in comparison
    assert "correct_predictions" in comparison
    assert "incorrect_predictions" in comparison
    assert "missed_opportunities" in comparison

    # Should have some matches
    total = (comparison["true_positives"] +
             comparison["false_positives"] +
             comparison["true_negatives"] +
             comparison["false_negatives"])
    assert total > 0


@pytest.mark.asyncio
async def test_compare_predictions_matches_correct_bullish(validator):
    """Test that correct bullish predictions are counted as TP"""
    predictions = [
        AgentPrediction(
            timestamp=datetime(2024, 10, 1, 12, 0),
            agent_name="test_agent",
            symbol="BTCUSDT",
            prediction=PredictionType.BULLISH,
            confidence=0.8,
            reasoning="Test",
            timeframe="1h",
            price_at_prediction=65000.0,
            metadata={}
        )
    ]

    optimal_trades = [
        OptimalTrade(
            entry_time=datetime(2024, 10, 1, 12, 5),  # Within 30min window
            exit_time=datetime(2024, 10, 1, 13, 0),
            direction="long",
            entry_price=65000.0,
            exit_price=66500.0,
            profit_percent=2.31,
            max_profit_percent=2.31,
            slippage_percent=0.1,
            fee_percent=0.1
        )
    ]

    comparison = await validator._compare_predictions_to_optimal(
        predictions=predictions,
        optimal_trades=optimal_trades,
        timeframe="1h"
    )

    assert comparison["true_positives"] == 1
    assert comparison["false_positives"] == 0
    assert comparison["false_negatives"] == 0


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_validate_agent_complete_workflow(validator, mock_chromadb, mock_db_connection):
    """Test complete validation workflow end-to-end"""
    validator.memory = mock_chromadb
    validator.db_connection = mock_db_connection

    result = await validator.validate_agent(
        agent_name="technical_analyst",
        symbol="BTCUSDT",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 2),
        timeframe="1h"
    )

    assert isinstance(result, ValidationResult)
    assert result.agent_name == "technical_analyst"
    assert result.symbol == "BTCUSDT"
    assert result.total_predictions > 0
    assert 0 <= result.accuracy <= 1.0
    assert 0 <= result.precision <= 1.0
    assert 0 <= result.recall <= 1.0
    assert 0 <= result.f1_score <= 1.0


@pytest.mark.asyncio
async def test_validate_multiple_agents(validator, mock_chromadb, mock_db_connection):
    """Test parallel validation of multiple agents"""
    validator.memory = mock_chromadb
    validator.db_connection = mock_db_connection

    agents = ["technical_analyst", "sentiment_analyst", "macro_analyst"]

    results = await validator.validate_multiple_agents(
        agent_names=agents,
        symbol="BTCUSDT",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 2),
        timeframe="1h"
    )

    assert len(results) == 3
    assert all(isinstance(r, ValidationResult) for r in results)
    assert {r.agent_name for r in results} == set(agents)


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.asyncio
async def test_validate_agent_no_predictions(validator, mock_db_connection):
    """Test validation when agent made no predictions"""
    mock_memory = MagicMock()
    mock_memory.query_similar.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]]
    }

    validator.memory = mock_memory
    validator.db_connection = mock_db_connection

    result = await validator.validate_agent(
        agent_name="silent_agent",
        symbol="BTCUSDT",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 2),
        timeframe="1h"
    )

    assert result.total_predictions == 0
    assert result.accuracy == 0.0
    assert result.precision == 0.0
    assert result.recall == 0.0


@pytest.mark.asyncio
async def test_validate_agent_all_correct(validator):
    """Test validation when all predictions are correct"""
    # Perfect agent - always predicts correctly
    mock_memory = MagicMock()
    mock_memory.query_similar.return_value = {
        "ids": [["pred1", "pred2"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[
            {
                "timestamp": "2024-10-01T12:00:00",
                "agent_name": "perfect_agent",
                "symbol": "BTCUSDT",
                "decision": "bullish",
                "confidence": 0.9,
                "price": 65000.0,
                "timeframe": "1h"
            },
            {
                "timestamp": "2024-10-01T14:00:00",
                "agent_name": "perfect_agent",
                "symbol": "BTCUSDT",
                "decision": "bearish",
                "confidence": 0.9,
                "price": 66500.0,
                "timeframe": "1h"
            },
        ]]
    }

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    base_time = datetime(2024, 10, 1, 12, 0)
    mock_cursor.fetchall.return_value = [
        (base_time, 65000, 66600, 64900, 66500, 1000),
        (base_time + timedelta(hours=2), 66500, 66700, 64900, 65000, 1100),
    ]
    mock_conn.cursor.return_value = mock_cursor

    validator.memory = mock_memory
    validator.db_connection = mock_conn

    result = await validator.validate_agent(
        agent_name="perfect_agent",
        symbol="BTCUSDT",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 2),
        timeframe="1h"
    )

    # Perfect agent should have high accuracy
    assert result.accuracy >= 0.8
    assert result.precision >= 0.8


@pytest.mark.asyncio
async def test_validate_agent_all_wrong(validator):
    """Test validation when all predictions are wrong"""
    # Terrible agent - always wrong
    mock_memory = MagicMock()
    mock_memory.query_similar.return_value = {
        "ids": [["pred1"]],
        "documents": [["doc1"]],
        "metadatas": [[
            {
                "timestamp": "2024-10-01T12:00:00",
                "agent_name": "terrible_agent",
                "symbol": "BTCUSDT",
                "decision": "bearish",  # Wrong - market goes up
                "confidence": 0.9,
                "price": 65000.0,
                "timeframe": "1h"
            },
        ]]
    }

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    base_time = datetime(2024, 10, 1, 12, 0)
    mock_cursor.fetchall.return_value = [
        (base_time, 65000, 66600, 64900, 66500, 1000),  # Big upward move
    ]
    mock_conn.cursor.return_value = mock_cursor

    validator.memory = mock_memory
    validator.db_connection = mock_conn

    result = await validator.validate_agent(
        agent_name="terrible_agent",
        symbol="BTCUSDT",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 2),
        timeframe="1h"
    )

    # Terrible agent should have low metrics
    assert result.false_positives > 0 or result.false_negatives > 0


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_validation_completes_within_time_limit(validator, mock_chromadb, mock_db_connection):
    """Test that validation completes within reasonable time"""
    import time

    validator.memory = mock_chromadb
    validator.db_connection = mock_db_connection

    start_time = time.time()

    result = await validator.validate_agent(
        agent_name="technical_analyst",
        symbol="BTCUSDT",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 2),
        timeframe="1h"
    )

    elapsed = time.time() - start_time

    # Should complete in under 5 seconds
    assert elapsed < 5.0
    assert result is not None


# ============================================================================
# Data Quality Tests
# ============================================================================

def test_validation_result_to_dict():
    """Test ValidationResult serialization"""
    result = ValidationResult(
        agent_name="test_agent",
        symbol="BTCUSDT",
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 10, 2),
        timeframe="1h",
        total_predictions=10,
        total_optimal_trades=8,
        true_positives=5,
        false_positives=2,
        true_negatives=2,
        false_negatives=1,
        accuracy=0.7,
        precision=0.71,
        recall=0.83,
        f1_score=0.77,
        confusion_matrix=[[5, 2], [1, 2]],
        agent_total_profit_percent=15.0,
        optimal_total_profit_percent=20.0,
        efficiency_ratio=0.75,
        correct_predictions=5,
        incorrect_predictions=3,
        missed_opportunities=2,
        avg_confidence_correct=0.85,
        avg_confidence_incorrect=0.60,
        prediction_distribution={"BULLISH": 7, "BEARISH": 2, "NEUTRAL": 1}
    )

    result_dict = result.to_dict()

    assert isinstance(result_dict, dict)
    assert result_dict["agent_name"] == "test_agent"
    assert result_dict["accuracy"] == 0.7
    assert result_dict["efficiency_ratio"] == 0.75
    assert "start_date" in result_dict
    assert "end_date" in result_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
