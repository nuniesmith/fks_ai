"""
Unit tests for ChromaDB TradingMemory class

Tests memory storage, retrieval, and similarity search
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


def test_trading_memory_initialization(mock_chromadb_client):
    """Test TradingMemory class initialization"""
    from src.services.ai.src.memory.chroma_client import TradingMemory

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        memory = TradingMemory()

        assert memory.client is not None
        assert memory.collection is not None


def test_add_insight_basic(mock_chromadb_client):
    """Test adding a simple insight to memory"""
    from src.services.ai.src.memory.chroma_client import TradingMemory

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        memory = TradingMemory()

        insight_id = memory.add_insight(
            decision="BUY BTCUSDT with 70% confidence",
            symbol="BTCUSDT",
            confidence=0.70,
            regime="bull"
        )

        assert insight_id.startswith('insight_')
        assert 'BTCUSDT' in insight_id or len(insight_id) > 10


def test_add_insight_with_metadata(mock_chromadb_client):
    """Test adding insight with full metadata"""
    from src.services.ai.src.memory.chroma_client import TradingMemory

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        memory = TradingMemory()

        metadata = {
            'symbol': 'ETHUSDT',
            'confidence': 0.65,
            'regime': 'sideways',
            'analyst_consensus': 'mixed'
        }

        memory.add_insight(
            decision="HOLD ETHUSDT, wait for breakout",
            **metadata
        )

        memory.collection.add.assert_called_once()
        call_args = memory.collection.add.call_args

        # Verify metadata was passed
        assert call_args[1]['metadatas'][0]['symbol'] == 'ETHUSDT'
        assert call_args[1]['metadatas'][0]['confidence'] == 0.65


def test_query_similar_basic(mock_chromadb_client):
    """Test querying similar past decisions"""
    from src.services.ai.src.memory.chroma_client import TradingMemory

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        memory = TradingMemory()

        results = memory.query_similar(
            query="Should I buy BTCUSDT?",
            n_results=3
        )

        assert len(results) > 0
        assert 'Past decision 1' in results[0]


def test_query_similar_with_filter(mock_chromadb_client):
    """Test querying with metadata filters"""
    from src.services.ai.src.memory.chroma_client import TradingMemory

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        memory = TradingMemory()

        memory.query_similar(
            query="BTC analysis",
            n_results=5,
            where={'symbol': 'BTCUSDT'}
        )

        memory.collection.query.assert_called_once()
        call_args = memory.collection.query.call_args

        assert call_args[1]['where'] == {'symbol': 'BTCUSDT'}
        assert call_args[1]['n_results'] == 5


def test_get_by_id_found(mock_chromadb_client):
    """Test retrieving insight by ID when exists"""
    from src.services.ai.src.memory.chroma_client import TradingMemory

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        memory = TradingMemory()

        result = memory.get_by_id('insight_1')

        assert result is not None
        assert result == 'Past decision'


def test_get_by_id_not_found(mock_chromadb_client):
    """Test retrieving non-existent insight"""
    from src.services.ai.src.memory.chroma_client import TradingMemory

    # Mock empty result
    mock_chromadb_client.get_or_create_collection().get.return_value = {
        'ids': [],
        'documents': [],
        'metadatas': []
    }

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        memory = TradingMemory()

        result = memory.get_by_id('nonexistent_id')

        assert result is None


def test_memory_timestamp_generation(mock_chromadb_client):
    """Test that timestamps are automatically added"""
    from src.services.ai.src.memory.chroma_client import TradingMemory

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        memory = TradingMemory()

        memory.add_insight(
            decision="Test decision",
            symbol="BTCUSDT",
            confidence=0.5
        )

        call_args = memory.collection.add.call_args
        metadata = call_args[1]['metadatas'][0]

        assert 'timestamp' in metadata
        # Verify timestamp is ISO format
        datetime.fromisoformat(metadata['timestamp'])


def test_memory_multiple_insights(mock_chromadb_client):
    """Test storing multiple insights"""
    from src.services.ai.src.memory.chroma_client import TradingMemory

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        memory = TradingMemory()

        decisions = [
            ("BUY BTCUSDT", "BTCUSDT", 0.70),
            ("SELL ETHUSDT", "ETHUSDT", 0.65),
            ("HOLD SOLUSDT", "SOLUSDT", 0.50)
        ]

        ids = []
        for decision, symbol, confidence in decisions:
            insight_id = memory.add_insight(decision, symbol, confidence)
            ids.append(insight_id)

        assert len(ids) == 3
        assert len(set(ids)) == 3  # All unique IDs


def test_query_similar_empty_results(mock_chromadb_client):
    """Test querying when no similar results found"""
    from src.services.ai.src.memory.chroma_client import TradingMemory

    # Mock empty results
    mock_chromadb_client.get_or_create_collection().query.return_value = {
        'ids': [[]],
        'documents': [[]],
        'metadatas': [[]],
        'distances': [[]]
    }

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        memory = TradingMemory()

        results = memory.query_similar("No results query")

        assert results == []


def test_memory_confidence_filtering(mock_chromadb_client):
    """Test filtering by confidence threshold"""
    from src.services.ai.src.memory.chroma_client import TradingMemory

    with patch('chromadb.PersistentClient', return_value=mock_chromadb_client):
        memory = TradingMemory()

        memory.query_similar(
            query="High confidence trades",
            n_results=10,
            where={'confidence': {'$gte': 0.7}}
        )

        call_args = memory.collection.query.call_args
        where_clause = call_args[1]['where']

        assert 'confidence' in where_clause
        assert where_clause['confidence']['$gte'] == 0.7
