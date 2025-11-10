"""
Integration tests for bot API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from src.api.routes import app


class TestBotAPIEndpoints:
    """Test bot API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing"""
        return {
            "data": [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 100.0,
                    "volume": 1000000
                }
                for _ in range(250)
            ]
        }
    
    def test_bots_health_endpoint(self, client):
        """Test bots health endpoint"""
        response = client.get("/ai/bots/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "bots" in data
        assert "StockBot" in data["bots"]
    
    @patch('src.api.routes.bots.StockBot')
    def test_stock_signal_endpoint(self, mock_stock_bot, client, sample_market_data):
        """Test StockBot signal endpoint"""
        # Mock StockBot
        mock_bot = mock_stock_bot.return_value
        mock_bot.analyze = AsyncMock(return_value={
            "signal": "BUY",
            "confidence": 0.75,
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 105.0,
            "reason": "Test signal"
        })
        mock_bot.validate_signal = lambda x: True
        mock_bot.get_strategy_name = lambda: "Test Strategy"
        
        response = client.post(
            "/ai/bots/stock/signal",
            json={
                "symbol": "AAPL.US",
                "market_data": sample_market_data
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["bot"] == "StockBot"
        assert data["signal"] == "BUY"
        assert data["confidence"] == 0.75
    
    @patch('src.api.routes.bots.ForexBot')
    def test_forex_signal_endpoint(self, mock_forex_bot, client, sample_market_data):
        """Test ForexBot signal endpoint"""
        # Mock ForexBot
        mock_bot = mock_forex_bot.return_value
        mock_bot.analyze = AsyncMock(return_value={
            "signal": "HOLD",
            "confidence": 0.5,
            "reason": "Test signal"
        })
        mock_bot.validate_signal = lambda x: True
        mock_bot.get_strategy_name = lambda: "Test Strategy"
        
        response = client.post(
            "/ai/bots/forex/signal",
            json={
                "symbol": "EURUSD",
                "market_data": sample_market_data
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["bot"] == "ForexBot"
        assert data["signal"] in ["BUY", "SELL", "HOLD"]
    
    @patch('src.api.routes.bots.CryptoBot')
    def test_crypto_signal_endpoint(self, mock_crypto_bot, client, sample_market_data):
        """Test CryptoBot signal endpoint"""
        # Mock CryptoBot
        mock_bot = mock_crypto_bot.return_value
        mock_bot.analyze = AsyncMock(return_value={
            "signal": "BUY",
            "confidence": 0.8,
            "entry_price": 50000.0,
            "stop_loss": 45000.0,
            "take_profit": 57500.0,
            "reason": "Test signal",
            "btc_priority": True
        })
        mock_bot.validate_signal = lambda x: True
        mock_bot.get_strategy_name = lambda: "Test Strategy"
        
        response = client.post(
            "/ai/bots/crypto/signal",
            json={
                "symbol": "BTC-USD",
                "market_data": sample_market_data
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["bot"] == "CryptoBot"
        assert data["signal"] == "BUY"
    
    @patch('src.api.routes.bots.consensus_node')
    @patch('src.api.routes.bots.stock_bot_node')
    @patch('src.api.routes.bots.forex_bot_node')
    @patch('src.api.routes.bots.crypto_bot_node')
    def test_consensus_endpoint(
        self,
        mock_crypto,
        mock_forex,
        mock_stock,
        mock_consensus,
        client,
        sample_market_data
    ):
        """Test consensus endpoint"""
        # Mock consensus node
        mock_consensus.return_value = {
            "consensus_signal": {
                "signal": "BUY",
                "confidence": 0.7,
                "reason": "Test consensus"
            },
            "bot_signals": {
                "stock": [],
                "forex": [],
                "crypto": []
            }
        }
        
        response = client.post(
            "/ai/bots/consensus",
            json={
                "symbol": "BTC-USD",
                "market_data": sample_market_data,
                "include_stock": True,
                "include_forex": True,
                "include_crypto": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "consensus_signal" in data
        assert "bot_signals" in data

