"""
Tests for StockBot
"""
import pytest
from unittest.mock import AsyncMock, patch
import pandas as pd
import numpy as np
from src.agents.stockbot import StockBot


class TestStockBot:
    """Test StockBot functionality"""
    
    def test_bot_initialization(self):
        """Test StockBot initialization"""
        bot = StockBot()
        assert bot.name == "StockBot"
        assert bot.get_strategy_name() == "Trend Following (Moving Averages)"
    
    def test_bot_custom_data_service_url(self):
        """Test StockBot with custom data service URL"""
        bot = StockBot(data_service_url="http://custom:8003")
        assert bot.data_service_url == "http://custom:8003"
    
    @pytest.mark.asyncio
    async def test_analyze_insufficient_data(self):
        """Test analyze with insufficient data"""
        bot = StockBot()
        
        # Empty data
        result = await bot.analyze("AAPL", {"data": []})
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 0.0
        assert "No data available" in result["reason"]
        
        # Not enough data
        data = [{"close": 100.0, "volume": 1000} for _ in range(50)]
        result = await bot.analyze("AAPL", {"data": data})
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 0.0
        assert "Insufficient data" in result["reason"]
    
    @pytest.mark.asyncio
    async def test_analyze_bullish_signal(self):
        """Test analyze with bullish conditions"""
        bot = StockBot()
        
        # Create sample data with bullish trend
        np.random.seed(42)
        base_price = 100.0
        prices = [base_price + i * 0.5 + np.random.normal(0, 1) for i in range(250)]
        volumes = [1000000 + np.random.normal(0, 100000) for _ in range(250)]
        
        data = [
            {
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "open": p,
                "high": p * 1.01,
                "low": p * 0.99,
                "close": p,
                "volume": int(v)
            }
            for i, (p, v) in enumerate(zip(prices, volumes))
        ]
        
        result = await bot.analyze("AAPL", {"data": data})
        
        # Should generate a signal (BUY, SELL, or HOLD)
        assert result["signal"] in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert "reason" in result
        assert "indicators" in result
    
    @pytest.mark.asyncio
    async def test_analyze_with_nan_values(self):
        """Test analyze with NaN values in data"""
        bot = StockBot()
        
        # Create data with some NaN values
        data = []
        for i in range(250):
            if i < 10:
                # First 10 rows have NaN
                data.append({
                    "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "volume": None
                })
            else:
                data.append({
                    "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                    "open": 100.0 + i * 0.1,
                    "high": 101.0 + i * 0.1,
                    "low": 99.0 + i * 0.1,
                    "close": 100.0 + i * 0.1,
                    "volume": 1000000
                })
        
        result = await bot.analyze("AAPL", {"data": data})
        
        # Should handle NaN values gracefully
        assert result["signal"] in ["BUY", "SELL", "HOLD"]
        assert "indicators" in result
    
    @pytest.mark.asyncio
    async def test_analyze_validate_signal(self):
        """Test that analyze returns valid signals"""
        bot = StockBot()
        
        # Create sufficient data
        np.random.seed(42)
        base_price = 100.0
        prices = [base_price + i * 0.1 + np.random.normal(0, 0.5) for i in range(250)]
        volumes = [1000000 + np.random.normal(0, 50000) for _ in range(250)]
        
        data = [
            {
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "open": p,
                "high": p * 1.01,
                "low": p * 0.99,
                "close": p,
                "volume": int(max(v, 1000))
            }
            for i, (p, v) in enumerate(zip(prices, volumes))
        ]
        
        result = await bot.analyze("AAPL", {"data": data})
        
        # Validate signal structure
        assert bot.validate_signal(result)
        
        # If BUY signal, check prices
        if result["signal"] == "BUY":
            assert "entry_price" in result
            assert "stop_loss" in result
            assert "take_profit" in result
            assert result["stop_loss"] < result["entry_price"]
            assert result["take_profit"] > result["entry_price"]
    
    @pytest.mark.asyncio
    async def test_analyze_indicators_present(self):
        """Test that analyze returns indicators"""
        bot = StockBot()
        
        # Create sufficient data
        np.random.seed(42)
        base_price = 100.0
        prices = [base_price + i * 0.1 for i in range(250)]
        volumes = [1000000 for _ in range(250)]
        
        data = [
            {
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "open": p,
                "high": p * 1.01,
                "low": p * 0.99,
                "close": p,
                "volume": v
            }
            for i, (p, v) in enumerate(zip(prices, volumes))
        ]
        
        result = await bot.analyze("AAPL", {"data": data})
        
        # Check indicators
        assert "indicators" in result
        indicators = result["indicators"]
        assert "ema5" in indicators
        assert "ema13" in indicators
        assert "ema50" in indicators
        assert "ema200" in indicators
        assert "macd_hist" in indicators
        assert "volume_ratio" in indicators

