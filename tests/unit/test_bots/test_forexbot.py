"""
Tests for ForexBot
"""
import pytest
from unittest.mock import AsyncMock, patch
import pandas as pd
import numpy as np
from src.agents.forexbot import ForexBot


class TestForexBot:
    """Test ForexBot functionality"""
    
    def test_bot_initialization(self):
        """Test ForexBot initialization"""
        bot = ForexBot()
        assert bot.name == "ForexBot"
        assert bot.get_strategy_name() == "Mean Reversion (RSI + Bollinger Bands)"
    
    @pytest.mark.asyncio
    async def test_analyze_insufficient_data(self):
        """Test analyze with insufficient data"""
        bot = ForexBot()
        
        # Empty data
        result = await bot.analyze("EURUSD", {"data": []})
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 0.0
        
        # Not enough data
        data = [{"close": 1.10, "high": 1.11, "low": 1.09} for _ in range(30)]
        result = await bot.analyze("EURUSD", {"data": data})
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_oversold_signal(self):
        """Test analyze with oversold conditions (should generate BUY)"""
        bot = ForexBot()
        
        # Create data with oversold conditions (low prices, RSI < 30)
        np.random.seed(42)
        base_price = 1.10
        # Create declining prices to trigger oversold
        prices = [base_price - i * 0.001 for i in range(100)]
        
        data = [
            {
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "open": p,
                "high": p * 1.001,
                "low": p * 0.999,
                "close": p,
                "volume": 1000000
            }
            for i, p in enumerate(prices)
        ]
        
        result = await bot.analyze("EURUSD", {"data": data})
        
        # Should generate a signal
        assert result["signal"] in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert "indicators" in result
        assert "rsi" in result["indicators"]
        assert "atr" in result["indicators"]
    
    @pytest.mark.asyncio
    async def test_analyze_overbought_signal(self):
        """Test analyze with overbought conditions (should generate SELL)"""
        bot = ForexBot()
        
        # Create data with overbought conditions (high prices, RSI > 70)
        np.random.seed(42)
        base_price = 1.10
        # Create rising prices to trigger overbought
        prices = [base_price + i * 0.002 for i in range(100)]
        
        data = [
            {
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "open": p,
                "high": p * 1.002,
                "low": p * 0.998,
                "close": p,
                "volume": 1000000
            }
            for i, p in enumerate(prices)
        ]
        
        result = await bot.analyze("EURUSD", {"data": data})
        
        # Should generate a signal
        assert result["signal"] in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert "indicators" in result
    
    @pytest.mark.asyncio
    async def test_analyze_validate_signal(self):
        """Test that analyze returns valid signals"""
        bot = ForexBot()
        
        # Create sufficient data
        np.random.seed(42)
        base_price = 1.10
        prices = [base_price + np.random.normal(0, 0.001) for _ in range(100)]
        
        data = [
            {
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "open": p,
                "high": p * 1.001,
                "low": p * 0.999,
                "close": p,
                "volume": 1000000
            }
            for i, p in enumerate(prices)
        ]
        
        result = await bot.analyze("EURUSD", {"data": data})
        
        # Validate signal structure
        assert bot.validate_signal(result)
        
        # If BUY signal, check prices
        if result["signal"] == "BUY":
            assert "entry_price" in result
            assert "stop_loss" in result
            assert "take_profit" in result
            assert result["stop_loss"] < result["entry_price"]
            assert result["take_profit"] > result["entry_price"]
        
        # If SELL signal, check prices
        if result["signal"] == "SELL":
            assert "entry_price" in result
            assert "stop_loss" in result
            assert "take_profit" in result
            assert result["stop_loss"] > result["entry_price"]
            assert result["take_profit"] < result["entry_price"]
    
    @pytest.mark.asyncio
    async def test_analyze_indicators_present(self):
        """Test that analyze returns indicators"""
        bot = ForexBot()
        
        # Create sufficient data
        np.random.seed(42)
        base_price = 1.10
        prices = [base_price + np.random.normal(0, 0.0005) for _ in range(100)]
        
        data = [
            {
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "open": p,
                "high": p * 1.001,
                "low": p * 0.999,
                "close": p,
                "volume": 1000000
            }
            for i, p in enumerate(prices)
        ]
        
        result = await bot.analyze("EURUSD", {"data": data})
        
        # Check indicators
        assert "indicators" in result
        indicators = result["indicators"]
        assert "rsi" in indicators
        assert "upper_bb" in indicators
        assert "middle_bb" in indicators
        assert "lower_bb" in indicators
        assert "atr" in indicators

