"""
Tests for CryptoBot
"""
import pytest
from unittest.mock import AsyncMock, patch
import pandas as pd
import numpy as np
from src.agents.cryptobot import CryptoBot


class TestCryptoBot:
    """Test CryptoBot functionality"""
    
    def test_bot_initialization(self):
        """Test CryptoBot initialization"""
        bot = CryptoBot()
        assert bot.name == "CryptoBot"
        assert bot.get_strategy_name() == "Breakout Trading (Donchian Channels + Volume)"
    
    def test_is_btc(self):
        """Test BTC symbol detection"""
        bot = CryptoBot()
        
        # BTC symbols
        assert bot.is_btc("BTC-USD") is True
        assert bot.is_btc("BTCUSDT") is True
        assert bot.is_btc("BTC/USD") is True
        assert bot.is_btc("BTC.US") is True
        assert bot.is_btc("BTC_USD") is True
        assert bot.is_btc("btc-usd") is True  # Case insensitive
        
        # Non-BTC symbols
        assert bot.is_btc("ETH-USD") is False
        assert bot.is_btc("AAPL") is False
        assert bot.is_btc("EURUSD") is False
    
    @pytest.mark.asyncio
    async def test_analyze_insufficient_data(self):
        """Test analyze with insufficient data"""
        bot = CryptoBot()
        
        # Empty data
        result = await bot.analyze("BTC-USD", {"data": []})
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 0.0
        
        # Not enough data
        data = [{"close": 50000.0, "high": 51000.0, "low": 49000.0, "volume": 1000} for _ in range(30)]
        result = await bot.analyze("BTC-USD", {"data": data})
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_btc_priority(self):
        """Test that BTC signals have priority (wider stops)"""
        bot = CryptoBot()
        
        # Create data with bullish breakout
        np.random.seed(42)
        base_price = 50000.0
        # Create prices that break above upper channel
        prices = [base_price + i * 100 for i in range(100)]
        volumes = [1000000 * (1.6 + np.random.normal(0, 0.1)) for _ in range(100)]  # High volume
        
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
        
        result = await bot.analyze("BTC-USD", {"data": data})
        
        # Check BTC priority flag
        if result["signal"] == "BUY":
            assert "btc_priority" in result
            assert result["btc_priority"] is True
            # BTC should have wider stops (10% vs 3%)
            stop_pct = abs(result["entry_price"] - result["stop_loss"]) / result["entry_price"]
            # Should be around 10% for BTC
            assert stop_pct >= 0.08  # Allow some tolerance
    
    @pytest.mark.asyncio
    async def test_analyze_non_btc_priority(self):
        """Test that non-BTC signals have standard stops"""
        bot = CryptoBot()
        
        # Create data with bullish breakout
        np.random.seed(42)
        base_price = 3000.0
        prices = [base_price + i * 10 for i in range(100)]
        volumes = [1000000 * (1.6 + np.random.normal(0, 0.1)) for _ in range(100)]
        
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
        
        result = await bot.analyze("ETH-USD", {"data": data})
        
        # Check non-BTC priority flag
        if result["signal"] == "BUY":
            assert "btc_priority" in result
            assert result["btc_priority"] is False
            # Non-BTC should have tighter stops (3% vs 10%)
            stop_pct = abs(result["entry_price"] - result["stop_loss"]) / result["entry_price"]
            # Should be around 3% for non-BTC
            assert stop_pct <= 0.05  # Allow some tolerance
    
    @pytest.mark.asyncio
    async def test_analyze_bullish_breakout(self):
        """Test analyze with bullish breakout conditions"""
        bot = CryptoBot()
        
        # Create data with bullish breakout
        np.random.seed(42)
        base_price = 50000.0
        prices = [base_price + i * 200 for i in range(100)]
        volumes = [1000000 * 2.0 for _ in range(100)]  # High volume
        
        data = [
            {
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "open": p,
                "high": p * 1.02,
                "low": p * 0.98,
                "close": p,
                "volume": int(v)
            }
            for i, (p, v) in enumerate(zip(prices, volumes))
        ]
        
        result = await bot.analyze("BTC-USD", {"data": data})
        
        # Should generate a signal
        assert result["signal"] in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert "indicators" in result
        assert "upper_channel" in result["indicators"]
        assert "lower_channel" in result["indicators"]
        assert "rsi" in result["indicators"]
        assert "volume_ratio" in result["indicators"]
    
    @pytest.mark.asyncio
    async def test_analyze_validate_signal(self):
        """Test that analyze returns valid signals"""
        bot = CryptoBot()
        
        # Create sufficient data
        np.random.seed(42)
        base_price = 50000.0
        prices = [base_price + np.random.normal(0, 500) for _ in range(100)]
        volumes = [1000000 + np.random.normal(0, 100000) for _ in range(100)]
        
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
        
        result = await bot.analyze("BTC-USD", {"data": data})
        
        # Validate signal structure
        assert bot.validate_signal(result)
        
        # If BUY signal, check prices
        if result["signal"] == "BUY":
            assert "entry_price" in result
            assert "stop_loss" in result
            assert "take_profit" in result
            assert result["stop_loss"] < result["entry_price"]
            assert result["take_profit"] > result["entry_price"]
            assert "btc_priority" in result
    
    @pytest.mark.asyncio
    async def test_analyze_indicators_present(self):
        """Test that analyze returns indicators"""
        bot = CryptoBot()
        
        # Create sufficient data
        np.random.seed(42)
        base_price = 50000.0
        prices = [base_price + np.random.normal(0, 200) for _ in range(100)]
        volumes = [1000000 for _ in range(100)]
        
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
        
        result = await bot.analyze("BTC-USD", {"data": data})
        
        # Check indicators
        assert "indicators" in result
        indicators = result["indicators"]
        assert "upper_channel" in indicators
        assert "lower_channel" in indicators
        assert "middle_channel" in indicators
        assert "rsi" in indicators
        assert "volume_ratio" in indicators

