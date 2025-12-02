"""
Tests for BaseTradingBot class
"""
import pytest
from unittest.mock import AsyncMock, patch, Mock
import httpx
from src.agents.base_bot import BaseTradingBot


class ConcreteTradingBot(BaseTradingBot):
    """Concrete implementation for testing"""
    
    def get_strategy_name(self) -> str:
        return "Test Strategy"
    
    async def analyze(self, symbol: str, market_data: dict) -> dict:
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "reason": "Test signal"
        }


class TestBaseTradingBot:
    """Test BaseTradingBot functionality"""
    
    def test_bot_initialization(self):
        """Test bot initialization"""
        bot = ConcreteTradingBot("TestBot")
        assert bot.name == "TestBot"
        assert bot.data_service_url == "http://fks_data:8003"
        assert bot.get_strategy_name() == "Test Strategy"
    
    def test_bot_custom_data_service_url(self):
        """Test bot with custom data service URL"""
        bot = ConcreteTradingBot("TestBot", data_service_url="http://custom:8003")
        assert bot.data_service_url == "http://custom:8003"
    
    @pytest.mark.asyncio
    async def test_fetch_market_data_success(self):
        """Test successful market data fetching"""
        bot = ConcreteTradingBot("TestBot")
        
        mock_response = {
            "data": [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 102.0,
                    "volume": 1000000
                }
            ]
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value.raise_for_status = Mock()
            mock_client.return_value.__aenter__.return_value.get.return_value.json.return_value = mock_response
            
            result = await bot.fetch_market_data("BTC-USD")
            assert "data" in result
            assert len(result["data"]) == 1
    
    @pytest.mark.asyncio
    async def test_fetch_market_data_failure(self):
        """Test market data fetching failure"""
        bot = ConcreteTradingBot("TestBot")
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = httpx.HTTPError("Connection error")
            
            result = await bot.fetch_market_data("BTC-USD")
            assert result == {"data": []}
    
    def test_calculate_risk(self):
        """Test risk calculation"""
        bot = ConcreteTradingBot("TestBot")
        
        # Standard risk calculation
        position_size = bot.calculate_risk(
            entry_price=100.0,
            stop_loss=98.0,  # 2% stop
            portfolio_value=10000.0,
            risk_pct=0.02  # 2% risk
        )
        
        # Expected: $200 risk / $2 price difference = 100 units
        # But max 10% = $1000 / $100 = 10 units
        # So should return min(100, 10) = 10
        assert position_size == 10.0
    
    def test_calculate_risk_zero_stop_loss(self):
        """Test risk calculation with zero stop loss difference"""
        bot = ConcreteTradingBot("TestBot")
        
        position_size = bot.calculate_risk(
            entry_price=100.0,
            stop_loss=100.0,  # No difference
            portfolio_value=10000.0,
            risk_pct=0.02
        )
        
        assert position_size == 0.0
    
    def test_validate_signal_valid_hold(self):
        """Test validating a valid HOLD signal"""
        bot = ConcreteTradingBot("TestBot")
        
        signal = {
            "signal": "HOLD",
            "confidence": 0.5,
            "reason": "No clear signal"
        }
        
        assert bot.validate_signal(signal) is True
    
    def test_validate_signal_valid_buy(self):
        """Test validating a valid BUY signal"""
        bot = ConcreteTradingBot("TestBot")
        
        signal = {
            "signal": "BUY",
            "confidence": 0.75,
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 105.0,
            "reason": "Bullish signal"
        }
        
        assert bot.validate_signal(signal) is True
    
    def test_validate_signal_invalid_missing_fields(self):
        """Test validating signal with missing fields"""
        bot = ConcreteTradingBot("TestBot")
        
        signal = {
            "signal": "BUY",
            "confidence": 0.75
            # Missing: entry_price, stop_loss, take_profit, reason
        }
        
        assert bot.validate_signal(signal) is False
    
    def test_validate_signal_invalid_signal_type(self):
        """Test validating signal with invalid signal type"""
        bot = ConcreteTradingBot("TestBot")
        
        signal = {
            "signal": "INVALID",
            "confidence": 0.5,
            "reason": "Test"
        }
        
        assert bot.validate_signal(signal) is False
    
    def test_validate_signal_invalid_confidence(self):
        """Test validating signal with invalid confidence"""
        bot = ConcreteTradingBot("TestBot")
        
        signal = {
            "signal": "HOLD",
            "confidence": 1.5,  # Invalid (> 1.0)
            "reason": "Test"
        }
        
        assert bot.validate_signal(signal) is False
    
    def test_validate_signal_invalid_stop_loss_buy(self):
        """Test validating BUY signal with invalid stop loss"""
        bot = ConcreteTradingBot("TestBot")
        
        signal = {
            "signal": "BUY",
            "confidence": 0.75,
            "entry_price": 100.0,
            "stop_loss": 102.0,  # Invalid (above entry)
            "take_profit": 105.0,
            "reason": "Test"
        }
        
        assert bot.validate_signal(signal) is False
    
    def test_validate_signal_invalid_take_profit_buy(self):
        """Test validating BUY signal with invalid take profit"""
        bot = ConcreteTradingBot("TestBot")
        
        signal = {
            "signal": "BUY",
            "confidence": 0.75,
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 95.0,  # Invalid (below entry)
            "reason": "Test"
        }
        
        assert bot.validate_signal(signal) is False
    
    @pytest.mark.asyncio
    async def test_analyze_abstract_method(self):
        """Test that analyze method is implemented"""
        bot = ConcreteTradingBot("TestBot")
        
        market_data = {"data": []}
        result = await bot.analyze("BTC-USD", market_data)
        
        assert "signal" in result
        assert "confidence" in result
        assert "reason" in result
        assert result["signal"] == "HOLD"

