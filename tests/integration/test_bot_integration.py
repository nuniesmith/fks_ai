"""
Integration tests for trading bot integration with LangGraph
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from agents.state import create_initial_state
from graph.bot_nodes import stock_bot_node, forex_bot_node, crypto_bot_node, run_bots_parallel
from graph.consensus_node import consensus_node
from graph.trading_graph import build_trading_graph, analyze_symbol


class TestBotIntegration:
    """Test bot integration with LangGraph"""
    
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
    
    @pytest.fixture
    def sample_state(self, sample_market_data):
        """Create sample agent state"""
        return create_initial_state("AAPL.US", sample_market_data)
    
    @pytest.mark.asyncio
    async def test_stock_bot_node(self, sample_state):
        """Test StockBot node integration"""
        result = await stock_bot_node(sample_state)
        assert "signals" in result
        # Should have signals if symbol is recognized as stock
        assert isinstance(result["signals"], list)
    
    @pytest.mark.asyncio
    async def test_forex_bot_node(self, sample_state):
        """Test ForexBot node integration"""
        # Change symbol to forex
        sample_state["symbol"] = "EURUSD"
        result = await forex_bot_node(sample_state)
        assert "signals" in result
        assert isinstance(result["signals"], list)
    
    @pytest.mark.asyncio
    async def test_crypto_bot_node(self, sample_state):
        """Test CryptoBot node integration"""
        # Change symbol to crypto
        sample_state["symbol"] = "BTC-USD"
        result = await crypto_bot_node(sample_state)
        assert "signals" in result
        assert isinstance(result["signals"], list)
    
    @pytest.mark.asyncio
    async def test_run_bots_parallel(self, sample_state):
        """Test running all bots in parallel"""
        result = await run_bots_parallel(sample_state)
        assert "signals" in result
        assert isinstance(result["signals"], list)
    
    @pytest.mark.asyncio
    async def test_consensus_node(self, sample_state):
        """Test consensus node"""
        # Add some mock signals
        sample_state["signals"] = [
            {
                "bot": "StockBot",
                "signal": "BUY",
                "confidence": 0.7,
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 105.0
            },
            {
                "bot": "CryptoBot",
                "signal": "BUY",
                "confidence": 0.8,
                "btc_priority": True,
                "entry_price": 50000.0,
                "stop_loss": 45000.0,
                "take_profit": 57500.0
            }
        ]
        
        result = await consensus_node(sample_state)
        assert "consensus_signal" in result
        assert "bot_signals" in result
        assert result["consensus_signal"]["signal"] in ["BUY", "SELL", "HOLD"]
    
    @pytest.mark.asyncio
    async def test_trading_graph_with_bots(self, sample_state):
        """Test trading graph with bots enabled"""
        graph = build_trading_graph(include_bots=True)
        assert graph is not None
    
    @pytest.mark.asyncio
    async def test_trading_graph_without_bots(self, sample_state):
        """Test trading graph without bots"""
        graph = build_trading_graph(include_bots=False)
        assert graph is not None
    
    @pytest.mark.asyncio
    @patch('graph.bot_nodes.StockBot')
    @patch('graph.bot_nodes.ForexBot')
    @patch('graph.bot_nodes.CryptoBot')
    async def test_bot_error_handling(self, mock_crypto, mock_forex, mock_stock, sample_state):
        """Test error handling in bot nodes"""
        # Make bots raise errors
        mock_stock.return_value.analyze = AsyncMock(side_effect=Exception("Bot error"))
        mock_forex.return_value.analyze = AsyncMock(side_effect=Exception("Bot error"))
        mock_crypto.return_value.analyze = AsyncMock(side_effect=Exception("Bot error"))
        
        # Should not crash, just return state with empty signals
        result = await run_bots_parallel(sample_state)
        assert "signals" in result
        assert isinstance(result["signals"], list)

