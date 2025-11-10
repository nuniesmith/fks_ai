"""
Base Trading Bot Class

Abstract base class for all trading bots (StockBot, ForexBot, CryptoBot).
Provides common functionality for data fetching, risk calculation, and signal generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import httpx
from loguru import logger


class BaseTradingBot(ABC):
    """Base class for all trading bots"""
    
    def __init__(self, name: str, data_service_url: str = "http://fks_data:8003"):
        """
        Initialize base trading bot
        
        Args:
            name: Bot name (e.g., "StockBot", "ForexBot", "CryptoBot")
            data_service_url: URL for fks_data service
        """
        self.name = name
        self.data_service_url = data_service_url
        self.logger = logger.bind(bot=name)
    
    @abstractmethod
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and generate signal
        
        Args:
            symbol: Trading symbol (e.g., "AAPL.US", "BTC-USD")
            market_data: Market data from fks_data service
        
        Returns:
            Dict with signal information:
            - signal: "BUY", "SELL", or "HOLD"
            - confidence: float (0.0-1.0)
            - entry_price: float (optional)
            - stop_loss: float (optional)
            - take_profit: float (optional)
            - reason: str (explanation)
            - indicators: dict (technical indicators)
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Return strategy name
        
        Returns:
            Strategy name (e.g., "Trend Following (Moving Averages)")
        """
        pass
    
    async def fetch_market_data(self, symbol: str, interval: str = "1h", limit: int = 100) -> Dict[str, Any]:
        """
        Fetch market data from fks_data service
        
        Args:
            symbol: Trading symbol
            interval: Data interval (1h, 1d, etc.)
            limit: Number of data points to fetch
        
        Returns:
            Market data dictionary with "data" key containing OHLCV data
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.data_service_url}/api/v1/data/{symbol}",
                    params={"interval": interval, "limit": limit}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return {"data": []}
    
    def calculate_risk(
        self,
        entry_price: float,
        stop_loss: float,
        portfolio_value: float,
        risk_pct: float = 0.02
    ) -> float:
        """
        Calculate position size based on risk
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            portfolio_value: Total portfolio value
            risk_pct: Risk percentage per trade (default: 2%)
        
        Returns:
            Position size in units
        """
        risk_amount = portfolio_value * risk_pct
        price_diff = abs(entry_price - stop_loss)
        if price_diff == 0:
            return 0.0
        position_size = risk_amount / price_diff
        # Max 10% per position
        max_position = portfolio_value * 0.1 / entry_price
        return min(position_size, max_position)
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate signal structure and values
        
        Args:
            signal: Signal dictionary
        
        Returns:
            True if signal is valid, False otherwise
        """
        required_fields = ["signal", "confidence", "reason"]
        for field in required_fields:
            if field not in signal:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Validate signal type
        if signal["signal"] not in ["BUY", "SELL", "HOLD"]:
            self.logger.error(f"Invalid signal type: {signal['signal']}")
            return False
        
        # Validate confidence
        confidence = signal.get("confidence", 0.0)
        if not 0.0 <= confidence <= 1.0:
            self.logger.error(f"Invalid confidence: {confidence}")
            return False
        
        # Validate prices if signal is BUY or SELL
        if signal["signal"] in ["BUY", "SELL"]:
            if "entry_price" not in signal or signal["entry_price"] <= 0:
                self.logger.error("Missing or invalid entry_price")
                return False
            if "stop_loss" not in signal or signal["stop_loss"] <= 0:
                self.logger.error("Missing or invalid stop_loss")
                return False
            if "take_profit" not in signal or signal["take_profit"] <= 0:
                self.logger.error("Missing or invalid take_profit")
                return False
            
            # Validate stop loss and take profit relative to entry
            entry = signal["entry_price"]
            stop = signal["stop_loss"]
            tp = signal["take_profit"]
            
            if signal["signal"] == "BUY":
                if stop >= entry:
                    self.logger.error("Stop loss must be below entry for BUY")
                    return False
                if tp <= entry:
                    self.logger.error("Take profit must be above entry for BUY")
                    return False
            elif signal["signal"] == "SELL":
                if stop <= entry:
                    self.logger.error("Stop loss must be above entry for SELL")
                    return False
                if tp >= entry:
                    self.logger.error("Take profit must be below entry for SELL")
                    return False
        
        return True

