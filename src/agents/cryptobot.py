"""
Crypto Trading Bot

Breakout trading bot for crypto markets with BTC priority and wider stops.
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Any
from .base_bot import BaseTradingBot


class CryptoBot(BaseTradingBot):
    """Crypto breakout trading bot with BTC priority"""
    
    def __init__(self, data_service_url: str = "http://fks_data:8003"):
        super().__init__("CryptoBot", data_service_url)
        self.btc_symbols = ["BTC-USD", "BTCUSDT", "BTC/USD", "BTC.US", "BTC_USD"]
    
    def get_strategy_name(self) -> str:
        return "Breakout Trading (Donchian Channels + Volume)"
    
    def is_btc(self, symbol: str) -> bool:
        """Check if symbol is BTC"""
        symbol_upper = symbol.upper()
        return any(btc in symbol_upper for btc in self.btc_symbols)
    
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze crypto data and generate signal"""
        data = market_data.get("data", [])
        if not data:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "No data available"
            }
        
        df = pd.DataFrame(data)
        if len(df) < 50:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "Insufficient data (need at least 50 bars)"
            }
        
        # Ensure numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Drop rows with NaN
        df = df.dropna(subset=["close", "high", "low", "volume"])
        
        if len(df) < 50:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "Insufficient valid data after cleaning"
            }
        
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values
        
        # Calculate indicators
        # Donchian Channels (20-period)
        try:
            upper_channel = pd.Series(high).rolling(20).max().values
            lower_channel = pd.Series(low).rolling(20).min().values
            middle_channel = (upper_channel + lower_channel) / 2
            
            rsi = talib.RSI(close, timeperiod=14)
            volume_avg = talib.SMA(volume, timeperiod=20)
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": f"Indicator calculation error: {e}"
            }
        
        # Get current values (last non-NaN)
        current_idx = -1
        while current_idx >= -len(close) and (
            np.isnan(upper_channel[current_idx]) or
            np.isnan(lower_channel[current_idx]) or
            np.isnan(rsi[current_idx]) or
            np.isnan(volume_avg[current_idx])
        ):
            current_idx -= 1
        
        if current_idx < -len(close):
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "No valid indicator values"
            }
        
        current_price = close[current_idx]
        current_upper = upper_channel[current_idx]
        current_lower = lower_channel[current_idx]
        current_rsi = rsi[current_idx]
        current_volume = volume[current_idx]
        avg_volume = volume_avg[current_idx] if not np.isnan(volume_avg[current_idx]) else current_volume
        
        # Breakout conditions
        bullish_breakout = current_price > current_upper
        bearish_breakout = current_price < current_lower
        volume_confirmation = current_volume > (avg_volume * 1.5)
        not_overbought = current_rsi < 70
        
        # BTC-specific rules (wider stops, long-term focus)
        is_btc_symbol = self.is_btc(symbol)
        
        if bullish_breakout and volume_confirmation and not_overbought:
            entry_price = float(current_price)
            
            if is_btc_symbol:
                # BTC: Wider stops for long-term holds
                stop_loss = entry_price * 0.90  # 10% stop
                take_profit = entry_price * 1.15  # 15% target
                confidence = 0.8  # Higher confidence for BTC
            else:
                # Other crypto: Standard stops
                stop_loss = entry_price * 0.97  # 3% stop
                take_profit = entry_price * 1.08  # 8% target
                confidence = 0.7
            
            return {
                "signal": "BUY",
                "confidence": float(confidence),
                "entry_price": entry_price,
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "reason": f"Bullish breakout above upper channel, volume confirmation",
                "btc_priority": is_btc_symbol,
                "indicators": {
                    "upper_channel": float(current_upper),
                    "lower_channel": float(current_lower),
                    "middle_channel": float(middle_channel[current_idx]),
                    "rsi": float(current_rsi),
                    "volume_ratio": float(current_volume / avg_volume) if avg_volume > 0 else 1.0
                }
            }
        
        elif bearish_breakout:
            return {
                "signal": "SELL",
                "confidence": 0.6,
                "reason": "Bearish breakout below lower channel",
                "btc_priority": is_btc_symbol,
                "indicators": {
                    "upper_channel": float(current_upper),
                    "lower_channel": float(current_lower),
                    "middle_channel": float(middle_channel[current_idx]),
                    "rsi": float(current_rsi),
                    "volume_ratio": float(current_volume / avg_volume) if avg_volume > 0 else 1.0
                }
            }
        
        else:
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "reason": "No breakout detected, waiting for confirmation",
                "btc_priority": is_btc_symbol,
                "indicators": {
                    "upper_channel": float(current_upper),
                    "lower_channel": float(current_lower),
                    "middle_channel": float(middle_channel[current_idx]),
                    "rsi": float(current_rsi),
                    "volume_ratio": float(current_volume / avg_volume) if avg_volume > 0 else 1.0
                }
            }

