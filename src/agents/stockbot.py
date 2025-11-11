"""
Stock Trading Bot

Trend-following bot for stock markets using moving averages and MACD.
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Any
from .base_bot import BaseTradingBot


class StockBot(BaseTradingBot):
    """Stock market trend-following bot"""
    
    def __init__(self, data_service_url: str = "http://fks_data:8003"):
        super().__init__("StockBot", data_service_url)
    
    def get_strategy_name(self) -> str:
        return "Trend Following (Moving Averages)"
    
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stock data and generate signal"""
        # Extract OHLCV data
        data = market_data.get("data", [])
        if not data:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "No data available"
            }
        
        df = pd.DataFrame(data)
        if len(df) < 200:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "Insufficient data (need at least 200 bars)"
            }
        
        # Ensure numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Drop rows with NaN
        df = df.dropna(subset=["close", "volume"])
        
        if len(df) < 200:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "Insufficient valid data after cleaning"
            }
        
        close = df["close"].values
        volume = df["volume"].values
        
        # Calculate indicators
        try:
            ema5 = talib.EMA(close, timeperiod=5)
            ema13 = talib.EMA(close, timeperiod=13)
            ema50 = talib.EMA(close, timeperiod=50)
            ema200 = talib.EMA(close, timeperiod=200)
            macd, macd_signal, macd_hist = talib.MACD(close)
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
            np.isnan(ema5[current_idx]) or
            np.isnan(ema13[current_idx]) or
            np.isnan(ema50[current_idx]) or
            np.isnan(ema200[current_idx])
        ):
            current_idx -= 1
        
        if current_idx < -len(close):
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "No valid indicator values"
            }
        
        current_price = close[current_idx]
        current_ema5 = ema5[current_idx]
        current_ema13 = ema13[current_idx]
        current_ema50 = ema50[current_idx]
        current_ema200 = ema200[current_idx]
        current_macd_hist = macd_hist[current_idx] if not np.isnan(macd_hist[current_idx]) else 0.0
        current_volume = volume[current_idx]
        avg_volume = volume_avg[current_idx] if not np.isnan(volume_avg[current_idx]) else current_volume
        
        # Entry conditions
        bullish_crossover = current_ema5 > current_ema13
        if current_idx > -len(ema5):
            bullish_crossover = bullish_crossover and ema5[current_idx - 1] <= ema13[current_idx - 1]
        
        uptrend = current_price > current_ema200
        macd_bullish = current_macd_hist > 0
        volume_confirmation = current_volume > avg_volume * 1.1
        
        # Calculate confidence
        confidence = 0.0
        if bullish_crossover:
            confidence += 0.3
        if uptrend:
            confidence += 0.3
        if macd_bullish:
            confidence += 0.2
        if volume_confirmation:
            confidence += 0.2
        
        # Generate signal
        if confidence >= 0.6:
            entry_price = float(current_price)
            stop_loss = entry_price * 0.98  # 2% stop
            take_profit = entry_price * 1.05  # 5% target
            
            return {
                "signal": "BUY",
                "confidence": min(float(confidence), 1.0),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "reason": "Bullish crossover, uptrend, MACD positive, volume confirmation",
                "indicators": {
                    "ema5": float(current_ema5),
                    "ema13": float(current_ema13),
                    "ema50": float(current_ema50),
                    "ema200": float(current_ema200),
                    "macd_hist": float(current_macd_hist),
                    "volume_ratio": float(current_volume / avg_volume) if avg_volume > 0 else 1.0
                }
            }
        elif confidence <= 0.3:
            return {
                "signal": "SELL",
                "confidence": min(float(1.0 - confidence), 1.0),
                "reason": "Bearish conditions detected",
                "indicators": {
                    "ema5": float(current_ema5),
                    "ema13": float(current_ema13),
                    "ema50": float(current_ema50),
                    "ema200": float(current_ema200),
                    "macd_hist": float(current_macd_hist),
                    "volume_ratio": float(current_volume / avg_volume) if avg_volume > 0 else 1.0
                }
            }
        else:
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "reason": "Mixed signals, waiting for confirmation",
                "indicators": {
                    "ema5": float(current_ema5),
                    "ema13": float(current_ema13),
                    "ema50": float(current_ema50),
                    "ema200": float(current_ema200),
                    "macd_hist": float(current_macd_hist),
                    "volume_ratio": float(current_volume / avg_volume) if avg_volume > 0 else 1.0
                }
            }

