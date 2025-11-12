"""
Forex Trading Bot

Mean-reversion bot for forex markets using RSI and Bollinger Bands.
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Any
from .base_bot import BaseTradingBot


class ForexBot(BaseTradingBot):
    """Forex mean-reversion trading bot"""
    
    def __init__(self, data_service_url: str = "http://fks_data:8003"):
        super().__init__("ForexBot", data_service_url)
    
    def get_strategy_name(self) -> str:
        return "Mean Reversion (RSI + Bollinger Bands)"
    
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze forex data and generate signal"""
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
        numeric_cols = ["open", "high", "low", "close"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Drop rows with NaN
        df = df.dropna(subset=["close", "high", "low"])
        
        if len(df) < 50:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "Insufficient valid data after cleaning"
            }
        
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        
        # Calculate indicators
        try:
            rsi = talib.RSI(close, timeperiod=14)
            upper_bb, middle_bb, lower_bb = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            atr = talib.ATR(high, low, close, timeperiod=14)
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
            np.isnan(rsi[current_idx]) or
            np.isnan(upper_bb[current_idx]) or
            np.isnan(lower_bb[current_idx]) or
            np.isnan(atr[current_idx])
        ):
            current_idx -= 1
        
        if current_idx < -len(close):
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "No valid indicator values"
            }
        
        current_price = close[current_idx]
        current_rsi = rsi[current_idx]
        current_upper_bb = upper_bb[current_idx]
        current_lower_bb = lower_bb[current_idx]
        current_atr = atr[current_idx]
        
        # Entry conditions (mean reversion)
        oversold = current_rsi < 30
        overbought = current_rsi > 70
        touch_lower_bb = current_price <= current_lower_bb
        touch_upper_bb = current_price >= current_upper_bb
        
        # Long signal (oversold + lower BB)
        if oversold and touch_lower_bb:
            entry_price = float(current_price)
            stop_loss = entry_price - (current_atr * 1.5)  # 1.5x ATR stop
            take_profit = entry_price + (current_atr * 2.0)  # 2x ATR target
            
            confidence = 0.7 + (30 - current_rsi) / 100  # Higher confidence for lower RSI
            confidence = min(float(confidence), 1.0)
            
            return {
                "signal": "BUY",
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "reason": f"Oversold (RSI={current_rsi:.2f}), touched lower Bollinger Band",
                "indicators": {
                    "rsi": float(current_rsi),
                    "upper_bb": float(current_upper_bb),
                    "middle_bb": float(middle_bb[current_idx]),
                    "lower_bb": float(current_lower_bb),
                    "atr": float(current_atr)
                }
            }
        
        # Short signal (overbought + upper BB)
        elif overbought and touch_upper_bb:
            entry_price = float(current_price)
            stop_loss = entry_price + (current_atr * 1.5)
            take_profit = entry_price - (current_atr * 2.0)
            
            confidence = 0.7 + (current_rsi - 70) / 100
            confidence = min(float(confidence), 1.0)
            
            return {
                "signal": "SELL",
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "reason": f"Overbought (RSI={current_rsi:.2f}), touched upper Bollinger Band",
                "indicators": {
                    "rsi": float(current_rsi),
                    "upper_bb": float(current_upper_bb),
                    "middle_bb": float(middle_bb[current_idx]),
                    "lower_bb": float(current_lower_bb),
                    "atr": float(current_atr)
                }
            }
        
        else:
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "reason": f"RSI={current_rsi:.2f}, no mean-reversion opportunity",
                "indicators": {
                    "rsi": float(current_rsi),
                    "upper_bb": float(current_upper_bb),
                    "middle_bb": float(middle_bb[current_idx]),
                    "lower_bb": float(current_lower_bb),
                    "atr": float(current_atr)
                }
            }

