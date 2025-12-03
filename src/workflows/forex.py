"""
Forex Workflow

Screening workflow for forex trading.
Focus: Major pairs for prop firm compatibility
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional, List, Dict, Any
import logging

from .base import (
    BaseWorkflow, 
    WorkflowResult, 
    SignalCandidate, 
    SignalDirection,
    SignalQuality,
    TechnicalContext,
)
from ..rules import RulesEngine, ForexGates

logger = logging.getLogger(__name__)


class ForexWorkflow(BaseWorkflow):
    """
    Forex market screening workflow.
    
    PROP FIRM COMPATIBLE:
    - 5% max daily drawdown
    - 10% max total drawdown
    - Consistent risk management
    
    Signal Generation Criteria:
    1. Technical: Session-based analysis, key levels
    2. Carry: Prefer positive carry trades
    3. News: Avoid 30 min before/after major releases
    """
    
    market_type = "forex"
    
    # Major pairs (most liquid, lowest spread)
    MAJOR_PAIRS = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "USDCAD", "NZDUSD"
    ]
    
    # Minor pairs (good liquidity)
    MINOR_PAIRS = [
        "EURGBP", "EURJPY", "GBPJPY", "AUDJPY",
        "EURAUD", "GBPCHF", "EURCHF"
    ]
    
    # Trading sessions (EST)
    LONDON_OPEN = time(3, 0)
    LONDON_CLOSE = time(12, 0)
    NY_OPEN = time(8, 0)
    NY_CLOSE = time(17, 0)
    
    def __init__(self):
        self.rules_engine = RulesEngine()
        self.forex_gates = ForexGates()
    
    @property
    def watchlist(self) -> List[str]:
        return self.MAJOR_PAIRS + self.MINOR_PAIRS
    
    def is_optimal_session(self, current_time: Optional[time] = None) -> bool:
        """Check if current time is optimal trading session."""
        if current_time is None:
            current_time = datetime.now().time()
        
        # London-NY overlap is best (8:00-12:00 EST)
        return time(8, 0) <= current_time <= time(12, 0)
    
    def get_session(self, current_time: Optional[time] = None) -> str:
        """Get current trading session."""
        if current_time is None:
            current_time = datetime.now().time()
        
        if self.LONDON_OPEN <= current_time < self.NY_OPEN:
            return "london"
        elif self.NY_OPEN <= current_time <= self.LONDON_CLOSE:
            return "london_ny_overlap"
        elif self.LONDON_CLOSE < current_time <= self.NY_CLOSE:
            return "new_york"
        else:
            return "asian"
    
    async def fetch_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch market data for forex pairs.
        
        In production, this would call:
        - Forex broker API
        - Economic calendar API
        - Carry rate data
        """
        # Placeholder - return mock data
        data = {}
        
        # Swap rates (positive = receive interest, negative = pay)
        swap_rates = {
            "EURUSD": {"long": -2.5, "short": 0.8},
            "GBPUSD": {"long": -1.8, "short": 0.5},
            "USDJPY": {"long": 1.2, "short": -1.8},
            "USDCHF": {"long": 0.6, "short": -1.2},
            "AUDUSD": {"long": -0.4, "short": -0.2},
            "USDCAD": {"long": 0.3, "short": -0.8},
            "NZDUSD": {"long": -0.3, "short": -0.1},
            "EURGBP": {"long": -0.8, "short": 0.3},
            "EURJPY": {"long": -1.5, "short": -0.5},
            "GBPJPY": {"long": -0.8, "short": -0.3},
            "AUDJPY": {"long": 0.5, "short": -0.8},
            "EURAUD": {"long": -2.0, "short": 1.0},
            "GBPCHF": {"long": -1.0, "short": 0.4},
            "EURCHF": {"long": -1.5, "short": 0.6},
        }
        
        for symbol in symbols:
            is_major = symbol in self.MAJOR_PAIRS
            
            data[symbol] = {
                "bid": 1.0850 if "EUR" in symbol else 1.2500 if "GBP" in symbol else 150.00,
                "ask": 1.0852 if "EUR" in symbol else 1.2502 if "GBP" in symbol else 150.02,
                "spread_pips": 0.8 if is_major else 1.5,
                "daily_range_pips": 80 if is_major else 100,
                "swap_rates": swap_rates.get(symbol, {"long": 0, "short": 0}),
                # Technical data
                "trend_4h": "up" if symbol in ["EURUSD", "GBPUSD", "AUDUSD"] else "down",
                "trend_1d": "up" if symbol in ["EURUSD", "GBPUSD", "AUDUSD"] else "down",
                "rsi_14": 55,
                "volume_ratio": 1.1,
                "atr_pips": 60 if is_major else 80,
                "key_levels": {
                    "support": [1.0800, 1.0750] if "EUR" in symbol else [1.2400, 1.2350],
                    "resistance": [1.0900, 1.0950] if "EUR" in symbol else [1.2600, 1.2650],
                },
                # Upcoming news
                "minutes_to_news": None,  # Will be populated from calendar
                "next_news_event": None,
            }
        
        return data
    
    async def identify_candidates(
        self, 
        market_data: Dict[str, Dict[str, Any]]
    ) -> List[SignalCandidate]:
        """
        Identify potential forex signals.
        
        Criteria:
        - Trend alignment
        - Good spread (< 2 pips for majors)
        - Optimal session timing
        - Prefer positive carry
        """
        candidates = []
        session = self.get_session()
        
        for symbol, data in market_data.items():
            # Check spread
            spread_pips = data.get("spread_pips", 10)
            if spread_pips > 2.0:
                continue  # Skip if spread too wide
            
            trend_4h = data.get("trend_4h", "sideways")
            trend_1d = data.get("trend_1d", "sideways")
            
            if trend_4h != trend_1d or trend_4h == "sideways":
                continue
            
            # Check for upcoming news
            minutes_to_news = data.get("minutes_to_news")
            if minutes_to_news is not None and abs(minutes_to_news) < 30:
                continue  # Skip during news blackout
            
            # Determine direction and check carry
            direction = SignalDirection.LONG if trend_4h == "up" else SignalDirection.SHORT
            swap_rates = data.get("swap_rates", {"long": 0, "short": 0})
            
            if direction == SignalDirection.LONG:
                carry_direction = "positive" if swap_rates["long"] > 0 else "negative"
            else:
                carry_direction = "positive" if swap_rates["short"] > 0 else "negative"
            
            # Calculate levels
            bid = data["bid"]
            ask = data["ask"]
            atr_pips = data.get("atr_pips", 60)
            pip_value = 0.0001 if "JPY" not in symbol else 0.01
            
            key_levels = data.get("key_levels", {})
            supports = key_levels.get("support", [])
            resistances = key_levels.get("resistance", [])
            
            if direction == SignalDirection.LONG:
                entry = ask  # Buy at ask
                stop = supports[0] if supports else entry - (atr_pips * pip_value * 1.5)
                target = resistances[0] if resistances else entry + (atr_pips * pip_value * 2)
            else:
                entry = bid  # Sell at bid
                stop = resistances[0] if resistances else entry + (atr_pips * pip_value * 1.5)
                target = supports[0] if supports else entry - (atr_pips * pip_value * 2)
            
            # Calculate R:R
            risk = abs(entry - stop)
            reward = abs(target - entry)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Skip if R:R below 1.5
            if rr_ratio < 1.5:
                continue
            
            # Calculate confidence
            confidence = 0.5
            if carry_direction == "positive":
                confidence += 0.15
            if session == "london_ny_overlap":
                confidence += 0.1
            if spread_pips < 1.0:
                confidence += 0.1
            if rr_ratio >= 2.0:
                confidence += 0.1
            
            confidence = min(max(confidence, 0.0), 1.0)
            
            # Position size - 1% risk per trade for prop firm compatibility
            risk_pips = abs(entry - stop) / pip_value
            
            candidate = SignalCandidate(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                position_size_pct=2.0,  # Conservative for prop firms
                confidence=confidence,
                timeframe="4H",
                reason=f"FX {direction.value}: Trend aligned, {carry_direction} carry, {session} session",
                technical=TechnicalContext(
                    trend_4h=trend_4h,
                    trend_1d=trend_1d,
                    key_levels=supports + resistances,
                    atr_pct=(atr_pips * pip_value / entry) * 100,
                    rsi=data.get("rsi_14", 50),
                    volume_ratio=data.get("volume_ratio", 1.0),
                ),
                metadata={
                    "spread_pips": spread_pips,
                    "carry_direction": carry_direction,
                    "swap_rate": swap_rates["long"] if direction == SignalDirection.LONG else swap_rates["short"],
                    "session": session,
                    "is_major": symbol in self.MAJOR_PAIRS,
                    "risk_pips": risk_pips,
                }
            )
            
            candidates.append(candidate)
        
        return candidates
    
    async def validate_candidate(
        self, 
        candidate: SignalCandidate,
        market_data: Dict[str, Any],
        account_balance: float
    ) -> tuple[bool, SignalQuality, List[str]]:
        """
        Validate a forex signal candidate.
        """
        warnings = []
        
        # Calculate risk/reward
        entry = candidate.entry_price
        stop = candidate.stop_loss
        target = candidate.take_profit
        
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Check forex gates
        gate_result = self.forex_gates.check_all(
            symbol=candidate.symbol,
            spread_pips=market_data.get("spread_pips", 2.0),
            leverage=20.0,  # Standard forex leverage
            daily_drawdown_pct=0.0,  # Current drawdown
            risk_reward_ratio=rr_ratio,
            minutes_to_news=market_data.get("minutes_to_news"),
            carry_direction=candidate.metadata.get("carry_direction"),
        )
        
        if not gate_result.passed:
            for gate in gate_result.gate_results:
                if gate.status.value == "fail":
                    warnings.append(f"Gate {gate.gate_id}: {gate.message}")
            return False, SignalQuality.REJECTED, warnings
        
        # Check hard rules
        pip_value = 0.0001 if "JPY" not in candidate.symbol else 0.01
        risk_pips = abs(entry - stop) / pip_value
        
        # Calculate risk % (assuming $10/pip for standard lot)
        # This is simplified - in production, calculate based on lot size
        risk_pct = (risk_pips * 10) / account_balance * 100
        
        result = self.rules_engine.validate_trade(
            symbol=candidate.symbol,
            market="forex",
            position_size_pct=candidate.position_size_pct,
            account_balance=account_balance,
            stop_loss_set=candidate.stop_loss > 0,
            risk_per_trade_pct=risk_pct,
            data_quality_pct=90.0,  # Forex data is typically high quality
            liquidity_usd=5_000_000_000,  # Forex is highly liquid
            leverage=20.0,
            trend_direction=candidate.technical.trend_4h if candidate.technical else None,
        )
        
        if not result.hard_rule_passed:
            for v in result.hard_violations:
                warnings.append(f"HR {v.rule_id.value}: {v.consequence}")
            return False, SignalQuality.REJECTED, warnings
        
        # Determine quality
        soft_warnings = result.soft_rule_warnings
        carry_positive = candidate.metadata.get("carry_direction") == "positive"
        optimal_session = candidate.metadata.get("session") == "london_ny_overlap"
        
        if candidate.confidence >= 0.75 and soft_warnings == 0 and carry_positive and optimal_session:
            quality = SignalQuality.A_PLUS
        elif candidate.confidence >= 0.7 and soft_warnings <= 1:
            quality = SignalQuality.A
        elif candidate.confidence >= 0.6:
            quality = SignalQuality.B
        else:
            quality = SignalQuality.C
        
        # Add soft rule warnings
        for check in result.soft_checks:
            if not check.passed:
                warnings.append(f"SR {check.rule_id.value}: {check.recommendation}")
        
        # Add carry warning if negative
        if not carry_positive:
            warnings.append(f"Negative carry: {candidate.metadata.get('swap_rate', 0):.2f} per day")
        
        return True, quality, warnings
