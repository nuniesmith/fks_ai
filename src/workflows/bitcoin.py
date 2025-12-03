"""
Bitcoin-Focused Workflow

Dedicated workflow for BTC spot trading with enhanced signals,
dynamic risk management, and comprehensive validation.

This is the primary focus workflow for building solid trading foundations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import logging
import time

from .base import (
    BaseWorkflow, 
    WorkflowResult, 
    SignalCandidate, 
    SignalDirection,
    SignalQuality,
    TechnicalContext,
)
from ..rules import RulesEngine, CryptoSpotGates

logger = logging.getLogger(__name__)


class BTCTrendState(Enum):
    """Bitcoin trend state classification."""
    STRONG_BULL = "strong_bull"      # Weekly, Daily, 4H all up
    BULL = "bull"                     # Daily, 4H up
    WEAK_BULL = "weak_bull"          # 4H up, Daily sideways
    NEUTRAL = "neutral"              # Mixed signals
    WEAK_BEAR = "weak_bear"          # 4H down, Daily sideways
    BEAR = "bear"                    # Daily, 4H down
    STRONG_BEAR = "strong_bear"      # Weekly, Daily, 4H all down


class ConfidenceLevel(Enum):
    """Signal confidence levels for dynamic risk."""
    VERY_HIGH = "very_high"   # 85%+: Full position
    HIGH = "high"             # 75-84%: 75% position
    MEDIUM = "medium"         # 65-74%: 50% position
    LOW = "low"               # 55-64%: 25% position
    VERY_LOW = "very_low"     # <55%: Skip or paper trade only


@dataclass
class DynamicRisk:
    """Dynamic risk parameters based on confidence."""
    confidence_level: ConfidenceLevel
    base_risk_pct: float
    adjusted_risk_pct: float
    position_multiplier: float
    max_position_pct: float
    recommended_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_level": self.confidence_level.value,
            "base_risk_pct": self.base_risk_pct,
            "adjusted_risk_pct": self.adjusted_risk_pct,
            "position_multiplier": self.position_multiplier,
            "max_position_pct": self.max_position_pct,
            "recommended_action": self.recommended_action,
        }


@dataclass
class OnChainBTCMetrics:
    """Bitcoin-specific on-chain metrics."""
    # Exchange flows
    exchange_netflow_btc: float = 0.0  # Positive = inflow (bearish), Negative = outflow (bullish)
    exchange_reserve_change_pct: float = 0.0
    
    # Whale activity
    whale_tx_count: int = 0  # Transactions > 100 BTC
    whale_accumulation_score: float = 0.0  # -1 to 1, positive = accumulating
    
    # Miner metrics
    miner_outflow_btc: float = 0.0
    miner_reserve_btc: float = 0.0
    hash_rate_change_pct: float = 0.0
    
    # Network health
    active_addresses: int = 0
    transaction_count: int = 0
    fees_btc: float = 0.0
    
    # Derivatives (for spot context)
    futures_funding_rate: float = 0.0  # Cross-reference with perps
    futures_oi_change_pct: float = 0.0
    
    def bullish_signals(self) -> int:
        """Count bullish on-chain signals."""
        count = 0
        if self.exchange_netflow_btc < -1000:  # Significant outflow
            count += 1
        if self.whale_accumulation_score > 0.3:
            count += 1
        if self.miner_outflow_btc < 100:  # Miners holding
            count += 1
        if self.hash_rate_change_pct > 0:
            count += 1
        if self.futures_funding_rate < 0.01:  # Not overheated
            count += 1
        return count
    
    def bearish_signals(self) -> int:
        """Count bearish on-chain signals."""
        count = 0
        if self.exchange_netflow_btc > 5000:  # Large inflow
            count += 1
        if self.whale_accumulation_score < -0.3:
            count += 1
        if self.miner_outflow_btc > 1000:  # Miners selling
            count += 1
        if self.hash_rate_change_pct < -5:
            count += 1
        if self.futures_funding_rate > 0.05:  # Overheated
            count += 1
        return count
    
    def net_score(self) -> float:
        """Net on-chain score (-1 to 1)."""
        bullish = self.bullish_signals()
        bearish = self.bearish_signals()
        total = bullish + bearish
        if total == 0:
            return 0.0
        return (bullish - bearish) / 5.0  # Normalize to -1 to 1


@dataclass
class BTCSignalContext:
    """Extended context for Bitcoin signals."""
    trend_state: BTCTrendState
    dynamic_risk: DynamicRisk
    on_chain: OnChainBTCMetrics
    market_regime: str  # "trending", "ranging", "volatile"
    support_levels: List[float]
    resistance_levels: List[float]
    next_halving_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend_state": self.trend_state.value,
            "dynamic_risk": self.dynamic_risk.to_dict(),
            "on_chain_net_score": self.on_chain.net_score(),
            "market_regime": self.market_regime,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "next_halving_days": self.next_halving_days,
        }


@dataclass
class TimingMetrics:
    """Performance timing metrics in milliseconds."""
    data_fetch_ms: float = 0.0
    technical_analysis_ms: float = 0.0
    on_chain_analysis_ms: float = 0.0
    candidate_identification_ms: float = 0.0
    validation_ms: float = 0.0
    risk_calculation_ms: float = 0.0
    total_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_fetch_ms": round(self.data_fetch_ms, 3),
            "technical_analysis_ms": round(self.technical_analysis_ms, 3),
            "on_chain_analysis_ms": round(self.on_chain_analysis_ms, 3),
            "candidate_identification_ms": round(self.candidate_identification_ms, 3),
            "validation_ms": round(self.validation_ms, 3),
            "risk_calculation_ms": round(self.risk_calculation_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


class BitcoinWorkflow(BaseWorkflow):
    """
    Bitcoin-focused screening workflow with dynamic risk management.
    
    This workflow is optimized for BTC spot trading with:
    - Multi-timeframe trend analysis
    - On-chain metrics integration
    - Dynamic position sizing based on confidence
    - Performance timing for optimization
    """
    
    market_type = "bitcoin"
    
    # Only Bitcoin
    SYMBOLS = ["BTCUSDT", "BTCUSD", "BTC-USD"]
    
    # Risk parameters by confidence level
    RISK_CONFIG = {
        ConfidenceLevel.VERY_HIGH: {"multiplier": 1.0, "base_risk": 2.0, "action": "Full position"},
        ConfidenceLevel.HIGH: {"multiplier": 0.75, "base_risk": 2.0, "action": "Reduced position"},
        ConfidenceLevel.MEDIUM: {"multiplier": 0.5, "base_risk": 1.5, "action": "Half position"},
        ConfidenceLevel.LOW: {"multiplier": 0.25, "base_risk": 1.0, "action": "Quarter position"},
        ConfidenceLevel.VERY_LOW: {"multiplier": 0.0, "base_risk": 0.5, "action": "Paper trade only"},
    }
    
    # Next halving estimate (April 2028)
    NEXT_HALVING = datetime(2028, 4, 15)
    
    def __init__(self, base_risk_pct: float = 2.0, max_position_pct: float = 10.0):
        self.rules_engine = RulesEngine()
        self.spot_gates = CryptoSpotGates()
        self.base_risk_pct = base_risk_pct
        self.max_position_pct = max_position_pct
        self.timing = TimingMetrics()
    
    @property
    def watchlist(self) -> List[str]:
        return self.SYMBOLS
    
    def _time_ms(self) -> float:
        """Get current time in milliseconds."""
        return time.perf_counter() * 1000
    
    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map confidence score to level."""
        if confidence >= 0.85:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.65:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.55:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def calculate_dynamic_risk(self, confidence: float) -> DynamicRisk:
        """Calculate dynamic risk parameters based on confidence."""
        start = self._time_ms()
        
        level = self.get_confidence_level(confidence)
        config = self.RISK_CONFIG[level]
        
        multiplier = config["multiplier"]
        base_risk = config["base_risk"]
        adjusted_risk = base_risk * multiplier
        max_position = self.max_position_pct * multiplier
        
        result = DynamicRisk(
            confidence_level=level,
            base_risk_pct=base_risk,
            adjusted_risk_pct=adjusted_risk,
            position_multiplier=multiplier,
            max_position_pct=max_position,
            recommended_action=config["action"],
        )
        
        self.timing.risk_calculation_ms += self._time_ms() - start
        return result
    
    def classify_trend(
        self, 
        trend_4h: str, 
        trend_1d: str, 
        trend_1w: str
    ) -> BTCTrendState:
        """Classify overall BTC trend state."""
        trends = (trend_1w, trend_1d, trend_4h)
        
        if all(t == "up" for t in trends):
            return BTCTrendState.STRONG_BULL
        elif trend_1d == "up" and trend_4h == "up":
            return BTCTrendState.BULL
        elif trend_4h == "up" and trend_1d in ("sideways", "neutral"):
            return BTCTrendState.WEAK_BULL
        elif all(t == "down" for t in trends):
            return BTCTrendState.STRONG_BEAR
        elif trend_1d == "down" and trend_4h == "down":
            return BTCTrendState.BEAR
        elif trend_4h == "down" and trend_1d in ("sideways", "neutral"):
            return BTCTrendState.WEAK_BEAR
        else:
            return BTCTrendState.NEUTRAL
    
    def detect_market_regime(
        self, 
        atr_pct: float, 
        volume_ratio: float,
        price_range_pct: float
    ) -> str:
        """Detect current market regime."""
        if atr_pct > 4.0 or volume_ratio > 2.0:
            return "volatile"
        elif price_range_pct < 3.0 and volume_ratio < 0.8:
            return "ranging"
        else:
            return "trending"
    
    def days_to_halving(self) -> int:
        """Calculate days until next halving."""
        return max(0, (self.NEXT_HALVING - datetime.now()).days)
    
    async def fetch_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch comprehensive Bitcoin market data.
        """
        start = self._time_ms()
        
        # In production, this would call:
        # - Exchange APIs (Binance, Coinbase)
        # - On-chain data (Glassnode, CryptoQuant)
        # - Derivatives data (Coinglass)
        
        data = {}
        
        for symbol in symbols:
            # Mock BTC price around 97k
            current_price = 97000.0
            
            data[symbol] = {
                "symbol": symbol,
                "price": current_price,
                "price_24h_ago": current_price * 0.98,
                "price_7d_ago": current_price * 0.95,
                "high_24h": current_price * 1.02,
                "low_24h": current_price * 0.97,
                "volume_24h_usd": 45_000_000_000,
                "volume_ratio": 1.15,
                "market_cap_usd": 1_900_000_000_000,
                "spread_pct": 0.01,
                "exchange_listings": 50,
                
                # Multi-timeframe trends
                "trend_4h": "up",
                "trend_1d": "up", 
                "trend_1w": "up",
                
                # Technical indicators
                "rsi_14": 62,
                "rsi_4h": 58,
                "macd_signal": "bullish",
                "ema_20": current_price * 0.98,
                "ema_50": current_price * 0.94,
                "ema_200": current_price * 0.85,
                
                # Volatility
                "atr_pct": 2.8,
                "bollinger_width": 0.05,
                
                # Key levels
                "support_levels": [95000, 92000, 88000, 85000],
                "resistance_levels": [100000, 105000, 110000, 120000],
                
                # On-chain data
                "on_chain": OnChainBTCMetrics(
                    exchange_netflow_btc=-2500,  # Outflow (bullish)
                    exchange_reserve_change_pct=-0.5,
                    whale_tx_count=85,
                    whale_accumulation_score=0.45,
                    miner_outflow_btc=150,
                    miner_reserve_btc=1_850_000,
                    hash_rate_change_pct=2.5,
                    active_addresses=950_000,
                    transaction_count=450_000,
                    fees_btc=125,
                    futures_funding_rate=0.008,
                    futures_oi_change_pct=3.5,
                ),
            }
        
        self.timing.data_fetch_ms = self._time_ms() - start
        return data
    
    async def identify_candidates(
        self, 
        market_data: Dict[str, Dict[str, Any]]
    ) -> List[SignalCandidate]:
        """
        Identify Bitcoin trading signals with comprehensive analysis.
        """
        start = self._time_ms()
        candidates = []
        
        for symbol, data in market_data.items():
            tech_start = self._time_ms()
            
            # Get trend state
            trend_state = self.classify_trend(
                data.get("trend_4h", "sideways"),
                data.get("trend_1d", "sideways"),
                data.get("trend_1w", "sideways"),
            )
            
            # Detect market regime
            market_regime = self.detect_market_regime(
                data.get("atr_pct", 2.0),
                data.get("volume_ratio", 1.0),
                ((data.get("high_24h", 0) - data.get("low_24h", 0)) / data["price"]) * 100
            )
            
            self.timing.technical_analysis_ms += self._time_ms() - tech_start
            
            # On-chain analysis
            onchain_start = self._time_ms()
            on_chain = data.get("on_chain", OnChainBTCMetrics())
            on_chain_score = on_chain.net_score() if isinstance(on_chain, OnChainBTCMetrics) else 0
            self.timing.on_chain_analysis_ms += self._time_ms() - onchain_start
            
            # Determine direction based on trend
            if trend_state in (BTCTrendState.STRONG_BULL, BTCTrendState.BULL):
                direction = SignalDirection.LONG
            elif trend_state in (BTCTrendState.STRONG_BEAR, BTCTrendState.BEAR):
                direction = SignalDirection.SHORT
            else:
                # For neutral/weak trends, use on-chain as tiebreaker
                if on_chain_score > 0.2:
                    direction = SignalDirection.LONG
                elif on_chain_score < -0.2:
                    direction = SignalDirection.SHORT
                else:
                    continue  # Skip neutral conditions
            
            # Calculate confidence
            confidence = 0.5
            
            # Trend strength contribution
            if trend_state == BTCTrendState.STRONG_BULL:
                confidence += 0.20
            elif trend_state == BTCTrendState.BULL:
                confidence += 0.15
            elif trend_state == BTCTrendState.WEAK_BULL:
                confidence += 0.05
            elif trend_state == BTCTrendState.STRONG_BEAR:
                confidence += 0.20 if direction == SignalDirection.SHORT else -0.10
            elif trend_state == BTCTrendState.BEAR:
                confidence += 0.15 if direction == SignalDirection.SHORT else -0.05
            
            # On-chain contribution
            confidence += on_chain_score * 0.15
            
            # Volume confirmation
            volume_ratio = data.get("volume_ratio", 1.0)
            if volume_ratio > 1.3:
                confidence += 0.10
            elif volume_ratio > 1.1:
                confidence += 0.05
            elif volume_ratio < 0.8:
                confidence -= 0.10
            
            # RSI contribution
            rsi = data.get("rsi_14", 50)
            if direction == SignalDirection.LONG:
                if 40 < rsi < 60:
                    confidence += 0.05  # Healthy zone
                elif rsi > 75:
                    confidence -= 0.15  # Overbought
            else:  # SHORT
                if 40 < rsi < 60:
                    confidence += 0.05
                elif rsi < 25:
                    confidence -= 0.15  # Oversold
            
            # Moving average alignment
            price = data["price"]
            if price > data.get("ema_20", price) > data.get("ema_50", price) > data.get("ema_200", price):
                confidence += 0.10 if direction == SignalDirection.LONG else -0.05
            
            confidence = max(min(confidence, 1.0), 0.0)
            
            # Calculate dynamic risk
            dynamic_risk = self.calculate_dynamic_risk(confidence)
            
            # Skip very low confidence signals
            if dynamic_risk.confidence_level == ConfidenceLevel.VERY_LOW:
                continue
            
            # Calculate entry/stop/target
            atr_pct = data.get("atr_pct", 2.0)
            supports = data.get("support_levels", [])
            resistances = data.get("resistance_levels", [])
            
            if direction == SignalDirection.LONG:
                entry = price
                # Use nearest support or 1.5 ATR
                stop = supports[0] if supports else price * (1 - atr_pct * 1.5 / 100)
                # Use nearest resistance or 3 ATR for 2:1 R:R
                target = resistances[0] if resistances else price * (1 + atr_pct * 3 / 100)
            else:
                entry = price
                stop = resistances[0] if resistances else price * (1 + atr_pct * 1.5 / 100)
                target = supports[0] if supports else price * (1 - atr_pct * 3 / 100)
            
            # Create signal context
            btc_context = BTCSignalContext(
                trend_state=trend_state,
                dynamic_risk=dynamic_risk,
                on_chain=on_chain if isinstance(on_chain, OnChainBTCMetrics) else OnChainBTCMetrics(),
                market_regime=market_regime,
                support_levels=supports,
                resistance_levels=resistances,
                next_halving_days=self.days_to_halving(),
            )
            
            candidate = SignalCandidate(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                position_size_pct=dynamic_risk.max_position_pct,
                confidence=confidence,
                timeframe="4H",
                reason=f"BTC {direction.value}: {trend_state.value}, on-chain {on_chain_score:+.2f}, {market_regime}",
                technical=TechnicalContext(
                    trend_4h=data.get("trend_4h", "sideways"),
                    trend_1d=data.get("trend_1d", "sideways"),
                    key_levels=supports[:2] + resistances[:2],
                    atr_pct=atr_pct,
                    rsi=rsi,
                    volume_ratio=volume_ratio,
                ),
                metadata={
                    "btc_context": btc_context.to_dict(),
                    "dynamic_risk": dynamic_risk.to_dict(),
                    "on_chain_score": on_chain_score,
                    "market_regime": market_regime,
                }
            )
            
            candidates.append(candidate)
        
        self.timing.candidate_identification_ms = self._time_ms() - start
        return candidates
    
    async def validate_candidate(
        self, 
        candidate: SignalCandidate,
        market_data: Dict[str, Any],
        account_balance: float
    ) -> Tuple[bool, SignalQuality, List[str]]:
        """
        Validate Bitcoin signal with comprehensive checks.
        """
        start = self._time_ms()
        warnings = []
        
        # Get dynamic risk from metadata
        dynamic_risk_data = candidate.metadata.get("dynamic_risk", {})
        adjusted_risk = dynamic_risk_data.get("adjusted_risk_pct", self.base_risk_pct)
        
        # Data quality check
        required_fields = ["price", "volume_24h_usd", "market_cap_usd", "spread_pct", "rsi_14"]
        available = sum(1 for f in required_fields if f in market_data and market_data[f])
        data_quality_pct = (available / len(required_fields)) * 100
        
        # Market gates check
        gate_result = self.spot_gates.check_all(
            symbol=candidate.symbol,
            volume_24h=market_data.get("volume_24h_usd", 0),
            market_cap=market_data.get("market_cap_usd", 0),
            spread_pct=market_data.get("spread_pct", 1.0),
            exchange_listings=market_data.get("exchange_listings", 1),
            current_concentration_pct=candidate.position_size_pct,
        )
        
        if not gate_result.passed:
            for gate in gate_result.gate_results:
                if gate.status.value == "fail":
                    warnings.append(f"Gate {gate.gate_id}: {gate.message}")
            self.timing.validation_ms += self._time_ms() - start
            return False, SignalQuality.REJECTED, warnings
        
        # Hard rules check
        result = self.rules_engine.validate_trade(
            symbol=candidate.symbol,
            market="crypto_spot",
            position_size_pct=candidate.position_size_pct,
            account_balance=account_balance,
            stop_loss_set=candidate.stop_loss > 0,
            risk_per_trade_pct=adjusted_risk,
            data_quality_pct=data_quality_pct,
            liquidity_usd=market_data.get("volume_24h_usd", 0),
            leverage=1.0,  # Spot
            trend_direction=candidate.technical.trend_4h if candidate.technical else None,
        )
        
        if not result.hard_rule_passed:
            for v in result.hard_violations:
                warnings.append(f"HR {v.rule_id.value}: {v.consequence}")
            self.timing.validation_ms += self._time_ms() - start
            return False, SignalQuality.REJECTED, warnings
        
        # Determine quality based on confidence level
        conf_level = candidate.metadata.get("dynamic_risk", {}).get("confidence_level", "medium")
        soft_warnings = result.soft_rule_warnings
        
        if conf_level == "very_high" and soft_warnings == 0:
            quality = SignalQuality.A_PLUS
        elif conf_level in ("very_high", "high") and soft_warnings <= 1:
            quality = SignalQuality.A
        elif conf_level in ("high", "medium") and soft_warnings <= 2:
            quality = SignalQuality.B
        elif conf_level == "medium":
            quality = SignalQuality.C
        else:
            quality = SignalQuality.D
        
        # Add soft rule warnings
        for check in result.soft_checks:
            if not check.passed:
                warnings.append(f"SR {check.rule_id.value}: {check.recommendation}")
        
        self.timing.validation_ms += self._time_ms() - start
        return True, quality, warnings
    
    async def run(
        self,
        symbols: Optional[List[str]] = None,
        account_balance: float = 10000.0,
        max_signals: int = 3,
    ) -> WorkflowResult:
        """
        Run Bitcoin workflow with timing metrics.
        """
        total_start = self._time_ms()
        
        # Reset timing metrics
        self.timing = TimingMetrics()
        
        # Run parent workflow
        result = await super().run(symbols, account_balance, max_signals)
        
        # Add timing to result
        self.timing.total_ms = self._time_ms() - total_start
        result.metadata = {
            "timing": self.timing.to_dict(),
            "workflow": "bitcoin",
            "focus": "BTC spot with dynamic risk",
        }
        
        return result
