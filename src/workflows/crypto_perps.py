"""
Crypto Perpetual Futures Workflow

Screening workflow for cryptocurrency perpetual futures trading.
Focus: BTC/ETH only (until rules mastered)
Max leverage: 5x
"""

from dataclasses import dataclass
from datetime import datetime
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
from ..rules import RulesEngine, CryptoPerpsGates

logger = logging.getLogger(__name__)


class CryptoPerpsWorkflow(BaseWorkflow):
    """
    Crypto Perpetual Futures screening workflow.
    
    STRICT RULES:
    - BTC/ETH ONLY until consistently profitable
    - Max 5x leverage
    - 50% buffer to liquidation minimum
    - Funding rate consideration
    
    Signal Generation Criteria:
    1. Technical: Strong trend, momentum confirmation
    2. Funding: Prefer positions that collect funding
    3. Risk: Strict leverage and liquidation management
    """
    
    market_type = "crypto_perps"
    
    # Only BTC/ETH for now
    ALLOWED_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
    MAX_LEVERAGE = 5.0
    MIN_LIQUIDATION_BUFFER = 0.50  # 50%
    
    def __init__(self):
        self.rules_engine = RulesEngine()
        self.perps_gates = CryptoPerpsGates()
    
    @property
    def watchlist(self) -> List[str]:
        return self.ALLOWED_SYMBOLS
    
    async def fetch_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch market data for crypto perps symbols.
        
        In production, this would call:
        - Exchange API (Binance Futures, Bybit, etc.)
        - Funding rate data
        - Open interest data
        """
        # Placeholder - return mock data
        data = {}
        for symbol in symbols:
            base = symbol.replace("USDT", "")
            
            # Skip if not in allowed list
            if symbol not in self.ALLOWED_SYMBOLS:
                continue
            
            data[symbol] = {
                "price": 50000 if base == "BTC" else 3000,
                "funding_rate_pct": 0.01 if base == "BTC" else 0.008,  # Per 8h
                "open_interest": 5_000_000_000 if base == "BTC" else 2_000_000_000,
                "volume_24h": 20_000_000_000 if base == "BTC" else 8_000_000_000,
                "long_short_ratio": 1.2 if base == "BTC" else 0.9,
                "liquidation_heatmap": {
                    "above": [52000, 55000] if base == "BTC" else [3200, 3500],
                    "below": [48000, 45000] if base == "BTC" else [2800, 2500],
                },
                # Technical data
                "trend_4h": "up",
                "trend_1d": "up",
                "rsi_14": 58,
                "volume_ratio": 1.3,
                "atr_pct": 3.0,
                "key_levels": {
                    "support": [48000, 45000] if base == "BTC" else [2800, 2600],
                    "resistance": [52000, 55000] if base == "BTC" else [3200, 3500],
                },
            }
        
        return data
    
    async def identify_candidates(
        self, 
        market_data: Dict[str, Dict[str, Any]]
    ) -> List[SignalCandidate]:
        """
        Identify potential crypto perps signals.
        
        Criteria:
        - Strong trend alignment
        - Funding rate favorable (or acceptable cost)
        - Liquidation clusters to target
        - Good R:R ratio
        """
        candidates = []
        
        for symbol, data in market_data.items():
            # Only BTC/ETH
            if symbol not in self.ALLOWED_SYMBOLS:
                continue
            
            trend_4h = data.get("trend_4h", "sideways")
            trend_1d = data.get("trend_1d", "sideways")
            
            # Need trend alignment for perps
            if trend_4h != trend_1d or trend_4h == "sideways":
                continue
            
            price = data["price"]
            funding_rate = data.get("funding_rate_pct", 0)
            long_short_ratio = data.get("long_short_ratio", 1.0)
            
            # Determine direction based on trend
            direction = SignalDirection.LONG if trend_4h == "up" else SignalDirection.SHORT
            
            # Check if funding is favorable
            # Longs pay shorts when funding is positive
            funding_favorable = (
                (direction == SignalDirection.LONG and funding_rate < 0.05) or
                (direction == SignalDirection.SHORT and funding_rate > -0.05)
            )
            
            # Calculate levels
            atr_pct = data.get("atr_pct", 3.0)
            key_levels = data.get("key_levels", {})
            supports = key_levels.get("support", [])
            resistances = key_levels.get("resistance", [])
            
            # For perps, use tighter stops with leverage
            if direction == SignalDirection.LONG:
                entry = price
                stop = supports[0] if supports else price * (1 - atr_pct / 100)
                target = resistances[0] if resistances else price * (1 + atr_pct / 100 * 2)
            else:
                entry = price
                stop = resistances[0] if resistances else price * (1 + atr_pct / 100)
                target = supports[0] if supports else price * (1 - atr_pct / 100 * 2)
            
            # Calculate confidence
            confidence = 0.5
            if funding_favorable:
                confidence += 0.15
            if data.get("volume_ratio", 0) > 1.2:
                confidence += 0.1
            rsi = data.get("rsi_14", 50)
            if 40 < rsi < 60:
                confidence += 0.1
            
            # Penalize crowded trades
            if direction == SignalDirection.LONG and long_short_ratio > 1.5:
                confidence -= 0.1  # Too crowded long
            elif direction == SignalDirection.SHORT and long_short_ratio < 0.7:
                confidence -= 0.1  # Too crowded short
            
            confidence = min(max(confidence, 0.0), 1.0)
            
            # Position size - use 10% of account with leverage consideration
            # With 5x leverage, 10% margin = 50% effective exposure
            position_size_pct = 10.0
            
            candidate = SignalCandidate(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                position_size_pct=position_size_pct,
                confidence=confidence,
                timeframe="4H",
                reason=f"Perps {direction.value}: Trend aligned, funding {'favorable' if funding_favorable else 'acceptable'}",
                technical=TechnicalContext(
                    trend_4h=trend_4h,
                    trend_1d=trend_1d,
                    key_levels=supports + resistances,
                    atr_pct=atr_pct,
                    rsi=rsi,
                    volume_ratio=data.get("volume_ratio", 1.0),
                ),
                metadata={
                    "funding_rate_pct": funding_rate,
                    "funding_favorable": funding_favorable,
                    "long_short_ratio": long_short_ratio,
                    "suggested_leverage": min(3.0, self.MAX_LEVERAGE),
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
        Validate a crypto perps signal candidate.
        """
        warnings = []
        
        # Calculate liquidation distance
        leverage = candidate.metadata.get("suggested_leverage", 3.0)
        entry = candidate.entry_price
        stop = candidate.stop_loss
        
        # Liquidation price calculation (simplified)
        # Long: liq_price = entry - (entry / leverage)
        # Short: liq_price = entry + (entry / leverage)
        if candidate.direction == SignalDirection.LONG:
            liq_price = entry * (1 - 1/leverage)
            liq_distance_pct = (entry - liq_price) / entry * 100
        else:
            liq_price = entry * (1 + 1/leverage)
            liq_distance_pct = (liq_price - entry) / entry * 100
        
        # Check perps-specific gates
        gate_result = self.perps_gates.check_all(
            symbol=candidate.symbol,
            leverage=leverage,
            funding_rate_pct=market_data.get("funding_rate_pct", 0),
            open_interest=market_data.get("open_interest", 0),
            position_size_pct=candidate.position_size_pct,
            liquidation_distance_pct=liq_distance_pct,
            account_equity=account_balance,
        )
        
        if not gate_result.passed:
            for gate in gate_result.gate_results:
                if gate.status.value == "fail":
                    warnings.append(f"Gate {gate.gate_id}: {gate.message}")
            return False, SignalQuality.REJECTED, warnings
        
        # Check hard rules
        result = self.rules_engine.validate_trade(
            symbol=candidate.symbol,
            market="crypto_perps",
            position_size_pct=candidate.position_size_pct,
            account_balance=account_balance,
            stop_loss_set=candidate.stop_loss > 0,
            risk_per_trade_pct=candidate.risk_pct,
            data_quality_pct=85.0,  # Assume good data for perps
            liquidity_usd=market_data.get("open_interest", 0),
            leverage=leverage,
            trend_direction=candidate.technical.trend_4h if candidate.technical else None,
        )
        
        if not result.hard_rule_passed:
            for v in result.hard_violations:
                warnings.append(f"HR {v.rule_id.value}: {v.consequence}")
            return False, SignalQuality.REJECTED, warnings
        
        # Determine quality
        soft_warnings = result.soft_rule_warnings
        funding_favorable = candidate.metadata.get("funding_favorable", False)
        
        # Boost quality if funding favorable
        if candidate.confidence >= 0.75 and soft_warnings == 0 and funding_favorable:
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
        
        # Add funding warning if not favorable
        if not funding_favorable:
            warnings.append(f"Funding rate against position: {market_data.get('funding_rate_pct', 0):.3f}%")
        
        return True, quality, warnings
