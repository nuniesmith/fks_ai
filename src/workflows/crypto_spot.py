"""
Crypto Spot Workflow

Screening workflow for cryptocurrency spot trading.
Focus: BTC/ETH majors, SOL/LINK/AVAX alts
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
from ..rules import RulesEngine, CryptoSpotGates

logger = logging.getLogger(__name__)


@dataclass
class OnChainMetrics:
    """On-chain metrics for crypto assets."""
    exchange_inflow_spike: bool = False
    exchange_outflow_spike: bool = False
    whale_accumulation: bool = False
    whale_distribution: bool = False
    active_addresses_change_pct: float = 0.0
    transaction_volume_change_pct: float = 0.0


class CryptoSpotWorkflow(BaseWorkflow):
    """
    Crypto Spot market screening workflow.
    
    Signal Generation Criteria:
    1. Technical: Trend alignment, key level bounces, volume confirmation
    2. On-chain: Whale accumulation, exchange outflows (bullish)
    3. Risk: Proper position sizing, R:R >= 2:1
    
    Priority Symbols:
    - Tier 1: BTC, ETH (higher position sizes allowed)
    - Tier 2: SOL, LINK, AVAX, DOT (standard sizing)
    - Tier 3: Other top 20 by market cap
    """
    
    market_type = "crypto_spot"
    
    # Watchlist by tier
    TIER_1 = ["BTCUSDT", "ETHUSDT"]
    TIER_2 = ["SOLUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "ATOMUSDT"]
    TIER_3 = ["BNBUSDT", "XRPUSDT", "ADAUSDT", "MATICUSDT", "DOGEUSDT"]
    
    def __init__(self):
        self.rules_engine = RulesEngine()
        self.spot_gates = CryptoSpotGates()
    
    @property
    def watchlist(self) -> List[str]:
        return self.TIER_1 + self.TIER_2 + self.TIER_3
    
    def get_tier(self, symbol: str) -> int:
        """Get tier for a symbol (1, 2, or 3)."""
        if symbol in self.TIER_1:
            return 1
        elif symbol in self.TIER_2:
            return 2
        else:
            return 3
    
    def get_max_position_pct(self, symbol: str) -> float:
        """Get max position size % based on tier."""
        tier = self.get_tier(symbol)
        if tier == 1:
            return 10.0  # BTC/ETH can be 10%
        elif tier == 2:
            return 5.0
        else:
            return 3.0
    
    async def fetch_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch market data for crypto spot symbols.
        
        In production, this would call:
        - Exchange API (Binance, Coinbase, etc.)
        - CoinGecko/CoinMarketCap for market cap
        - Glassnode/IntoTheBlock for on-chain
        """
        # Placeholder - return mock data for now
        # TODO: Implement actual data fetching
        data = {}
        for symbol in symbols:
            base = symbol.replace("USDT", "")
            
            # Mock data - replace with actual API calls
            data[symbol] = {
                "price": 50000 if base == "BTC" else 3000 if base == "ETH" else 100,
                "volume_24h": 500_000_000 if base in ["BTC", "ETH"] else 50_000_000,
                "market_cap": 1_000_000_000_000 if base == "BTC" else 300_000_000_000 if base == "ETH" else 10_000_000_000,
                "spread_pct": 0.05,
                "exchange_listings": 10 if base in ["BTC", "ETH"] else 5,
                "change_24h_pct": 2.5,
                "volume_ratio": 1.2,  # Current vs 20-day avg
                "rsi_14": 55,
                "trend_4h": "up",
                "trend_1d": "up",
                "atr_pct": 2.5,
                "key_levels": {
                    "support": [48000, 45000] if base == "BTC" else [2800, 2600] if base == "ETH" else [90, 85],
                    "resistance": [52000, 55000] if base == "BTC" else [3200, 3500] if base == "ETH" else [110, 120],
                },
                "on_chain": OnChainMetrics(
                    whale_accumulation=True if base in ["BTC", "ETH"] else False,
                    exchange_outflow_spike=True if base == "BTC" else False,
                ),
            }
        
        return data
    
    async def identify_candidates(
        self, 
        market_data: Dict[str, Dict[str, Any]]
    ) -> List[SignalCandidate]:
        """
        Identify potential crypto spot signals.
        
        Criteria:
        - Trend alignment (4H and 1D same direction)
        - Volume confirmation (> 1.0x average)
        - RSI not overbought/oversold
        - Near key support/resistance
        - On-chain bullish (optional boost)
        """
        candidates = []
        
        for symbol, data in market_data.items():
            # Skip if no trend alignment
            trend_4h = data.get("trend_4h", "sideways")
            trend_1d = data.get("trend_1d", "sideways")
            
            if trend_4h == "sideways" or trend_1d == "sideways":
                continue
            
            if trend_4h != trend_1d:
                continue  # Need alignment
            
            # Skip if low volume
            volume_ratio = data.get("volume_ratio", 0)
            if volume_ratio < 0.8:
                continue
            
            # Skip if RSI extreme
            rsi = data.get("rsi_14", 50)
            if rsi > 75 or rsi < 25:
                continue
            
            # Determine direction and levels
            price = data["price"]
            direction = SignalDirection.LONG if trend_4h == "up" else SignalDirection.SHORT
            
            key_levels = data.get("key_levels", {})
            supports = key_levels.get("support", [])
            resistances = key_levels.get("resistance", [])
            
            # Calculate entry, stop, target
            atr_pct = data.get("atr_pct", 2.0)
            
            if direction == SignalDirection.LONG:
                entry = price
                stop = supports[0] if supports else price * (1 - atr_pct / 100 * 1.5)
                target = resistances[0] if resistances else price * (1 + atr_pct / 100 * 3)
            else:
                entry = price
                stop = resistances[0] if resistances else price * (1 + atr_pct / 100 * 1.5)
                target = supports[0] if supports else price * (1 - atr_pct / 100 * 3)
            
            # Calculate confidence
            confidence = 0.5  # Base
            if volume_ratio > 1.2:
                confidence += 0.1
            if 40 < rsi < 60:
                confidence += 0.1
            
            # On-chain boost
            on_chain = data.get("on_chain", OnChainMetrics())
            if isinstance(on_chain, OnChainMetrics):
                if on_chain.whale_accumulation:
                    confidence += 0.1
                if on_chain.exchange_outflow_spike:
                    confidence += 0.1
                if on_chain.whale_distribution:
                    confidence -= 0.15
            
            confidence = min(max(confidence, 0.0), 1.0)
            
            # Create candidate
            candidate = SignalCandidate(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                position_size_pct=self.get_max_position_pct(symbol),
                confidence=confidence,
                timeframe="4H",
                reason=f"Trend aligned ({trend_4h}), volume confirmed ({volume_ratio:.1f}x)",
                technical=TechnicalContext(
                    trend_4h=trend_4h,
                    trend_1d=trend_1d,
                    key_levels=supports + resistances,
                    atr_pct=atr_pct,
                    rsi=rsi,
                    volume_ratio=volume_ratio,
                ),
                metadata={
                    "tier": self.get_tier(symbol),
                    "on_chain_bullish": on_chain.whale_accumulation if isinstance(on_chain, OnChainMetrics) else False,
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
        Validate a crypto spot signal candidate.
        """
        warnings = []
        
        # Calculate data quality %
        required_fields = ["price", "volume_24h", "market_cap", "spread_pct", "rsi_14"]
        available = sum(1 for f in required_fields if f in market_data and market_data[f])
        data_quality_pct = (available / len(required_fields)) * 100
        
        # Check market gates
        gate_result = self.spot_gates.check_all(
            symbol=candidate.symbol,
            volume_24h=market_data.get("volume_24h", 0),
            market_cap=market_data.get("market_cap", 0),
            spread_pct=market_data.get("spread_pct", 1.0),
            exchange_listings=market_data.get("exchange_listings", 1),
            current_concentration_pct=candidate.position_size_pct,
        )
        
        if not gate_result.passed:
            for gate in gate_result.gate_results:
                if gate.status.value == "fail":
                    warnings.append(f"Gate {gate.gate_id}: {gate.message}")
            return False, SignalQuality.REJECTED, warnings
        
        # Check hard rules
        result = self.rules_engine.validate_trade(
            symbol=candidate.symbol,
            market="crypto_spot",
            position_size_pct=candidate.position_size_pct,
            account_balance=account_balance,
            stop_loss_set=candidate.stop_loss > 0,
            risk_per_trade_pct=candidate.risk_pct,
            data_quality_pct=data_quality_pct,
            liquidity_usd=market_data.get("volume_24h", 0),
            leverage=1.0,  # Spot = no leverage
            trend_direction=candidate.technical.trend_4h if candidate.technical else None,
        )
        
        if not result.hard_rule_passed:
            for v in result.hard_violations:
                warnings.append(f"HR {v.rule_id.value}: {v.consequence}")
            return False, SignalQuality.REJECTED, warnings
        
        # Determine quality based on confidence and warnings
        soft_warnings = result.soft_rule_warnings
        if candidate.confidence >= 0.8 and soft_warnings == 0:
            quality = SignalQuality.A_PLUS
        elif candidate.confidence >= 0.7 and soft_warnings <= 1:
            quality = SignalQuality.A
        elif candidate.confidence >= 0.6 and soft_warnings <= 2:
            quality = SignalQuality.B
        elif candidate.confidence >= 0.5:
            quality = SignalQuality.C
        else:
            quality = SignalQuality.D
        
        # Add soft rule warnings
        for check in result.soft_checks:
            if not check.passed:
                warnings.append(f"SR {check.rule_id.value}: {check.recommendation}")
        
        return True, quality, warnings
