"""
Futures Workflow

Screening workflow for futures trading.
Focus: ES/NQ micros initially, then full contracts
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
from ..rules import RulesEngine, FuturesGates

logger = logging.getLogger(__name__)


@dataclass
class COTData:
    """Commitment of Traders data."""
    commercial_net: int  # Commercials net position
    commercial_change: int  # Weekly change
    commercial_extreme: bool  # At historical extreme
    speculator_net: int  # Large specs net position
    speculator_sentiment: str  # "bullish", "bearish", "neutral"


class FuturesWorkflow(BaseWorkflow):
    """
    Futures market screening workflow.
    
    FOCUS:
    - Start with micros (MES, MNQ) to learn
    - Graduate to full contracts (ES, NQ) when consistent
    - Also monitor: CL (crude), GC (gold)
    
    Signal Generation Criteria:
    1. Technical: Market structure, key levels
    2. COT: Align with commercial positioning
    3. Session: Prefer RTH (Regular Trading Hours)
    """
    
    market_type = "futures"
    
    # Primary watchlist - start with micros
    MICRO_CONTRACTS = ["MES", "MNQ", "MCL", "MGC"]
    
    # Full contracts (graduate to these)
    FULL_CONTRACTS = ["ES", "NQ", "CL", "GC", "ZB", "ZN"]
    
    # Regular Trading Hours (EST)
    RTH_START = time(9, 30)
    RTH_END = time(16, 0)
    
    # Contract specifications
    CONTRACT_SPECS = {
        "MES": {"tick_size": 0.25, "tick_value": 1.25, "margin": 1500, "full_size": "ES"},
        "MNQ": {"tick_size": 0.25, "tick_value": 0.50, "margin": 1800, "full_size": "NQ"},
        "ES": {"tick_size": 0.25, "tick_value": 12.50, "margin": 15000},
        "NQ": {"tick_size": 0.25, "tick_value": 5.00, "margin": 18000},
        "CL": {"tick_size": 0.01, "tick_value": 10.00, "margin": 8000},
        "GC": {"tick_size": 0.10, "tick_value": 10.00, "margin": 10000},
        "MCL": {"tick_size": 0.01, "tick_value": 1.00, "margin": 800, "full_size": "CL"},
        "MGC": {"tick_size": 0.10, "tick_value": 1.00, "margin": 1000, "full_size": "GC"},
    }
    
    def __init__(self, use_micros: bool = True):
        self.rules_engine = RulesEngine()
        self.futures_gates = FuturesGates()
        self.use_micros = use_micros
    
    @property
    def watchlist(self) -> List[str]:
        return self.MICRO_CONTRACTS if self.use_micros else self.FULL_CONTRACTS
    
    def is_rth(self, current_time: Optional[time] = None) -> bool:
        """Check if current time is Regular Trading Hours."""
        if current_time is None:
            current_time = datetime.now().time()
        return self.RTH_START <= current_time <= self.RTH_END
    
    def get_contract_spec(self, symbol: str) -> Dict[str, Any]:
        """Get contract specifications."""
        # Strip month/year code if present (e.g., ESZ24 -> ES)
        base_symbol = ''.join(c for c in symbol if c.isalpha())[:3]
        return self.CONTRACT_SPECS.get(base_symbol, {})
    
    async def fetch_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch market data for futures symbols.
        
        In production, this would call:
        - Futures broker API (Interactive Brokers, etc.)
        - COT report data
        - Volume profile data
        """
        # Placeholder - return mock data
        data = {}
        
        # Mock COT data
        cot_data = {
            "ES": COTData(commercial_net=-50000, commercial_change=-5000, commercial_extreme=False, speculator_net=30000, speculator_sentiment="bullish"),
            "NQ": COTData(commercial_net=-35000, commercial_change=-3000, commercial_extreme=False, speculator_net=25000, speculator_sentiment="bullish"),
            "CL": COTData(commercial_net=150000, commercial_change=10000, commercial_extreme=True, speculator_net=-100000, speculator_sentiment="bearish"),
            "GC": COTData(commercial_net=-80000, commercial_change=5000, commercial_extreme=False, speculator_net=60000, speculator_sentiment="bullish"),
        }
        
        for symbol in symbols:
            spec = self.get_contract_spec(symbol)
            base_symbol = ''.join(c for c in symbol if c.isalpha())[:3]
            full_symbol = spec.get("full_size", base_symbol)
            
            # Determine prices based on contract
            if base_symbol in ["ES", "MES"]:
                price = 5050.0
                daily_range = 50
            elif base_symbol in ["NQ", "MNQ"]:
                price = 17500.0
                daily_range = 200
            elif base_symbol in ["CL", "MCL"]:
                price = 75.0
                daily_range = 2.0
            elif base_symbol in ["GC", "MGC"]:
                price = 2050.0
                daily_range = 20
            else:
                price = 100.0
                daily_range = 5
            
            data[symbol] = {
                "price": price,
                "daily_volume": 1_500_000 if base_symbol in ["ES", "NQ"] else 500_000,
                "tick_size": spec.get("tick_size", 0.25),
                "tick_value": spec.get("tick_value", 12.50),
                "margin": spec.get("margin", 10000),
                "daily_range": daily_range,
                "is_micro": base_symbol.startswith("M"),
                # Technical data
                "trend_4h": "up",
                "trend_1d": "up",
                "rsi_14": 58,
                "volume_ratio": 1.2,
                "atr_points": daily_range * 0.6,
                "key_levels": {
                    "support": [price - daily_range * 0.5, price - daily_range],
                    "resistance": [price + daily_range * 0.5, price + daily_range],
                },
                # COT data (from full contract)
                "cot": cot_data.get(full_symbol, None),
            }
        
        return data
    
    async def identify_candidates(
        self, 
        market_data: Dict[str, Dict[str, Any]]
    ) -> List[SignalCandidate]:
        """
        Identify potential futures signals.
        
        Criteria:
        - Trend alignment with higher timeframes
        - Volume confirmation
        - COT alignment (if available)
        - Prefer RTH for better fills
        """
        candidates = []
        is_rth = self.is_rth()
        
        for symbol, data in market_data.items():
            trend_4h = data.get("trend_4h", "sideways")
            trend_1d = data.get("trend_1d", "sideways")
            
            # Need trend alignment
            if trend_4h != trend_1d or trend_4h == "sideways":
                continue
            
            # Check volume
            volume_ratio = data.get("volume_ratio", 0)
            if volume_ratio < 0.8:
                continue
            
            price = data["price"]
            tick_size = data.get("tick_size", 0.25)
            atr_points = data.get("atr_points", 10)
            
            direction = SignalDirection.LONG if trend_4h == "up" else SignalDirection.SHORT
            
            # Check COT alignment
            cot = data.get("cot")
            cot_aligned = True
            if cot:
                if direction == SignalDirection.LONG and cot.speculator_sentiment == "bearish":
                    cot_aligned = False
                elif direction == SignalDirection.SHORT and cot.speculator_sentiment == "bullish":
                    cot_aligned = False
            
            # Calculate levels
            key_levels = data.get("key_levels", {})
            supports = key_levels.get("support", [])
            resistances = key_levels.get("resistance", [])
            
            if direction == SignalDirection.LONG:
                entry = price
                stop = supports[0] if supports else price - atr_points * 1.5
                target = resistances[0] if resistances else price + atr_points * 2
            else:
                entry = price
                stop = resistances[0] if resistances else price + atr_points * 1.5
                target = supports[0] if supports else price - atr_points * 2
            
            # Calculate confidence
            confidence = 0.5
            if is_rth:
                confidence += 0.1
            if cot_aligned:
                confidence += 0.15
            if volume_ratio > 1.2:
                confidence += 0.1
            
            # Boost for micros (learning phase)
            if data.get("is_micro", False):
                confidence += 0.05  # Slight boost for appropriate sizing
            
            confidence = min(max(confidence, 0.0), 1.0)
            
            # Position sizing - number of contracts based on account
            # For $10K account with micro ES at $1500 margin
            # 2 contracts = $3000 margin = 30% margin utilization
            suggested_contracts = 2 if data.get("is_micro", False) else 1
            
            candidate = SignalCandidate(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                position_size_pct=30.0,  # Margin utilization %
                confidence=confidence,
                timeframe="4H",
                reason=f"Futures {direction.value}: Trend aligned, {'RTH' if is_rth else 'ETH'}, COT {'aligned' if cot_aligned else 'contrary'}",
                technical=TechnicalContext(
                    trend_4h=trend_4h,
                    trend_1d=trend_1d,
                    key_levels=supports + resistances,
                    atr_pct=(atr_points / price) * 100,
                    rsi=data.get("rsi_14", 50),
                    volume_ratio=volume_ratio,
                ),
                metadata={
                    "is_rth": is_rth,
                    "is_micro": data.get("is_micro", False),
                    "suggested_contracts": suggested_contracts,
                    "margin_per_contract": data.get("margin", 10000),
                    "tick_value": data.get("tick_value", 12.50),
                    "cot_aligned": cot_aligned,
                    "cot_data": {
                        "commercial_net": cot.commercial_net,
                        "speculator_sentiment": cot.speculator_sentiment,
                    } if cot else None,
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
        Validate a futures signal candidate.
        """
        warnings = []
        
        # Calculate margin usage
        num_contracts = candidate.metadata.get("suggested_contracts", 1)
        margin_per_contract = candidate.metadata.get("margin_per_contract", 10000)
        total_margin = num_contracts * margin_per_contract
        margin_pct = (total_margin / account_balance) * 100
        
        # Check futures gates
        gate_result = self.futures_gates.check_all(
            symbol=candidate.symbol,
            daily_volume=market_data.get("daily_volume", 0),
            margin_used_pct=margin_pct,
            num_contracts=num_contracts,
            is_rth=candidate.metadata.get("is_rth", False),
            cot_data={
                "commercial_net": market_data.get("cot", COTData(0, 0, False, 0, "neutral")).commercial_net,
                "commercial_extreme": market_data.get("cot", COTData(0, 0, False, 0, "neutral")).commercial_extreme,
            } if market_data.get("cot") else None,
        )
        
        if not gate_result.passed:
            for gate in gate_result.gate_results:
                if gate.status.value == "fail":
                    warnings.append(f"Gate {gate.gate_id}: {gate.message}")
            return False, SignalQuality.REJECTED, warnings
        
        # Calculate risk in dollars
        entry = candidate.entry_price
        stop = candidate.stop_loss
        tick_value = candidate.metadata.get("tick_value", 12.50)
        tick_size = market_data.get("tick_size", 0.25)
        
        risk_points = abs(entry - stop)
        risk_ticks = risk_points / tick_size
        risk_dollars = risk_ticks * tick_value * num_contracts
        risk_pct = (risk_dollars / account_balance) * 100
        
        # Check hard rules
        result = self.rules_engine.validate_trade(
            symbol=candidate.symbol,
            market="futures",
            position_size_pct=margin_pct,
            account_balance=account_balance,
            stop_loss_set=candidate.stop_loss > 0,
            risk_per_trade_pct=risk_pct,
            data_quality_pct=85.0,
            liquidity_usd=market_data.get("daily_volume", 0) * market_data.get("price", 100),
            leverage=1.0,  # Futures margin is not leverage in the same sense
            trend_direction=candidate.technical.trend_4h if candidate.technical else None,
        )
        
        if not result.hard_rule_passed:
            for v in result.hard_violations:
                warnings.append(f"HR {v.rule_id.value}: {v.consequence}")
            return False, SignalQuality.REJECTED, warnings
        
        # Determine quality
        soft_warnings = result.soft_rule_warnings
        cot_aligned = candidate.metadata.get("cot_aligned", True)
        is_rth = candidate.metadata.get("is_rth", False)
        
        if candidate.confidence >= 0.75 and soft_warnings == 0 and cot_aligned and is_rth:
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
        
        # Add RTH warning if trading outside hours
        if not is_rth:
            warnings.append("Trading outside RTH - expect wider spreads")
        
        # Add risk info
        candidate.metadata["risk_dollars"] = risk_dollars
        candidate.metadata["risk_pct"] = risk_pct
        
        return True, quality, warnings
