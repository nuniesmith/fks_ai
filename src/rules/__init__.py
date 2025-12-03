"""
FKS Trading Rules Engine

A comprehensive rules system for enforcing trading discipline across all asset classes.
Implements both hard rules (non-negotiable) and soft rules (guidelines with override capability).

Architecture:
- hard_rules.py: 12 non-negotiable trading rules (HR-01 to HR-12)
- soft_rules.py: 10 guideline rules that can be overridden with justification
- market_gates.py: Market-specific hard gates for each asset class

Usage:
    from src.rules import RulesEngine, MarketType
    
    engine = RulesEngine()
    is_valid, violations = engine.validate_trade(
        symbol="BTCUSDT",
        market="crypto_spot",
        position_size_pct=5.0,
        account_balance=10000.0,
        stop_loss_set=True,
        risk_per_trade_pct=1.5,
        data_quality_pct=85.0,
        liquidity_usd=100_000_000.0,
        leverage=3.0,
    )
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

# Import submodules
from .hard_rules import HardRules, HardRuleViolation, HardRuleConfig, HardRuleID
from .soft_rules import SoftRules, SoftRuleCheck, SoftRuleOverride, SoftRuleID, SoftRuleConfig
from .market_gates import (
    get_market_gates,
    MarketGates,
    GateResult,
    GateStatus,
    MarketGateResults,
    CryptoSpotGates,
    CryptoPerpsGates,
    ForexGates,
    FuturesGates,
    StocksGates,
)


class MarketType(Enum):
    """Supported market types for trading."""
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_PERPS = "crypto_perps"
    FOREX = "forex"
    FUTURES = "futures"
    STOCKS = "stocks"


@dataclass
class TradeValidationResult:
    """Result of a complete trade validation."""
    is_valid: bool
    hard_rule_passed: bool
    market_gate_passed: bool
    soft_rule_warnings: int
    hard_violations: List[HardRuleViolation]
    gate_results: Optional[MarketGateResults]
    soft_checks: List[SoftRuleCheck]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "hard_rule_passed": self.hard_rule_passed,
            "market_gate_passed": self.market_gate_passed,
            "soft_rule_warnings": self.soft_rule_warnings,
            "hard_violations": [v.to_dict() for v in self.hard_violations],
            "gate_results": self.gate_results.to_dict() if self.gate_results else None,
            "soft_checks": [c.to_dict() for c in self.soft_checks],
            "timestamp": self.timestamp.isoformat()
        }


class RulesEngine:
    """
    Main rules engine that coordinates all rule checks.
    
    This is the primary interface for validating trades against the
    FKS trading framework rules.
    
    Rules Hierarchy:
    1. Hard Rules (HR-01 to HR-12): Non-negotiable, block trades
    2. Market Gates: Asset-class specific requirements, block trades
    3. Soft Rules (SR-01 to SR-10): Guidelines, can be overridden with justification
    """
    
    def __init__(
        self, 
        hard_config: Optional[HardRuleConfig] = None,
        soft_config: Optional[SoftRuleConfig] = None
    ):
        self.hard_rules = HardRules(hard_config)
        self.soft_rules = SoftRules(soft_config)
        self._market_gates: Dict[str, MarketGates] = {}
    
    def get_market_gates(self, market_type: str) -> MarketGates:
        """Get or create market gates for a market type."""
        if market_type not in self._market_gates:
            self._market_gates[market_type] = get_market_gates(market_type)
        return self._market_gates[market_type]
    
    def validate_trade(
        self,
        # Required parameters for hard rules
        symbol: str,
        market: str,
        position_size_pct: float,
        account_balance: float,
        stop_loss_set: bool,
        risk_per_trade_pct: float,
        data_quality_pct: float,
        liquidity_usd: float,
        leverage: float,
        # Optional hard rule parameters
        is_adding_to_loser: bool = False,
        upcoming_news: Optional[List[Dict]] = None,
        daily_pnl_pct: Optional[float] = None,
        weekly_pnl_pct: Optional[float] = None,
        total_24h_pnl_pct: Optional[float] = None,
        # Soft rule parameters
        trend_direction: Optional[str] = None,
        entry_timeframe: str = "1H",
        session_minutes_elapsed: int = 60,
        existing_positions: Optional[List[Dict]] = None,
        has_confirmation: bool = True,
        timeframes_analyzed: int = 2,
        trade_documented: bool = True,
        # Market gate parameters (passed as kwargs)
        **market_gate_kwargs
    ) -> TradeValidationResult:
        """
        Validate a trade against all rules.
        
        Returns TradeValidationResult with full details.
        Trade is valid only if hard rules AND market gates pass.
        Soft rule failures are warnings only.
        """
        # 1. Check hard rules
        hard_passed, hard_violations = self.hard_rules.check_all(
            symbol=symbol,
            market=market,
            position_size_pct=position_size_pct,
            account_balance=account_balance,
            stop_loss_set=stop_loss_set,
            risk_per_trade_pct=risk_per_trade_pct,
            data_quality_pct=data_quality_pct,
            liquidity_usd=liquidity_usd,
            leverage=leverage,
            is_adding_to_loser=is_adding_to_loser,
            upcoming_news=upcoming_news,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl_pct=weekly_pnl_pct,
            total_24h_pnl_pct=total_24h_pnl_pct,
        )
        
        # 2. Check market-specific gates
        gate_results = None
        gate_passed = True
        try:
            gates = self.get_market_gates(market)
            gate_results = gates.check_all(symbol=symbol, **market_gate_kwargs)
            gate_passed = gate_results.passed
        except (ValueError, TypeError) as e:
            # Market gates not configured or missing parameters
            pass
        
        # 3. Check soft rules (warnings only)
        soft_checks, soft_failed = self.soft_rules.check_all(
            symbol=symbol,
            market=market,
            trend_direction=trend_direction,
            entry_timeframe=entry_timeframe,
            session_minutes_elapsed=session_minutes_elapsed,
            existing_positions=existing_positions,
            has_confirmation=has_confirmation,
            timeframes_analyzed=timeframes_analyzed,
            trade_documented=trade_documented,
        )
        
        # Trade is valid if hard rules AND market gates pass
        is_valid = hard_passed and gate_passed
        
        return TradeValidationResult(
            is_valid=is_valid,
            hard_rule_passed=hard_passed,
            market_gate_passed=gate_passed,
            soft_rule_warnings=soft_failed,
            hard_violations=hard_violations,
            gate_results=gate_results,
            soft_checks=soft_checks,
            timestamp=datetime.now()
        )
    
    def check_hard_rules_only(
        self,
        symbol: str,
        market: str,
        position_size_pct: float,
        account_balance: float,
        stop_loss_set: bool,
        risk_per_trade_pct: float,
        data_quality_pct: float,
        liquidity_usd: float,
        leverage: float,
        **kwargs
    ) -> tuple[bool, List[HardRuleViolation]]:
        """Check only hard rules (useful for quick validation)."""
        return self.hard_rules.check_all(
            symbol=symbol,
            market=market,
            position_size_pct=position_size_pct,
            account_balance=account_balance,
            stop_loss_set=stop_loss_set,
            risk_per_trade_pct=risk_per_trade_pct,
            data_quality_pct=data_quality_pct,
            liquidity_usd=liquidity_usd,
            leverage=leverage,
            **kwargs
        )
    
    def check_market_gates_only(self, market_type: str, **kwargs) -> MarketGateResults:
        """Check only market gates for a specific market."""
        gates = self.get_market_gates(market_type)
        return gates.check_all(**kwargs)
    
    def record_soft_override(
        self,
        rule_id: SoftRuleID,
        rule_name: str,
        recommendation: str,
        override_reason: str,
        approved_by: str = "manual"
    ) -> SoftRuleOverride:
        """Record a soft rule override for audit purposes."""
        return self.soft_rules.record_override(
            rule_id=rule_id,
            rule_name=rule_name,
            recommendation=recommendation,
            override_reason=override_reason,
            approved_by=approved_by
        )
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status (pauses, violations, etc.)."""
        return {
            "trading_paused_until": self.hard_rules.trading_paused_until.isoformat() if self.hard_rules.trading_paused_until else None,
            "is_trading_allowed": self.hard_rules.trading_paused_until is None or datetime.now() >= self.hard_rules.trading_paused_until,
            "recent_hard_violations": len(self.hard_rules.violations),
            "soft_overrides_count": len(self.soft_rules.overrides),
        }


# Export main classes
__all__ = [
    # Core types
    "MarketType",
    "TradeValidationResult",
    # Main engine
    "RulesEngine",
    # Hard rules
    "HardRules",
    "HardRuleViolation",
    "HardRuleConfig",
    "HardRuleID",
    # Soft rules
    "SoftRules",
    "SoftRuleCheck",
    "SoftRuleOverride",
    "SoftRuleID",
    "SoftRuleConfig",
    # Market gates
    "get_market_gates",
    "MarketGates",
    "GateResult",
    "GateStatus",
    "MarketGateResults",
    "CryptoSpotGates",
    "CryptoPerpsGates",
    "ForexGates",
    "FuturesGates",
    "StocksGates",
]
