"""
Soft Rules - Trading Guidelines

These are best practices that CAN be overridden with justification.
All overrides are logged for review.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SoftRuleID(Enum):
    """Identifiers for all soft rules."""
    SR_01_TRADE_WITH_TREND = "SR-01"
    SR_02_AVOID_SESSION_OPEN = "SR-02"
    SR_03_PARTIAL_PROFITS = "SR-03"
    SR_04_PREFER_LIQUID_PAIRS = "SR-04"
    SR_05_CHECK_CORRELATION = "SR-05"
    SR_06_WAIT_CONFIRMATION = "SR-06"
    SR_07_OPTIMAL_HOURS = "SR-07"
    SR_08_MULTIPLE_TIMEFRAMES = "SR-08"
    SR_09_DOCUMENT_TRADES = "SR-09"
    SR_10_WEEKLY_REVIEW = "SR-10"


@dataclass
class SoftRuleOverride:
    """Records when a soft rule is overridden."""
    rule_id: SoftRuleID
    rule_name: str
    recommendation: str
    override_reason: str
    timestamp: datetime
    approved_by: str = "manual"  # or "system" for automated overrides
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id.value,
            "rule_name": self.rule_name,
            "recommendation": self.recommendation,
            "override_reason": self.override_reason,
            "timestamp": self.timestamp.isoformat(),
            "approved_by": self.approved_by
        }


@dataclass
class SoftRuleCheck:
    """Result of a soft rule check."""
    rule_id: SoftRuleID
    rule_name: str
    passed: bool
    recommendation: str
    override_conditions: List[str]
    details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id.value,
            "rule_name": self.rule_name,
            "passed": self.passed,
            "recommendation": self.recommendation,
            "override_conditions": self.override_conditions,
            "details": self.details
        }


@dataclass
class SoftRuleConfig:
    """Configuration for soft rules."""
    # SR-01: Trade with trend
    trend_timeframe: str = "4H"  # Minimum timeframe for trend
    
    # SR-02: Avoid session open
    session_open_avoid_minutes: int = 30
    
    # SR-03: Partial profits
    partial_profit_rr: float = 1.0  # Take partials at 1:1
    partial_profit_pct: float = 50.0  # Take 50% off
    
    # SR-04: Prefer liquid pairs
    top_pairs_count: int = 20  # Top N by volume
    
    # SR-05: Check correlation
    max_correlation: float = 0.7  # Max correlation between positions
    
    # SR-06: Wait for confirmation
    confirmation_candles: int = 1  # Wait N candles
    
    # SR-07: Optimal trading hours
    optimal_hours: Dict[str, List[tuple]] = field(default_factory=lambda: {
        "forex": [(8, 12), (13, 17)],  # EST: London-NY overlap
        "crypto": [(0, 24)],  # 24/7 but prefer high volume
        "stocks": [(9, 30), (15, 16)],  # EST: Open and close
        "futures": [(8, 12), (13, 17)],  # CME active hours
    })
    
    # SR-08: Multiple timeframe analysis
    required_timeframes: int = 2  # Check at least 2 timeframes
    
    # SR-09: Document trades
    documentation_required: bool = True
    
    # SR-10: Weekly review
    weekly_review_required: bool = True


class SoftRules:
    """
    Evaluates soft (guideline) trading rules.
    
    Unlike hard rules, soft rules can be overridden with justification.
    All overrides are logged for later review and pattern analysis.
    """
    
    def __init__(self, config: Optional[SoftRuleConfig] = None):
        self.config = config or SoftRuleConfig()
        self.overrides: List[SoftRuleOverride] = []
        
    def check_all(
        self,
        symbol: str,
        market: str,
        trend_direction: Optional[str] = None,  # "up", "down", "sideways"
        entry_timeframe: str = "1H",
        session_minutes_elapsed: int = 60,
        current_rr: float = 0.0,
        existing_positions: Optional[List[Dict]] = None,
        has_confirmation: bool = True,
        current_hour: Optional[int] = None,
        timeframes_analyzed: int = 1,
        trade_documented: bool = True,
    ) -> tuple[List[SoftRuleCheck], int]:
        """
        Check all soft rules.
        
        Returns:
            (checks, failed_count): List of check results and count of failures
        """
        checks = []
        
        # SR-01: Trade with trend
        checks.append(self._check_trend_alignment(trend_direction, entry_timeframe))
        
        # SR-02: Avoid session open
        checks.append(self._check_session_open(session_minutes_elapsed))
        
        # SR-03: Partial profits
        checks.append(self._check_partial_profits(current_rr))
        
        # SR-04: Prefer liquid pairs
        checks.append(self._check_liquid_pairs(symbol, market))
        
        # SR-05: Check correlation
        checks.append(self._check_correlation(symbol, existing_positions))
        
        # SR-06: Wait for confirmation
        checks.append(self._check_confirmation(has_confirmation))
        
        # SR-07: Optimal hours
        checks.append(self._check_optimal_hours(market, current_hour))
        
        # SR-08: Multiple timeframes
        checks.append(self._check_multiple_timeframes(timeframes_analyzed))
        
        # SR-09: Document trades
        checks.append(self._check_documentation(trade_documented))
        
        # SR-10: Weekly review (always passes during trading, checked separately)
        checks.append(SoftRuleCheck(
            rule_id=SoftRuleID.SR_10_WEEKLY_REVIEW,
            rule_name="Weekly Review",
            passed=True,
            recommendation="Review performance weekly",
            override_conditions=[],
            details="Checked separately at end of week"
        ))
        
        failed_count = sum(1 for c in checks if not c.passed)
        return checks, failed_count
    
    def _check_trend_alignment(
        self, 
        trend_direction: Optional[str], 
        entry_timeframe: str
    ) -> SoftRuleCheck:
        """SR-01: Trade with the trend on 4H+ timeframe."""
        # Parse timeframe to minutes
        tf_map = {"1M": 1, "5M": 5, "15M": 15, "30M": 30, "1H": 60, "4H": 240, "1D": 1440}
        entry_mins = tf_map.get(entry_timeframe.upper(), 60)
        trend_mins = tf_map.get(self.config.trend_timeframe, 240)
        
        passed = True
        details = None
        
        if trend_direction == "sideways":
            passed = False
            details = "No clear trend identified"
        elif entry_mins < trend_mins:
            details = f"Entry on {entry_timeframe}, ensure {self.config.trend_timeframe}+ trend aligned"
        
        return SoftRuleCheck(
            rule_id=SoftRuleID.SR_01_TRADE_WITH_TREND,
            rule_name="Trade With Trend",
            passed=passed,
            recommendation=f"Align trades with {self.config.trend_timeframe}+ trend direction",
            override_conditions=["Strong reversal pattern", "Mean reversion strategy", "News-driven setup"],
            details=details
        )
    
    def _check_session_open(self, minutes_elapsed: int) -> SoftRuleCheck:
        """SR-02: Avoid trading first 30 minutes of session."""
        passed = minutes_elapsed >= self.config.session_open_avoid_minutes
        
        return SoftRuleCheck(
            rule_id=SoftRuleID.SR_02_AVOID_SESSION_OPEN,
            rule_name="Avoid Session Open",
            passed=passed,
            recommendation=f"Wait {self.config.session_open_avoid_minutes} minutes after session open",
            override_conditions=["News-driven setup", "Gap fill play", "Pre-planned entry"],
            details=f"Minutes elapsed: {minutes_elapsed}" if not passed else None
        )
    
    def _check_partial_profits(self, current_rr: float) -> SoftRuleCheck:
        """SR-03: Take partial profits at 1:1 RR."""
        passed = current_rr < self.config.partial_profit_rr  # Not yet at take-profit level
        
        return SoftRuleCheck(
            rule_id=SoftRuleID.SR_03_PARTIAL_PROFITS,
            rule_name="Partial Profits",
            passed=passed,
            recommendation=f"Take {self.config.partial_profit_pct}% profit at {self.config.partial_profit_rr}:1 RR",
            override_conditions=["Strong momentum continuation", "Breakout with volume", "Runner position"],
            details=f"Current RR: {current_rr:.2f}" if current_rr >= self.config.partial_profit_rr else None
        )
    
    def _check_liquid_pairs(self, symbol: str, market: str) -> SoftRuleCheck:
        """SR-04: Prefer liquid pairs (top 20 by volume)."""
        # Define top liquid pairs by market
        top_pairs = {
            "crypto": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC",
                      "LINK", "UNI", "LTC", "ATOM", "FIL", "APT", "ARB", "OP", "INJ", "SUI"],
            "forex": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
                     "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "EURCHF", "GBPCHF",
                     "AUDCAD", "NZDJPY", "CADJPY", "AUDNZD", "EURCAD", "GBPAUD"],
            "futures": ["ES", "NQ", "CL", "GC", "SI", "NG", "ZB", "ZN", "6E", "6B",
                       "6J", "BTC", "MBT", "MES", "MNQ", "MCL", "MGC", "RTY", "YM", "ZC"]
        }
        
        market_key = market.lower().replace("_", "").replace("spot", "").replace("perp", "")
        if "crypto" in market_key:
            market_key = "crypto"
        elif "forex" in market_key or "fx" in market_key:
            market_key = "forex"
        else:
            market_key = "futures"
        
        symbol_clean = symbol.upper().replace("USDT", "").replace("USD", "").replace("PERP", "")
        is_top_pair = symbol_clean in top_pairs.get(market_key, [])
        
        return SoftRuleCheck(
            rule_id=SoftRuleID.SR_04_PREFER_LIQUID_PAIRS,
            rule_name="Prefer Liquid Pairs",
            passed=is_top_pair,
            recommendation=f"Trade top {self.config.top_pairs_count} pairs by volume",
            override_conditions=["Specific alpha opportunity", "Sector rotation play", "Arbitrage setup"],
            details=f"{symbol} is {'in' if is_top_pair else 'NOT in'} top {self.config.top_pairs_count} for {market_key}"
        )
    
    def _check_correlation(
        self, 
        symbol: str, 
        existing_positions: Optional[List[Dict]]
    ) -> SoftRuleCheck:
        """SR-05: Check correlation before adding positions."""
        if not existing_positions:
            return SoftRuleCheck(
                rule_id=SoftRuleID.SR_05_CHECK_CORRELATION,
                rule_name="Check Correlation",
                passed=True,
                recommendation="Check correlation with existing positions",
                override_conditions=["Uncorrelated catalyst", "Hedging position", "Different asset class"],
                details="No existing positions"
            )
        
        # Simple correlation check based on symbol similarity
        # In production, use actual correlation calculation
        correlated_symbols = []
        symbol_base = symbol.upper().replace("USDT", "").replace("USD", "")
        
        for pos in existing_positions:
            pos_symbol = pos.get("symbol", "").upper().replace("USDT", "").replace("USD", "")
            
            # Check for same base symbol
            if pos_symbol == symbol_base:
                correlated_symbols.append(pos_symbol)
                continue
            
            # Check for known correlations
            known_correlations = {
                "BTC": ["ETH", "SOL", "BNB"],  # Crypto correlations
                "EURUSD": ["GBPUSD", "AUDUSD"],  # Forex correlations
                "ES": ["NQ", "YM", "RTY"],  # Index correlations
            }
            
            for base, correlated in known_correlations.items():
                if symbol_base == base and pos_symbol in correlated:
                    correlated_symbols.append(pos_symbol)
                elif pos_symbol == base and symbol_base in correlated:
                    correlated_symbols.append(pos_symbol)
        
        passed = len(correlated_symbols) == 0
        
        return SoftRuleCheck(
            rule_id=SoftRuleID.SR_05_CHECK_CORRELATION,
            rule_name="Check Correlation",
            passed=passed,
            recommendation=f"Max correlation: {self.config.max_correlation}",
            override_conditions=["Uncorrelated catalyst", "Hedging position", "Calculated risk accepted"],
            details=f"Correlated with: {', '.join(correlated_symbols)}" if correlated_symbols else None
        )
    
    def _check_confirmation(self, has_confirmation: bool) -> SoftRuleCheck:
        """SR-06: Wait for confirmation candle."""
        return SoftRuleCheck(
            rule_id=SoftRuleID.SR_06_WAIT_CONFIRMATION,
            rule_name="Wait for Confirmation",
            passed=has_confirmation,
            recommendation=f"Wait for {self.config.confirmation_candles} confirmation candle(s)",
            override_conditions=["Gap/momentum setup", "Pre-planned entry level", "Breaking news"],
            details="No confirmation yet" if not has_confirmation else None
        )
    
    def _check_optimal_hours(self, market: str, current_hour: Optional[int]) -> SoftRuleCheck:
        """SR-07: Trade during optimal hours."""
        if current_hour is None:
            current_hour = datetime.now().hour
        
        market_key = market.lower()
        if "crypto" in market_key:
            market_key = "crypto"
        elif "forex" in market_key or "fx" in market_key:
            market_key = "forex"
        elif "stock" in market_key:
            market_key = "stocks"
        else:
            market_key = "futures"
        
        optimal_ranges = self.config.optimal_hours.get(market_key, [(0, 24)])
        is_optimal = any(start <= current_hour < end for start, end in optimal_ranges)
        
        return SoftRuleCheck(
            rule_id=SoftRuleID.SR_07_OPTIMAL_HOURS,
            rule_name="Optimal Trading Hours",
            passed=is_optimal,
            recommendation=f"Trade during optimal hours: {optimal_ranges}",
            override_conditions=["Overnight swing", "24h crypto market", "News-driven move"],
            details=f"Current hour: {current_hour}, optimal: {optimal_ranges}" if not is_optimal else None
        )
    
    def _check_multiple_timeframes(self, timeframes_analyzed: int) -> SoftRuleCheck:
        """SR-08: Use multiple timeframe analysis."""
        passed = timeframes_analyzed >= self.config.required_timeframes
        
        return SoftRuleCheck(
            rule_id=SoftRuleID.SR_08_MULTIPLE_TIMEFRAMES,
            rule_name="Multiple Timeframe Analysis",
            passed=passed,
            recommendation=f"Analyze at least {self.config.required_timeframes} timeframes",
            override_conditions=["Scalping exception", "Clear setup on single TF", "Time-sensitive entry"],
            details=f"Analyzed: {timeframes_analyzed}" if not passed else None
        )
    
    def _check_documentation(self, trade_documented: bool) -> SoftRuleCheck:
        """SR-09: Document every trade."""
        return SoftRuleCheck(
            rule_id=SoftRuleID.SR_09_DOCUMENT_TRADES,
            rule_name="Document Trades",
            passed=trade_documented,
            recommendation="Document entry reason, setup, and plan",
            override_conditions=["Quick scalp (document later)", "Emergency exit"],
            details="Trade not documented" if not trade_documented else None
        )
    
    def record_override(
        self,
        rule_id: SoftRuleID,
        rule_name: str,
        recommendation: str,
        override_reason: str,
        approved_by: str = "manual"
    ) -> SoftRuleOverride:
        """Record a soft rule override."""
        override = SoftRuleOverride(
            rule_id=rule_id,
            rule_name=rule_name,
            recommendation=recommendation,
            override_reason=override_reason,
            timestamp=datetime.now(),
            approved_by=approved_by
        )
        self.overrides.append(override)
        logger.info(f"Soft rule override: {rule_id.value} - {override_reason}")
        return override
    
    def get_override_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of overrides for analysis."""
        from collections import Counter
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        recent = [o for o in self.overrides if o.timestamp > cutoff]
        
        by_rule = Counter(o.rule_id.value for o in recent)
        by_approver = Counter(o.approved_by for o in recent)
        
        return {
            "total_overrides": len(recent),
            "by_rule": dict(by_rule),
            "by_approver": dict(by_approver),
            "most_overridden": by_rule.most_common(3),
            "period_days": days
        }
