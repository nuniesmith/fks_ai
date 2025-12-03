"""
Hard Rules - Non-Negotiable Trading Rules

These rules MUST NEVER be broken. Any violation triggers immediate action.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HardRuleID(Enum):
    """Identifiers for all hard rules."""
    HR_01_MAX_POSITION_SIZE = "HR-01"
    HR_02_MAX_DAILY_DRAWDOWN = "HR-02"
    HR_03_MAX_WEEKLY_DRAWDOWN = "HR-03"
    HR_04_STOP_LOSS_REQUIRED = "HR-04"
    HR_05_MAX_RISK_PER_TRADE = "HR-05"
    HR_06_NO_REVENGE_TRADING = "HR-06"
    HR_07_NO_NEWS_TRADING = "HR-07"
    HR_08_MIN_DATA_QUALITY = "HR-08"
    HR_09_MIN_LIQUIDITY = "HR-09"
    HR_10_NO_AVERAGING_DOWN = "HR-10"
    HR_11_MAX_LEVERAGE = "HR-11"
    HR_12_EMERGENCY_KILL_SWITCH = "HR-12"


@dataclass
class HardRuleViolation:
    """Records a hard rule violation."""
    rule_id: HardRuleID
    rule_name: str
    threshold: Any
    actual_value: Any
    timestamp: datetime
    consequence: str
    details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id.value,
            "rule_name": self.rule_name,
            "threshold": str(self.threshold),
            "actual_value": str(self.actual_value),
            "timestamp": self.timestamp.isoformat(),
            "consequence": self.consequence,
            "details": self.details
        }


@dataclass
class HardRuleConfig:
    """Configuration for hard rules with market-specific overrides."""
    # HR-01: Maximum position size (% of account)
    max_position_size_pct: float = 5.0
    max_position_size_btc_eth: float = 10.0  # Exception for BTC/ETH
    
    # HR-02: Maximum daily drawdown (% of account)
    max_daily_drawdown_pct: float = 3.0
    
    # HR-03: Maximum weekly drawdown (% of account)
    max_weekly_drawdown_pct: float = 6.0
    
    # HR-04: Stop-loss required (always True)
    stop_loss_required: bool = True
    
    # HR-05: Maximum risk per trade (% of account)
    max_risk_per_trade_pct: float = 2.0
    
    # HR-06: Revenge trading cooldown (minutes)
    revenge_trading_cooldown_mins: int = 30
    consecutive_losses_trigger: int = 2
    
    # HR-07: News blackout period (minutes before/after)
    news_blackout_minutes: int = 15
    high_impact_events: List[str] = field(default_factory=lambda: [
        "NFP", "FOMC", "CPI", "GDP", "ECB", "BOJ", "BOE", "RBA"
    ])
    
    # HR-08: Minimum data quality (% of metrics available)
    min_data_quality_pct: float = 66.0
    
    # HR-09: Minimum liquidity (USD daily notional)
    min_liquidity_usd: float = 1_000_000.0
    min_liquidity_crypto: float = 50_000_000.0
    min_liquidity_forex: float = 5_000_000_000.0
    
    # HR-10: No averaging down (always True)
    no_averaging_down: bool = True
    
    # HR-11: Maximum leverage
    max_leverage_crypto: float = 5.0
    max_leverage_forex: float = 30.0
    max_leverage_futures: float = 20.0
    
    # HR-12: Emergency kill switch (% loss in 24h)
    emergency_kill_switch_pct: float = 10.0


class HardRules:
    """
    Enforces hard (non-negotiable) trading rules.
    
    Any violation of a hard rule triggers immediate action:
    - HR-01 to HR-03: Position/trading restrictions
    - HR-04 to HR-05: Trade rejection
    - HR-06: Mandatory break
    - HR-07 to HR-09: Trade rejection
    - HR-10: Immediate stop
    - HR-11: Platform-enforced
    - HR-12: All positions closed
    """
    
    def __init__(self, config: Optional[HardRuleConfig] = None):
        self.config = config or HardRuleConfig()
        self.violations: List[HardRuleViolation] = []
        self.recent_losses: List[datetime] = []
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.trading_paused_until: Optional[datetime] = None
        
    def check_all(
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
        is_adding_to_loser: bool = False,
        upcoming_news: Optional[List[Dict]] = None,
        daily_pnl_pct: Optional[float] = None,
        weekly_pnl_pct: Optional[float] = None,
        total_24h_pnl_pct: Optional[float] = None,
    ) -> tuple[bool, List[HardRuleViolation]]:
        """
        Check all hard rules for a proposed trade.
        
        Returns:
            (passes_all, violations): Tuple of pass/fail and list of violations
        """
        violations = []
        
        # HR-01: Maximum position size
        violation = self._check_position_size(symbol, position_size_pct)
        if violation:
            violations.append(violation)
            
        # HR-02: Maximum daily drawdown
        if daily_pnl_pct is not None:
            violation = self._check_daily_drawdown(daily_pnl_pct)
            if violation:
                violations.append(violation)
                
        # HR-03: Maximum weekly drawdown
        if weekly_pnl_pct is not None:
            violation = self._check_weekly_drawdown(weekly_pnl_pct)
            if violation:
                violations.append(violation)
                
        # HR-04: Stop-loss required
        violation = self._check_stop_loss(stop_loss_set)
        if violation:
            violations.append(violation)
            
        # HR-05: Maximum risk per trade
        violation = self._check_risk_per_trade(risk_per_trade_pct)
        if violation:
            violations.append(violation)
            
        # HR-06: No revenge trading
        violation = self._check_revenge_trading()
        if violation:
            violations.append(violation)
            
        # HR-07: No news trading
        if upcoming_news:
            violation = self._check_news_blackout(upcoming_news)
            if violation:
                violations.append(violation)
                
        # HR-08: Minimum data quality
        violation = self._check_data_quality(data_quality_pct)
        if violation:
            violations.append(violation)
            
        # HR-09: Minimum liquidity
        violation = self._check_liquidity(market, liquidity_usd)
        if violation:
            violations.append(violation)
            
        # HR-10: No averaging down
        violation = self._check_averaging_down(is_adding_to_loser)
        if violation:
            violations.append(violation)
            
        # HR-11: Maximum leverage
        violation = self._check_leverage(market, leverage)
        if violation:
            violations.append(violation)
            
        # HR-12: Emergency kill switch
        if total_24h_pnl_pct is not None:
            violation = self._check_emergency_kill_switch(total_24h_pnl_pct)
            if violation:
                violations.append(violation)
        
        self.violations.extend(violations)
        return len(violations) == 0, violations
    
    def _check_position_size(self, symbol: str, size_pct: float) -> Optional[HardRuleViolation]:
        """HR-01: Check maximum position size."""
        is_btc_eth = symbol.upper() in ["BTC", "ETH", "BTCUSDT", "ETHUSDT", "BTCUSD", "ETHUSD"]
        max_size = self.config.max_position_size_btc_eth if is_btc_eth else self.config.max_position_size_pct
        
        if size_pct > max_size:
            return HardRuleViolation(
                rule_id=HardRuleID.HR_01_MAX_POSITION_SIZE,
                rule_name="Maximum Position Size",
                threshold=f"{max_size}%",
                actual_value=f"{size_pct}%",
                timestamp=datetime.now(),
                consequence="Reduce position size or reject trade",
                details=f"Symbol: {symbol}, BTC/ETH exception: {is_btc_eth}"
            )
        return None
    
    def _check_daily_drawdown(self, daily_pnl_pct: float) -> Optional[HardRuleViolation]:
        """HR-02: Check maximum daily drawdown."""
        if daily_pnl_pct < -self.config.max_daily_drawdown_pct:
            self.trading_paused_until = datetime.now() + timedelta(days=1)
            return HardRuleViolation(
                rule_id=HardRuleID.HR_02_MAX_DAILY_DRAWDOWN,
                rule_name="Maximum Daily Drawdown",
                threshold=f"-{self.config.max_daily_drawdown_pct}%",
                actual_value=f"{daily_pnl_pct}%",
                timestamp=datetime.now(),
                consequence="STOP TRADING FOR THE DAY",
                details=f"Trading paused until: {self.trading_paused_until.isoformat()}"
            )
        return None
    
    def _check_weekly_drawdown(self, weekly_pnl_pct: float) -> Optional[HardRuleViolation]:
        """HR-03: Check maximum weekly drawdown."""
        if weekly_pnl_pct < -self.config.max_weekly_drawdown_pct:
            self.trading_paused_until = datetime.now() + timedelta(weeks=1)
            return HardRuleViolation(
                rule_id=HardRuleID.HR_03_MAX_WEEKLY_DRAWDOWN,
                rule_name="Maximum Weekly Drawdown",
                threshold=f"-{self.config.max_weekly_drawdown_pct}%",
                actual_value=f"{weekly_pnl_pct}%",
                timestamp=datetime.now(),
                consequence="STOP TRADING FOR THE WEEK",
                details=f"Trading paused until: {self.trading_paused_until.isoformat()}"
            )
        return None
    
    def _check_stop_loss(self, stop_loss_set: bool) -> Optional[HardRuleViolation]:
        """HR-04: Stop-loss is ALWAYS required."""
        if not stop_loss_set:
            return HardRuleViolation(
                rule_id=HardRuleID.HR_04_STOP_LOSS_REQUIRED,
                rule_name="Stop-Loss Required",
                threshold="True",
                actual_value="False",
                timestamp=datetime.now(),
                consequence="TRADE REJECTED - Set stop-loss before entry",
                details="100% of trades must have a stop-loss. No exceptions."
            )
        return None
    
    def _check_risk_per_trade(self, risk_pct: float) -> Optional[HardRuleViolation]:
        """HR-05: Check maximum risk per trade."""
        if risk_pct > self.config.max_risk_per_trade_pct:
            return HardRuleViolation(
                rule_id=HardRuleID.HR_05_MAX_RISK_PER_TRADE,
                rule_name="Maximum Risk Per Trade",
                threshold=f"{self.config.max_risk_per_trade_pct}%",
                actual_value=f"{risk_pct}%",
                timestamp=datetime.now(),
                consequence="Reduce position size to meet risk limit",
                details="Adjust stop-loss distance or reduce size"
            )
        return None
    
    def _check_revenge_trading(self) -> Optional[HardRuleViolation]:
        """HR-06: Check for revenge trading after consecutive losses."""
        # Check if currently in cooldown
        if self.trading_paused_until and datetime.now() < self.trading_paused_until:
            return HardRuleViolation(
                rule_id=HardRuleID.HR_06_NO_REVENGE_TRADING,
                rule_name="No Revenge Trading",
                threshold=f"{self.config.consecutive_losses_trigger} consecutive losses",
                actual_value=f"In cooldown until {self.trading_paused_until.isoformat()}",
                timestamp=datetime.now(),
                consequence=f"WAIT {self.config.revenge_trading_cooldown_mins} minutes",
                details="Take a break, review losses, then return with clear mind"
            )
        
        # Check recent losses
        now = datetime.now()
        recent = [t for t in self.recent_losses if now - t < timedelta(hours=1)]
        
        if len(recent) >= self.config.consecutive_losses_trigger:
            self.trading_paused_until = now + timedelta(minutes=self.config.revenge_trading_cooldown_mins)
            return HardRuleViolation(
                rule_id=HardRuleID.HR_06_NO_REVENGE_TRADING,
                rule_name="No Revenge Trading",
                threshold=f"{self.config.consecutive_losses_trigger} consecutive losses",
                actual_value=f"{len(recent)} recent losses",
                timestamp=now,
                consequence=f"MANDATORY {self.config.revenge_trading_cooldown_mins} MINUTE BREAK",
                details=f"Trading paused until: {self.trading_paused_until.isoformat()}"
            )
        return None
    
    def record_loss(self):
        """Record a trading loss for revenge trading detection."""
        self.recent_losses.append(datetime.now())
        # Keep only last hour of losses
        cutoff = datetime.now() - timedelta(hours=1)
        self.recent_losses = [t for t in self.recent_losses if t > cutoff]
    
    def _check_news_blackout(self, upcoming_news: List[Dict]) -> Optional[HardRuleViolation]:
        """HR-07: Check for high-impact news blackout."""
        now = datetime.now()
        blackout_start = timedelta(minutes=self.config.news_blackout_minutes)
        
        for event in upcoming_news:
            event_time = event.get("datetime")
            event_name = event.get("name", "Unknown")
            impact = event.get("impact", "").upper()
            
            if impact != "HIGH":
                continue
                
            # Check if event is a high-impact type
            is_high_impact = any(
                hi in event_name.upper() 
                for hi in self.config.high_impact_events
            )
            
            if not is_high_impact:
                continue
            
            if isinstance(event_time, str):
                event_time = datetime.fromisoformat(event_time)
            
            time_to_event = event_time - now
            
            if -blackout_start <= time_to_event <= blackout_start:
                return HardRuleViolation(
                    rule_id=HardRuleID.HR_07_NO_NEWS_TRADING,
                    rule_name="No News Trading",
                    threshold=f"±{self.config.news_blackout_minutes} minutes of high-impact news",
                    actual_value=f"{event_name} in {time_to_event}",
                    timestamp=now,
                    consequence="TRADE REJECTED - Wait for news to pass",
                    details=f"Event: {event_name}, Time: {event_time.isoformat()}"
                )
        return None
    
    def _check_data_quality(self, quality_pct: float) -> Optional[HardRuleViolation]:
        """HR-08: Check minimum data quality."""
        if quality_pct < self.config.min_data_quality_pct:
            return HardRuleViolation(
                rule_id=HardRuleID.HR_08_MIN_DATA_QUALITY,
                rule_name="Minimum Data Quality",
                threshold=f"{self.config.min_data_quality_pct}%",
                actual_value=f"{quality_pct}%",
                timestamp=datetime.now(),
                consequence="SKIP - Insufficient data for analysis",
                details="Wait for more data or skip this asset"
            )
        return None
    
    def _check_liquidity(self, market: str, liquidity_usd: float) -> Optional[HardRuleViolation]:
        """HR-09: Check minimum liquidity."""
        market_lower = market.lower()
        
        if "crypto" in market_lower:
            min_liq = self.config.min_liquidity_crypto
        elif "forex" in market_lower or "fx" in market_lower:
            min_liq = self.config.min_liquidity_forex
        else:
            min_liq = self.config.min_liquidity_usd
        
        if liquidity_usd < min_liq:
            return HardRuleViolation(
                rule_id=HardRuleID.HR_09_MIN_LIQUIDITY,
                rule_name="Minimum Liquidity",
                threshold=f"${min_liq:,.0f}",
                actual_value=f"${liquidity_usd:,.0f}",
                timestamp=datetime.now(),
                consequence="SKIP - Insufficient liquidity",
                details=f"Market: {market}, required: ${min_liq:,.0f} daily"
            )
        return None
    
    def _check_averaging_down(self, is_adding_to_loser: bool) -> Optional[HardRuleViolation]:
        """HR-10: Never average down on losing positions."""
        if is_adding_to_loser:
            return HardRuleViolation(
                rule_id=HardRuleID.HR_10_NO_AVERAGING_DOWN,
                rule_name="No Averaging Down",
                threshold="Never add to losers",
                actual_value="Attempting to add to losing position",
                timestamp=datetime.now(),
                consequence="TRADE REJECTED - Close loser or wait",
                details="Adding to losers compounds losses. Exit or hold."
            )
        return None
    
    def _check_leverage(self, market: str, leverage: float) -> Optional[HardRuleViolation]:
        """HR-11: Check maximum leverage by market."""
        market_lower = market.lower()
        
        if "crypto" in market_lower:
            max_lev = self.config.max_leverage_crypto
        elif "forex" in market_lower or "fx" in market_lower:
            max_lev = self.config.max_leverage_forex
        else:
            max_lev = self.config.max_leverage_futures
        
        if leverage > max_lev:
            return HardRuleViolation(
                rule_id=HardRuleID.HR_11_MAX_LEVERAGE,
                rule_name="Maximum Leverage",
                threshold=f"{max_lev}x",
                actual_value=f"{leverage}x",
                timestamp=datetime.now(),
                consequence="Reduce leverage",
                details=f"Market: {market}, max allowed: {max_lev}x"
            )
        return None
    
    def _check_emergency_kill_switch(self, total_24h_pnl_pct: float) -> Optional[HardRuleViolation]:
        """HR-12: Emergency kill switch for catastrophic losses."""
        if total_24h_pnl_pct < -self.config.emergency_kill_switch_pct:
            return HardRuleViolation(
                rule_id=HardRuleID.HR_12_EMERGENCY_KILL_SWITCH,
                rule_name="Emergency Kill Switch",
                threshold=f"-{self.config.emergency_kill_switch_pct}%",
                actual_value=f"{total_24h_pnl_pct}%",
                timestamp=datetime.now(),
                consequence="🚨 CLOSE ALL POSITIONS IMMEDIATELY 🚨",
                details="Catastrophic loss detected. Stop all trading, review strategy."
            )
        return None
    
    def is_trading_allowed(self) -> tuple[bool, Optional[str]]:
        """Check if trading is currently allowed."""
        if self.trading_paused_until and datetime.now() < self.trading_paused_until:
            return False, f"Trading paused until {self.trading_paused_until.isoformat()}"
        return True, None
    
    def get_recent_violations(self, hours: int = 24) -> List[HardRuleViolation]:
        """Get violations from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [v for v in self.violations if v.timestamp > cutoff]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "trading_allowed": self.is_trading_allowed()[0],
            "trading_paused_until": self.trading_paused_until.isoformat() if self.trading_paused_until else None,
            "recent_losses_count": len(self.recent_losses),
            "violations_24h": len(self.get_recent_violations(24)),
            "config": {
                "max_position_size_pct": self.config.max_position_size_pct,
                "max_daily_drawdown_pct": self.config.max_daily_drawdown_pct,
                "max_weekly_drawdown_pct": self.config.max_weekly_drawdown_pct,
                "max_risk_per_trade_pct": self.config.max_risk_per_trade_pct,
            }
        }
