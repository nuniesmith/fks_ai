"""
Market-Specific Gates

Hard gates by asset class with specific thresholds per the FKS Trading Framework.
These are NON-NEGOTIABLE requirements for each market type.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Status of a gate check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"  # Gate not applicable


@dataclass
class GateResult:
    """Result of a single gate check."""
    gate_id: str
    gate_name: str
    status: GateStatus
    message: str
    value: Optional[Any] = None
    threshold: Optional[Any] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "gate_name": self.gate_name,
            "status": self.status.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "details": self.details
        }


@dataclass
class MarketGateResults:
    """Combined results from all gates for a market."""
    market_type: str
    passed: bool
    gate_results: List[GateResult]
    failed_gates: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_type": self.market_type,
            "passed": self.passed,
            "gate_results": [g.to_dict() for g in self.gate_results],
            "failed_gates": self.failed_gates,
            "timestamp": self.timestamp.isoformat()
        }


class MarketGates(ABC):
    """Abstract base class for market-specific gates."""
    
    @property
    @abstractmethod
    def market_type(self) -> str:
        pass
    
    @abstractmethod
    def check_all(self, **kwargs) -> MarketGateResults:
        """Check all gates for this market type."""
        pass


# =============================================================================
# CRYPTO SPOT GATES
# =============================================================================

class CryptoSpotGates(MarketGates):
    """
    Hard gates for Crypto Spot trading.
    
    Focus: BTC/ETH majors, SOL/LINK/AVAX alts
    Priority: High (bootstrap capital source)
    """
    
    market_type = "crypto_spot"
    
    # Gate thresholds
    MIN_24H_VOLUME_USD = 10_000_000  # $10M
    MIN_MARKET_CAP_USD = 100_000_000  # $100M
    MAX_SPREAD_PCT = 0.3  # 0.3%
    MIN_EXCHANGE_LISTINGS = 3
    MAX_CONCENTRATION_PCT = 40  # 40% max in single position
    
    def check_all(
        self,
        symbol: str,
        volume_24h: float,
        market_cap: float,
        spread_pct: float,
        exchange_listings: int,
        current_concentration_pct: float,
        on_chain_data: Optional[Dict] = None,
        **kwargs
    ) -> MarketGateResults:
        """Check all crypto spot gates."""
        results = []
        
        # G1: Volume Gate
        results.append(self._check_volume(symbol, volume_24h))
        
        # G2: Market Cap Gate
        results.append(self._check_market_cap(symbol, market_cap))
        
        # G3: Spread Gate
        results.append(self._check_spread(symbol, spread_pct))
        
        # G4: Exchange Listings Gate
        results.append(self._check_listings(symbol, exchange_listings))
        
        # G5: Concentration Gate
        results.append(self._check_concentration(symbol, current_concentration_pct))
        
        # G6: On-Chain Health (if data available)
        results.append(self._check_on_chain(symbol, on_chain_data))
        
        # Calculate overall pass/fail
        failed_gates = [r.gate_id for r in results if r.status == GateStatus.FAIL]
        passed = len(failed_gates) == 0
        
        return MarketGateResults(
            market_type=self.market_type,
            passed=passed,
            gate_results=results,
            failed_gates=failed_gates
        )
    
    def _check_volume(self, symbol: str, volume_24h: float) -> GateResult:
        """G1: 24h volume must exceed $10M."""
        passed = volume_24h >= self.MIN_24H_VOLUME_USD
        return GateResult(
            gate_id="CS-G1",
            gate_name="24h Volume",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Volume: ${volume_24h:,.0f}" if passed else f"Volume ${volume_24h:,.0f} below ${self.MIN_24H_VOLUME_USD:,.0f} minimum",
            value=volume_24h,
            threshold=self.MIN_24H_VOLUME_USD
        )
    
    def _check_market_cap(self, symbol: str, market_cap: float) -> GateResult:
        """G2: Market cap must exceed $100M."""
        passed = market_cap >= self.MIN_MARKET_CAP_USD
        return GateResult(
            gate_id="CS-G2",
            gate_name="Market Cap",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Market cap: ${market_cap:,.0f}" if passed else f"Market cap ${market_cap:,.0f} below ${self.MIN_MARKET_CAP_USD:,.0f} minimum",
            value=market_cap,
            threshold=self.MIN_MARKET_CAP_USD
        )
    
    def _check_spread(self, symbol: str, spread_pct: float) -> GateResult:
        """G3: Bid-ask spread must be under 0.3%."""
        passed = spread_pct <= self.MAX_SPREAD_PCT
        return GateResult(
            gate_id="CS-G3",
            gate_name="Spread",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Spread: {spread_pct:.2f}%" if passed else f"Spread {spread_pct:.2f}% exceeds {self.MAX_SPREAD_PCT}% maximum",
            value=spread_pct,
            threshold=self.MAX_SPREAD_PCT
        )
    
    def _check_listings(self, symbol: str, exchange_listings: int) -> GateResult:
        """G4: Must be listed on at least 3 major exchanges."""
        passed = exchange_listings >= self.MIN_EXCHANGE_LISTINGS
        return GateResult(
            gate_id="CS-G4",
            gate_name="Exchange Listings",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Listed on {exchange_listings} exchanges" if passed else f"Only {exchange_listings} listings, need {self.MIN_EXCHANGE_LISTINGS}+",
            value=exchange_listings,
            threshold=self.MIN_EXCHANGE_LISTINGS
        )
    
    def _check_concentration(self, symbol: str, concentration_pct: float) -> GateResult:
        """G5: Max 40% portfolio concentration in single position."""
        passed = concentration_pct <= self.MAX_CONCENTRATION_PCT
        return GateResult(
            gate_id="CS-G5",
            gate_name="Concentration",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Concentration: {concentration_pct:.1f}%" if passed else f"Concentration {concentration_pct:.1f}% exceeds {self.MAX_CONCENTRATION_PCT}% max",
            value=concentration_pct,
            threshold=self.MAX_CONCENTRATION_PCT
        )
    
    def _check_on_chain(self, symbol: str, on_chain_data: Optional[Dict]) -> GateResult:
        """G6: On-chain health check (whale movement, exchange flows)."""
        if not on_chain_data:
            return GateResult(
                gate_id="CS-G6",
                gate_name="On-Chain Health",
                status=GateStatus.SKIP,
                message="On-chain data not available"
            )
        
        # Check for large exchange inflows (bearish signal)
        exchange_inflow_spike = on_chain_data.get("exchange_inflow_spike", False)
        whale_distribution = on_chain_data.get("whale_distribution", False)
        
        issues = []
        if exchange_inflow_spike:
            issues.append("Exchange inflow spike detected")
        if whale_distribution:
            issues.append("Whale distribution detected")
        
        passed = len(issues) == 0
        return GateResult(
            gate_id="CS-G6",
            gate_name="On-Chain Health",
            status=GateStatus.PASS if passed else GateStatus.WARN,  # Warn, not fail
            message="On-chain metrics healthy" if passed else "; ".join(issues),
            details=on_chain_data
        )


# =============================================================================
# CRYPTO PERPS GATES
# =============================================================================

class CryptoPerpsGates(MarketGates):
    """
    Hard gates for Crypto Perpetual Futures trading.
    
    Focus: BTC/ETH perps only (until rules mastered)
    Priority: High (leverage instrument)
    """
    
    market_type = "crypto_perps"
    
    # Gate thresholds
    MAX_LEVERAGE = 5  # Max 5x until consistent
    MAX_FUNDING_RATE_PCT = 0.1  # 0.1% per 8h
    MIN_OPEN_INTEREST_USD = 50_000_000  # $50M
    MAX_POSITION_SIZE_PCT = 10  # 10% of account per position
    MIN_LIQUIDATION_BUFFER_PCT = 50  # 50% buffer to liquidation
    ALLOWED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BTCUSD", "ETHUSD"]
    
    def check_all(
        self,
        symbol: str,
        leverage: float,
        funding_rate_pct: float,
        open_interest: float,
        position_size_pct: float,
        liquidation_distance_pct: float,
        account_equity: float,
        **kwargs
    ) -> MarketGateResults:
        """Check all crypto perps gates."""
        results = []
        
        # G1: Symbol Whitelist
        results.append(self._check_symbol(symbol))
        
        # G2: Leverage Gate
        results.append(self._check_leverage(leverage))
        
        # G3: Funding Rate Gate
        results.append(self._check_funding_rate(symbol, funding_rate_pct))
        
        # G4: Open Interest Gate
        results.append(self._check_open_interest(symbol, open_interest))
        
        # G5: Position Size Gate
        results.append(self._check_position_size(position_size_pct))
        
        # G6: Liquidation Buffer Gate
        results.append(self._check_liquidation_buffer(liquidation_distance_pct))
        
        failed_gates = [r.gate_id for r in results if r.status == GateStatus.FAIL]
        passed = len(failed_gates) == 0
        
        return MarketGateResults(
            market_type=self.market_type,
            passed=passed,
            gate_results=results,
            failed_gates=failed_gates
        )
    
    def _check_symbol(self, symbol: str) -> GateResult:
        """G1: Only BTC/ETH perps allowed."""
        symbol_upper = symbol.upper().replace("-", "").replace("_", "")
        allowed = any(s in symbol_upper for s in ["BTC", "ETH"])
        return GateResult(
            gate_id="CP-G1",
            gate_name="Symbol Whitelist",
            status=GateStatus.PASS if allowed else GateStatus.FAIL,
            message=f"{symbol} allowed" if allowed else f"{symbol} not in whitelist (BTC/ETH only)",
            value=symbol,
            threshold=self.ALLOWED_SYMBOLS
        )
    
    def _check_leverage(self, leverage: float) -> GateResult:
        """G2: Max 5x leverage."""
        passed = leverage <= self.MAX_LEVERAGE
        return GateResult(
            gate_id="CP-G2",
            gate_name="Leverage",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Leverage: {leverage}x" if passed else f"Leverage {leverage}x exceeds {self.MAX_LEVERAGE}x max",
            value=leverage,
            threshold=self.MAX_LEVERAGE
        )
    
    def _check_funding_rate(self, symbol: str, funding_rate_pct: float) -> GateResult:
        """G3: Funding rate must be below 0.1% per 8h."""
        passed = abs(funding_rate_pct) <= self.MAX_FUNDING_RATE_PCT
        direction = "positive" if funding_rate_pct > 0 else "negative"
        return GateResult(
            gate_id="CP-G3",
            gate_name="Funding Rate",
            status=GateStatus.PASS if passed else GateStatus.WARN,  # Warn, not hard fail
            message=f"Funding: {funding_rate_pct:.3f}%" if passed else f"Funding {funding_rate_pct:.3f}% ({direction}) exceeds ±{self.MAX_FUNDING_RATE_PCT}%",
            value=funding_rate_pct,
            threshold=self.MAX_FUNDING_RATE_PCT
        )
    
    def _check_open_interest(self, symbol: str, open_interest: float) -> GateResult:
        """G4: Open interest must exceed $50M."""
        passed = open_interest >= self.MIN_OPEN_INTEREST_USD
        return GateResult(
            gate_id="CP-G4",
            gate_name="Open Interest",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"OI: ${open_interest:,.0f}" if passed else f"OI ${open_interest:,.0f} below ${self.MIN_OPEN_INTEREST_USD:,.0f} min",
            value=open_interest,
            threshold=self.MIN_OPEN_INTEREST_USD
        )
    
    def _check_position_size(self, position_size_pct: float) -> GateResult:
        """G5: Max 10% of account per position."""
        passed = position_size_pct <= self.MAX_POSITION_SIZE_PCT
        return GateResult(
            gate_id="CP-G5",
            gate_name="Position Size",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Position: {position_size_pct:.1f}% of account" if passed else f"Position {position_size_pct:.1f}% exceeds {self.MAX_POSITION_SIZE_PCT}% max",
            value=position_size_pct,
            threshold=self.MAX_POSITION_SIZE_PCT
        )
    
    def _check_liquidation_buffer(self, liquidation_distance_pct: float) -> GateResult:
        """G6: Must have 50%+ buffer to liquidation price."""
        passed = liquidation_distance_pct >= self.MIN_LIQUIDATION_BUFFER_PCT
        return GateResult(
            gate_id="CP-G6",
            gate_name="Liquidation Buffer",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Liq buffer: {liquidation_distance_pct:.1f}%" if passed else f"Liq buffer {liquidation_distance_pct:.1f}% below {self.MIN_LIQUIDATION_BUFFER_PCT}% min",
            value=liquidation_distance_pct,
            threshold=self.MIN_LIQUIDATION_BUFFER_PCT
        )


# =============================================================================
# FOREX GATES
# =============================================================================

class ForexGates(MarketGates):
    """
    Hard gates for Forex trading.
    
    Focus: Major pairs (EURUSD, GBPUSD, USDJPY, etc.)
    Priority: Medium (prop firm path)
    """
    
    market_type = "forex"
    
    # Gate thresholds
    MAJOR_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    MINOR_PAIRS = ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "EURCHF", "GBPCHF"]
    MAX_SPREAD_PIPS = 2.0  # 2 pips max for majors
    MAX_LEVERAGE = 30  # 30:1 max (prop firm standard)
    MAX_DRAWDOWN_PCT = 5  # 5% max daily drawdown (prop firm rule)
    MIN_RISK_REWARD = 1.5  # 1.5:1 minimum
    NO_TRADE_WINDOW_MINUTES = 30  # No trades 30 min before/after news
    
    def check_all(
        self,
        symbol: str,
        spread_pips: float,
        leverage: float,
        daily_drawdown_pct: float,
        risk_reward_ratio: float,
        minutes_to_news: Optional[int] = None,
        carry_direction: Optional[str] = None,  # "positive", "negative", "neutral"
        **kwargs
    ) -> MarketGateResults:
        """Check all forex gates."""
        results = []
        
        # G1: Pair Whitelist (majors/minors only)
        results.append(self._check_pair(symbol))
        
        # G2: Spread Gate
        results.append(self._check_spread(symbol, spread_pips))
        
        # G3: Leverage Gate
        results.append(self._check_leverage(leverage))
        
        # G4: Drawdown Gate (prop firm compatibility)
        results.append(self._check_drawdown(daily_drawdown_pct))
        
        # G5: Risk/Reward Gate
        results.append(self._check_risk_reward(risk_reward_ratio))
        
        # G6: News Blackout Gate
        results.append(self._check_news_blackout(minutes_to_news))
        
        # G7: Carry Direction (soft gate - warn only)
        results.append(self._check_carry(symbol, carry_direction))
        
        failed_gates = [r.gate_id for r in results if r.status == GateStatus.FAIL]
        passed = len(failed_gates) == 0
        
        return MarketGateResults(
            market_type=self.market_type,
            passed=passed,
            gate_results=results,
            failed_gates=failed_gates
        )
    
    def _check_pair(self, symbol: str) -> GateResult:
        """G1: Only major/minor pairs allowed."""
        symbol_clean = symbol.upper().replace("/", "").replace("_", "")
        is_major = symbol_clean in self.MAJOR_PAIRS
        is_minor = symbol_clean in self.MINOR_PAIRS
        allowed = is_major or is_minor
        
        return GateResult(
            gate_id="FX-G1",
            gate_name="Pair Whitelist",
            status=GateStatus.PASS if allowed else GateStatus.FAIL,
            message=f"{symbol} ({'major' if is_major else 'minor'})" if allowed else f"{symbol} not in allowed pairs",
            value=symbol,
            threshold=self.MAJOR_PAIRS + self.MINOR_PAIRS
        )
    
    def _check_spread(self, symbol: str, spread_pips: float) -> GateResult:
        """G2: Spread must be under 2 pips for majors."""
        passed = spread_pips <= self.MAX_SPREAD_PIPS
        return GateResult(
            gate_id="FX-G2",
            gate_name="Spread",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Spread: {spread_pips:.1f} pips" if passed else f"Spread {spread_pips:.1f} pips exceeds {self.MAX_SPREAD_PIPS} pip max",
            value=spread_pips,
            threshold=self.MAX_SPREAD_PIPS
        )
    
    def _check_leverage(self, leverage: float) -> GateResult:
        """G3: Max 30:1 leverage (prop firm standard)."""
        passed = leverage <= self.MAX_LEVERAGE
        return GateResult(
            gate_id="FX-G3",
            gate_name="Leverage",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Leverage: {leverage:.0f}:1" if passed else f"Leverage {leverage:.0f}:1 exceeds {self.MAX_LEVERAGE}:1 max",
            value=leverage,
            threshold=self.MAX_LEVERAGE
        )
    
    def _check_drawdown(self, daily_drawdown_pct: float) -> GateResult:
        """G4: Max 5% daily drawdown (prop firm rule)."""
        passed = daily_drawdown_pct <= self.MAX_DRAWDOWN_PCT
        return GateResult(
            gate_id="FX-G4",
            gate_name="Daily Drawdown",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Drawdown: {daily_drawdown_pct:.2f}%" if passed else f"Drawdown {daily_drawdown_pct:.2f}% exceeds {self.MAX_DRAWDOWN_PCT}% prop firm limit",
            value=daily_drawdown_pct,
            threshold=self.MAX_DRAWDOWN_PCT
        )
    
    def _check_risk_reward(self, risk_reward_ratio: float) -> GateResult:
        """G5: Minimum 1.5:1 risk/reward."""
        passed = risk_reward_ratio >= self.MIN_RISK_REWARD
        return GateResult(
            gate_id="FX-G5",
            gate_name="Risk/Reward",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"R:R {risk_reward_ratio:.1f}:1" if passed else f"R:R {risk_reward_ratio:.1f}:1 below {self.MIN_RISK_REWARD}:1 minimum",
            value=risk_reward_ratio,
            threshold=self.MIN_RISK_REWARD
        )
    
    def _check_news_blackout(self, minutes_to_news: Optional[int]) -> GateResult:
        """G6: No trades 30 minutes before/after major news."""
        if minutes_to_news is None:
            return GateResult(
                gate_id="FX-G6",
                gate_name="News Blackout",
                status=GateStatus.SKIP,
                message="News schedule not provided"
            )
        
        passed = abs(minutes_to_news) > self.NO_TRADE_WINDOW_MINUTES
        return GateResult(
            gate_id="FX-G6",
            gate_name="News Blackout",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"News in {minutes_to_news} min - OK" if passed else f"News in {minutes_to_news} min - blackout period",
            value=minutes_to_news,
            threshold=self.NO_TRADE_WINDOW_MINUTES
        )
    
    def _check_carry(self, symbol: str, carry_direction: Optional[str]) -> GateResult:
        """G7: Prefer positive carry direction (soft gate)."""
        if carry_direction is None:
            return GateResult(
                gate_id="FX-G7",
                gate_name="Carry Direction",
                status=GateStatus.SKIP,
                message="Carry data not provided"
            )
        
        preferred = carry_direction == "positive"
        return GateResult(
            gate_id="FX-G7",
            gate_name="Carry Direction",
            status=GateStatus.PASS if preferred else GateStatus.WARN,
            message=f"Carry: {carry_direction}" if preferred else f"Trading against carry ({carry_direction})",
            value=carry_direction
        )


# =============================================================================
# FUTURES GATES
# =============================================================================

class FuturesGates(MarketGates):
    """
    Hard gates for Futures trading.
    
    Focus: ES/NQ micros initially, then full contracts
    Priority: Medium (prop firm path)
    """
    
    market_type = "futures"
    
    # Gate thresholds
    ALLOWED_CONTRACTS = ["ES", "NQ", "MES", "MNQ", "CL", "GC", "MCL", "MGC", "ZB", "ZN"]
    MIN_VOLUME_CONTRACTS = 10000  # Daily volume
    MAX_MARGIN_PCT = 50  # Max 50% of account as margin
    MAX_CONTRACTS_PER_TRADE = {
        "MES": 10, "MNQ": 10, "MCL": 5, "MGC": 5,  # Micros
        "ES": 2, "NQ": 2, "CL": 2, "GC": 2, "ZB": 5, "ZN": 5  # Full
    }
    RTH_START = time(9, 30)  # Regular trading hours
    RTH_END = time(16, 0)
    
    def check_all(
        self,
        symbol: str,
        daily_volume: int,
        margin_used_pct: float,
        num_contracts: int,
        is_rth: Optional[bool] = None,
        cot_data: Optional[Dict] = None,  # Commitment of Traders
        **kwargs
    ) -> MarketGateResults:
        """Check all futures gates."""
        results = []
        
        # G1: Contract Whitelist
        results.append(self._check_contract(symbol))
        
        # G2: Volume Gate
        results.append(self._check_volume(symbol, daily_volume))
        
        # G3: Margin Gate
        results.append(self._check_margin(margin_used_pct))
        
        # G4: Position Size Gate
        results.append(self._check_position_size(symbol, num_contracts))
        
        # G5: Trading Hours (prefer RTH)
        results.append(self._check_trading_hours(is_rth))
        
        # G6: COT Alignment (soft gate)
        results.append(self._check_cot(symbol, cot_data))
        
        failed_gates = [r.gate_id for r in results if r.status == GateStatus.FAIL]
        passed = len(failed_gates) == 0
        
        return MarketGateResults(
            market_type=self.market_type,
            passed=passed,
            gate_results=results,
            failed_gates=failed_gates
        )
    
    def _check_contract(self, symbol: str) -> GateResult:
        """G1: Only allowed contracts."""
        symbol_clean = symbol.upper().replace("/", "").replace("_", "")
        # Extract base contract (remove month/year codes)
        for contract in self.ALLOWED_CONTRACTS:
            if symbol_clean.startswith(contract):
                return GateResult(
                    gate_id="FUT-G1",
                    gate_name="Contract Whitelist",
                    status=GateStatus.PASS,
                    message=f"{symbol} allowed ({contract})",
                    value=symbol,
                    threshold=self.ALLOWED_CONTRACTS
                )
        
        return GateResult(
            gate_id="FUT-G1",
            gate_name="Contract Whitelist",
            status=GateStatus.FAIL,
            message=f"{symbol} not in allowed contracts",
            value=symbol,
            threshold=self.ALLOWED_CONTRACTS
        )
    
    def _check_volume(self, symbol: str, daily_volume: int) -> GateResult:
        """G2: Minimum daily volume."""
        passed = daily_volume >= self.MIN_VOLUME_CONTRACTS
        return GateResult(
            gate_id="FUT-G2",
            gate_name="Daily Volume",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Volume: {daily_volume:,} contracts" if passed else f"Volume {daily_volume:,} below {self.MIN_VOLUME_CONTRACTS:,} min",
            value=daily_volume,
            threshold=self.MIN_VOLUME_CONTRACTS
        )
    
    def _check_margin(self, margin_used_pct: float) -> GateResult:
        """G3: Max 50% of account as margin."""
        passed = margin_used_pct <= self.MAX_MARGIN_PCT
        return GateResult(
            gate_id="FUT-G3",
            gate_name="Margin Usage",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Margin: {margin_used_pct:.1f}%" if passed else f"Margin {margin_used_pct:.1f}% exceeds {self.MAX_MARGIN_PCT}% max",
            value=margin_used_pct,
            threshold=self.MAX_MARGIN_PCT
        )
    
    def _check_position_size(self, symbol: str, num_contracts: int) -> GateResult:
        """G4: Max contracts per trade."""
        symbol_upper = symbol.upper()
        max_contracts = 1  # Default
        
        for contract, max_c in self.MAX_CONTRACTS_PER_TRADE.items():
            if contract in symbol_upper:
                max_contracts = max_c
                break
        
        passed = num_contracts <= max_contracts
        return GateResult(
            gate_id="FUT-G4",
            gate_name="Position Size",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"{num_contracts} contracts" if passed else f"{num_contracts} contracts exceeds {max_contracts} max for {symbol}",
            value=num_contracts,
            threshold=max_contracts
        )
    
    def _check_trading_hours(self, is_rth: Optional[bool]) -> GateResult:
        """G5: Prefer regular trading hours."""
        if is_rth is None:
            # Determine from current time
            now = datetime.now().time()
            is_rth = self.RTH_START <= now <= self.RTH_END
        
        return GateResult(
            gate_id="FUT-G5",
            gate_name="Trading Hours",
            status=GateStatus.PASS if is_rth else GateStatus.WARN,
            message="Regular trading hours" if is_rth else "Outside RTH - reduced liquidity",
            value=is_rth,
            details={"rth_start": str(self.RTH_START), "rth_end": str(self.RTH_END)}
        )
    
    def _check_cot(self, symbol: str, cot_data: Optional[Dict]) -> GateResult:
        """G6: COT data alignment (soft gate)."""
        if not cot_data:
            return GateResult(
                gate_id="FUT-G6",
                gate_name="COT Alignment",
                status=GateStatus.SKIP,
                message="COT data not available"
            )
        
        # Check commercial positioning
        commercial_net = cot_data.get("commercial_net", 0)
        commercial_extreme = cot_data.get("commercial_extreme", False)
        
        if commercial_extreme:
            return GateResult(
                gate_id="FUT-G6",
                gate_name="COT Alignment",
                status=GateStatus.WARN,
                message=f"Commercial extreme positioning: {commercial_net:+,}",
                value=commercial_net,
                details=cot_data
            )
        
        return GateResult(
            gate_id="FUT-G6",
            gate_name="COT Alignment",
            status=GateStatus.PASS,
            message=f"COT commercial net: {commercial_net:+,}",
            value=commercial_net,
            details=cot_data
        )


# =============================================================================
# STOCKS GATES (Lower Priority - For Later)
# =============================================================================

class StocksGates(MarketGates):
    """
    Hard gates for Stocks trading.
    
    Focus: Long-term dividend positions
    Priority: LOW (not active trading, just long-term)
    """
    
    market_type = "stocks"
    
    # Gate thresholds
    MIN_MARKET_CAP_USD = 10_000_000_000  # $10B (large cap only)
    MIN_AVG_VOLUME = 1_000_000  # 1M shares/day
    MIN_DIVIDEND_YIELD = 0.02  # 2% minimum yield
    MAX_PAYOUT_RATIO = 0.80  # 80% max payout ratio
    MIN_YEARS_DIVIDEND = 5  # 5+ years consecutive dividends
    
    def check_all(
        self,
        symbol: str,
        market_cap: float,
        avg_volume: int,
        dividend_yield: float,
        payout_ratio: float,
        years_dividend: int,
        **kwargs
    ) -> MarketGateResults:
        """Check all stocks gates (for long-term positions)."""
        results = []
        
        # G1: Market Cap Gate
        results.append(self._check_market_cap(symbol, market_cap))
        
        # G2: Volume Gate
        results.append(self._check_volume(symbol, avg_volume))
        
        # G3: Dividend Yield Gate
        results.append(self._check_dividend_yield(symbol, dividend_yield))
        
        # G4: Payout Ratio Gate
        results.append(self._check_payout_ratio(symbol, payout_ratio))
        
        # G5: Dividend History Gate
        results.append(self._check_dividend_history(symbol, years_dividend))
        
        failed_gates = [r.gate_id for r in results if r.status == GateStatus.FAIL]
        passed = len(failed_gates) == 0
        
        return MarketGateResults(
            market_type=self.market_type,
            passed=passed,
            gate_results=results,
            failed_gates=failed_gates
        )
    
    def _check_market_cap(self, symbol: str, market_cap: float) -> GateResult:
        """G1: Large cap only ($10B+)."""
        passed = market_cap >= self.MIN_MARKET_CAP_USD
        return GateResult(
            gate_id="STK-G1",
            gate_name="Market Cap",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Market cap: ${market_cap/1e9:.1f}B" if passed else f"Market cap ${market_cap/1e9:.1f}B below ${self.MIN_MARKET_CAP_USD/1e9:.0f}B min",
            value=market_cap,
            threshold=self.MIN_MARKET_CAP_USD
        )
    
    def _check_volume(self, symbol: str, avg_volume: int) -> GateResult:
        """G2: Minimum average volume."""
        passed = avg_volume >= self.MIN_AVG_VOLUME
        return GateResult(
            gate_id="STK-G2",
            gate_name="Average Volume",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Avg volume: {avg_volume/1e6:.1f}M" if passed else f"Avg volume {avg_volume/1e6:.1f}M below {self.MIN_AVG_VOLUME/1e6:.0f}M min",
            value=avg_volume,
            threshold=self.MIN_AVG_VOLUME
        )
    
    def _check_dividend_yield(self, symbol: str, dividend_yield: float) -> GateResult:
        """G3: Minimum 2% dividend yield."""
        passed = dividend_yield >= self.MIN_DIVIDEND_YIELD
        return GateResult(
            gate_id="STK-G3",
            gate_name="Dividend Yield",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"Yield: {dividend_yield*100:.2f}%" if passed else f"Yield {dividend_yield*100:.2f}% below {self.MIN_DIVIDEND_YIELD*100:.0f}% min",
            value=dividend_yield,
            threshold=self.MIN_DIVIDEND_YIELD
        )
    
    def _check_payout_ratio(self, symbol: str, payout_ratio: float) -> GateResult:
        """G4: Sustainable payout ratio (<80%)."""
        passed = payout_ratio <= self.MAX_PAYOUT_RATIO
        return GateResult(
            gate_id="STK-G4",
            gate_name="Payout Ratio",
            status=GateStatus.PASS if passed else GateStatus.WARN,  # Warn, not fail
            message=f"Payout: {payout_ratio*100:.0f}%" if passed else f"Payout {payout_ratio*100:.0f}% above {self.MAX_PAYOUT_RATIO*100:.0f}% - sustainability risk",
            value=payout_ratio,
            threshold=self.MAX_PAYOUT_RATIO
        )
    
    def _check_dividend_history(self, symbol: str, years_dividend: int) -> GateResult:
        """G5: Minimum 5 years consecutive dividends."""
        passed = years_dividend >= self.MIN_YEARS_DIVIDEND
        return GateResult(
            gate_id="STK-G5",
            gate_name="Dividend History",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=f"{years_dividend} years dividends" if passed else f"{years_dividend} years below {self.MIN_YEARS_DIVIDEND} year minimum",
            value=years_dividend,
            threshold=self.MIN_YEARS_DIVIDEND
        )


# =============================================================================
# GATE FACTORY
# =============================================================================

def get_market_gates(market_type: str) -> MarketGates:
    """Factory function to get gates for a market type."""
    gates_map = {
        "crypto_spot": CryptoSpotGates,
        "crypto_perps": CryptoPerpsGates,
        "forex": ForexGates,
        "futures": FuturesGates,
        "stocks": StocksGates,
    }
    
    # Normalize market type
    market_key = market_type.lower().replace("-", "_").replace(" ", "_")
    
    if market_key not in gates_map:
        raise ValueError(f"Unknown market type: {market_type}. Valid: {list(gates_map.keys())}")
    
    return gates_map[market_key]()
