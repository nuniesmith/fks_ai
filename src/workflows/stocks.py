"""
Stocks Workflow

Screening workflow for equities trading.
Lower priority but included for completeness.
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
from ..rules import RulesEngine, StocksGates

logger = logging.getLogger(__name__)


@dataclass
class FundamentalData:
    """Basic fundamental data for stock screening."""
    market_cap: float  # In billions
    pe_ratio: float
    eps_growth: float  # YoY %
    revenue_growth: float  # YoY %
    debt_to_equity: float
    sector: str
    industry: str


class StocksWorkflow(BaseWorkflow):
    """
    Stocks market screening workflow.
    
    NOTE: Lower priority in the trading framework.
    Included for diversification and learning purposes.
    
    FOCUS:
    - Large cap, liquid names
    - Sector rotation opportunities
    - Swing trades (not day trading)
    """
    
    market_type = "stocks"
    
    # Market hours (EST)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    
    # Minimum criteria
    MIN_MARKET_CAP = 10_000_000_000  # $10B
    MIN_AVG_VOLUME = 1_000_000  # shares
    
    # Sector ETFs for rotation analysis
    SECTOR_ETFS = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Materials": "XLB",
        "Industrials": "XLI",
        "Real Estate": "XLRE",
        "Communications": "XLC",
    }
    
    def __init__(self, watchlist: Optional[List[str]] = None):
        self.rules_engine = RulesEngine()
        self.stocks_gates = StocksGates()
        # Default to major tech + diversified names
        self._watchlist = watchlist or [
            # Large cap tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
            # Diversified
            "JPM", "V", "UNH", "XOM", "JNJ",
            # Sector ETFs
            "SPY", "QQQ",
        ]
    
    @property
    def watchlist(self) -> List[str]:
        return self._watchlist
    
    def is_market_hours(self, current_time: Optional[time] = None) -> bool:
        """Check if market is open."""
        if current_time is None:
            current_time = datetime.now().time()
        return self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE
    
    async def fetch_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch market data for stock symbols.
        
        In production, this would call:
        - Market data API (yfinance, polygon, etc.)
        - Fundamental data sources
        - Sector rotation analysis
        """
        data = {}
        
        # Mock fundamental data
        fundamentals = {
            "AAPL": FundamentalData(3000, 28, 15, 8, 1.5, "Technology", "Consumer Electronics"),
            "MSFT": FundamentalData(2800, 35, 20, 12, 0.5, "Technology", "Software"),
            "GOOGL": FundamentalData(1800, 25, 25, 10, 0.2, "Technology", "Internet"),
            "AMZN": FundamentalData(1600, 60, 30, 15, 0.8, "Consumer Discretionary", "E-commerce"),
            "NVDA": FundamentalData(1200, 65, 100, 80, 0.4, "Technology", "Semiconductors"),
            "META": FundamentalData(900, 30, 35, 20, 0.3, "Communications", "Social Media"),
            "JPM": FundamentalData(500, 11, 10, 5, 1.0, "Financials", "Banks"),
            "V": FundamentalData(450, 28, 15, 10, 0.6, "Financials", "Payment Processing"),
            "UNH": FundamentalData(500, 22, 12, 8, 0.7, "Healthcare", "Insurance"),
            "XOM": FundamentalData(400, 12, 20, 15, 0.3, "Energy", "Oil & Gas"),
            "JNJ": FundamentalData(400, 15, 5, 3, 0.4, "Healthcare", "Pharmaceuticals"),
        }
        
        for symbol in symbols:
            # Skip ETFs for fundamental analysis
            is_etf = symbol in ["SPY", "QQQ"] or symbol.startswith("XL")
            
            # Determine mock price based on symbol
            prices = {
                "AAPL": 190.0, "MSFT": 415.0, "GOOGL": 145.0, "AMZN": 180.0,
                "NVDA": 900.0, "META": 500.0, "JPM": 185.0, "V": 275.0,
                "UNH": 520.0, "XOM": 105.0, "JNJ": 155.0,
                "SPY": 510.0, "QQQ": 440.0,
            }
            price = prices.get(symbol, 100.0)
            
            data[symbol] = {
                "price": price,
                "avg_volume": 50_000_000 if symbol in ["AAPL", "MSFT", "SPY"] else 10_000_000,
                "market_cap": fundamentals.get(symbol, FundamentalData(50, 20, 10, 5, 0.5, "Other", "Other")).market_cap * 1e9,
                "is_etf": is_etf,
                # Technical data
                "trend_daily": "up",
                "trend_weekly": "up",
                "rsi_14": 55,
                "macd_signal": "bullish",
                "above_50ma": True,
                "above_200ma": True,
                "volume_ratio": 1.1,
                "atr_pct": 1.5,
                "relative_strength": 1.05,  # vs SPY
                # Fundamental (if not ETF)
                "fundamentals": fundamentals.get(symbol, None),
            }
        
        return data
    
    async def identify_candidates(
        self, 
        market_data: Dict[str, Dict[str, Any]]
    ) -> List[SignalCandidate]:
        """
        Identify potential stock signals.
        
        Criteria for swing trades:
        - Uptrend on daily/weekly
        - Above major moving averages
        - Positive relative strength
        - Volume confirmation
        """
        candidates = []
        is_market_open = self.is_market_hours()
        
        for symbol, data in market_data.items():
            # Skip if below minimum requirements
            if data.get("market_cap", 0) < self.MIN_MARKET_CAP:
                continue
            if data.get("avg_volume", 0) < self.MIN_AVG_VOLUME:
                continue
            
            trend_daily = data.get("trend_daily", "sideways")
            trend_weekly = data.get("trend_weekly", "sideways")
            
            # Need uptrend for long bias (we don't short stocks)
            if trend_daily != "up" or trend_weekly != "up":
                continue
            
            # Need to be above moving averages
            if not data.get("above_50ma", False) or not data.get("above_200ma", False):
                continue
            
            price = data["price"]
            atr_pct = data.get("atr_pct", 2.0)
            
            # Calculate levels
            entry = price
            stop = price * (1 - atr_pct * 1.5 / 100)  # 1.5x ATR stop
            target = price * (1 + atr_pct * 3 / 100)  # 2:1 R:R
            
            # Calculate confidence
            confidence = 0.5
            
            # Trend alignment
            if trend_daily == trend_weekly:
                confidence += 0.1
            
            # Relative strength
            rel_strength = data.get("relative_strength", 1.0)
            if rel_strength > 1.1:
                confidence += 0.15
            elif rel_strength > 1.0:
                confidence += 0.05
            
            # MACD confirmation
            if data.get("macd_signal") == "bullish":
                confidence += 0.1
            
            # Volume
            if data.get("volume_ratio", 0) > 1.2:
                confidence += 0.05
            
            # Fundamental boost (for stocks, not ETFs)
            fundamentals = data.get("fundamentals")
            if fundamentals:
                if fundamentals.eps_growth > 15:
                    confidence += 0.05
                if fundamentals.revenue_growth > 10:
                    confidence += 0.05
            
            confidence = min(max(confidence, 0.0), 1.0)
            
            candidate = SignalCandidate(
                symbol=symbol,
                direction=SignalDirection.LONG,  # We only go long stocks
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                position_size_pct=10.0,  # Stocks get smaller allocation
                confidence=confidence,
                timeframe="1D",
                reason=f"Stock swing: Uptrend, RS={rel_strength:.2f}, {'Above MAs' if data.get('above_200ma') else 'Below MAs'}",
                technical=TechnicalContext(
                    trend_4h=trend_daily,  # Use daily as 4h equivalent
                    trend_1d=trend_weekly,  # Use weekly as 1d equivalent
                    key_levels=[stop, target],
                    atr_pct=atr_pct,
                    rsi=data.get("rsi_14", 50),
                    volume_ratio=data.get("volume_ratio", 1.0),
                ),
                metadata={
                    "is_etf": data.get("is_etf", False),
                    "market_cap": data.get("market_cap", 0),
                    "relative_strength": rel_strength,
                    "fundamentals": {
                        "sector": fundamentals.sector,
                        "pe_ratio": fundamentals.pe_ratio,
                        "eps_growth": fundamentals.eps_growth,
                    } if fundamentals else None,
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
        Validate a stock signal candidate.
        """
        warnings = []
        
        # Calculate position value
        position_value = account_balance * (candidate.position_size_pct / 100)
        shares = int(position_value / candidate.entry_price)
        actual_position = shares * candidate.entry_price
        
        # Check stocks gates
        gate_result = self.stocks_gates.check_all(
            symbol=candidate.symbol,
            market_cap=market_data.get("market_cap", 0),
            avg_volume=market_data.get("avg_volume", 0),
            position_pct=candidate.position_size_pct,
            relative_strength=candidate.metadata.get("relative_strength", 1.0),
        )
        
        if not gate_result.passed:
            for gate in gate_result.gate_results:
                if gate.status.value == "fail":
                    warnings.append(f"Gate {gate.gate_id}: {gate.message}")
            return False, SignalQuality.REJECTED, warnings
        
        # Calculate risk
        risk_pct = ((candidate.entry_price - candidate.stop_loss) / candidate.entry_price) * candidate.position_size_pct
        
        # Check hard rules
        result = self.rules_engine.validate_trade(
            symbol=candidate.symbol,
            market="stocks",
            position_size_pct=candidate.position_size_pct,
            account_balance=account_balance,
            stop_loss_set=candidate.stop_loss > 0,
            risk_per_trade_pct=risk_pct,
            data_quality_pct=90.0,  # Stock data is reliable
            liquidity_usd=market_data.get("avg_volume", 0) * market_data.get("price", 100),
            leverage=1.0,  # No leverage for stocks
            trend_direction=candidate.technical.trend_4h if candidate.technical else None,
        )
        
        if not result.hard_rule_passed:
            for v in result.hard_violations:
                warnings.append(f"HR {v.rule_id.value}: {v.consequence}")
            return False, SignalQuality.REJECTED, warnings
        
        # Determine quality
        soft_warnings = result.soft_rule_warnings
        rel_strength = candidate.metadata.get("relative_strength", 1.0)
        
        if candidate.confidence >= 0.75 and soft_warnings == 0 and rel_strength > 1.1:
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
        
        # Stocks typically get B quality at best (lower priority market)
        if quality == SignalQuality.A_PLUS:
            quality = SignalQuality.A
        
        # Add position info
        candidate.metadata["shares"] = shares
        candidate.metadata["position_value"] = actual_position
        candidate.metadata["risk_pct"] = risk_pct
        
        return True, quality, warnings
