"""
Base Workflow Classes

Abstract base class and common types for all screening workflows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SignalQuality(Enum):
    """Quality rating for generated signals."""
    A_PLUS = "A+"  # 90%+ confidence, all checks pass
    A = "A"        # 80-89% confidence, minor soft rule warnings
    B = "B"        # 70-79% confidence, some warnings
    C = "C"        # 60-69% confidence, multiple warnings
    D = "D"        # Below 60%, significant issues
    REJECTED = "REJECTED"  # Failed hard rules, not tradeable


class SignalDirection(Enum):
    """Signal direction."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class TechnicalContext:
    """Technical analysis context for a signal."""
    trend_4h: str  # "up", "down", "sideways"
    trend_1d: str
    key_levels: List[float]  # Support/resistance
    atr_pct: float  # ATR as % of price
    rsi: float
    volume_ratio: float  # Current vs average
    pattern: Optional[str] = None  # Detected pattern
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend_4h": self.trend_4h,
            "trend_1d": self.trend_1d,
            "key_levels": self.key_levels,
            "atr_pct": self.atr_pct,
            "rsi": self.rsi,
            "volume_ratio": self.volume_ratio,
            "pattern": self.pattern,
        }


@dataclass
class SignalCandidate:
    """A potential trading signal before validation."""
    symbol: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float  # Suggested position size
    confidence: float  # 0-1
    timeframe: str
    reason: str
    technical: Optional[TechnicalContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def risk_reward_ratio(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0
    
    @property
    def risk_pct(self) -> float:
        """Risk as % of entry price."""
        return abs(self.entry_price - self.stop_loss) / self.entry_price * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size_pct": self.position_size_pct,
            "confidence": self.confidence,
            "risk_reward_ratio": self.risk_reward_ratio,
            "timeframe": self.timeframe,
            "reason": self.reason,
            "technical": self.technical.to_dict() if self.technical else None,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowResult:
    """Result of a screening workflow run."""
    market_type: str
    symbols_scanned: int
    candidates_found: int
    signals_generated: List[SignalCandidate]
    rejected_count: int
    rejected_reasons: Dict[str, int]  # Reason -> count
    execution_time_ms: float
    timestamp: datetime
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_type": self.market_type,
            "symbols_scanned": self.symbols_scanned,
            "candidates_found": self.candidates_found,
            "signals_generated": [s.to_dict() for s in self.signals_generated],
            "rejected_count": self.rejected_count,
            "rejected_reasons": self.rejected_reasons,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "errors": self.errors,
            "metadata": self.metadata,
        }


class BaseWorkflow(ABC):
    """
    Abstract base class for market screening workflows.
    
    Each workflow implements the screening logic for its specific market:
    1. Fetch relevant data (prices, volume, indicators)
    2. Identify potential signals
    3. Validate against rules engine
    4. Generate qualified signals
    """
    
    @property
    @abstractmethod
    def market_type(self) -> str:
        """Return the market type this workflow handles."""
        pass
    
    @property
    @abstractmethod
    def watchlist(self) -> List[str]:
        """Return the default watchlist for this market."""
        pass
    
    @abstractmethod
    async def fetch_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch market data for the given symbols.
        
        Returns dict of symbol -> data dict containing:
        - price, volume, spread, etc.
        - Market-specific data (funding rate, on-chain, etc.)
        """
        pass
    
    @abstractmethod
    async def identify_candidates(
        self, 
        market_data: Dict[str, Dict[str, Any]]
    ) -> List[SignalCandidate]:
        """
        Identify potential trading signals from market data.
        
        Returns list of SignalCandidate objects before validation.
        """
        pass
    
    @abstractmethod
    async def validate_candidate(
        self, 
        candidate: SignalCandidate,
        market_data: Dict[str, Any],
        account_balance: float
    ) -> tuple[bool, SignalQuality, List[str]]:
        """
        Validate a candidate signal against rules.
        
        Returns:
            (is_valid, quality, warnings): Validation result
        """
        pass
    
    async def run(
        self,
        symbols: Optional[List[str]] = None,
        account_balance: float = 10000.0,
        max_signals: int = 5,
    ) -> WorkflowResult:
        """
        Run the full screening workflow.
        
        Args:
            symbols: List of symbols to scan (defaults to watchlist)
            account_balance: Account balance for position sizing
            max_signals: Maximum number of signals to return
        
        Returns:
            WorkflowResult with all findings
        """
        import time
        start_time = time.time()
        
        symbols = symbols or self.watchlist
        errors = []
        rejected_reasons: Dict[str, int] = {}
        
        # 1. Fetch market data
        try:
            market_data = await self.fetch_market_data(symbols)
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            errors.append(f"Data fetch error: {str(e)}")
            market_data = {}
        
        # 2. Identify candidates
        try:
            candidates = await self.identify_candidates(market_data)
        except Exception as e:
            logger.error(f"Error identifying candidates: {e}")
            errors.append(f"Candidate identification error: {str(e)}")
            candidates = []
        
        # 3. Validate candidates
        validated_signals = []
        rejected_count = 0
        
        for candidate in candidates:
            try:
                symbol_data = market_data.get(candidate.symbol, {})
                is_valid, quality, warnings = await self.validate_candidate(
                    candidate, symbol_data, account_balance
                )
                
                if is_valid and quality != SignalQuality.REJECTED:
                    candidate.metadata["quality"] = quality.value
                    candidate.metadata["warnings"] = warnings
                    validated_signals.append(candidate)
                else:
                    rejected_count += 1
                    reason = warnings[0] if warnings else "Unknown"
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                    
            except Exception as e:
                logger.error(f"Error validating {candidate.symbol}: {e}")
                errors.append(f"Validation error for {candidate.symbol}: {str(e)}")
                rejected_count += 1
        
        # 4. Sort by confidence and limit
        validated_signals.sort(key=lambda s: s.confidence, reverse=True)
        validated_signals = validated_signals[:max_signals]
        
        execution_time = (time.time() - start_time) * 1000
        
        return WorkflowResult(
            market_type=self.market_type,
            symbols_scanned=len(symbols),
            candidates_found=len(candidates),
            signals_generated=validated_signals,
            rejected_count=rejected_count,
            rejected_reasons=rejected_reasons,
            execution_time_ms=execution_time,
            timestamp=datetime.now(),
            errors=errors,
        )
