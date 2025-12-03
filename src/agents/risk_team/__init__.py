"""
Risk Team Agents

Three-perspective risk assessment for position sizing.
Based on ai-investment-agent risk team architecture.

The risk team provides three independent views:
- Risky Analyst: Aggressive, maximizes position for high-conviction plays
- Safe Analyst: Conservative, prioritizes capital preservation
- Neutral Analyst: Balanced, seeks optimal risk-adjusted sizing

Their recommendations are aggregated to produce final position sizing.
"""

import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .risky import risky_analyst, generate_risky_assessment
from .safe import safe_analyst, generate_safe_assessment
from .neutral import neutral_analyst, generate_neutral_assessment


__all__ = [
    # Agents
    "risky_analyst",
    "safe_analyst", 
    "neutral_analyst",
    # Functions
    "generate_risky_assessment",
    "generate_safe_assessment",
    "generate_neutral_assessment",
    "run_risk_team",
    # Classes
    "RiskTeamResult",
    "PositionSizing",
]


@dataclass
class PositionSizing:
    """Position sizing recommendation from risk team."""
    aggressive_pct: float  # Risky analyst's recommendation
    conservative_pct: float  # Safe analyst's recommendation
    balanced_pct: float  # Neutral analyst's recommendation
    recommended_pct: float  # Final recommendation (typically balanced)
    max_position_pct: float  # Hard cap based on risks
    
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 15.0
    risk_reward_ratio: float = 3.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggressive_pct": self.aggressive_pct,
            "conservative_pct": self.conservative_pct,
            "balanced_pct": self.balanced_pct,
            "recommended_pct": self.recommended_pct,
            "max_position_pct": self.max_position_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "risk_reward_ratio": self.risk_reward_ratio,
        }


@dataclass
class RiskTeamResult:
    """Complete result from risk team assessment."""
    symbol: str
    risky_assessment: Dict[str, Any]
    safe_assessment: Dict[str, Any]
    neutral_assessment: Dict[str, Any]
    position_sizing: PositionSizing
    risk_factors: List[str]
    duration_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "risky_assessment": self.risky_assessment,
            "safe_assessment": self.safe_assessment,
            "neutral_assessment": self.neutral_assessment,
            "position_sizing": self.position_sizing.to_dict(),
            "risk_factors": self.risk_factors,
            "duration_ms": self.duration_ms,
        }


async def run_risk_team(
    symbol: str,
    manager_decision: Dict[str, Any],
    data_gathering_result: Dict[str, Any] = None,
    debate_result: Dict[str, Any] = None,
) -> RiskTeamResult:
    """
    Run all three risk analysts in parallel and aggregate their recommendations.
    
    Args:
        symbol: Trading symbol
        manager_decision: Research Manager's decision dict
        data_gathering_result: Result from parallel_gather_data()
        debate_result: Result from run_full_debate()
    
    Returns:
        RiskTeamResult with aggregated position sizing
    
    Example:
        >>> result = await run_risk_team(
        ...     "AAPL",
        ...     manager_decision=debate["manager_decision"],
        ...     data_gathering_result=data.to_dict(),
        ...     debate_result=debate
        ... )
        >>> print(f"Recommended position: {result.position_sizing.recommended_pct}%")
    """
    from datetime import datetime
    start_time = datetime.now()
    
    # Run all three assessments in parallel
    risky_task = generate_risky_assessment(
        symbol, manager_decision, data_gathering_result, debate_result
    )
    safe_task = generate_safe_assessment(
        symbol, manager_decision, data_gathering_result, debate_result
    )
    neutral_task = generate_neutral_assessment(
        symbol, manager_decision, data_gathering_result, debate_result
    )
    
    results = await asyncio.gather(
        risky_task, safe_task, neutral_task,
        return_exceptions=True
    )
    
    # Handle results
    risky_result = results[0] if not isinstance(results[0], Exception) else {"position_pct": 0, "error": str(results[0])}
    safe_result = results[1] if not isinstance(results[1], Exception) else {"position_pct": 0, "error": str(results[1])}
    neutral_result = results[2] if not isinstance(results[2], Exception) else {"position_pct": 0, "error": str(results[2])}
    
    # Extract position percentages
    aggressive_pct = risky_result.get("position_pct", 0)
    conservative_pct = safe_result.get("position_pct", 0)
    balanced_pct = neutral_result.get("position_pct", 0)
    
    # Aggregate risk factors
    risk_factors = []
    risk_factors.extend(risky_result.get("risk_factors", []))
    risk_factors.extend(safe_result.get("risk_factors", []))
    risk_factors.extend(neutral_result.get("risk_factors", []))
    risk_factors = list(set(risk_factors))  # Deduplicate
    
    # Calculate max position based on risk factors
    max_position = 10.0  # Default max
    if len(risk_factors) > 3:
        max_position = 5.0
    elif len(risk_factors) > 5:
        max_position = 3.0
    
    # Cap all positions at max
    aggressive_pct = min(aggressive_pct, max_position * 1.2)  # Allow risky slight overage
    conservative_pct = min(conservative_pct, max_position)
    balanced_pct = min(balanced_pct, max_position)
    
    # Recommended is typically the balanced view, but bounded
    recommended_pct = balanced_pct
    
    position_sizing = PositionSizing(
        aggressive_pct=aggressive_pct,
        conservative_pct=conservative_pct,
        balanced_pct=balanced_pct,
        recommended_pct=recommended_pct,
        max_position_pct=max_position,
        stop_loss_pct=neutral_result.get("stop_loss_pct", 5.0),
        take_profit_pct=neutral_result.get("take_profit_pct", 15.0),
        risk_reward_ratio=neutral_result.get("risk_reward_ratio", 3.0),
    )
    
    duration_ms = (datetime.now() - start_time).total_seconds() * 1000
    
    return RiskTeamResult(
        symbol=symbol,
        risky_assessment=risky_result,
        safe_assessment=safe_result,
        neutral_assessment=neutral_result,
        position_sizing=position_sizing,
        risk_factors=risk_factors,
        duration_ms=duration_ms,
    )
