"""
Debater Agents Module

Bull/Bear debate system with strict role commitment and hard scoring gates.
Based on ai-investment-agent research team architecture.

Key Features:
- Strict role commitment (Bull argues HARD, Bear argues HARD)
- 2-round debate structure (initial case + rebuttal)
- Hard scoring gates (Health ≥50%, Growth ≥50%, P/E gates, Liquidity)
- Research Manager synthesizes with gate enforcement
"""

from .bear import bear_agent, generate_bear_case, BEAR_AGENT_PROMPT
from .bull import bull_agent, generate_bull_case, BULL_AGENT_PROMPT
from .manager import (
    manager_agent,
    synthesize_debate,
    evaluate_thesis_gates,
    ThesisGates,
    MANAGER_AGENT_PROMPT
)

__all__ = [
    # Agents
    "bull_agent",
    "bear_agent",
    "manager_agent",
    # Functions
    "generate_bull_case",
    "generate_bear_case",
    "synthesize_debate",
    "evaluate_thesis_gates",
    # Classes
    "ThesisGates",
    # Prompts (for inspection/debugging)
    "BULL_AGENT_PROMPT",
    "BEAR_AGENT_PROMPT",
    "MANAGER_AGENT_PROMPT",
    # Convenience function
    "run_full_debate",
]


async def run_full_debate(
    symbol: str,
    analyst_insights: list,
    data_gathering_result: dict = None,
    market_regime: str = "unknown",
    rounds: int = 2
) -> dict:
    """
    Run a complete bull/bear debate with optional 2-round structure.
    
    Args:
        symbol: Trading symbol
        analyst_insights: List of analyst analysis strings
        data_gathering_result: Result from parallel_gather_data()
        market_regime: Current market regime
        rounds: Number of debate rounds (1 or 2)
    
    Returns:
        Dict with complete debate results and final decision
    
    Example:
        >>> from agents.data_gathering import parallel_gather_data
        >>> from agents.debaters import run_full_debate
        >>> 
        >>> data = await parallel_gather_data("AAPL")
        >>> insights = [
        ...     data.market_analysis["report"],
        ...     data.fundamentals_analysis["report"],
        ...     data.sentiment_analysis["report"],
        ...     data.news_analysis["report"],
        ... ]
        >>> result = await run_full_debate("AAPL", insights, data.to_dict())
        >>> print(f"Decision: {result['final_decision']}")
    """
    # Round 1: Initial arguments
    bull_r1 = await generate_bull_case(
        analyst_insights,
        data_gathering_result=data_gathering_result,
        debate_round=1
    )
    
    bear_r1 = await generate_bear_case(
        analyst_insights,
        data_gathering_result=data_gathering_result,
        debate_round=1
    )
    
    bull_r2 = None
    bear_r2 = None
    
    # Round 2: Rebuttals (if enabled)
    if rounds >= 2:
        # Bull responds to Bear's round 1
        bull_r2 = await generate_bull_case(
            analyst_insights,
            data_gathering_result=data_gathering_result,
            bear_argument=bear_r1["argument"],
            debate_round=2
        )
        
        # Bear responds to Bull's round 1
        bear_r2 = await generate_bear_case(
            analyst_insights,
            data_gathering_result=data_gathering_result,
            bull_argument=bull_r1["argument"],
            debate_round=2
        )
    
    # Manager synthesizes
    decision = await synthesize_debate(
        bull_argument=bull_r1["argument"],
        bear_argument=bear_r1["argument"],
        data_gathering_result=data_gathering_result,
        market_regime=market_regime,
        bull_round_2=bull_r2["argument"] if bull_r2 else None,
        bear_round_2=bear_r2["argument"] if bear_r2 else None
    )
    
    return {
        "symbol": symbol,
        "rounds": rounds,
        "bull_round_1": bull_r1,
        "bear_round_1": bear_r1,
        "bull_round_2": bull_r2,
        "bear_round_2": bear_r2,
        "manager_decision": decision,
        "final_decision": decision["decision"],
        "thesis_gates": decision["thesis_gates"],
        "gates_all_pass": decision["gates_all_pass"],
        "compliance_pct": decision["compliance_pct"],
    }
