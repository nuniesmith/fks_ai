"""
Manager Agent - Objective Synthesis

Synthesizes Bull and Bear debates into final trading decision.
"""

from agents.base import create_agent

MANAGER_AGENT_PROMPT = """You are the Manager Agent, responsible for objectively synthesizing
Bull and Bear debate into a final trading decision.

Your role:
1. Evaluate both Bull and Bear arguments fairly
2. Consider market regime context (bull/bear/sideways)
3. Weigh confidence levels and data quality
4. Make final BUY/SELL/HOLD decision
5. Determine position size based on conviction

Evaluation criteria:
- **Data Quality**: Which side has stronger factual support?
- **Logic**: Which argument is more internally consistent?
- **Risk/Reward**: What's the asymmetric bet here?
- **Regime Fit**: Does this align with current market conditions?
- **Edge**: Do we have informational or analytical advantage?

Output format:
**MANAGER DECISION for [SYMBOL]**

**Final Decision**: BUY / SELL / HOLD
**Confidence**: [0-1]
**Position Size**: [0-10]% of capital

**Winning Argument**: Bull / Bear / Neither
**Reasoning**:
[2-3 sentences on why you chose this decision]

**Bull Case Strengths**:
- [Strength 1]
- [Strength 2]

**Bear Case Strengths**:
- [Strength 1]
- [Strength 2]

**Deciding Factors**:
[What tipped the scales? What was the key insight?]

**Execution Plan**:
- Entry: $[price]
- Stop-Loss: $[price] ([%] below entry)
- Take-Profit: $[price] ([%] above entry)
- Time Horizon: [intraday / swing / position]

**Contingencies**:
[If X happens, do Y. If Z happens, exit.]

Be objective. Your job is to make the BEST decision, not to please either side.
Sometimes HOLD is the right answer when risk/reward is unclear."""

# Create manager agent
manager_agent = create_agent(
    role="Manager Agent",
    system_prompt=MANAGER_AGENT_PROMPT,
    temperature=0.3  # Lower temperature for more consistent decision-making
)


async def synthesize_debate(
    bull_argument: str,
    bear_argument: str,
    market_regime: str = "unknown",
    additional_context: dict = None
) -> dict:
    """
    Synthesize Bull and Bear arguments into final decision.

    Args:
        bull_argument: Bull agent's argument
        bear_argument: Bear agent's argument
        market_regime: Current market regime (bull/bear/sideways)
        additional_context: Additional context dict

    Returns:
        Dict with final decision

    Example:
        >>> decision = await synthesize_debate(
        ...     bull_argument="Strong support, Fed dovish...",
        ...     bear_argument="Overbought, macro risks...",
        ...     market_regime="bull"
        ... )
    """
    # Format context
    context = additional_context or {}
    context_text = "\n".join([f"- {k}: {v}" for k, v in context.items()])

    prompt = f"""**BULL ARGUMENT**:
{bull_argument}

---

**BEAR ARGUMENT**:
{bear_argument}

---

**MARKET REGIME**: {market_regime}

**ADDITIONAL CONTEXT**:
{context_text or 'None'}

---

Synthesize these arguments and make your final decision using the format in your system prompt.
Consider: Which side has better data? What's the regime? What's the risk/reward?"""

    response = await manager_agent.ainvoke({"input": prompt})

    return {
        "agent": "manager",
        "decision": response.content,
        "raw_response": response,
        "inputs": {
            "bull_argument": bull_argument,
            "bear_argument": bear_argument,
            "regime": market_regime
        }
    }
