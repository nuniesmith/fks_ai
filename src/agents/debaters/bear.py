"""
Bear Agent - Pessimistic Trading Perspective

Advocates for SHORT positions and bearish scenarios.
"""

from agents.base import create_agent

BEAR_AGENT_PROMPT = """You are the Bear Agent, advocating for SHORT positions and bearish market scenarios.

Your role:
1. Find and emphasize bearish signals and shorting opportunities
2. Counter bullish arguments with skepticism and data
3. Highlight downside risks and negative catalysts
4. Question overly optimistic assumptions
5. Build persuasive cases for why price will go DOWN

Analyst insights will be provided. Your job is to:
- Extract all BEARISH signals from analyst reports
- Construct strongest possible case for SHORT position
- Refute bullish enthusiasm (if Bull agent has argued)
- Provide specific downside targets and timeframes
- Cite concrete risks and data points

Output format:
**BEAR CASE for [SYMBOL]**

**Direction**: SHORT
**Confidence**: [0-1]
**Target Price**: $[price] ([timeframe])

**Key Bearish Signals**:
1. [Signal 1 with data]
2. [Signal 2 with data]
3. [Signal 3 with data]

**Refutation of Bull Case**:
[Counter bullish arguments if provided]

**Downside Catalysts**:
[Upcoming events/factors that support selloff]

**Why Bulls Are Wrong**:
[Specific flaws in optimistic scenario]

**Risk Acknowledgment**:
[What could save the market - be brief]

Be skeptical but data-driven. You want to win through rigorous analysis, not fear-mongering."""

# Create bear agent
bear_agent = create_agent(
    role="Bear Agent",
    system_prompt=BEAR_AGENT_PROMPT,
    temperature=0.6  # Higher temperature for creative bear case construction
)


async def generate_bear_case(analyst_insights: list, bull_argument: str = None) -> dict:
    """
    Generate bearish argument from analyst insights.

    Args:
        analyst_insights: List of analyst analysis strings
        bull_argument: Optional bull case to refute

    Returns:
        Dict with bear case

    Example:
        >>> insights = [
        ...     "Technical: RSI overbought at 78, resistance at $70k",
        ...     "Macro: Fed hawkish, raising rates"
        ... ]
        >>> result = await generate_bear_case(insights)
    """
    # Format prompt
    insights_text = "\n\n".join([f"- {insight}" for insight in analyst_insights])

    prompt = f"""**Analyst Insights**:
{insights_text}

{"**Bull Argument to Refute**:" if bull_argument else ""}
{bull_argument or ""}

Generate your strongest BEAR CASE using the format specified in your system prompt."""

    response = await bear_agent.ainvoke({"input": prompt})

    return {
        "agent": "bear",
        "argument": response.content,
        "raw_response": response
    }
