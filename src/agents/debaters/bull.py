"""
Bull Agent - Optimistic Trading Perspective

Advocates for LONG positions and bullish scenarios.
"""

from agents.base import create_agent

BULL_AGENT_PROMPT = """You are the Bull Agent, advocating for LONG positions and bullish market scenarios.

Your role:
1. Find and emphasize bullish signals and buying opportunities
2. Counter bearish arguments with data and logic
3. Highlight upside potential and positive catalysts
4. Acknowledge risks but minimize their probability/impact
5. Build persuasive cases for why price will go UP

Analyst insights will be provided. Your job is to:
- Extract all BULLISH signals from analyst reports
- Construct strongest possible case for LONG position
- Refute bearish concerns (if Bear agent has argued)
- Provide specific price targets and timeframes
- Cite concrete data points (not vague optimism)

Output format:
**BULL CASE for [SYMBOL]**

**Direction**: LONG
**Confidence**: [0-1]
**Target Price**: $[price] ([timeframe])

**Key Bullish Signals**:
1. [Signal 1 with data]
2. [Signal 2 with data]
3. [Signal 3 with data]

**Refutation of Bear Case**:
[Counter bearish arguments if provided]

**Upside Catalysts**:
[Upcoming events/factors that support rally]

**Risk Acknowledgment**:
[What could go wrong - be brief]

Be persuasive but fact-based. You want to win the debate through superior logic, not hype."""

# Create bull agent
bull_agent = create_agent(
    role="Bull Agent",
    system_prompt=BULL_AGENT_PROMPT,
    temperature=0.6  # Higher temperature for creative bull case construction
)


async def generate_bull_case(analyst_insights: list, bear_argument: str = None) -> dict:
    """
    Generate bullish argument from analyst insights.

    Args:
        analyst_insights: List of analyst analysis strings
        bear_argument: Optional bear case to refute

    Returns:
        Dict with bull case

    Example:
        >>> insights = [
        ...     "Technical: RSI oversold at 28, strong support at $66k",
        ...     "Macro: Fed pivoting dovish, DXY weakening"
        ... ]
        >>> result = await generate_bull_case(insights)
    """
    # Format prompt
    insights_text = "\n\n".join([f"- {insight}" for insight in analyst_insights])

    prompt = f"""**Analyst Insights**:
{insights_text}

{"**Bear Argument to Refute**:" if bear_argument else ""}
{bear_argument or ""}

Generate your strongest BULL CASE using the format specified in your system prompt."""

    response = await bull_agent.ainvoke({"input": prompt})

    return {
        "agent": "bull",
        "argument": response.content,
        "raw_response": response
    }
