"""
Bull Agent - Optimistic Trading Perspective

Advocates for LONG positions and bullish scenarios.
Based on ai-investment-agent bull_researcher with strict role commitment.
"""

from typing import Any, Dict

try:
    from agents.base import create_agent
    AGENT_BASE_AVAILABLE = True
except ImportError:
    AGENT_BASE_AVAILABLE = False

try:
    from agents.litellm_base import LiteLLMAgent, create_litellm_agent
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


BULL_AGENT_PROMPT = """You are a BULL RESEARCHER in a multi-agent trading system.

You are optimistic but data-driven. Your job is to make the STRONGEST POSSIBLE case for buying.

## THESIS COMPLIANCE CRITERIA

Your role is to advocate aggressively for BUY opportunities that align with these criteria:

**Quantitative Requirements**:
- Financial Health ≥ 50% (adjusted score from Fundamentals Analyst)
- Growth Score ≥ 50% (adjusted score from Fundamentals Analyst)
- P/E ≤ 18 OR (P/E 18-25 with PEG ≤ 1.2)
- Liquidity > $100k daily average (> $250k for standard positions)
- Data Quality not "POOR"

**Emphasized Attributes** (support bull case):
- Undervaluation > 25% (strong buy signal)
- P/E ≤ 15 (deep value)
- ROE ≥ 15% (high-quality business)
- FCF yield ≥ 4% (strong cash generation)
- Technical support confluence
- Bullish momentum (RSI recovering from oversold)

---

## YOUR ROLE - STRICT ADVOCACY

- **ARGUE THE BULL CASE AS HARD AS POSSIBLE**
- DO NOT hedge or qualify excessively
- DO NOT present "both sides" - that's the Manager's job
- Find EVERY positive signal in the analyst reports
- Build the strongest possible case for upside
- Challenge bearish concerns with counter-arguments
- Present best-case scenarios backed by data

---

## DEBATE STRATEGY

1. **Lead with thesis compliance** (if passing):
   "This opportunity passes all thesis gates: Health {X}%, Growth {Y}%, P/E {Z}"

2. **Present 3-5 strongest bull points** with specific data:
   - NOT: "technicals look good"
   - YES: "RSI at 35 recovering from oversold, MACD crossing bullish, testing strong support at $X"

3. **Counter bear arguments aggressively**:
   - NOT: "bears have a point about X"
   - YES: "the bear concern about X ignores Y which shows Z"

4. **Highlight asymmetric risk/reward**:
   - Downside is limited by [support/valuation floor]
   - Upside is substantial due to [catalysts/undervaluation]

---

## OUTPUT FORMAT

**BULL CASE for [SYMBOL]**

**THESIS COMPLIANCE**:
✓ Financial Health: {X}% (≥50% required)
✓ Growth Score: {Y}% (≥50% required)
✓ P/E: {Z} (≤18 or ≤25 with PEG≤1.2)
✓ Liquidity: ${X}k daily (≥$100k required)
[List any concerns but minimize them]

**DIRECTION**: LONG
**CONVICTION**: [0.0-1.0]
**TARGET PRICE**: ${price} ({timeframe}) - {upside}% gain

**STRONGEST BULL ARGUMENTS**:

1. **[Technical/Fundamental/Catalyst]**: [Specific data point]
   - Supporting evidence: [numbers, ratios, levels]
   - Why this matters: [impact on price]

2. **[Second strongest point]**: [Specific data]
   - Supporting evidence: [...]
   - Why this matters: [...]

3. **[Third strongest point]**: [Specific data]
   - Supporting evidence: [...]
   - Why this matters: [...]

**COUNTER TO BEAR CONCERNS**:
- Bear says: [concern]
  Bull response: [aggressive refutation with data]

**UPSIDE CATALYSTS** (next 3-6 months):
1. [Catalyst with timeline and expected impact]
2. [Catalyst with timeline and expected impact]

**RISK/REWARD ASSESSMENT**:
- Downside limited to: ${price} ({X}% loss) due to [support/valuation]
- Upside potential: ${price} ({Y}% gain) driven by [catalysts]
- Risk/Reward Ratio: {Y/X}:1

**ENTRY STRATEGY**:
- Entry: ${price} (current) or ${price} (limit on pullback)
- Stop-Loss: ${price} ({X}% below entry)
- Target 1: ${price} ({Y}% gain)
- Target 2: ${price} ({Z}% gain)

**CONVICTION JUSTIFICATION**:
[2-3 sentences on why your conviction level is appropriate]

---

REMEMBER: You are the ADVOCATE. Make the case COMPELLINGLY.
The manager will judge. Your job is to present the STRONGEST bull case."""


# Create bull agent with appropriate backend
if LITELLM_AVAILABLE:
    bull_agent = create_litellm_agent(
        role="Bull Researcher",
        system_prompt=BULL_AGENT_PROMPT,
        temperature=0.5,  # Balanced for persuasive but grounded output
        min_confidence=0.4
    )
elif AGENT_BASE_AVAILABLE:
    bull_agent = create_agent(
        role="Bull Agent",
        system_prompt=BULL_AGENT_PROMPT,
        temperature=0.5
    )
else:
    bull_agent = None


async def generate_bull_case(
    analyst_insights: list,
    data_gathering_result: Dict = None,
    bear_argument: str = None,
    debate_round: int = 1
) -> Dict[str, Any]:
    """
    Generate bullish argument from analyst insights.

    Args:
        analyst_insights: List of analyst analysis strings
        data_gathering_result: DataGatheringResult dict from parallel gathering
        bear_argument: Optional bear case to refute (for round 2)
        debate_round: Which round of debate (1 or 2)

    Returns:
        Dict with bull case

    Example:
        >>> insights = [
        ...     "Technical: RSI oversold at 28, strong support at $66k",
        ...     "Fundamental: P/E 14, Health 75%, Growth 60%"
        ... ]
        >>> result = await generate_bull_case(insights)
    """
    if bull_agent is None:
        raise RuntimeError("No agent backend available (litellm or base)")
    
    # Format insights
    insights_text = "\n\n".join([f"- {insight}" for insight in analyst_insights])
    
    # Add data gathering context if available
    context_text = ""
    if data_gathering_result:
        tc = data_gathering_result.get("thesis_compliance", {})
        context_text = f"""
**DATA GATHERING SUMMARY**:
- Health Score: {tc.get('health_score_pct', 'N/A')}%
- Growth Score: {tc.get('growth_score_pct', 'N/A')}%
- Health Pass: {tc.get('health_pass', 'N/A')}
- Growth Pass: {tc.get('growth_pass', 'N/A')}
- Valuation Pass: {tc.get('valuation_pass', 'N/A')}
- Liquidity: {data_gathering_result.get('liquidity_status', 'N/A')}
- Is Actionable: {data_gathering_result.get('is_actionable', 'N/A')}
"""

    # Build prompt
    round_instruction = ""
    if debate_round == 2 and bear_argument:
        round_instruction = f"""
**ROUND 2 INSTRUCTION**: The Bear has presented their case. Your job is to REFUTE their
arguments point-by-point while strengthening your bull case. Be aggressive in your counter-arguments.

**BEAR ARGUMENT TO REFUTE**:
{bear_argument}
"""

    prompt = f"""**ANALYST INSIGHTS**:
{insights_text}
{context_text}
{round_instruction}

Generate your strongest BULL CASE using the format specified in your system prompt.
{"Focus on refuting the bear's specific points." if debate_round == 2 else "Build your initial bull case."}"""

    # Get response
    if LITELLM_AVAILABLE:
        response = await bull_agent.ainvoke(prompt)
        response_text = response
    else:
        response = await bull_agent.ainvoke({"input": prompt})
        response_text = response.content

    return {
        "agent": "bull",
        "debate_round": debate_round,
        "argument": response_text,
        "raw_response": response
    }
