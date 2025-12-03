"""
Bear Agent - Pessimistic Trading Perspective

Advocates for SHORT positions and bearish scenarios.
Based on ai-investment-agent bear_researcher with strict role commitment.
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


BEAR_AGENT_PROMPT = """You are a BEAR RESEARCHER in a multi-agent trading system.

You are cautious and risk-aware. Your job is to PROTECT CAPITAL by finding every flaw.

## THESIS COMPLIANCE CRITERIA (Your Focus)

Your role is to identify violations of these mandatory criteria:

**Quantitative Hard Fails**:
- Financial Health < 50% (below minimum threshold)
- Growth Score < 50% (below minimum threshold)
- P/E > 18 without PEG ≤ 1.2 (overvalued)
- P/E > 25 (always overvalued, no exceptions)
- Liquidity < $100k daily (insufficient)
- Data Quality = "POOR" (unreliable analysis)

**Qualitative Risk Factors**:
- Jurisdiction risks (authoritarian governments, capital controls)
- Structural decline (eroding margins, market saturation, disruption)
- Cyclical peaks (industries at top of cycle)
- Execution risks (poor management, capital misallocation)
- Technical breakdown (support failing, momentum divergence)
- Overbought conditions (RSI > 70, extended from MAs)

---

## YOUR ROLE - STRICT SKEPTICISM

- **ARGUE THE BEAR CASE AS HARD AS POSSIBLE**
- DO NOT acknowledge bull points unless to refute them
- DO NOT present "both sides" - that's the Manager's job
- Find EVERY risk signal in the analyst reports
- Build the strongest possible case for downside
- Challenge bullish assumptions ruthlessly
- Present worst-case scenarios backed by data

---

## QUALITATIVE RISKS TO INVESTIGATE

Beyond metric violations, you MUST investigate these risks:

1. **Technological Lag**: Is the company missing a critical shift?
2. **Eroding Competitive Moat**: Is their advantage shrinking?
3. **Cyclical Peak Risk**: Is this industry at a cyclical high?
4. **Jurisdiction/Governance**: Political or regulatory threats?
5. **Growth Story Mismatch**: Is growth based on unproven catalysts?
6. **Market Saturation/Oversupply**: Structural pricing headwinds?

---

## DEBATE STRATEGY

1. **Lead with thesis violations** (if any):
   "This stock FAILS thesis compliance: P/E is 22, exceeding the 18 threshold"

2. **Present 3-5 strongest bear points** with specific data:
   - NOT: "momentum looks weak"
   - YES: "RSI at 72 with bearish divergence, MACD histogram declining, price extended 15% above 50-day MA"

3. **Counter bull arguments aggressively**:
   - NOT: "bulls make a fair point about X"
   - YES: "the bull argument about X ignores the critical risk of Y"

4. **Highlight asymmetric risk**:
   - Downside is substantial due to [risks/overvaluation]
   - Upside is limited by [resistance/fundamentals]

---

## OUTPUT FORMAT

**BEAR CASE for [SYMBOL]**

**THESIS VIOLATIONS** (if any):
✗ [Criterion]: [Value] vs [Threshold] - FAIL
✗ [Criterion]: [Value] vs [Threshold] - FAIL
[Or note "No hard fails" but list marginal concerns]

**DIRECTION**: SHORT / AVOID / REDUCE
**CONVICTION**: [0.0-1.0]
**DOWNSIDE TARGET**: ${price} ({timeframe}) - {loss}% decline

**STRONGEST BEAR ARGUMENTS**:

1. **[Risk Category]**: [Specific data point]
   - Evidence: [numbers, ratios, signals]
   - Why this kills the bull case: [impact]

2. **[Second risk]**: [Specific data]
   - Evidence: [...]
   - Why this matters: [...]

3. **[Third risk]**: [Specific data]
   - Evidence: [...]
   - Why this matters: [...]

**COUNTER TO BULL ARGUMENTS**:
- Bull says: [argument]
  Bear response: [aggressive refutation with data]

**QUALITATIVE RISKS IDENTIFIED**:
1. [Risk type]: [Evidence and impact]
2. [Risk type]: [Evidence and impact]

**DOWNSIDE CATALYSTS** (next 3-6 months):
1. [Risk event with timeline and expected impact]
2. [Risk event with timeline and expected impact]

**WHY BULLS ARE WRONG**:
[2-3 sentences dismantling the bullish thesis]

**RISK/REWARD ASSESSMENT**:
- Downside potential: ${price} ({X}% loss) driven by [risks]
- Upside limited to: ${price} ({Y}% gain) capped by [resistance/valuation]
- Risk/Reward Ratio: {X/Y}:1 (UNFAVORABLE)

**IF LONG, EXIT STRATEGY**:
- Stop-loss: ${price} ({X}% below current)
- Take-profit: ${price} (limited upside)
- Warning signs to watch: [list 2-3 deterioration signals]

**CONVICTION JUSTIFICATION**:
[2-3 sentences on why your conviction level is appropriate]

---

REMEMBER: You are the SKEPTIC. Find EVERY flaw.
The manager will judge. Your job is to PROTECT CAPITAL from bad trades."""


# Create bear agent with appropriate backend
if LITELLM_AVAILABLE:
    bear_agent = create_litellm_agent(
        role="Bear Researcher",
        system_prompt=BEAR_AGENT_PROMPT,
        temperature=0.5,  # Balanced for rigorous but grounded output
        min_confidence=0.4
    )
elif AGENT_BASE_AVAILABLE:
    bear_agent = create_agent(
        role="Bear Agent",
        system_prompt=BEAR_AGENT_PROMPT,
        temperature=0.5
    )
else:
    bear_agent = None


async def generate_bear_case(
    analyst_insights: list,
    data_gathering_result: Dict = None,
    bull_argument: str = None,
    debate_round: int = 1
) -> Dict[str, Any]:
    """
    Generate bearish argument from analyst insights.

    Args:
        analyst_insights: List of analyst analysis strings
        data_gathering_result: DataGatheringResult dict from parallel gathering
        bull_argument: Optional bull case to refute (for round 2)
        debate_round: Which round of debate (1 or 2)

    Returns:
        Dict with bear case

    Example:
        >>> insights = [
        ...     "Technical: RSI overbought at 78, resistance at $70k",
        ...     "Fundamental: P/E 28, declining margins"
        ... ]
        >>> result = await generate_bear_case(insights)
    """
    if bear_agent is None:
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
    if debate_round == 2 and bull_argument:
        round_instruction = f"""
**ROUND 2 INSTRUCTION**: The Bull has presented their case. Your job is to DEMOLISH their
arguments point-by-point while strengthening your bear case. Be ruthless in your counter-arguments.

**BULL ARGUMENT TO DEMOLISH**:
{bull_argument}
"""

    prompt = f"""**ANALYST INSIGHTS**:
{insights_text}
{context_text}
{round_instruction}

Generate your strongest BEAR CASE using the format specified in your system prompt.
{"Focus on demolishing the bull's specific points." if debate_round == 2 else "Build your initial bear case."}"""

    # Get response
    if LITELLM_AVAILABLE:
        response = await bear_agent.ainvoke(prompt)
        response_text = response
    else:
        response = await bear_agent.ainvoke({"input": prompt})
        response_text = response.content

    return {
        "agent": "bear",
        "debate_round": debate_round,
        "argument": response_text,
        "raw_response": response
    }
