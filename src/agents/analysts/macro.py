"""
Macro Analyst Agent

Specializes in macroeconomic indicators, correlations, and fundamental analysis.
"""

from agents.base import create_agent

MACRO_ANALYST_PROMPT = """You are a Macro Analyst specializing in macroeconomic indicators,
central bank policy, interest rates, inflation, and cross-asset correlations.

Your expertise includes:
1. Central bank policy (Fed, ECB, BOJ) and interest rate expectations
2. Inflation metrics (CPI, PPI, PCE) and impact on asset classes
3. GDP growth, employment data, and economic cycles
4. Currency correlations (DXY impact on crypto/commodities)
5. Stock market correlations (S&P 500, Nasdaq relationships)
6. Commodity relationships (Gold, Oil, Bitcoin as digital gold)
7. Risk-on vs. risk-off regime identification

Given macro data, provide:
- **Economic Regime**: Expansion, Slowdown, Recession, Recovery
- **Direction**: BUY, SELL, or HOLD based on macro backdrop
- **Confidence**: 0-1 (strength of macro signals)
- **Key Macro Drivers**: 2-3 most important economic factors
- **Correlations**: How this asset relates to broader markets
- **Policy Impact**: Expected Fed/central bank actions and effects
- **Tail Risks**: Black swan events or regime changes to watch

Think like Ray Dalio: Consider debt cycles, inflation/deflation dynamics, and currency debasement."""

# Create macro analyst agent
macro_analyst = create_agent(
    role="Macro Analyst",
    system_prompt=MACRO_ANALYST_PROMPT,
    temperature=0.4  # Medium-low temperature for thoughtful macro analysis
)


async def analyze_macro(macro_data: dict) -> dict:
    """
    Run macroeconomic analysis on fundamentals data.

    Args:
        macro_data: Dict with economic indicators, rates, correlations

    Returns:
        Dict with analysis results

    Example:
        >>> macro_data = {
        ...     "symbol": "BTCUSDT",
        ...     "cpi_yoy": 3.2,
        ...     "fed_funds_rate": 5.25,
        ...     "dxy": 104.5,
        ...     "spx_correlation": -0.65,
        ...     "gold_correlation": 0.72
        ... }
        >>> result = await analyze_macro(macro_data)
    """
    # Format macro data for prompt
    prompt = f"""Analyze macroeconomic backdrop for {macro_data.get('symbol', 'UNKNOWN')}:

Inflation:
- CPI (YoY): {macro_data.get('cpi_yoy', 'N/A')}%
- Core CPI: {macro_data.get('core_cpi', 'N/A')}%
- PPI: {macro_data.get('ppi', 'N/A')}%

Interest Rates:
- Fed Funds Rate: {macro_data.get('fed_funds_rate', 'N/A')}%
- Expected Rate Change: {macro_data.get('rate_change_expectation', 'N/A')}

Currency & Correlations:
- DXY (Dollar Index): {macro_data.get('dxy', 'N/A')}
- S&P 500 Correlation: {macro_data.get('spx_correlation', 'N/A')}
- Gold Correlation: {macro_data.get('gold_correlation', 'N/A')}

Economic Data:
- GDP Growth: {macro_data.get('gdp_growth', 'N/A')}%
- Unemployment: {macro_data.get('unemployment', 'N/A')}%
- Leading Economic Index: {macro_data.get('lei', 'N/A')}

Additional Context:
{macro_data.get('additional_context', 'None')}

Provide macro analysis with economic regime, direction, confidence, key drivers,
correlations, policy impact, and tail risks."""

    response = await macro_analyst.ainvoke({"input": prompt})

    return {
        "agent": "macro_analyst",
        "symbol": macro_data.get('symbol'),
        "analysis": response.content,
        "raw_response": response
    }
