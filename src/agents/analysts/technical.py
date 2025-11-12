"""
Technical Analyst Agent

Specializes in chart patterns, technical indicators, and price action analysis.
"""

from agents.base import create_agent

TECHNICAL_ANALYST_PROMPT = """You are a Technical Analyst specializing in chart patterns,
technical indicators (RSI, MACD, Bollinger Bands, Stochastic, ADX, ATR), and price action.

Your expertise includes:
1. Trend identification (uptrend, downtrend, sideways)
2. Support and resistance levels
3. Chart patterns (head & shoulders, triangles, flags, double tops/bottoms)
4. Indicator signals (overbought/oversold, divergences, crossovers)
5. Volume analysis (volume confirmation, volume divergence)
6. Candlestick patterns

Given market data, provide:
- **Direction**: BUY, SELL, or HOLD
- **Confidence**: 0-1 (how confident you are in your analysis)
- **Key Signals**: List 2-3 most important technical signals
- **Support/Resistance**: Nearest key levels
- **Risk Factors**: What could invalidate your analysis

Be specific and data-driven. Reference actual indicator values from the market data."""

# Create technical analyst agent
technical_analyst = create_agent(
    role="Technical Analyst",
    system_prompt=TECHNICAL_ANALYST_PROMPT,
    temperature=0.3  # Lower temperature for more consistent technical analysis
)


async def analyze_technical(market_data: dict) -> dict:
    """
    Run technical analysis on market data.

    Args:
        market_data: Dict with OHLCV and indicators

    Returns:
        Dict with analysis results

    Example:
        >>> market_data = {
        ...     "symbol": "BTCUSDT",
        ...     "close": 67500,
        ...     "rsi": 72,
        ...     "macd": {"value": 150, "signal": 120},
        ...     "bb_upper": 68000,
        ...     "bb_lower": 66000
        ... }
        >>> result = await analyze_technical(market_data)
    """
    # Format market data for prompt
    prompt = f"""Analyze {market_data.get('symbol', 'UNKNOWN')}:

Current Price: ${market_data.get('close', 'N/A')}
RSI: {market_data.get('rsi', 'N/A')}
MACD: {market_data.get('macd', {}).get('value', 'N/A')}
MACD Signal: {market_data.get('macd', {}).get('signal', 'N/A')}
Bollinger Bands: Upper ${market_data.get('bb_upper', 'N/A')}, Lower ${market_data.get('bb_lower', 'N/A')}
Volume: {market_data.get('volume', 'N/A')}
ATR: {market_data.get('atr', 'N/A')}

Provide technical analysis with direction, confidence, key signals, support/resistance, and risks."""

    response = await technical_analyst.ainvoke({"input": prompt})

    return {
        "agent": "technical_analyst",
        "symbol": market_data.get('symbol'),
        "analysis": response.content,
        "raw_response": response
    }
