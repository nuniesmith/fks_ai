"""
Market Analyst Agent

Pure technical analyst specializing in price action, indicators, and liquidity.
Based on ai-investment-agent market_analyst with FKS integration.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass
from loguru import logger

try:
    from agents.litellm_base import LiteLLMAgent, create_litellm_agent
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from .tools import data_tools


MARKET_ANALYST_PROMPT = """You are a PURE TECHNICAL ANALYST specializing in quantitative price analysis.

## YOUR EXCLUSIVE DOMAIN

**Market structure and price action ONLY**:
- Price trends, support/resistance, chart patterns
- Technical indicators: RSI, MACD, Bollinger Bands, moving averages
- Volume analysis and momentum
- Volatility measurements and trading ranges
- Specific entry/exit price levels
- **LIQUIDITY ASSESSMENT** (critical)

## STRICT BOUNDARIES - DO NOT:
- Analyze news events or sentiment (News/Sentiment Analyst domains)
- Discuss company fundamentals (Fundamentals Analyst domain)
- Make investment recommendations (Research Manager domain)

## OUTPUT STRUCTURE

### LIQUIDITY ASSESSMENT (Priority #1)
**Daily Volume**: $[X]M USD
**Liquidity Status**: [HIGH/MEDIUM/LOW]
**Assessment**: [PASS if >$100k, MARGINAL if $50k-100k, FAIL if <$50k]

### TREND & PRICE ACTION
**Current Trend**: [Uptrend/Downtrend/Sideways] since [timeframe]
**Price**: [Current price]
**vs MAs**: 50-day: [%], 200-day: [%]

### KEY LEVELS
**Support**: [Price levels]
**Resistance**: [Price levels]

### MOMENTUM
**RSI**: [Value] ([Overbought/Neutral/Oversold])
**MACD**: [Signal - Bullish/Bearish/Neutral]
**Bollinger**: [Position - Upper/Middle/Lower band]

### VOLUME
**Average**: [Value]
**Trend**: [Increasing/Decreasing/Stable]

### ENTRY/EXIT RECOMMENDATIONS
**Entry Approach**: [Immediate/Pullback/Scaled] at [Levels]
**Stop Loss**: [Price] ([%] below entry)
**Targets**: [Price levels with % gains]

### SUMMARY
**Liquidity**: [PASS/MARGINAL/FAIL] - $[X]M daily
**Technical Setup**: [Bullish/Neutral/Bearish]
**Entry Timing**: [Recommendation]

Confidence: [0.0-1.0]
"""


class MarketAnalystAgent:
    """
    Market Analyst Agent for technical analysis.
    
    Gathers technical data from fks_data and provides analysis.
    """
    
    def __init__(self, temperature: float = 0.3, min_confidence: float = 0.6):
        self.temperature = temperature
        self.min_confidence = min_confidence
        self._agent: Optional[LiteLLMAgent] = None
    
    @property
    def agent(self) -> LiteLLMAgent:
        """Lazy-load the LLM agent."""
        if self._agent is None:
            if LITELLM_AVAILABLE:
                self._agent = create_litellm_agent(
                    role="Market Analyst",
                    system_prompt=MARKET_ANALYST_PROMPT,
                    temperature=self.temperature,
                    min_confidence=self.min_confidence
                )
            else:
                raise RuntimeError("LiteLLM module not available")
        return self._agent
    
    async def analyze(self, symbol: str, provided_data: Dict = None) -> Dict[str, Any]:
        """
        Run technical analysis on a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "RELIANCE")
            provided_data: Optional pre-fetched technical data
            
        Returns:
            Dict with technical analysis
        """
        # Fetch data if not provided
        if provided_data is None:
            technical_data = await data_tools.get_technical_data(symbol)
        else:
            technical_data = provided_data
        
        # Format data for prompt
        prompt = self._format_prompt(symbol, technical_data)
        
        # Get analysis from LLM
        try:
            response = await self.agent.ainvoke(prompt)
            
            return {
                "agent": "market_analyst",
                "symbol": symbol,
                "report": response,
                "data": technical_data,
                "data_quality": "GOOD" if technical_data.get("data_source") != "unavailable" else "POOR"
            }
        except Exception as e:
            logger.error(f"Market analyst error for {symbol}: {e}")
            return {
                "agent": "market_analyst",
                "symbol": symbol,
                "report": f"Error: {str(e)}",
                "data": technical_data,
                "data_quality": "ERROR"
            }
    
    def _format_prompt(self, symbol: str, data: Dict) -> str:
        """Format technical data into analysis prompt."""
        
        # Handle missing data gracefully
        def safe_get(key, default="N/A"):
            val = data.get(key)
            return val if val is not None else default
        
        return f"""Analyze {symbol} using the following technical data:

## PRICE DATA
- Current Price: ${safe_get('close')}
- Daily Volume: {safe_get('volume')}
- Daily Volume USD: ${safe_get('daily_volume_usd', 0):,.0f}
- Liquidity Status: {safe_get('liquidity_status')}

## TECHNICAL INDICATORS
- RSI (14): {safe_get('rsi')}
- MACD: {safe_get('macd')}
- MACD Signal: {safe_get('macd_signal')}
- Bollinger Upper: ${safe_get('bollinger_upper')}
- Bollinger Lower: ${safe_get('bollinger_lower')}
- ATR: {safe_get('atr')}

## MOVING AVERAGES
- SMA 20: ${safe_get('sma_20')}
- SMA 50: ${safe_get('sma_50')}
- EMA 12: ${safe_get('ema_12')}
- EMA 26: ${safe_get('ema_26')}

## KEY LEVELS
- Support: {safe_get('support_levels', [])}
- Resistance: {safe_get('resistance_levels', [])}

## TREND
- Current Trend: {safe_get('trend')}

Data Source: {safe_get('data_source')}

Provide your technical analysis following the output structure in your system prompt.
Focus on LIQUIDITY first, then technical setup and entry/exit levels.
"""


# Factory function for convenience
def create_market_analyst(
    temperature: float = 0.3,
    min_confidence: float = 0.6
) -> MarketAnalystAgent:
    """Create a Market Analyst agent."""
    return MarketAnalystAgent(
        temperature=temperature,
        min_confidence=min_confidence
    )


# Singleton instance
market_analyst = MarketAnalystAgent()
