"""
Fundamentals Analyst Agent

Pure fundamentals analyst specializing in financial metrics, health/growth scoring.
Based on ai-investment-agent fundamentals_analyst with FKS integration.
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


FUNDAMENTALS_ANALYST_PROMPT = """You are a PURE FUNDAMENTALS ANALYST specializing in financial metrics and valuation.

## YOUR EXCLUSIVE DOMAIN

**Financial analysis ONLY**:
- Valuation ratios (P/E, P/B, PEG, EV/EBITDA)
- Profitability metrics (ROE, ROA, margins)
- Financial health (debt/equity, current ratio, cash flow)
- Growth metrics (revenue, earnings, margin trends)
- Analyst coverage count
- **HEALTH SCORE** calculation (0-12 points)
- **GROWTH SCORE** calculation (0-6 points)

## STRICT BOUNDARIES - DO NOT:
- Analyze price charts or technical indicators (Market Analyst domain)
- Discuss social sentiment (Sentiment Analyst domain)
- Interpret news events (News Analyst domain)
- Make final investment recommendations (Research Manager domain)

## THESIS COMPLIANCE CRITERIA

**Hard Thresholds**:
- Health Score ≥ 50% (6/12 points minimum)
- Growth Score ≥ 50% (3/6 points minimum)
- P/E ≤ 18 OR (P/E 18-25 with PEG ≤ 1.2)
- Liquidity > $100k daily (Market Analyst reports this)

## OUTPUT STRUCTURE

### VALUATION METRICS
**P/E Ratio**: [Value] ([Below/At/Above] thesis threshold of 18)
**PEG Ratio**: [Value] (Target: ≤1.2 if P/E > 18)
**P/B Ratio**: [Value]
**EV/EBITDA**: [Value]

### HEALTH SCORE: [X]/12 ([Y]%)
Profitability (0-4):
- ROE: [Value] → [Points]
- ROA: [Value] → [Points]

Leverage (0-4):
- Debt/Equity: [Value] → [Points]
- Current Ratio: [Value] → [Points]

Cash Flow (0-4):
- Free Cash Flow: [Positive/Negative] → [Points]
- Operating Cash Flow: [Positive/Negative] → [Points]

**Health Status**: [PASS if ≥50%, FAIL if <50%]

### GROWTH SCORE: [X]/6 ([Y]%)
- Revenue Growth: [Value] → [Points]
- Earnings Growth: [Value] → [Points]
- Gross Margin: [Value] → [Points]

**Growth Status**: [PASS if ≥50%, FAIL if <50%]

### ANALYST COVERAGE
**Count**: [Number] analysts
**Status**: [UNDISCOVERED if <5, EMERGING if 5-10, WELL-KNOWN if >10]

### THESIS COMPLIANCE
✓/✗ Health Score: [Pass/Fail] ([X]% vs 50% threshold)
✓/✗ Growth Score: [Pass/Fail] ([X]% vs 50% threshold)
✓/✗ Valuation: [Pass/Fail] (P/E=[X], PEG=[Y])

### DATA_BLOCK (Structured Data)
```
HEALTH_SCORE: [X]/12
HEALTH_PCT: [Y]%
GROWTH_SCORE: [X]/6
GROWTH_PCT: [Y]%
PE_RATIO: [Value]
PEG_RATIO: [Value]
VALUATION_PASS: [TRUE/FALSE]
HEALTH_PASS: [TRUE/FALSE]
GROWTH_PASS: [TRUE/FALSE]
THESIS_COMPLIANCE: [FULL/PARTIAL/FAIL]
```

Confidence: [0.0-1.0]
"""


class FundamentalsAnalystAgent:
    """
    Fundamentals Analyst Agent for financial metrics analysis.
    
    Gathers fundamentals data from fks_data and provides health/growth scoring.
    """
    
    def __init__(self, temperature: float = 0.2, min_confidence: float = 0.6):
        self.temperature = temperature
        self.min_confidence = min_confidence
        self._agent: Optional[LiteLLMAgent] = None
    
    @property
    def agent(self) -> LiteLLMAgent:
        """Lazy-load the LLM agent."""
        if self._agent is None:
            if LITELLM_AVAILABLE:
                self._agent = create_litellm_agent(
                    role="Fundamentals Analyst",
                    system_prompt=FUNDAMENTALS_ANALYST_PROMPT,
                    temperature=self.temperature,
                    min_confidence=self.min_confidence
                )
            else:
                raise RuntimeError("LiteLLM module not available")
        return self._agent
    
    async def analyze(self, symbol: str, provided_data: Dict = None) -> Dict[str, Any]:
        """
        Run fundamental analysis on a symbol.
        
        Args:
            symbol: Trading symbol
            provided_data: Optional pre-fetched fundamentals data
            
        Returns:
            Dict with fundamental analysis
        """
        # Fetch data if not provided
        if provided_data is None:
            fundamentals_data = await data_tools.get_fundamentals_data(symbol)
        else:
            fundamentals_data = provided_data
        
        # Format data for prompt
        prompt = self._format_prompt(symbol, fundamentals_data)
        
        # Get analysis from LLM
        try:
            response = await self.agent.ainvoke(prompt)
            
            # Extract structured data from response
            thesis_compliance = self._extract_thesis_compliance(fundamentals_data)
            
            return {
                "agent": "fundamentals_analyst",
                "symbol": symbol,
                "report": response,
                "data": fundamentals_data,
                "thesis_compliance": thesis_compliance,
                "data_quality": "GOOD" if fundamentals_data.get("data_source") != "unavailable" else "POOR"
            }
        except Exception as e:
            logger.error(f"Fundamentals analyst error for {symbol}: {e}")
            return {
                "agent": "fundamentals_analyst",
                "symbol": symbol,
                "report": f"Error: {str(e)}",
                "data": fundamentals_data,
                "thesis_compliance": {"status": "ERROR"},
                "data_quality": "ERROR"
            }
    
    def _format_prompt(self, symbol: str, data: Dict) -> str:
        """Format fundamentals data into analysis prompt."""
        
        def safe_get(key, default="N/A"):
            val = data.get(key)
            return val if val is not None else default
        
        def safe_pct(key, default="N/A"):
            val = data.get(key)
            if val is not None:
                return f"{val * 100:.1f}%"
            return default
        
        return f"""Analyze {symbol} fundamentals using the following data:

## VALUATION
- P/E Ratio: {safe_get('pe_ratio')}
- PEG Ratio: {safe_get('peg_ratio')}
- Price/Book: {safe_get('price_to_book')}
- Market Cap: ${safe_get('market_cap', 0):,.0f}

## PROFITABILITY
- ROE: {safe_pct('roe')}
- ROA: {safe_pct('roa')}
- Gross Margin: {safe_pct('gross_margin')}
- Operating Margin: {safe_pct('operating_margin')}

## FINANCIAL HEALTH
- Debt/Equity: {safe_get('debt_to_equity')}
- Current Ratio: {safe_get('current_ratio')}
- Total Cash: ${safe_get('total_cash', 0):,.0f}
- Total Debt: ${safe_get('total_debt', 0):,.0f}

## CASH FLOW
- Operating Cash Flow: ${safe_get('operating_cash_flow', 0):,.0f}
- Free Cash Flow: ${safe_get('free_cash_flow', 0):,.0f}

## GROWTH
- Revenue Growth (YoY): {safe_pct('revenue_growth')}
- Earnings Growth: {safe_pct('earnings_growth')}

## COVERAGE
- Analyst Coverage: {safe_get('analyst_coverage', 0)} analysts

## PRE-CALCULATED SCORES
- Health Score: {safe_get('health_score', 0)}/12 ({safe_get('health_score_pct', 0)}%)
- Growth Score: {safe_get('growth_score', 0)}/6 ({safe_get('growth_score_pct', 0)}%)
- Valuation Pass: {safe_get('valuation_pass', False)}
- Health Pass: {safe_get('health_pass', False)}
- Growth Pass: {safe_get('growth_pass', False)}

Data Source: {safe_get('data_source')}

Provide your fundamental analysis following the output structure in your system prompt.
Calculate and verify the HEALTH and GROWTH scores, and assess THESIS COMPLIANCE.
"""
    
    def _extract_thesis_compliance(self, data: Dict) -> Dict[str, Any]:
        """Extract thesis compliance from data."""
        health_pass = data.get("health_pass", False)
        growth_pass = data.get("growth_pass", False)
        valuation_pass = data.get("valuation_pass", False)
        
        if health_pass and growth_pass and valuation_pass:
            status = "FULL"
        elif (health_pass and growth_pass) or (health_pass and valuation_pass):
            status = "PARTIAL"
        else:
            status = "FAIL"
        
        return {
            "status": status,
            "health_pass": health_pass,
            "growth_pass": growth_pass,
            "valuation_pass": valuation_pass,
            "health_score_pct": data.get("health_score_pct", 0),
            "growth_score_pct": data.get("growth_score_pct", 0),
        }


# Factory function
def create_fundamentals_analyst(
    temperature: float = 0.2,
    min_confidence: float = 0.6
) -> FundamentalsAnalystAgent:
    """Create a Fundamentals Analyst agent."""
    return FundamentalsAnalystAgent(
        temperature=temperature,
        min_confidence=min_confidence
    )


# Singleton instance
fundamentals_analyst = FundamentalsAnalystAgent()
