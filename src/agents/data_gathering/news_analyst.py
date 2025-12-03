"""
News Analyst Agent

Pure news analyst specializing in corporate events and news interpretation.
Based on ai-investment-agent news_analyst with FKS integration.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

try:
    from agents.litellm_base import LiteLLMAgent, create_litellm_agent
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from .tools import data_tools


NEWS_ANALYST_PROMPT = """You are a PURE NEWS ANALYST specializing in corporate events and news interpretation.

## YOUR EXCLUSIVE DOMAIN

**News and events ONLY**:
- Corporate announcements (earnings, guidance, restructuring)
- M&A activity (acquisitions, divestitures, mergers)
- Management changes (CEO, CFO, board changes)
- Regulatory filings (SEC, FDA, antitrust)
- Industry news affecting the specific company
- Macroeconomic events with direct company impact
- Competitive landscape changes
- Product launches and recalls

## STRICT BOUNDARIES - DO NOT:
- Analyze price charts or technical indicators (Market Analyst domain)
- Discuss financial metrics or valuations (Fundamentals Analyst domain)
- Interpret social sentiment (Sentiment Analyst domain)
- Make final investment recommendations (Research Manager domain)

## NEWS IMPACT FRAMEWORK

Categorize each news item by:

**Impact Level**:
- **HIGH**: Material event likely to move stock >5%
- **MEDIUM**: Meaningful event, likely 2-5% impact
- **LOW**: Minor event, <2% expected impact

**Time Horizon**:
- **IMMEDIATE**: Impact within 1-2 trading sessions
- **SHORT-TERM**: Impact over 1-4 weeks
- **LONG-TERM**: Impact over months/quarters

**Sentiment**:
- **POSITIVE**: Fundamentally bullish news
- **NEGATIVE**: Fundamentally bearish news
- **NEUTRAL**: Informational, no clear direction
- **MIXED**: Contains both positive and negative elements

## OUTPUT STRUCTURE

### NEWS SUMMARY: {Symbol}
**Analysis Period**: [Date range of news analyzed]
**News Volume**: [High/Medium/Low] - [Number] relevant articles

### HIGH IMPACT EVENTS
[For each high-impact news item]
**[Date]**: [Headline]
- Impact: HIGH | Time: [Horizon] | Sentiment: [Direction]
- Summary: [Brief summary]
- Implications: [What this means for the stock]

### MEDIUM IMPACT EVENTS
[List medium-impact news]

### UPCOMING CATALYSTS
- **Earnings Date**: [Date if known]
- **Ex-Dividend Date**: [Date if applicable]
- **Conference/Events**: [Upcoming investor events]
- **Regulatory Deadlines**: [Any pending decisions]

### RISK EVENTS
- [List any pending litigation, investigations, or threats]
- Probability: [High/Medium/Low]
- Potential Impact: [Estimate]

### NEWS-DRIVEN THESIS
**Overall News Tone**: [Positive/Negative/Neutral/Mixed]
**Key Theme**: [Primary narrative emerging from news]
**Catalyst Timeline**: [Near-term catalyst expectation]

### CORPORATE ACTIONS
- **Recent Insider Activity**: [Buys/Sells if available]
- **Buyback Status**: [Active/None/Announced]
- **Dividend Changes**: [Increased/Decreased/Unchanged/None]

### KEY OBSERVATIONS
1. [Most important news-driven insight]
2. [Secondary insight]
3. [Third insight]

### DATA_BLOCK (Structured Data)
```
NEWS_VOLUME: [Count]
HIGH_IMPACT_COUNT: [Count]
OVERALL_NEWS_TONE: [POSITIVE/NEGATIVE/NEUTRAL/MIXED]
UPCOMING_EARNINGS: [DATE or NONE]
PENDING_CATALYSTS: [TRUE/FALSE]
RISK_EVENTS_PRESENT: [TRUE/FALSE]
NEWS_RECENCY: [FRESH/STALE] (Fresh = news within 3 days)
```

Confidence: [0.0-1.0]
"""


class NewsAnalystAgent:
    """
    News Analyst Agent for corporate events and news interpretation.
    
    Gathers news from multiple sources and provides event-driven insights.
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
                    role="News Analyst",
                    system_prompt=NEWS_ANALYST_PROMPT,
                    temperature=self.temperature,
                    min_confidence=self.min_confidence
                )
            else:
                raise RuntimeError("LiteLLM module not available")
        return self._agent
    
    async def analyze(self, symbol: str, provided_data: Dict = None) -> Dict[str, Any]:
        """
        Run news analysis on a symbol.
        
        Args:
            symbol: Trading symbol
            provided_data: Optional pre-fetched news data
            
        Returns:
            Dict with news analysis
        """
        # Fetch data if not provided
        if provided_data is None:
            news_data = await data_tools.get_news_data(symbol, days=7)
        else:
            news_data = provided_data
        
        # Format data for prompt
        prompt = self._format_prompt(symbol, news_data)
        
        # Get analysis from LLM
        try:
            response = await self.agent.ainvoke(prompt)
            
            return {
                "agent": "news_analyst",
                "symbol": symbol,
                "report": response,
                "data": news_data,
                "news_summary": self._extract_news_summary(news_data),
                "data_quality": "GOOD" if news_data.get("articles") else "LIMITED"
            }
        except Exception as e:
            logger.error(f"News analyst error for {symbol}: {e}")
            return {
                "agent": "news_analyst",
                "symbol": symbol,
                "report": f"Error: {str(e)}",
                "data": news_data,
                "news_summary": {"tone": "UNKNOWN", "volume": 0},
                "data_quality": "ERROR"
            }
    
    def _format_prompt(self, symbol: str, data: Dict) -> str:
        """Format news data into analysis prompt."""
        
        articles = data.get("articles", [])
        corporate_actions = data.get("corporate_actions", {})
        upcoming_events = data.get("upcoming_events", {})
        
        # Format articles
        articles_text = ""
        if articles:
            for i, article in enumerate(articles[:15], 1):  # Limit to 15 most recent
                published = article.get("published_at", "Unknown date")
                headline = article.get("headline", article.get("title", "No headline"))
                source = article.get("source", "Unknown")
                summary = article.get("summary", article.get("description", ""))[:300]
                
                articles_text += f"""
**{i}. [{published}] {headline}**
Source: {source}
{summary}
---
"""
        else:
            articles_text = "No recent news articles found."
        
        return f"""Analyze {symbol} news using the following data:

## RECENT NEWS ARTICLES
{articles_text}

## CORPORATE ACTIONS
- Earnings Date: {upcoming_events.get('earnings_date', 'N/A')}
- Ex-Dividend Date: {upcoming_events.get('ex_dividend_date', 'N/A')}
- Last Dividend: ${corporate_actions.get('last_dividend', 'N/A')}
- Active Buyback: {corporate_actions.get('buyback_active', False)}
- Recent Insider Activity: {corporate_actions.get('insider_activity', 'N/A')}

## UPCOMING EVENTS
- Investor Day: {upcoming_events.get('investor_day', 'None scheduled')}
- Product Launch: {upcoming_events.get('product_launch', 'None known')}
- Regulatory Decision: {upcoming_events.get('regulatory_decision', 'None pending')}

## RISK INDICATORS
- Pending Litigation: {data.get('risk_indicators', {}).get('litigation', 'None known')}
- Regulatory Investigation: {data.get('risk_indicators', {}).get('investigation', 'None known')}
- Credit Watch: {data.get('risk_indicators', {}).get('credit_watch', 'N/A')}

Data Source: {data.get('data_source', 'fks_data')}
Analysis Period: Last {data.get('days_analyzed', 7)} days

Provide your news analysis following the output structure in your system prompt.
Focus on material events and upcoming catalysts that could impact the stock.
"""
    
    def _extract_news_summary(self, data: Dict) -> Dict[str, Any]:
        """Extract summary metrics from news data."""
        articles = data.get("articles", [])
        
        # Count articles and categorize
        high_impact = 0
        positive = 0
        negative = 0
        
        for article in articles:
            sentiment = article.get("sentiment", "").lower()
            impact = article.get("impact", "").lower()
            
            if impact == "high":
                high_impact += 1
            if sentiment == "positive":
                positive += 1
            elif sentiment == "negative":
                negative += 1
        
        # Determine overall tone
        if positive > negative * 1.5:
            tone = "POSITIVE"
        elif negative > positive * 1.5:
            tone = "NEGATIVE"
        elif positive > 0 or negative > 0:
            tone = "MIXED"
        else:
            tone = "NEUTRAL"
        
        # Check news recency
        if articles:
            try:
                most_recent = articles[0].get("published_at", "")
                # Simple check - if contains today's date or yesterday
                recency = "FRESH" if most_recent else "STALE"
            except:
                recency = "UNKNOWN"
        else:
            recency = "NONE"
        
        return {
            "tone": tone,
            "volume": len(articles),
            "high_impact_count": high_impact,
            "recency": recency,
            "has_upcoming_catalysts": bool(data.get("upcoming_events", {}).get("earnings_date"))
        }


# Factory function
def create_news_analyst(
    temperature: float = 0.2,
    min_confidence: float = 0.6
) -> NewsAnalystAgent:
    """Create a News Analyst agent."""
    return NewsAnalystAgent(
        temperature=temperature,
        min_confidence=min_confidence
    )


# Singleton instance
news_analyst = NewsAnalystAgent()
