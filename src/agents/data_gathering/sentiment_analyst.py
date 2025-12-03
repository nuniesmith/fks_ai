"""
Sentiment Analyst Agent

Pure sentiment analyst specializing in social media and behavioral analysis.
Based on ai-investment-agent sentiment_analyst with FKS integration.
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


SENTIMENT_ANALYST_PROMPT = """You are a PURE SENTIMENT ANALYST specializing in social and behavioral signals.

## YOUR EXCLUSIVE DOMAIN

**Social/behavioral analysis ONLY**:
- Social media sentiment (Twitter/X, Reddit, StockTwits)
- Retail investor positioning and chatter
- Sentiment trends and momentum
- Community engagement metrics
- Institutional vs retail sentiment divergence
- Options flow sentiment (put/call ratios if available)

## STRICT BOUNDARIES - DO NOT:
- Analyze price charts or technical indicators (Market Analyst domain)
- Discuss financial statements (Fundamentals Analyst domain)
- Interpret corporate news or events (News Analyst domain)
- Make final investment recommendations (Research Manager domain)

## SENTIMENT SCORING METHODOLOGY

Score social sentiment on a scale of -100 to +100:
- **+75 to +100**: Euphoric - Extreme bullish sentiment, potential top
- **+50 to +74**: Very Bullish - Strong positive sentiment
- **+25 to +49**: Bullish - Moderately positive
- **0 to +24**: Slightly Bullish - Cautiously optimistic
- **-24 to 0**: Slightly Bearish - Cautiously pessimistic
- **-49 to -25**: Bearish - Moderately negative
- **-74 to -50**: Very Bearish - Strong negative sentiment
- **-100 to -75**: Capitulation - Extreme fear, potential bottom

## OUTPUT STRUCTURE

### OVERALL SENTIMENT: [Sentiment Level] ([Score]/100)

### SOCIAL MEDIA BREAKDOWN

**StockTwits** (if available):
- Sentiment: [Bullish/Bearish/Neutral]
- Message Volume: [High/Medium/Low]
- Watchlist Count: [Number]
- Trending Status: [Yes/No]

**Reddit** (r/stocks, r/investing, r/wallstreetbets):
- Mention Frequency: [High/Medium/Low/None]
- Discussion Tone: [Bullish/Bearish/Neutral/Mixed]
- Notable Threads: [Yes/No]

**Twitter/X**:
- Activity Level: [High/Medium/Low]
- Influencer Mentions: [Yes/No]
- Sentiment Trend: [Improving/Stable/Declining]

### BEHAVIORAL INDICATORS

**Retail Positioning**:
- Interest Level: [High/Medium/Low]
- FOMO Signals: [Present/Absent]
- Fear Indicators: [Present/Absent]

**Sentiment Momentum**:
- 7-Day Trend: [Improving/Stable/Declining]
- Volume vs Sentiment Correlation: [Aligned/Divergent]

**Contrarian Signals**:
- Extreme Readings: [Overbought/Oversold/Normal]
- Potential Reversal Setup: [Yes/No]

### KEY OBSERVATIONS
1. [Primary sentiment insight]
2. [Secondary sentiment insight]
3. [Behavioral pattern or anomaly]

### SENTIMENT RISK ASSESSMENT
- **Crowded Trade Risk**: [High/Medium/Low] - How crowded is the sentiment?
- **Sentiment Reversal Risk**: [High/Medium/Low] - How likely is a sentiment flip?
- **Information Asymmetry**: [High/Medium/Low] - Are insiders/institutions diverging from retail?

### DATA_BLOCK (Structured Data)
```
OVERALL_SENTIMENT_SCORE: [Score]
SENTIMENT_LABEL: [Label]
STOCKTWITS_BULLISH_PCT: [Pct]
REDDIT_MENTIONS: [Count]
TWITTER_SENTIMENT: [Score]
VOLUME_SENTIMENT_ALIGNED: [TRUE/FALSE]
CROWDED_TRADE_RISK: [HIGH/MEDIUM/LOW]
REVERSAL_RISK: [HIGH/MEDIUM/LOW]
```

Confidence: [0.0-1.0]
"""


class SentimentAnalystAgent:
    """
    Sentiment Analyst Agent for social and behavioral analysis.
    
    Gathers sentiment data from social APIs and provides behavioral insights.
    """
    
    def __init__(self, temperature: float = 0.3, min_confidence: float = 0.5):
        self.temperature = temperature
        self.min_confidence = min_confidence
        self._agent: Optional[LiteLLMAgent] = None
    
    @property
    def agent(self) -> LiteLLMAgent:
        """Lazy-load the LLM agent."""
        if self._agent is None:
            if LITELLM_AVAILABLE:
                self._agent = create_litellm_agent(
                    role="Sentiment Analyst",
                    system_prompt=SENTIMENT_ANALYST_PROMPT,
                    temperature=self.temperature,
                    min_confidence=self.min_confidence
                )
            else:
                raise RuntimeError("LiteLLM module not available")
        return self._agent
    
    async def analyze(self, symbol: str, provided_data: Dict = None) -> Dict[str, Any]:
        """
        Run sentiment analysis on a symbol.
        
        Args:
            symbol: Trading symbol
            provided_data: Optional pre-fetched sentiment data
            
        Returns:
            Dict with sentiment analysis
        """
        # Fetch data if not provided
        if provided_data is None:
            sentiment_data = await data_tools.get_sentiment_data(symbol)
        else:
            sentiment_data = provided_data
        
        # Format data for prompt
        prompt = self._format_prompt(symbol, sentiment_data)
        
        # Get analysis from LLM
        try:
            response = await self.agent.ainvoke(prompt)
            
            return {
                "agent": "sentiment_analyst",
                "symbol": symbol,
                "report": response,
                "data": sentiment_data,
                "overall_sentiment": self._calculate_overall_sentiment(sentiment_data),
                "data_quality": "GOOD" if sentiment_data.get("data_source") != "unavailable" else "LIMITED"
            }
        except Exception as e:
            logger.error(f"Sentiment analyst error for {symbol}: {e}")
            return {
                "agent": "sentiment_analyst",
                "symbol": symbol,
                "report": f"Error: {str(e)}",
                "data": sentiment_data,
                "overall_sentiment": {"score": 0, "label": "UNKNOWN"},
                "data_quality": "ERROR"
            }
    
    def _format_prompt(self, symbol: str, data: Dict) -> str:
        """Format sentiment data into analysis prompt."""
        
        stocktwits = data.get("stocktwits", {})
        reddit = data.get("reddit", {})
        twitter = data.get("twitter", {})
        
        return f"""Analyze {symbol} sentiment using the following social data:

## STOCKTWITS DATA
- Sentiment: {stocktwits.get('sentiment', 'N/A')}
- Bullish %: {stocktwits.get('bullish_pct', 'N/A')}
- Message Volume: {stocktwits.get('message_volume', 'N/A')}
- Watchlist Count: {stocktwits.get('watchlist_count', 'N/A')}
- Trending: {stocktwits.get('trending', False)}

## REDDIT DATA
- Mentions (7d): {reddit.get('mention_count', 'N/A')}
- Overall Tone: {reddit.get('overall_tone', 'N/A')}
- Active Subreddits: {reddit.get('active_subreddits', [])}
- Top Comment Sentiment: {reddit.get('top_comment_sentiment', 'N/A')}

## TWITTER/X DATA
- Volume (24h): {twitter.get('volume_24h', 'N/A')}
- Sentiment Score: {twitter.get('sentiment_score', 'N/A')}
- Trending Topics: {twitter.get('trending', False)}
- Influencer Activity: {twitter.get('influencer_mentions', 'N/A')}

## AGGREGATE METRICS
- Combined Sentiment Score: {data.get('aggregate_score', 'N/A')}
- 7-Day Sentiment Trend: {data.get('trend_7d', 'N/A')}
- Retail Interest Level: {data.get('retail_interest', 'N/A')}

Data Source: {data.get('data_source', 'fks_data')}
Coverage Quality: {data.get('coverage_quality', 'standard')}

Provide your sentiment analysis following the output structure in your system prompt.
Focus on actionable behavioral insights and sentiment-based risk assessment.
"""
    
    def _calculate_overall_sentiment(self, data: Dict) -> Dict[str, Any]:
        """Calculate overall sentiment from components."""
        scores = []
        
        # StockTwits
        st_bullish = data.get("stocktwits", {}).get("bullish_pct")
        if st_bullish is not None:
            scores.append((st_bullish - 50) * 2)  # Convert 0-100% to -100 to +100
        
        # Twitter
        twitter_score = data.get("twitter", {}).get("sentiment_score")
        if twitter_score is not None:
            scores.append(twitter_score)
        
        # Reddit (qualitative to quantitative)
        reddit_tone = data.get("reddit", {}).get("overall_tone", "").lower()
        if reddit_tone == "bullish":
            scores.append(50)
        elif reddit_tone == "bearish":
            scores.append(-50)
        elif reddit_tone == "neutral":
            scores.append(0)
        
        # Aggregate
        if data.get("aggregate_score") is not None:
            scores.append(data["aggregate_score"])
        
        if not scores:
            return {"score": 0, "label": "UNKNOWN"}
        
        avg_score = sum(scores) / len(scores)
        
        # Determine label
        if avg_score >= 75:
            label = "EUPHORIC"
        elif avg_score >= 50:
            label = "VERY_BULLISH"
        elif avg_score >= 25:
            label = "BULLISH"
        elif avg_score >= 0:
            label = "SLIGHTLY_BULLISH"
        elif avg_score >= -25:
            label = "SLIGHTLY_BEARISH"
        elif avg_score >= -50:
            label = "BEARISH"
        elif avg_score >= -75:
            label = "VERY_BEARISH"
        else:
            label = "CAPITULATION"
        
        return {"score": round(avg_score, 1), "label": label}


# Factory function
def create_sentiment_analyst(
    temperature: float = 0.3,
    min_confidence: float = 0.5
) -> SentimentAnalystAgent:
    """Create a Sentiment Analyst agent."""
    return SentimentAnalystAgent(
        temperature=temperature,
        min_confidence=min_confidence
    )


# Singleton instance
sentiment_analyst = SentimentAnalystAgent()
