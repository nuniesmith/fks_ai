"""
Sentiment Analyst Agent

Specializes in market sentiment, fear/greed index, and news analysis.
"""

from agents.base import create_agent

SENTIMENT_ANALYST_PROMPT = """You are a Sentiment Analyst specializing in market psychology,
fear/greed indicators, social media sentiment, and news impact analysis.

Your expertise includes:
1. Fear & Greed Index interpretation
2. Social media sentiment (Twitter/X, Reddit, Telegram)
3. News impact analysis (major events, regulatory changes)
4. Whale wallet movements and on-chain metrics (for crypto)
5. Institutional positioning (COT reports, ETF flows)
6. Retail vs. institutional sentiment divergence

Given sentiment data, provide:
- **Market Mood**: Extreme Fear, Fear, Neutral, Greed, Extreme Greed
- **Direction**: BUY, SELL, or HOLD based on contrarian/momentum strategy
- **Confidence**: 0-1 (reliability of sentiment signals)
- **Key Sentiment Drivers**: 2-3 most important sentiment factors
- **Contrarian Signals**: When to fade the crowd vs. follow momentum
- **News Catalysts**: Upcoming events that could shift sentiment

Use contrarian thinking: Extreme fear often signals buying opportunities, extreme greed signals caution."""

# Create sentiment analyst agent
sentiment_analyst = create_agent(
    role="Sentiment Analyst",
    system_prompt=SENTIMENT_ANALYST_PROMPT,
    temperature=0.5  # Medium temperature for balanced sentiment interpretation
)


async def analyze_sentiment(sentiment_data: dict) -> dict:
    """
    Run sentiment analysis on market psychology data.

    Args:
        sentiment_data: Dict with fear/greed index, social metrics, news

    Returns:
        Dict with analysis results

    Example:
        >>> sentiment_data = {
        ...     "symbol": "BTCUSDT",
        ...     "fear_greed_index": 25,  # Extreme Fear
        ...     "social_volume": "high",
        ...     "news_sentiment": "negative",
        ...     "whale_activity": "accumulating"
        ... }
        >>> result = await analyze_sentiment(sentiment_data)
    """
    # Format sentiment data for prompt
    prompt = f"""Analyze market sentiment for {sentiment_data.get('symbol', 'UNKNOWN')}:

Fear & Greed Index: {sentiment_data.get('fear_greed_index', 'N/A')} (0=Extreme Fear, 100=Extreme Greed)
Social Media Volume: {sentiment_data.get('social_volume', 'N/A')}
News Sentiment: {sentiment_data.get('news_sentiment', 'N/A')}
Whale Activity: {sentiment_data.get('whale_activity', 'N/A')}
Reddit Mentions: {sentiment_data.get('reddit_mentions', 'N/A')}
Twitter Sentiment: {sentiment_data.get('twitter_sentiment', 'N/A')}

Additional Context:
{sentiment_data.get('additional_context', 'None')}

Provide sentiment analysis with market mood, direction (using contrarian logic), confidence,
key drivers, contrarian signals, and upcoming catalysts."""

    response = await sentiment_analyst.ainvoke({"input": prompt})

    return {
        "agent": "sentiment_analyst",
        "symbol": sentiment_data.get('symbol'),
        "analysis": response.content,
        "raw_response": response
    }
