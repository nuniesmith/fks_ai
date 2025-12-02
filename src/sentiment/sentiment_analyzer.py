"""
Sentiment Analyzer for Financial Markets

Provides sentiment analysis using FinBERT (specialized for financial text) with
real-time news aggregation from CryptoPanic and NewsAPI. Implements Redis caching
to reduce API costs and improve response times.

Target Performance:
- F1 Score: >0.85 on financial sentiment classification
- Latency: <500ms with caching
- Cost: <$0.01 per sentiment query with caching

Examples:
    >>> analyzer = SentimentAnalyzer()
    >>> sentiment = analyzer.get_sentiment_from_news("BTC")
    >>> print(f"BTC Sentiment: {sentiment}")  # Output: -0.2 (bearish)
"""

import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

# Third-party imports
import requests
from transformers import pipeline
import redis

# Configure logging
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Financial sentiment analyzer using FinBERT and multiple news sources.
    
    Attributes:
        classifier: HuggingFace FinBERT pipeline for sentiment analysis
        cache: Redis client for caching sentiment scores
        cache_ttl: Time-to-live for cached entries (default: 300 seconds)
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        redis_host: str = None,
        redis_port: int = 6379,
        cache_ttl: int = 300,
    ):
        """
        Initialize sentiment analyzer with model and cache configuration.
        
        Args:
            model_name: HuggingFace model identifier (default: ProsusAI/finbert)
            redis_host: Redis server hostname (default: from env or 'redis')
            redis_port: Redis server port (default: 6379)
            cache_ttl: Cache time-to-live in seconds (default: 300 = 5 min)
        """
        self.cache_ttl = cache_ttl
        
        # Initialize FinBERT sentiment classifier
        logger.info(f"Loading sentiment model: {model_name}")
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=-1  # CPU inference (-1 = CPU, 0+ = GPU)
            )
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise
        
        # Initialize Redis cache
        redis_host = redis_host or os.getenv("REDIS_HOST", "redis")
        try:
            self.cache = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            # Test connection
            self.cache.ping()
            logger.info(f"Redis cache connected: {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"Redis cache unavailable: {e}. Operating without cache.")
            self.cache = None
    
    def get_sentiment_from_news(
        self,
        symbol: str,
        max_headlines: int = 5,
        sources: Optional[List[str]] = None
    ) -> float:
        """
        Get aggregated sentiment score from recent news headlines.
        
        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH", "AAPL")
            max_headlines: Number of recent headlines to analyze (default: 5)
            sources: List of news sources to query (default: ["cryptopanic"])
        
        Returns:
            Sentiment score from -1 (bearish) to 1 (bullish), 0 is neutral
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> sentiment = analyzer.get_sentiment_from_news("BTC", max_headlines=10)
            >>> if sentiment > 0.5:
            ...     print("Strong bullish sentiment")
        """
        # Check cache first
        cache_key = f"sentiment:{symbol}"
        if self.cache:
            try:
                cached = self.cache.get(cache_key)
                if cached:
                    logger.info(f"Cache hit for {symbol}")
                    return float(cached)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # Fetch headlines from news sources
        sources = sources or ["cryptopanic"]
        headlines = []
        
        for source in sources:
            try:
                if source == "cryptopanic":
                    headlines.extend(self._fetch_cryptopanic(symbol, max_headlines))
                elif source == "newsapi":
                    headlines.extend(self._fetch_newsapi(symbol, max_headlines))
                else:
                    logger.warning(f"Unknown news source: {source}")
            except Exception as e:
                logger.error(f"Failed to fetch from {source}: {e}")
        
        if not headlines:
            logger.warning(f"No headlines found for {symbol}")
            return 0.0
        
        # Analyze sentiment for each headline
        sentiments = []
        for headline in headlines[:max_headlines]:
            try:
                result = self.classifier(headline)[0]
                label = result['label']
                
                # Convert FinBERT labels to numeric scores
                if label == 'positive':
                    sentiments.append(1.0)
                elif label == 'negative':
                    sentiments.append(-1.0)
                else:  # neutral
                    sentiments.append(0.0)
                    
            except Exception as e:
                logger.error(f"Sentiment analysis failed for headline: {e}")
        
        # Calculate average sentiment
        if not sentiments:
            return 0.0
        
        score = sum(sentiments) / len(sentiments)
        
        # Cache the result
        if self.cache:
            try:
                self.cache.setex(cache_key, self.cache_ttl, score)
                logger.info(f"Cached sentiment for {symbol}: {score:.2f}")
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")
        
        return score
    
    def _fetch_cryptopanic(self, symbol: str, limit: int) -> List[str]:
        """
        Fetch news headlines from CryptoPanic API.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            limit: Maximum number of headlines to fetch
            
        Returns:
            List of news headlines
        """
        api_key = os.getenv("CRYPTOPANIC_API_KEY")
        if not api_key:
            logger.warning("CRYPTOPANIC_API_KEY not set, using public API (limited)")
            api_url = f"https://cryptopanic.com/api/v1/posts/?currencies={symbol}&kind=news"
        else:
            api_url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&currencies={symbol}&kind=news"
        
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            headlines = [
                post['title']
                for post in data.get('results', [])[:limit]
            ]
            
            logger.info(f"Fetched {len(headlines)} headlines from CryptoPanic for {symbol}")
            return headlines
            
        except Exception as e:
            logger.error(f"CryptoPanic API error: {e}")
            return []
    
    def _fetch_newsapi(self, symbol: str, limit: int) -> List[str]:
        """
        Fetch news headlines from NewsAPI.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of headlines to fetch
            
        Returns:
            List of news headlines
        """
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            logger.warning("NEWSAPI_KEY not set, skipping NewsAPI")
            return []
        
        # Map crypto symbols to search queries
        query_map = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum",
            # Add more mappings as needed
        }
        query = query_map.get(symbol, symbol)
        
        api_url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": api_key,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": limit,
        }
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            headlines = [
                article['title']
                for article in data.get('articles', [])[:limit]
            ]
            
            logger.info(f"Fetched {len(headlines)} headlines from NewsAPI for {symbol}")
            return headlines
            
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []
    
    def get_sentiment_score(
        self,
        text: str
    ) -> Dict[str, float]:
        """
        Get detailed sentiment score for arbitrary text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment label and confidence score
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> result = analyzer.get_sentiment_score("Bitcoin reaches new all-time high!")
            >>> print(result)  # {'label': 'positive', 'score': 0.95, 'numeric': 1.0}
        """
        try:
            result = self.classifier(text)[0]
            
            # Convert to numeric score
            if result['label'] == 'positive':
                numeric = 1.0
            elif result['label'] == 'negative':
                numeric = -1.0
            else:
                numeric = 0.0
            
            return {
                'label': result['label'],
                'confidence': result['score'],
                'numeric': numeric,
                'text': text[:100],  # Return snippet for logging
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                'label': 'neutral',
                'confidence': 0.0,
                'numeric': 0.0,
                'text': text[:100],
            }


# Convenience functions for backward compatibility and simple use cases
_global_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_from_news(symbol: str, max_headlines: int = 5) -> float:
    """
    Global convenience function to get sentiment from news.
    Uses a singleton analyzer instance.
    
    Args:
        symbol: Trading symbol (e.g., "BTC", "ETH")
        max_headlines: Number of headlines to analyze
        
    Returns:
        Sentiment score from -1 (bearish) to 1 (bullish)
    """
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = SentimentAnalyzer()
    return _global_analyzer.get_sentiment_from_news(symbol, max_headlines)


def get_sentiment_score(text: str) -> Dict[str, float]:
    """
    Global convenience function to get sentiment score for text.
    Uses a singleton analyzer instance.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment details
    """
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = SentimentAnalyzer()
    return _global_analyzer.get_sentiment_score(text)
