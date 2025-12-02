"""
Configuration for Sentiment Analysis Module

Set these environment variables in your .env file or deployment environment:

# CryptoPanic API (for crypto news)
CRYPTOPANIC_API_KEY=your_api_key_here

# NewsAPI (for general financial news)
NEWSAPI_KEY=your_api_key_here

# OpenAI (for GPT-4 multi-agent system)
OPENAI_API_KEY=your_api_key_here

# Redis Cache
REDIS_HOST=redis
REDIS_PORT=6379

# Model Configuration
SENTIMENT_MODEL=ProsusAI/finbert  # FinBERT for financial sentiment
SENTIMENT_CACHE_TTL=300  # 5 minutes cache

Usage in Python:
    from src.services.ai.src.sentiment import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer()
    sentiment = analyzer.get_sentiment_from_news("BTC")
"""

import os
from typing import Dict


class SentimentConfig:
    """Configuration class for sentiment analysis settings."""
    
    # Model settings
    MODEL_NAME = os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")
    DEVICE = int(os.getenv("SENTIMENT_DEVICE", "-1"))  # -1 for CPU, 0+ for GPU
    
    # Cache settings
    REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    CACHE_TTL = int(os.getenv("SENTIMENT_CACHE_TTL", "300"))  # 5 minutes
    
    # API keys
    CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Analysis settings
    MAX_HEADLINES = int(os.getenv("SENTIMENT_MAX_HEADLINES", "5"))
    CONFIDENCE_THRESHOLD = float(os.getenv("SENTIMENT_CONFIDENCE_THRESHOLD", "0.6"))
    
    @classmethod
    def to_dict(cls) -> Dict[str, any]:
        """Export configuration as dictionary."""
        return {
            "model_name": cls.MODEL_NAME,
            "device": cls.DEVICE,
            "redis_host": cls.REDIS_HOST,
            "redis_port": cls.REDIS_PORT,
            "cache_ttl": cls.CACHE_TTL,
            "max_headlines": cls.MAX_HEADLINES,
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
            "has_cryptopanic_key": bool(cls.CRYPTOPANIC_API_KEY),
            "has_newsapi_key": bool(cls.NEWSAPI_KEY),
            "has_openai_key": bool(cls.OPENAI_API_KEY),
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        warnings = []
        
        if not cls.CRYPTOPANIC_API_KEY:
            warnings.append("CRYPTOPANIC_API_KEY not set - using limited public API")
        
        if not cls.NEWSAPI_KEY:
            warnings.append("NEWSAPI_KEY not set - NewsAPI source disabled")
        
        if not cls.OPENAI_API_KEY:
            warnings.append("OPENAI_API_KEY not set - Multi-agent features disabled")
        
        if warnings:
            import logging
            logger = logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(warning)
        
        return True  # Non-blocking warnings


# Validate on import
SentimentConfig.validate()
