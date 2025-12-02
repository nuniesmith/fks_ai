"""
Sentiment Analysis Module for FKS Trading Platform

This module provides LLM-powered sentiment analysis for crypto/stock trading signals.
Uses FinBERT for financial sentiment analysis with caching to reduce API costs.

Key Features:
- Real-time news sentiment scoring (-1 to 1 scale)
- Social media sentiment analysis
- Redis caching (5min TTL) for cost optimization
- Multi-source aggregation (CryptoPanic, NewsAPI, Twitter)
"""

from .sentiment_analyzer import (
    get_sentiment_from_news,
    get_sentiment_score,
    SentimentAnalyzer,
)

__all__ = [
    "get_sentiment_from_news",
    "get_sentiment_score",
    "SentimentAnalyzer",
]
