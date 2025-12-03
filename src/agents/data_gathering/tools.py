"""
Data Gathering Tools

HTTP tools for fetching data from fks_data service.
Used by parallel analyst agents.
"""

import os
import asyncio
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import httpx
from loguru import logger


# Configuration
FKS_DATA_URL = os.getenv("FKS_DATA_URL", "http://fks-data:8006")
FKS_DATA_TIMEOUT = int(os.getenv("FKS_DATA_TIMEOUT", "30"))


@dataclass
class DataQuality:
    """Data quality assessment."""
    status: str  # "GOOD", "MARGINAL", "POOR"
    missing_fields: List[str]
    coverage_pct: float
    
    @property
    def is_acceptable(self) -> bool:
        return self.status in ("GOOD", "MARGINAL")


class DataGatheringTools:
    """
    Tools for fetching data from fks_data service.
    
    Provides methods for each analyst type:
    - get_technical_data() - Market analyst
    - get_sentiment_data() - Sentiment analyst
    - get_news_data() - News analyst
    - get_fundamentals_data() - Fundamentals analyst
    """
    
    def __init__(self, base_url: str = None, timeout: int = None):
        self.base_url = base_url or FKS_DATA_URL
        self.timeout = timeout or FKS_DATA_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Accept": "application/json"}
            )
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def get_technical_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch technical analysis data.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "RELIANCE")
            
        Returns:
            Dict with technical indicators and price data
        """
        try:
            client = await self._get_client()
            
            # Try multiple endpoints
            endpoints = [
                f"/data/technical/{symbol}",
                f"/api/v1/technical/{symbol}",
                f"/technical/{symbol}",
            ]
            
            for endpoint in endpoints:
                try:
                    response = await client.get(
                        endpoint,
                        params={
                            "indicators": "rsi,macd,bollinger,atr,volume,sma,ema",
                            "period": "1d"
                        }
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return self._enrich_technical_data(data, symbol)
                except httpx.HTTPError:
                    continue
            
            # Fallback: return empty structure
            logger.warning(f"Technical data not available for {symbol}")
            return self._empty_technical_data(symbol)
            
        except Exception as e:
            logger.error(f"Error fetching technical data for {symbol}: {e}")
            return self._empty_technical_data(symbol)
    
    def _enrich_technical_data(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Enrich technical data with derived fields."""
        data["symbol"] = symbol
        data["data_source"] = "fks_data"
        
        # Calculate liquidity if volume and price available
        volume = data.get("volume", 0)
        price = data.get("close", data.get("last", 0))
        if volume and price:
            data["daily_volume_usd"] = volume * price
            data["liquidity_status"] = (
                "HIGH" if data["daily_volume_usd"] > 1_000_000 else
                "MEDIUM" if data["daily_volume_usd"] > 100_000 else
                "LOW"
            )
        
        return data
    
    def _empty_technical_data(self, symbol: str) -> Dict[str, Any]:
        """Return empty technical data structure."""
        return {
            "symbol": symbol,
            "data_source": "unavailable",
            "close": None,
            "rsi": None,
            "macd": None,
            "macd_signal": None,
            "bollinger_upper": None,
            "bollinger_lower": None,
            "atr": None,
            "volume": None,
            "sma_20": None,
            "sma_50": None,
            "ema_12": None,
            "ema_26": None,
            "support_levels": [],
            "resistance_levels": [],
            "trend": None,
            "daily_volume_usd": 0,
            "liquidity_status": "UNKNOWN",
        }
    
    async def get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch sentiment analysis data.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with sentiment scores and social metrics
        """
        try:
            client = await self._get_client()
            
            endpoints = [
                f"/data/sentiment/{symbol}",
                f"/api/v1/sentiment/{symbol}",
                f"/sentiment/{symbol}",
            ]
            
            for endpoint in endpoints:
                try:
                    response = await client.get(endpoint)
                    if response.status_code == 200:
                        data = response.json()
                        return self._enrich_sentiment_data(data, symbol)
                except httpx.HTTPError:
                    continue
            
            logger.warning(f"Sentiment data not available for {symbol}")
            return self._empty_sentiment_data(symbol)
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data for {symbol}: {e}")
            return self._empty_sentiment_data(symbol)
    
    def _enrich_sentiment_data(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Enrich sentiment data with derived fields."""
        data["symbol"] = symbol
        data["data_source"] = "fks_data"
        
        # Calculate undiscovered status
        social_volume = data.get("social_volume", 0)
        analyst_count = data.get("analyst_coverage", 0)
        
        if social_volume < 10 and analyst_count < 5:
            data["discovery_status"] = "UNDISCOVERED"
        elif social_volume < 50 and analyst_count < 10:
            data["discovery_status"] = "EMERGING"
        else:
            data["discovery_status"] = "WELL_KNOWN"
        
        return data
    
    def _empty_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Return empty sentiment data structure."""
        return {
            "symbol": symbol,
            "data_source": "unavailable",
            "social_score": None,
            "news_sentiment": None,
            "social_volume": 0,
            "bullish_pct": None,
            "bearish_pct": None,
            "neutral_pct": None,
            "analyst_coverage": 0,
            "retail_flow": None,
            "discovery_status": "UNKNOWN",
        }
    
    async def get_news_data(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """
        Fetch recent news data.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of articles
            
        Returns:
            Dict with news articles and summary
        """
        try:
            client = await self._get_client()
            
            endpoints = [
                f"/data/news/{symbol}",
                f"/api/v1/news/{symbol}",
                f"/news/{symbol}",
            ]
            
            for endpoint in endpoints:
                try:
                    response = await client.get(
                        endpoint,
                        params={"limit": limit}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return self._enrich_news_data(data, symbol)
                except httpx.HTTPError:
                    continue
            
            logger.warning(f"News data not available for {symbol}")
            return self._empty_news_data(symbol)
            
        except Exception as e:
            logger.error(f"Error fetching news data for {symbol}: {e}")
            return self._empty_news_data(symbol)
    
    def _enrich_news_data(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Enrich news data with derived fields."""
        data["symbol"] = symbol
        data["data_source"] = "fks_data"
        
        articles = data.get("articles", [])
        data["article_count"] = len(articles)
        
        # Simple sentiment from headlines
        positive_keywords = ["surge", "jump", "rise", "gain", "bullish", "growth", "profit"]
        negative_keywords = ["fall", "drop", "decline", "bearish", "loss", "crash", "warning"]
        
        positive_count = 0
        negative_count = 0
        
        for article in articles:
            headline = (article.get("title", "") + " " + article.get("summary", "")).lower()
            if any(kw in headline for kw in positive_keywords):
                positive_count += 1
            if any(kw in headline for kw in negative_keywords):
                negative_count += 1
        
        total = positive_count + negative_count
        if total > 0:
            data["news_sentiment_score"] = (positive_count - negative_count) / total
        else:
            data["news_sentiment_score"] = 0
        
        return data
    
    def _empty_news_data(self, symbol: str) -> Dict[str, Any]:
        """Return empty news data structure."""
        return {
            "symbol": symbol,
            "data_source": "unavailable",
            "articles": [],
            "article_count": 0,
            "news_sentiment_score": 0,
            "summary": None,
        }
    
    async def get_fundamentals_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental analysis data.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with financial metrics and scores
        """
        try:
            client = await self._get_client()
            
            endpoints = [
                f"/data/fundamentals/{symbol}",
                f"/api/v1/fundamentals/{symbol}",
                f"/fundamentals/{symbol}",
            ]
            
            for endpoint in endpoints:
                try:
                    response = await client.get(endpoint)
                    if response.status_code == 200:
                        data = response.json()
                        return self._enrich_fundamentals_data(data, symbol)
                except httpx.HTTPError:
                    continue
            
            logger.warning(f"Fundamentals data not available for {symbol}")
            return self._empty_fundamentals_data(symbol)
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals data for {symbol}: {e}")
            return self._empty_fundamentals_data(symbol)
    
    def _enrich_fundamentals_data(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Enrich fundamentals data with health and growth scores."""
        data["symbol"] = symbol
        data["data_source"] = "fks_data"
        
        # Calculate Health Score (out of 12 points)
        # Based on ai-investment-agent criteria
        health_score = 0
        
        # Profitability (4 points)
        roe = data.get("roe", 0) or 0
        if roe > 0.15:
            health_score += 2
        elif roe > 0.10:
            health_score += 1
        
        roa = data.get("roa", 0) or 0
        if roa > 0.08:
            health_score += 2
        elif roa > 0.05:
            health_score += 1
        
        # Leverage (4 points)
        debt_equity = data.get("debt_to_equity", 999) or 999
        if debt_equity < 0.5:
            health_score += 2
        elif debt_equity < 1.0:
            health_score += 1
        
        current_ratio = data.get("current_ratio", 0) or 0
        if current_ratio > 2.0:
            health_score += 2
        elif current_ratio > 1.5:
            health_score += 1
        
        # Cash Flow (4 points)
        fcf = data.get("free_cash_flow", 0) or 0
        if fcf > 0:
            health_score += 2
        
        ocf = data.get("operating_cash_flow", 0) or 0
        if ocf > 0:
            health_score += 2
        
        data["health_score"] = health_score
        data["health_score_pct"] = round((health_score / 12) * 100, 1)
        
        # Calculate Growth Score (out of 6 points)
        growth_score = 0
        
        revenue_growth = data.get("revenue_growth", 0) or 0
        if revenue_growth > 0.20:
            growth_score += 2
        elif revenue_growth > 0.10:
            growth_score += 1
        
        earnings_growth = data.get("earnings_growth", 0) or 0
        if earnings_growth > 0.25:
            growth_score += 2
        elif earnings_growth > 0.15:
            growth_score += 1
        
        gross_margin = data.get("gross_margin", 0) or 0
        if gross_margin > 0.40:
            growth_score += 2
        elif gross_margin > 0.30:
            growth_score += 1
        
        data["growth_score"] = growth_score
        data["growth_score_pct"] = round((growth_score / 6) * 100, 1)
        
        # Thesis compliance checks
        pe = data.get("pe_ratio")
        peg = data.get("peg_ratio")
        
        data["valuation_pass"] = (
            (pe is not None and pe <= 18) or
            (pe is not None and pe <= 25 and peg is not None and peg <= 1.2)
        )
        
        data["health_pass"] = data["health_score_pct"] >= 50
        data["growth_pass"] = data["growth_score_pct"] >= 50
        
        return data
    
    def _empty_fundamentals_data(self, symbol: str) -> Dict[str, Any]:
        """Return empty fundamentals data structure."""
        return {
            "symbol": symbol,
            "data_source": "unavailable",
            "pe_ratio": None,
            "peg_ratio": None,
            "roe": None,
            "roa": None,
            "debt_to_equity": None,
            "current_ratio": None,
            "revenue_growth": None,
            "earnings_growth": None,
            "gross_margin": None,
            "free_cash_flow": None,
            "operating_cash_flow": None,
            "market_cap": None,
            "analyst_coverage": None,
            "health_score": 0,
            "health_score_pct": 0,
            "growth_score": 0,
            "growth_score_pct": 0,
            "valuation_pass": False,
            "health_pass": False,
            "growth_pass": False,
        }
    
    async def gather_all_data(self, symbol: str) -> Dict[str, Any]:
        """
        Gather all data types in parallel.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with all data categories
        """
        # Run all data fetches in parallel
        technical, sentiment, news, fundamentals = await asyncio.gather(
            self.get_technical_data(symbol),
            self.get_sentiment_data(symbol),
            self.get_news_data(symbol),
            self.get_fundamentals_data(symbol),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(technical, Exception):
            logger.error(f"Technical data failed: {technical}")
            technical = self._empty_technical_data(symbol)
        if isinstance(sentiment, Exception):
            logger.error(f"Sentiment data failed: {sentiment}")
            sentiment = self._empty_sentiment_data(symbol)
        if isinstance(news, Exception):
            logger.error(f"News data failed: {news}")
            news = self._empty_news_data(symbol)
        if isinstance(fundamentals, Exception):
            logger.error(f"Fundamentals data failed: {fundamentals}")
            fundamentals = self._empty_fundamentals_data(symbol)
        
        # Assess overall data quality
        data_quality = self._assess_data_quality(technical, sentiment, news, fundamentals)
        
        return {
            "symbol": symbol,
            "technical": technical,
            "sentiment": sentiment,
            "news": news,
            "fundamentals": fundamentals,
            "data_quality": data_quality,
        }
    
    def _assess_data_quality(
        self,
        technical: Dict,
        sentiment: Dict,
        news: Dict,
        fundamentals: Dict
    ) -> DataQuality:
        """Assess overall data quality."""
        missing = []
        
        # Check critical fields
        if technical.get("data_source") == "unavailable":
            missing.append("technical")
        if fundamentals.get("data_source") == "unavailable":
            missing.append("fundamentals")
        
        # Non-critical but useful
        if sentiment.get("data_source") == "unavailable":
            missing.append("sentiment")
        if news.get("data_source") == "unavailable":
            missing.append("news")
        
        coverage = (4 - len(missing)) / 4
        
        if len(missing) == 0:
            status = "GOOD"
        elif len(missing) <= 1 and "fundamentals" not in missing:
            status = "MARGINAL"
        else:
            status = "POOR"
        
        return DataQuality(
            status=status,
            missing_fields=missing,
            coverage_pct=coverage * 100
        )


# Singleton instance
data_tools = DataGatheringTools()
