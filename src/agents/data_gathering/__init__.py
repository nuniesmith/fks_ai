"""
Parallel Data Gathering Agents

Implements 4 parallel analyst agents that gather data from fks_data service:
- Market Analyst (technical analysis)
- Sentiment Analyst (social/news sentiment)
- News Analyst (recent news/press releases)
- Fundamentals Analyst (financial metrics)

Based on ai-investment-agent architecture for parallel data gathering.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger

from .market_analyst import MarketAnalystAgent, market_analyst
from .sentiment_analyst import SentimentAnalystAgent, sentiment_analyst
from .news_analyst import NewsAnalystAgent, news_analyst
from .fundamentals_analyst import FundamentalsAnalystAgent, fundamentals_analyst
from .tools import DataGatheringTools, data_tools

__all__ = [
    # Agent classes
    "MarketAnalystAgent",
    "SentimentAnalystAgent", 
    "NewsAnalystAgent",
    "FundamentalsAnalystAgent",
    # Singleton instances
    "market_analyst",
    "sentiment_analyst",
    "news_analyst",
    "fundamentals_analyst",
    # Tools
    "DataGatheringTools",
    "data_tools",
    # Main entry point
    "parallel_gather_data",
    "DataGatheringResult",
]


class DataGatheringResult:
    """Result of parallel data gathering."""
    
    def __init__(
        self,
        symbol: str,
        trade_date: str,
        market_analysis: Dict[str, Any],
        sentiment_analysis: Dict[str, Any],
        news_analysis: Dict[str, Any],
        fundamentals_analysis: Dict[str, Any],
        raw_data: Dict[str, Any],
        errors: List[str],
        duration_ms: float,
    ):
        self.symbol = symbol
        self.trade_date = trade_date
        self.market_analysis = market_analysis
        self.sentiment_analysis = sentiment_analysis
        self.news_analysis = news_analysis
        self.fundamentals_analysis = fundamentals_analysis
        self.raw_data = raw_data
        self.errors = errors
        self.duration_ms = duration_ms
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def thesis_compliance(self) -> Dict[str, Any]:
        """Extract thesis compliance from fundamentals analysis."""
        return self.fundamentals_analysis.get("thesis_compliance", {})
    
    @property
    def liquidity_status(self) -> str:
        """Extract liquidity status from market analysis."""
        return self.market_analysis.get("data", {}).get("liquidity_status", "UNKNOWN")
    
    @property
    def is_actionable(self) -> bool:
        """Check if the analysis is actionable for trading."""
        tc = self.thesis_compliance
        return (
            tc.get("status") in ("FULL", "PARTIAL") and
            self.liquidity_status in ("HIGH", "MEDIUM") and
            not self.has_errors
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "trade_date": self.trade_date,
            "market_analysis": self.market_analysis,
            "sentiment_analysis": self.sentiment_analysis,
            "news_analysis": self.news_analysis,
            "fundamentals_analysis": self.fundamentals_analysis,
            "raw_data": self.raw_data,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "thesis_compliance": self.thesis_compliance,
            "liquidity_status": self.liquidity_status,
            "is_actionable": self.is_actionable,
        }


async def parallel_gather_data(
    symbol: str,
    trade_date: str = None,
    pre_fetch_data: bool = True,
) -> DataGatheringResult:
    """
    Run all 4 analyst agents in parallel.
    
    This is the main entry point for the data gathering phase of the
    AI Investment Agent workflow. It:
    1. Pre-fetches data from fks_data service
    2. Runs all 4 analysts in parallel with the shared data
    3. Returns consolidated results with thesis compliance
    
    Args:
        symbol: Trading symbol (e.g., "AAPL", "RELIANCE.NS")
        trade_date: Analysis date (default: today)
        pre_fetch_data: If True, fetch data first then pass to agents
                       If False, let agents fetch their own data
    
    Returns:
        DataGatheringResult with all analyst outputs
    
    Example:
        >>> result = await parallel_gather_data("AAPL")
        >>> if result.is_actionable:
        >>>     # Proceed to bull/bear debate
        >>>     pass
    """
    trade_date = trade_date or datetime.now().strftime("%Y-%m-%d")
    start_time = datetime.now()
    errors = []
    
    logger.info(f"Starting parallel data gathering for {symbol} on {trade_date}")
    
    # Phase 1: Pre-fetch data from fks_data
    raw_data = {}
    if pre_fetch_data:
        try:
            raw_data = await data_tools.gather_all_data(symbol)
            logger.info(f"Data quality for {symbol}: {raw_data.get('data_quality')}")
        except Exception as e:
            logger.error(f"Data pre-fetch failed: {e}")
            errors.append(f"data_fetch: {str(e)}")
    
    # Phase 2: Run all analysts in parallel
    analyst_tasks = [
        _run_analyst(
            market_analyst, "market", symbol,
            raw_data.get("technical") if pre_fetch_data else None
        ),
        _run_analyst(
            sentiment_analyst, "sentiment", symbol,
            raw_data.get("sentiment") if pre_fetch_data else None
        ),
        _run_analyst(
            news_analyst, "news", symbol,
            raw_data.get("news") if pre_fetch_data else None
        ),
        _run_analyst(
            fundamentals_analyst, "fundamentals", symbol,
            raw_data.get("fundamentals") if pre_fetch_data else None
        ),
    ]
    
    results = await asyncio.gather(*analyst_tasks, return_exceptions=True)
    
    # Process results
    market_result = _extract_result(results[0], "market_analyst", errors)
    sentiment_result = _extract_result(results[1], "sentiment_analyst", errors)
    news_result = _extract_result(results[2], "news_analyst", errors)
    fundamentals_result = _extract_result(results[3], "fundamentals_analyst", errors)
    
    duration_ms = (datetime.now() - start_time).total_seconds() * 1000
    logger.info(f"Parallel gathering complete for {symbol} in {duration_ms:.0f}ms")
    
    return DataGatheringResult(
        symbol=symbol,
        trade_date=trade_date,
        market_analysis=market_result,
        sentiment_analysis=sentiment_result,
        news_analysis=news_result,
        fundamentals_analysis=fundamentals_result,
        raw_data=raw_data,
        errors=errors,
        duration_ms=duration_ms,
    )


async def _run_analyst(
    analyst,
    analyst_type: str,
    symbol: str,
    provided_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """Run a single analyst with error handling."""
    try:
        return await analyst.analyze(symbol, provided_data=provided_data)
    except Exception as e:
        logger.error(f"{analyst_type} analyst failed for {symbol}: {e}")
        return {
            "agent": f"{analyst_type}_analyst",
            "symbol": symbol,
            "report": f"Error: {str(e)}",
            "data": {},
            "data_quality": "ERROR"
        }


def _extract_result(
    result: Any,
    agent_name: str,
    errors: List[str]
) -> Dict[str, Any]:
    """Extract result from asyncio.gather output."""
    if isinstance(result, Exception):
        errors.append(f"{agent_name}: {str(result)}")
        return {
            "agent": agent_name,
            "report": f"Error: {str(result)}",
            "data": {},
            "data_quality": "ERROR"
        }
    return result


# Convenience function for batch processing
async def batch_gather_data(
    symbols: List[str],
    trade_date: str = None,
    max_concurrent: int = 5,
) -> List[DataGatheringResult]:
    """
    Gather data for multiple symbols with concurrency control.
    
    Args:
        symbols: List of trading symbols
        trade_date: Analysis date
        max_concurrent: Maximum concurrent symbol processing
    
    Returns:
        List of DataGatheringResult for each symbol
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_gather(symbol: str) -> DataGatheringResult:
        async with semaphore:
            return await parallel_gather_data(symbol, trade_date)
    
    tasks = [bounded_gather(s) for s in symbols]
    return await asyncio.gather(*tasks, return_exceptions=True)
