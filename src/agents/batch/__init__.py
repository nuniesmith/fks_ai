"""
Batch Screening Module

JSON batch output mode for screening 100+ tickers.
Produces structured output for ranking and notification.

Based on ai-investment-agent workflow with fks_ai integration.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

__all__ = [
    "BatchScreener",
    "ScreeningResult",
    "DataQuality",
    "OpportunityRank",
    "BatchConfig",
    "run_batch_screening",
]


class DataQuality(str, Enum):
    """Data quality assessment for screening."""
    GOOD = "GOOD"
    MARGINAL = "MARGINAL"
    POOR = "POOR"


@dataclass
class OpportunityRank:
    """Ranked opportunity from batch screening."""
    rank: int
    symbol: str
    conviction_score: float  # 0-100
    health_score: float  # 0-100
    growth_score: float  # 0-100
    pe_ratio: Optional[float]
    peg_ratio: Optional[float]
    daily_liquidity: float  # USD
    data_quality: DataQuality
    position_size_aggressive: float  # % of portfolio
    position_size_conservative: float
    position_size_balanced: float
    direction: str  # "LONG" or "SHORT" or "NEUTRAL"
    target_price: Optional[float]
    upside_pct: Optional[float]
    hard_fails: List[str] = field(default_factory=list)
    bull_summary: str = ""
    bear_summary: str = ""
    manager_verdict: str = ""
    reasoning_trace_id: Optional[str] = None
    
    def passes_thesis(self) -> bool:
        """Check if opportunity passes all thesis gates."""
        return len(self.hard_fails) == 0
    
    def is_actionable(self) -> bool:
        """Check if opportunity is actionable (passes gates + good data)."""
        return self.passes_thesis() and self.data_quality != DataQuality.POOR
    
    def weighted_score(self) -> float:
        """Calculate weighted score for ranking."""
        quality_weights = {
            DataQuality.GOOD: 1.0,
            DataQuality.MARGINAL: 0.7,
            DataQuality.POOR: 0.0,
        }
        return self.conviction_score * quality_weights[self.data_quality]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "symbol": self.symbol,
            "conviction_score": self.conviction_score,
            "health_score": self.health_score,
            "growth_score": self.growth_score,
            "pe_ratio": self.pe_ratio,
            "peg_ratio": self.peg_ratio,
            "daily_liquidity": self.daily_liquidity,
            "data_quality": self.data_quality.value,
            "position_sizing": {
                "aggressive_pct": self.position_size_aggressive,
                "conservative_pct": self.position_size_conservative,
                "balanced_pct": self.position_size_balanced,
            },
            "direction": self.direction,
            "target_price": self.target_price,
            "upside_pct": self.upside_pct,
            "passes_thesis": self.passes_thesis(),
            "is_actionable": self.is_actionable(),
            "hard_fails": self.hard_fails,
            "bull_summary": self.bull_summary,
            "bear_summary": self.bear_summary,
            "manager_verdict": self.manager_verdict,
            "reasoning_trace_id": self.reasoning_trace_id,
        }
    
    def to_discord_line(self, include_emoji: bool = True) -> str:
        """Format as single Discord line for notification."""
        emoji = ""
        if include_emoji:
            if self.conviction_score >= 90:
                emoji = "⭐⭐⭐⭐⭐ "
            elif self.conviction_score >= 80:
                emoji = "⭐⭐⭐⭐ "
            elif self.conviction_score >= 70:
                emoji = "⭐⭐⭐ "
            elif self.conviction_score >= 60:
                emoji = "⭐⭐ "
            else:
                emoji = "⭐ "
        
        return (
            f"{self.rank}. **{self.symbol}** - Conviction: {self.conviction_score:.0f}/100 {emoji}\n"
            f"   Health: {self.health_score:.0f}% | Growth: {self.growth_score:.0f}% | "
            f"Liquidity: ${self.daily_liquidity/1e6:.1f}M daily\n"
            f"   Position: {self.position_size_aggressive:.1f}% (Aggressive) / "
            f"{self.position_size_balanced:.1f}% (Balanced)\n"
            f"   Data Quality: {self.data_quality.value}"
        )


@dataclass
class BatchConfig:
    """Configuration for batch screening."""
    # Concurrency settings
    max_concurrent: int = 10
    timeout_per_symbol_secs: float = 60.0
    
    # Thesis gates
    min_health_score: float = 50.0
    min_growth_score: float = 50.0
    max_pe_ratio: float = 18.0
    max_pe_with_peg: float = 25.0
    max_peg_ratio: float = 1.2
    min_liquidity_usd: float = 100_000.0
    
    # Output settings
    top_n_results: int = 10
    include_failures: bool = False
    include_reasoning_trace: bool = True
    
    # Data quality thresholds
    good_data_min_analysts: int = 5
    marginal_data_min_analysts: int = 3


@dataclass
class ScreeningResult:
    """Complete result from batch screening."""
    timestamp: str
    total_screened: int
    passed_thesis: int
    failed_thesis: int
    poor_data_quality: int
    top_opportunities: List[OpportunityRank]
    failed_opportunities: List[OpportunityRank]
    duration_secs: float
    config: BatchConfig
    errors: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_screened": self.total_screened,
                "passed_thesis": self.passed_thesis,
                "failed_thesis": self.failed_thesis,
                "poor_data_quality": self.poor_data_quality,
                "duration_secs": self.duration_secs,
            },
            "top_opportunities": [o.to_dict() for o in self.top_opportunities],
            "failed_opportunities": [o.to_dict() for o in self.failed_opportunities] if self.config.include_failures else [],
            "errors": self.errors,
        }
    
    def to_discord_message(self) -> str:
        """Format as Discord notification message."""
        date_str = datetime.now(timezone.utc).strftime("%b %d, %Y")
        
        lines = [
            f"🚨 **DAILY OPPORTUNITIES** - {date_str}",
            f"",
            f"Screened: {self.total_screened} | Passed: {self.passed_thesis} | "
            f"Failed: {self.failed_thesis} | Poor Data: {self.poor_data_quality}",
            f"",
        ]
        
        if self.top_opportunities:
            for opp in self.top_opportunities[:5]:
                lines.append(opp.to_discord_line())
                lines.append("")
        else:
            lines.append("_No opportunities meeting all criteria today._")
        
        if self.poor_data_quality > 0:
            lines.append("")
            lines.append(f"⚠️ {self.poor_data_quality} symbols skipped due to poor data quality")
        
        return "\n".join(lines)


class BatchScreener:
    """
    Batch screener for 100+ tickers with JSON output.
    
    Usage:
        screener = BatchScreener(config=BatchConfig())
        result = await screener.screen(symbols=["AAPL", "GOOGL", ...])
        print(result.to_dict())  # JSON output
        print(result.to_discord_message())  # Discord notification
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    async def screen(self, symbols: List[str]) -> ScreeningResult:
        """
        Screen multiple symbols concurrently.
        
        Args:
            symbols: List of ticker symbols to screen
            
        Returns:
            ScreeningResult with ranked opportunities
        """
        start_time = time.time()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Run screening tasks concurrently
        tasks = [self._screen_single(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        opportunities: List[OpportunityRank] = []
        errors: List[Dict[str, str]] = []
        poor_data_count = 0
        
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                errors.append({
                    "symbol": symbol,
                    "error": str(result),
                })
            elif result is None:
                # Skipped due to error
                pass
            elif result.data_quality == DataQuality.POOR:
                poor_data_count += 1
                if self.config.include_failures:
                    opportunities.append(result)
            else:
                opportunities.append(result)
        
        # Sort by weighted score (conviction × data quality weight)
        opportunities.sort(key=lambda x: x.weighted_score(), reverse=True)
        
        # Assign ranks
        for i, opp in enumerate(opportunities, 1):
            opp.rank = i
        
        # Split into passed/failed
        passed = [o for o in opportunities if o.is_actionable()]
        failed = [o for o in opportunities if not o.is_actionable()]
        
        duration = time.time() - start_time
        
        return ScreeningResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_screened=len(symbols),
            passed_thesis=len(passed),
            failed_thesis=len(failed),
            poor_data_quality=poor_data_count,
            top_opportunities=passed[:self.config.top_n_results],
            failed_opportunities=failed if self.config.include_failures else [],
            duration_secs=duration,
            config=self.config,
            errors=errors,
        )
    
    async def _screen_single(self, symbol: str) -> Optional[OpportunityRank]:
        """Screen a single symbol with semaphore limiting."""
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    self._run_full_analysis(symbol),
                    timeout=self.config.timeout_per_symbol_secs
                )
            except asyncio.TimeoutError:
                return None
            except Exception:
                return None
    
    async def _run_full_analysis(self, symbol: str) -> OpportunityRank:
        """
        Run full analysis pipeline for a single symbol.
        
        This integrates:
        1. Data gathering (parallel analysts)
        2. Bull/Bear debate
        3. Manager synthesis
        4. Risk team sizing
        """
        # Import here to avoid circular imports
        try:
            from agents.data_gathering import parallel_gather_data
            from agents.debaters import run_full_debate
            from agents.risk_team import run_risk_team
        except ImportError:
            # Fallback for standalone testing
            return await self._mock_analysis(symbol)
        
        # Phase 1: Data Gathering
        data_result = await parallel_gather_data(symbol)
        
        # Check data quality early
        data_quality = self._assess_data_quality(data_result)
        if data_quality == DataQuality.POOR:
            return OpportunityRank(
                rank=0,
                symbol=symbol,
                conviction_score=0,
                health_score=data_result.get("fundamentals", {}).get("health_score", 0),
                growth_score=data_result.get("fundamentals", {}).get("growth_score", 0),
                pe_ratio=data_result.get("fundamentals", {}).get("pe_ratio"),
                peg_ratio=data_result.get("fundamentals", {}).get("peg_ratio"),
                daily_liquidity=data_result.get("market", {}).get("daily_volume_usd", 0),
                data_quality=DataQuality.POOR,
                position_size_aggressive=0,
                position_size_conservative=0,
                position_size_balanced=0,
                direction="NEUTRAL",
                target_price=None,
                upside_pct=None,
                hard_fails=["POOR_DATA_QUALITY"],
            )
        
        # Phase 2: Bull/Bear Debate
        debate_result = await run_full_debate(symbol, data_result)
        
        # Phase 3: Risk Team Sizing
        risk_result = await run_risk_team(symbol, data_result, debate_result)
        
        # Extract hard fails from thesis gates
        hard_fails = self._check_thesis_gates(data_result)
        
        # Build final result
        fundamentals = data_result.get("fundamentals", {})
        market = data_result.get("market", {})
        manager = debate_result.get("manager", {})
        
        return OpportunityRank(
            rank=0,  # Will be set after sorting
            symbol=symbol,
            conviction_score=manager.get("conviction_score", 50),
            health_score=fundamentals.get("health_score", 0),
            growth_score=fundamentals.get("growth_score", 0),
            pe_ratio=fundamentals.get("pe_ratio"),
            peg_ratio=fundamentals.get("peg_ratio"),
            daily_liquidity=market.get("daily_volume_usd", 0),
            data_quality=data_quality,
            position_size_aggressive=risk_result.position_sizing.aggressive_pct,
            position_size_conservative=risk_result.position_sizing.conservative_pct,
            position_size_balanced=risk_result.position_sizing.balanced_pct,
            direction=manager.get("direction", "NEUTRAL"),
            target_price=manager.get("target_price"),
            upside_pct=manager.get("upside_pct"),
            hard_fails=hard_fails,
            bull_summary=debate_result.get("bull", {}).get("summary", ""),
            bear_summary=debate_result.get("bear", {}).get("summary", ""),
            manager_verdict=manager.get("verdict", ""),
            reasoning_trace_id=debate_result.get("trace_id"),
        )
    
    async def _mock_analysis(self, symbol: str) -> OpportunityRank:
        """Mock analysis for testing without full pipeline."""
        import random
        
        # Simulate some delay
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        health = random.uniform(30, 90)
        growth = random.uniform(30, 90)
        pe = random.uniform(10, 35)
        liquidity = random.uniform(50_000, 5_000_000)
        
        hard_fails = []
        if health < self.config.min_health_score:
            hard_fails.append(f"HEALTH_BELOW_{self.config.min_health_score}")
        if growth < self.config.min_growth_score:
            hard_fails.append(f"GROWTH_BELOW_{self.config.min_growth_score}")
        if pe > self.config.max_pe_ratio:
            hard_fails.append(f"PE_ABOVE_{self.config.max_pe_ratio}")
        if liquidity < self.config.min_liquidity_usd:
            hard_fails.append(f"LIQUIDITY_BELOW_{self.config.min_liquidity_usd}")
        
        # Data quality based on random factor
        quality_roll = random.random()
        if quality_roll < 0.1:
            data_quality = DataQuality.POOR
        elif quality_roll < 0.3:
            data_quality = DataQuality.MARGINAL
        else:
            data_quality = DataQuality.GOOD
        
        conviction = random.uniform(40, 95) if not hard_fails else random.uniform(20, 50)
        
        return OpportunityRank(
            rank=0,
            symbol=symbol,
            conviction_score=conviction,
            health_score=health,
            growth_score=growth,
            pe_ratio=pe,
            peg_ratio=pe / max(growth / 10, 0.1),  # Simplified PEG
            daily_liquidity=liquidity,
            data_quality=data_quality,
            position_size_aggressive=random.uniform(5, 10) if not hard_fails else 0,
            position_size_conservative=random.uniform(1, 3) if not hard_fails else 0,
            position_size_balanced=random.uniform(2, 5) if not hard_fails else 0,
            direction="LONG" if conviction > 60 else "NEUTRAL",
            target_price=random.uniform(100, 500),
            upside_pct=random.uniform(5, 30),
            hard_fails=hard_fails,
            bull_summary=f"Mock bull case for {symbol}",
            bear_summary=f"Mock bear case for {symbol}",
            manager_verdict=f"Mock verdict for {symbol}",
        )
    
    def _assess_data_quality(self, data_result: Dict[str, Any]) -> DataQuality:
        """Assess data quality from analyst coverage and data completeness."""
        fundamentals = data_result.get("fundamentals", {})
        
        analyst_count = fundamentals.get("analyst_count", 0)
        has_pe = fundamentals.get("pe_ratio") is not None
        has_revenue = fundamentals.get("revenue") is not None
        
        if analyst_count >= self.config.good_data_min_analysts and has_pe and has_revenue:
            return DataQuality.GOOD
        elif analyst_count >= self.config.marginal_data_min_analysts and (has_pe or has_revenue):
            return DataQuality.MARGINAL
        else:
            return DataQuality.POOR
    
    def _check_thesis_gates(self, data_result: Dict[str, Any]) -> List[str]:
        """Check thesis gates and return list of failures."""
        fundamentals = data_result.get("fundamentals", {})
        market = data_result.get("market", {})
        
        hard_fails = []
        
        # Health score gate
        health = fundamentals.get("health_score", 0)
        if health < self.config.min_health_score:
            hard_fails.append(f"HEALTH_{health:.0f}_BELOW_{self.config.min_health_score}")
        
        # Growth score gate
        growth = fundamentals.get("growth_score", 0)
        if growth < self.config.min_growth_score:
            hard_fails.append(f"GROWTH_{growth:.0f}_BELOW_{self.config.min_growth_score}")
        
        # P/E gate (with PEG exception)
        pe = fundamentals.get("pe_ratio")
        peg = fundamentals.get("peg_ratio")
        if pe is not None:
            if pe > self.config.max_pe_with_peg:
                hard_fails.append(f"PE_{pe:.1f}_ABOVE_MAX_{self.config.max_pe_with_peg}")
            elif pe > self.config.max_pe_ratio:
                if peg is None or peg > self.config.max_peg_ratio:
                    hard_fails.append(f"PE_{pe:.1f}_ABOVE_{self.config.max_pe_ratio}_WITHOUT_PEG")
        
        # Liquidity gate
        liquidity = market.get("daily_volume_usd", 0)
        if liquidity < self.config.min_liquidity_usd:
            hard_fails.append(f"LIQUIDITY_{liquidity/1000:.0f}K_BELOW_{self.config.min_liquidity_usd/1000:.0f}K")
        
        return hard_fails


async def run_batch_screening(
    symbols: List[str],
    config: Optional[BatchConfig] = None
) -> ScreeningResult:
    """
    Convenience function to run batch screening.
    
    Args:
        symbols: List of ticker symbols to screen
        config: Optional screening configuration
        
    Returns:
        ScreeningResult with ranked opportunities
    """
    screener = BatchScreener(config=config)
    return await screener.screen(symbols)
