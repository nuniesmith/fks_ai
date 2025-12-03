"""
AI-Powered Trading Analysis Endpoint

Uses LiteLLM for intelligent market analysis and trading decisions.
Replaces the multi-agent debate system with a single efficient prompt.

Based on Agentic-Trader research showing:
- Single-agent with good prompts outperforms multi-agent debate
- 75% token reduction while maintaining quality
- Faster response times (critical for trading)
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/trading", tags=["trading-analysis"])

# Try to import LiteLLM client
try:
    from llm import llm_client, get_llm_config
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    llm_client = None
    get_llm_config = None
    logger.warning("LLM module not available for trading analysis")


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TimeHorizon(str, Enum):
    SCALP = "scalp"      # 5-30 minutes
    SWING = "swing"      # 4h-1d
    POSITION = "position" # 1d-1w


class TradingAnalysisRequest(BaseModel):
    """Request for AI trading analysis."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT, ETHUSDT)")
    current_price: float = Field(..., description="Current market price")
    
    # Technical indicators (optional but recommended)
    rsi: Optional[float] = Field(None, ge=0, le=100, description="RSI value (0-100)")
    macd: Optional[Dict[str, float]] = Field(None, description="MACD data {macd, signal, histogram}")
    ema_9: Optional[float] = Field(None, description="9-period EMA")
    ema_21: Optional[float] = Field(None, description="21-period EMA")
    ema_50: Optional[float] = Field(None, description="50-period EMA")
    volume_ratio: Optional[float] = Field(None, description="Volume vs 20-period average")
    atr: Optional[float] = Field(None, description="Average True Range for volatility")
    
    # Price action
    support_levels: Optional[List[float]] = Field(None, description="Key support levels")
    resistance_levels: Optional[List[float]] = Field(None, description="Key resistance levels")
    recent_high: Optional[float] = Field(None, description="Recent swing high")
    recent_low: Optional[float] = Field(None, description="Recent swing low")
    
    # Context
    time_horizon: TimeHorizon = Field(TimeHorizon.SWING, description="Trading time horizon")
    market_sentiment: Optional[str] = Field(None, description="Overall market sentiment (bullish/bearish/neutral)")
    news_summary: Optional[str] = Field(None, description="Recent news summary")
    
    # Risk parameters
    account_balance: Optional[float] = Field(None, description="Account balance for position sizing")
    max_risk_pct: float = Field(0.02, ge=0.005, le=0.05, description="Max risk per trade (1-5%)")


class TradingAnalysisResponse(BaseModel):
    """AI trading analysis response."""
    symbol: str
    signal: SignalType
    confidence: float = Field(..., ge=0, le=1, description="Confidence in signal (0-1)")
    
    # Entry/Exit
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Position sizing
    position_size_pct: float = Field(..., description="Recommended position size as % of account")
    risk_reward_ratio: float
    
    # Analysis
    rationale: str = Field(..., description="Reasoning behind the decision")
    key_factors: List[str] = Field(..., description="Key factors influencing decision")
    risks: List[str] = Field(..., description="Key risks to monitor")
    
    # Metadata
    model: str
    timestamp: datetime
    analysis_time_ms: int


def build_analysis_prompt(request: TradingAnalysisRequest) -> str:
    """Build a comprehensive prompt for trading analysis."""
    
    # Build technical analysis section
    tech_analysis = []
    if request.rsi is not None:
        if request.rsi < 30:
            tech_analysis.append(f"RSI: {request.rsi:.1f} (OVERSOLD)")
        elif request.rsi > 70:
            tech_analysis.append(f"RSI: {request.rsi:.1f} (OVERBOUGHT)")
        else:
            tech_analysis.append(f"RSI: {request.rsi:.1f}")
    
    if request.macd:
        macd_val = request.macd.get("macd", 0)
        signal = request.macd.get("signal", 0)
        hist = request.macd.get("histogram", macd_val - signal)
        trend = "bullish" if hist > 0 else "bearish"
        tech_analysis.append(f"MACD: {trend} (hist={hist:.4f})")
    
    if request.ema_9 and request.ema_21:
        if request.ema_9 > request.ema_21:
            tech_analysis.append(f"EMA 9/21: Bullish crossover (9={request.ema_9:.2f} > 21={request.ema_21:.2f})")
        else:
            tech_analysis.append(f"EMA 9/21: Bearish (9={request.ema_9:.2f} < 21={request.ema_21:.2f})")
    
    if request.volume_ratio:
        if request.volume_ratio > 1.5:
            tech_analysis.append(f"Volume: HIGH ({request.volume_ratio:.1f}x average)")
        elif request.volume_ratio < 0.5:
            tech_analysis.append(f"Volume: LOW ({request.volume_ratio:.1f}x average)")
        else:
            tech_analysis.append(f"Volume: Normal ({request.volume_ratio:.1f}x average)")
    
    if request.atr:
        tech_analysis.append(f"ATR (volatility): {request.atr:.4f}")
    
    # Build price levels section
    price_levels = []
    if request.support_levels:
        price_levels.append(f"Support: {', '.join([f'{s:.2f}' for s in request.support_levels[:3]])}")
    if request.resistance_levels:
        price_levels.append(f"Resistance: {', '.join([f'{r:.2f}' for r in request.resistance_levels[:3]])}")
    if request.recent_high:
        price_levels.append(f"Recent High: {request.recent_high:.2f}")
    if request.recent_low:
        price_levels.append(f"Recent Low: {request.recent_low:.2f}")
    
    prompt = f"""You are an expert cryptocurrency/forex trader analyzing {request.symbol}.

CURRENT MARKET DATA:
- Symbol: {request.symbol}
- Price: {request.current_price}
- Time Horizon: {request.time_horizon.value}

TECHNICAL ANALYSIS:
{chr(10).join(f'- {t}' for t in tech_analysis) if tech_analysis else '- No technical data provided'}

PRICE LEVELS:
{chr(10).join(f'- {p}' for p in price_levels) if price_levels else '- No price levels provided'}

{f'MARKET SENTIMENT: {request.market_sentiment}' if request.market_sentiment else ''}
{f'NEWS: {request.news_summary}' if request.news_summary else ''}

RISK PARAMETERS:
- Max Risk: {request.max_risk_pct*100:.1f}% per trade

Based on this analysis, provide a trading decision in the following JSON format:
{{
    "signal": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "entry_price": <price>,
    "stop_loss": <price>,
    "take_profit": <price>,
    "position_size_pct": 0.01 to 0.05,
    "rationale": "<brief explanation>",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risks": ["risk1", "risk2"]
}}

IMPORTANT:
- For HOLD signals, use current price for entry, and set stop_loss/take_profit to 0
- Stop loss should be within 2% of entry for scalps, 3% for swings
- Risk/reward ratio should be at least 1.5:1 for a valid trade
- Be conservative - if uncertain, signal HOLD
- Position size should reflect confidence (lower confidence = smaller size)

Respond with ONLY the JSON object, no other text."""

    return prompt


@router.post("/analyze", response_model=TradingAnalysisResponse)
async def analyze_trading_opportunity(request: TradingAnalysisRequest):
    """
    AI-powered trading analysis using LiteLLM.
    
    This endpoint provides intelligent market analysis by combining:
    - Technical indicator interpretation
    - Price action analysis
    - Risk management calculations
    - Market context awareness
    
    Returns a trading signal with entry/exit levels and position sizing.
    """
    import time
    start_time = time.time()
    
    if not LLM_AVAILABLE or llm_client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM not available for trading analysis"
        )
    
    try:
        # Build the analysis prompt
        prompt = build_analysis_prompt(request)
        
        # Get LLM response
        response = await llm_client.acomplete(
            prompt,
            system_prompt="You are an expert quantitative trader. Analyze markets and provide precise trading decisions in JSON format only.",
            temperature=0.1,  # Low temperature for consistency
            max_tokens=500
        )
        
        # Parse JSON response
        import json
        import re
        
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if not json_match:
            logger.error(f"No JSON found in LLM response: {response[:200]}")
            raise HTTPException(status_code=500, detail="Invalid LLM response format")
        
        try:
            analysis = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}, response: {response[:200]}")
            raise HTTPException(status_code=500, detail="Failed to parse LLM response")
        
        # Calculate risk/reward ratio
        signal = analysis.get("signal", "HOLD")
        entry = analysis.get("entry_price", request.current_price)
        stop = analysis.get("stop_loss", entry)
        target = analysis.get("take_profit", entry)
        
        if signal == "HOLD":
            rr_ratio = 0.0
        elif signal == "BUY":
            risk = abs(entry - stop) if stop else 0
            reward = abs(target - entry) if target else 0
            rr_ratio = reward / risk if risk > 0 else 0.0
        else:  # SELL
            risk = abs(stop - entry) if stop else 0
            reward = abs(entry - target) if target else 0
            rr_ratio = reward / risk if risk > 0 else 0.0
        
        config = get_llm_config()
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return TradingAnalysisResponse(
            symbol=request.symbol,
            signal=SignalType(signal),
            confidence=analysis.get("confidence", 0.5),
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            position_size_pct=analysis.get("position_size_pct", 0.01),
            risk_reward_ratio=round(rr_ratio, 2),
            rationale=analysis.get("rationale", "No rationale provided"),
            key_factors=analysis.get("key_factors", []),
            risks=analysis.get("risks", []),
            model=config.model,
            timestamp=datetime.now(),
            analysis_time_ms=elapsed_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trading analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/quick-signal")
async def quick_signal(
    symbol: str,
    price: float,
    rsi: Optional[float] = None,
    ema_trend: Optional[str] = None,  # "bullish" or "bearish"
    volume: Optional[str] = None  # "high", "low", "normal"
):
    """
    Quick trading signal for simple scenarios.
    
    Faster than full analysis - good for scanning multiple symbols.
    """
    if not LLM_AVAILABLE or llm_client is None:
        raise HTTPException(status_code=503, detail="LLM not available")
    
    import time
    start_time = time.time()
    
    # Build simple prompt
    context = f"Symbol: {symbol}, Price: {price}"
    if rsi:
        context += f", RSI: {rsi}"
    if ema_trend:
        context += f", EMA Trend: {ema_trend}"
    if volume:
        context += f", Volume: {volume}"
    
    prompt = f"""{context}

Quick analysis - respond with JSON only:
{{"signal": "BUY"/"SELL"/"HOLD", "confidence": 0.0-1.0, "reason": "<10 words>"}}"""
    
    try:
        response = await llm_client.acomplete(
            prompt,
            system_prompt="Expert trader. JSON only.",
            temperature=0.1,
            max_tokens=100
        )
        
        import json
        import re
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"signal": "HOLD", "confidence": 0.5, "reason": "Parse error"}
        
        config = get_llm_config()
        
        return {
            "symbol": symbol,
            "signal": result.get("signal", "HOLD"),
            "confidence": result.get("confidence", 0.5),
            "reason": result.get("reason", ""),
            "model": config.model,
            "time_ms": int((time.time() - start_time) * 1000)
        }
        
    except Exception as e:
        logger.error(f"Quick signal failed: {e}")
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0.5,
            "reason": f"Error: {str(e)}",
            "model": "error",
            "time_ms": 0
        }


@router.post("/batch-scan")
async def batch_scan(
    symbols: List[str],
    prices: Dict[str, float],
    rsi_values: Optional[Dict[str, float]] = None
):
    """
    Scan multiple symbols for trading opportunities.
    
    Returns signals sorted by confidence (highest first).
    """
    if not LLM_AVAILABLE or llm_client is None:
        raise HTTPException(status_code=503, detail="LLM not available")
    
    import asyncio
    
    async def analyze_symbol(symbol: str):
        price = prices.get(symbol, 0)
        rsi = rsi_values.get(symbol) if rsi_values else None
        
        result = await quick_signal(symbol, price, rsi)
        return result
    
    # Analyze all symbols concurrently
    results = await asyncio.gather(*[
        analyze_symbol(s) for s in symbols
    ], return_exceptions=True)
    
    # Filter out errors and sort by confidence
    valid_results = []
    for r in results:
        if isinstance(r, dict) and r.get("signal") != "HOLD":
            valid_results.append(r)
    
    valid_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    return {
        "total_scanned": len(symbols),
        "signals_found": len(valid_results),
        "results": valid_results
    }
