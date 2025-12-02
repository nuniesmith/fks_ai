"""
Signal Enhancement Endpoint for fks_app

Provides AI enhancement for trading signals with graceful degradation
if full AI agents are not available.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["signal-enhancement"])


def _get_current_timestamp():
    """Get current timestamp in EST/EDT timezone"""
    from utils.timezone import now_est

    return now_est()


class SignalEnhancementRequest(BaseModel):
    """Request model for signal enhancement"""

    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    base_signal: Dict[str, Any] = Field(..., description="Base signal from strategy")
    market_data: Dict[str, Any] = Field(
        ..., description="Market data (price, indicators, etc.)"
    )
    use_full_agents: bool = Field(
        default=False, description="Use full multi-agent analysis if available"
    )


class SignalEnhancementResponse(BaseModel):
    """Response model for signal enhancement"""

    enhanced: bool
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Enhanced confidence score"
    )
    analysts_output: Optional[Dict[str, str]] = None
    debate_output: Optional[Dict[str, str]] = None
    final_decision: Optional[str] = None
    enhancement_notes: Optional[str] = None
    timestamp: datetime


def enhance_signal_simple(
    base_signal: Dict[str, Any],
    market_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Simple signal enhancement without full AI agents."""
    enhanced_signal = base_signal.copy()

    # Get base confidence
    base_confidence = base_signal.get("confidence", 0.5)

    # Simple enhancement factors
    price = market_data.get("price", 0)
    volume = market_data.get("volume", 0)
    rsi = market_data.get("rsi")
    macd = market_data.get("macd")

    confidence_boost = 0.0
    enhancement_notes: list[str] = []

    # Volume confirmation
    if volume and volume > 0:
        avg_volume = market_data.get("avg_volume", volume)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        if volume_ratio > 1.3:
            confidence_boost += 0.06
            enhancement_notes.append("High volume confirmation")
        elif volume_ratio > 1.15:
            confidence_boost += 0.03
            enhancement_notes.append("Above-average volume")
        elif volume_ratio < 0.7:
            confidence_boost -= 0.06
            enhancement_notes.append("Low volume - reduced confidence")
        elif volume_ratio < 0.85:
            confidence_boost -= 0.03
            enhancement_notes.append("Below-average volume")

    # RSI confirmation
    if rsi is not None:
        signal_type = base_signal.get("signal_type", "").upper()
        if signal_type == "BUY":
            if rsi < 35:
                confidence_boost += 0.06
                enhancement_notes.append("RSI strongly supports BUY signal")
            elif rsi < 40:
                confidence_boost += 0.03
                enhancement_notes.append("RSI supports BUY signal")
            elif rsi > 75:
                confidence_boost -= 0.12
                enhancement_notes.append("RSI overbought - strong conflicting signal")
            elif rsi > 70:
                confidence_boost -= 0.06
                enhancement_notes.append("RSI overbought - conflicting signal")
        elif signal_type == "SELL":
            if rsi > 65:
                confidence_boost += 0.06
                enhancement_notes.append("RSI strongly supports SELL signal")
            elif rsi > 60:
                confidence_boost += 0.03
                enhancement_notes.append("RSI supports SELL signal")
            elif rsi < 25:
                confidence_boost -= 0.12
                enhancement_notes.append("RSI oversold - strong conflicting signal")
            elif rsi < 30:
                confidence_boost -= 0.06
                enhancement_notes.append("RSI oversold - conflicting signal")

    # MACD confirmation
    if macd and isinstance(macd, dict):
        macd_value = macd.get("macd", 0)
        signal_line = macd.get("signal", 0)
        macd_hist = macd.get("histogram", macd_value - signal_line)
        signal_type = base_signal.get("signal_type", "").upper()

        if signal_type == "BUY":
            if macd_value > signal_line and macd_hist > 0:
                if macd_hist > abs(signal_line) * 0.1:
                    confidence_boost += 0.06
                    enhancement_notes.append("MACD strong bullish crossover")
                else:
                    confidence_boost += 0.03
                    enhancement_notes.append("MACD bullish crossover")
        elif signal_type == "SELL":
            if macd_value < signal_line and macd_hist < 0:
                if abs(macd_hist) > abs(signal_line) * 0.1:
                    confidence_boost += 0.06
                    enhancement_notes.append("MACD strong bearish crossover")
                else:
                    confidence_boost += 0.03
                    enhancement_notes.append("MACD bearish crossover")

    enhanced_confidence = max(0.0, min(1.0, base_confidence + confidence_boost))

    return {
        "enhanced": True,
        "confidence": enhanced_confidence,
        "analysts_output": {
            "technical": (
                f"Simple enhancement: {', '.join(enhancement_notes) if enhancement_notes else 'Basic confirmation'}"
            ),
            "risk": "Standard risk assessment",
        },
        "debate_output": None,
        "final_decision": base_signal.get("signal_type", "HOLD"),
        "enhancement_notes": (
            "; ".join(enhancement_notes)
            if enhancement_notes
            else "Basic signal confirmation"
        ),
        "timestamp": _get_current_timestamp(),
    }


async def enhance_signal_with_agents(
    symbol: str,
    base_signal: Dict[str, Any],
    market_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Full signal enhancement using multi-agent system, with fallback."""
    try:
        from graph.trading_graph import analyze_symbol
        from agents.state import create_initial_state

        logger.info(f"Using full agent system for {symbol}")

        state = create_initial_state(symbol, market_data)
        final_state = await analyze_symbol(symbol, market_data)

        analysts_output = {
            "technical": (
                final_state.get("messages", [{}])[0].get("content", "")
                if final_state.get("messages")
                else ""
            ),
            "sentiment": (
                final_state.get("messages", [{}])[1].get("content", "")
                if len(final_state.get("messages", [])) > 1
                else ""
            ),
            "macro": (
                final_state.get("messages", [{}])[2].get("content", "")
                if len(final_state.get("messages", [])) > 2
                else ""
            ),
            "risk": (
                final_state.get("messages", [{}])[3].get("content", "")
                if len(final_state.get("messages", [])) > 3
                else ""
            ),
        }

        debate_output = {
            "bull": (
                final_state.get("debates", [""])[0]
                if final_state.get("debates")
                else ""
            ),
            "bear": (
                final_state.get("debates", ["", ""])[1]
                if len(final_state.get("debates", [])) > 1
                else ""
            ),
        }

        ai_confidence = final_state.get(
            "confidence", base_signal.get("confidence", 0.5)
        )
        base_confidence = base_signal.get("confidence", 0.5)
        enhanced_confidence = (base_confidence * 0.6 + ai_confidence * 0.4)

        return {
            "enhanced": True,
            "confidence": enhanced_confidence,
            "analysts_output": analysts_output,
            "debate_output": debate_output,
            "final_decision": final_state.get(
                "final_decision", base_signal.get("signal_type", "HOLD")
            ),
            "enhancement_notes": "Full multi-agent analysis completed",
            "timestamp": _get_current_timestamp(),
        }

    except ImportError as e:
        logger.warning(
            f"Full agent system not available: {e}. Using simple enhancement."
        )
        return enhance_signal_simple(base_signal, market_data)
    except Exception as e:
        logger.warning(f"Agent system failed: {e}. Falling back to simple enhancement.")
        return enhance_signal_simple(base_signal, market_data)


@router.post("/enhance-signal", response_model=SignalEnhancementResponse)
async def enhance_signal(request: SignalEnhancementRequest):
    """Enhance a trading signal with AI analysis."""
    try:
        logger.info(f"Enhancing signal for {request.symbol}")

        if request.use_full_agents:
            result = await enhance_signal_with_agents(
                request.symbol,
                request.base_signal,
                request.market_data,
            )
        else:
            result = enhance_signal_simple(
                request.base_signal,
                request.market_data,
            )

        return SignalEnhancementResponse(**result)

    except Exception as e:
        logger.error(
            f"Signal enhancement failed for {request.symbol}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Signal enhancement failed: {str(e)}",
        )


class AnalyzeRequest(BaseModel):
    """Request model for /ai/analyze endpoint (compatible with signal pipeline)"""

    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    market_data: Dict[str, Any] = Field(
        ..., description="Market data (price, indicators, etc.)"
    )


@router.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Analysis endpoint used by the signal pipeline."""
    try:
        logger.info(f"Analyzing {request.symbol} for signal pipeline")

        base_signal = {
            "signal_type": "HOLD",
            "confidence": 0.5,
            "entry_price": request.market_data.get("price", 0),
        }

        result = enhance_signal_simple(base_signal, request.market_data)

        return {
            "confidence": result["confidence"],
            "analysts_output": result.get("analysts_output", {}),
            "debate_output": result.get("debate_output", {}),
            "final_decision": result.get("final_decision", "HOLD"),
        }

    except Exception as e:
        logger.error(f"Analysis failed for {request.symbol}: {e}", exc_info=True)
        return {
            "confidence": 0.5,
            "analysts_output": {},
            "debate_output": {},
            "final_decision": "HOLD",
        }
