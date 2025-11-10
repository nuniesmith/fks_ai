"""
API Routes for Trading Bots

Endpoints for StockBot, ForexBot, CryptoBot, and consensus signals.
"""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime

from agents.stockbot import StockBot
from agents.forexbot import ForexBot
from agents.cryptobot import CryptoBot
from graph.consensus_node import consensus_node
from graph.bot_nodes import _prepare_market_data
from agents.state import create_initial_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/bots", tags=["trading-bots"])


# Request/Response Models
class BotSignalRequest(BaseModel):
    """Request model for bot signal generation"""
    symbol: str = Field(..., description="Trading symbol (e.g., AAPL, BTC-USD, EURUSD)")
    market_data: Dict[str, Any] = Field(..., description="Market data (OHLCV, indicators)")


class BotSignalResponse(BaseModel):
    """Response model for bot signal"""
    symbol: str
    bot: str
    signal: str
    confidence: float
    strategy: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str
    indicators: Optional[Dict[str, Any]] = None
    timestamp: datetime


class ConsensusRequest(BaseModel):
    """Request model for consensus signal"""
    symbol: str = Field(..., description="Trading symbol")
    market_data: Dict[str, Any] = Field(..., description="Market data")
    include_stock: bool = Field(True, description="Include StockBot")
    include_forex: bool = Field(True, description="Include ForexBot")
    include_crypto: bool = Field(True, description="Include CryptoBot")


class ConsensusResponse(BaseModel):
    """Response model for consensus signal"""
    symbol: str
    consensus_signal: Dict[str, Any]
    bot_signals: Dict[str, Any]
    timestamp: datetime


@router.post("/stock/signal", response_model=BotSignalResponse)
async def get_stock_signal(request: BotSignalRequest):
    """
    Get StockBot signal for a stock symbol.
    
    Args:
        request: BotSignalRequest with symbol and market_data
    
    Returns:
        BotSignalResponse with StockBot signal
    """
    try:
        # Prepare market data for bot
        bot_market_data = _prepare_market_data(request.market_data)
        
        bot = StockBot()
        signal = await bot.analyze(request.symbol, bot_market_data)
        
        if not bot.validate_signal(signal):
            raise HTTPException(status_code=400, detail="Invalid signal generated")
        
        return BotSignalResponse(
            symbol=request.symbol,
            bot="StockBot",
            signal=signal.get("signal", "HOLD"),
            confidence=signal.get("confidence", 0.0),
            strategy=bot.get_strategy_name(),
            entry_price=signal.get("entry_price"),
            stop_loss=signal.get("stop_loss"),
            take_profit=signal.get("take_profit"),
            reason=signal.get("reason", ""),
            indicators=signal.get("indicators"),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error generating StockBot signal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forex/signal", response_model=BotSignalResponse)
async def get_forex_signal(request: BotSignalRequest):
    """
    Get ForexBot signal for a forex pair.
    
    Args:
        request: BotSignalRequest with symbol and market_data
    
    Returns:
        BotSignalResponse with ForexBot signal
    """
    try:
        # Prepare market data for bot
        bot_market_data = _prepare_market_data(request.market_data)
        
        bot = ForexBot()
        signal = await bot.analyze(request.symbol, bot_market_data)
        
        if not bot.validate_signal(signal):
            raise HTTPException(status_code=400, detail="Invalid signal generated")
        
        return BotSignalResponse(
            symbol=request.symbol,
            bot="ForexBot",
            signal=signal.get("signal", "HOLD"),
            confidence=signal.get("confidence", 0.0),
            strategy=bot.get_strategy_name(),
            entry_price=signal.get("entry_price"),
            stop_loss=signal.get("stop_loss"),
            take_profit=signal.get("take_profit"),
            reason=signal.get("reason", ""),
            indicators=signal.get("indicators"),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error generating ForexBot signal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crypto/signal", response_model=BotSignalResponse)
async def get_crypto_signal(request: BotSignalRequest):
    """
    Get CryptoBot signal for a crypto symbol.
    
    Args:
        request: BotSignalRequest with symbol and market_data
    
    Returns:
        BotSignalResponse with CryptoBot signal
    """
    try:
        # Prepare market data for bot
        bot_market_data = _prepare_market_data(request.market_data)
        
        bot = CryptoBot()
        signal = await bot.analyze(request.symbol, bot_market_data)
        
        if not bot.validate_signal(signal):
            raise HTTPException(status_code=400, detail="Invalid signal generated")
        
        return BotSignalResponse(
            symbol=request.symbol,
            bot="CryptoBot",
            signal=signal.get("signal", "HOLD"),
            confidence=signal.get("confidence", 0.0),
            strategy=bot.get_strategy_name(),
            entry_price=signal.get("entry_price"),
            stop_loss=signal.get("stop_loss"),
            take_profit=signal.get("take_profit"),
            reason=signal.get("reason", ""),
            indicators=signal.get("indicators"),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error generating CryptoBot signal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consensus", response_model=ConsensusResponse)
async def get_consensus(request: ConsensusRequest):
    """
    Get consensus signal from multiple bots.
    
    Applies BTC priority rules (50-60% allocation) and aggregates signals.
    
    Args:
        request: ConsensusRequest with symbol, market_data, and bot flags
    
    Returns:
        ConsensusResponse with consensus signal and individual bot signals
    """
    try:
        # Prepare market data for bots
        bot_market_data = _prepare_market_data(request.market_data)
        
        # Create initial state
        state = create_initial_state(request.symbol, bot_market_data)
        state["signals"] = []
        
        # Run bots based on flags
        if request.include_stock:
            from graph.bot_nodes import stock_bot_node
            state = await stock_bot_node(state)
        
        if request.include_forex:
            from graph.bot_nodes import forex_bot_node
            state = await forex_bot_node(state)
        
        if request.include_crypto:
            from graph.bot_nodes import crypto_bot_node
            state = await crypto_bot_node(state)
        
        # Generate consensus
        state = await consensus_node(state)
        
        return ConsensusResponse(
            symbol=request.symbol,
            consensus_signal=state.get("consensus_signal", {}),
            bot_signals=state.get("bot_signals", {}),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error generating consensus: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def bots_health():
    """Health check for trading bots"""
    return {
        "status": "healthy",
        "bots": ["StockBot", "ForexBot", "CryptoBot"],
        "timestamp": datetime.utcnow()
    }

