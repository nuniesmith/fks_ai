"""
Bot Nodes for Multi-Agent Trading System

LangGraph nodes for StockBot, ForexBot, and CryptoBot integration.
"""

import asyncio
from typing import Dict, Any
from loguru import logger

from agents.state import AgentState
from agents.stockbot import StockBot
from agents.forexbot import ForexBot
from agents.cryptobot import CryptoBot


async def stock_bot_node(state: AgentState) -> AgentState:
    """
    StockBot node for stock market analysis.
    
    Args:
        state: Current agent state with market_data and symbol
    
    Returns:
        Updated state with StockBot signal
    """
    try:
        symbol = state.get("symbol", "")
        market_data = state.get("market_data", {})
        
        # Check if symbol is a stock (basic heuristic)
        if not _is_stock_symbol(symbol):
            logger.debug(f"Skipping StockBot for non-stock symbol: {symbol}")
            if "signals" not in state:
                state["signals"] = []
            return state
        
        # Prepare market data for bot (bots expect {"data": [...]} format)
        # If market_data doesn't have "data" key, create it from available data
        bot_market_data = _prepare_market_data(market_data)
        
        # Initialize StockBot
        bot = StockBot()
        
        # Analyze with StockBot
        signal = await bot.analyze(symbol, bot_market_data)
        
        # Validate signal
        if bot.validate_signal(signal):
            # Add bot metadata
            signal["bot"] = "StockBot"
            signal["strategy"] = bot.get_strategy_name()
            
            # Add to signals list
            if "signals" not in state:
                state["signals"] = []
            state["signals"].append(signal)
            
            logger.info(f"StockBot signal for {symbol}: {signal.get('signal')} (confidence: {signal.get('confidence', 0):.2f})")
        else:
            logger.warning(f"Invalid StockBot signal for {symbol}: {signal}")
        
        return state
    except Exception as e:
        logger.error(f"Error in StockBot node: {e}", exc_info=True)
        # Return state unchanged on error
        return state


async def forex_bot_node(state: AgentState) -> AgentState:
    """
    ForexBot node for forex market analysis.
    
    Args:
        state: Current agent state with market_data and symbol
    
    Returns:
        Updated state with ForexBot signal
    """
    try:
        symbol = state.get("symbol", "")
        market_data = state.get("market_data", {})
        
        # Check if symbol is forex (basic heuristic)
        if not _is_forex_symbol(symbol):
            logger.debug(f"Skipping ForexBot for non-forex symbol: {symbol}")
            if "signals" not in state:
                state["signals"] = []
            return state
        
        # Prepare market data for bot
        bot_market_data = _prepare_market_data(market_data)
        
        # Initialize ForexBot
        bot = ForexBot()
        
        # Analyze with ForexBot
        signal = await bot.analyze(symbol, bot_market_data)
        
        # Validate signal
        if bot.validate_signal(signal):
            # Add bot metadata
            signal["bot"] = "ForexBot"
            signal["strategy"] = bot.get_strategy_name()
            
            # Add to signals list
            if "signals" not in state:
                state["signals"] = []
            state["signals"].append(signal)
            
            logger.info(f"ForexBot signal for {symbol}: {signal.get('signal')} (confidence: {signal.get('confidence', 0):.2f})")
        else:
            logger.warning(f"Invalid ForexBot signal for {symbol}: {signal}")
        
        return state
    except Exception as e:
        logger.error(f"Error in ForexBot node: {e}", exc_info=True)
        # Return state unchanged on error
        return state


async def crypto_bot_node(state: AgentState) -> AgentState:
    """
    CryptoBot node for crypto market analysis.
    
    Args:
        state: Current agent state with market_data and symbol
    
    Returns:
        Updated state with CryptoBot signal
    """
    try:
        symbol = state.get("symbol", "")
        market_data = state.get("market_data", {})
        
        # Check if symbol is crypto (basic heuristic)
        if not _is_crypto_symbol(symbol):
            logger.debug(f"Skipping CryptoBot for non-crypto symbol: {symbol}")
            if "signals" not in state:
                state["signals"] = []
            return state
        
        # Prepare market data for bot
        bot_market_data = _prepare_market_data(market_data)
        
        # Initialize CryptoBot
        bot = CryptoBot()
        
        # Analyze with CryptoBot
        signal = await bot.analyze(symbol, bot_market_data)
        
        # Validate signal
        if bot.validate_signal(signal):
            # Add bot metadata
            signal["bot"] = "CryptoBot"
            signal["strategy"] = bot.get_strategy_name()
            signal["btc_priority"] = bot.is_btc(symbol)
            
            # Add to signals list
            if "signals" not in state:
                state["signals"] = []
            state["signals"].append(signal)
            
            logger.info(f"CryptoBot signal for {symbol}: {signal.get('signal')} (confidence: {signal.get('confidence', 0):.2f})")
        else:
            logger.warning(f"Invalid CryptoBot signal for {symbol}: {signal}")
        
        return state
    except Exception as e:
        logger.error(f"Error in CryptoBot node: {e}", exc_info=True)
        # Return state unchanged on error
        return state


async def run_bots_parallel(state: AgentState) -> AgentState:
    """
    Run all trading bots in parallel.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with all bot signals
    """
    # Create copies of state for each bot (to avoid mutations)
    state_copy1 = dict(state)
    state_copy2 = dict(state)
    state_copy3 = dict(state)
    
    # Initialize signals list if not present
    if "signals" not in state:
        state["signals"] = []
    
    # Run bots in parallel
    results = await asyncio.gather(
        stock_bot_node(state_copy1),
        forex_bot_node(state_copy2),
        crypto_bot_node(state_copy3),
        return_exceptions=True
    )
    
    # Aggregate signals from all bots
    all_signals = state.get("signals", [])
    for result in results:
        if isinstance(result, dict) and "signals" in result:
            all_signals.extend(result["signals"])
        elif isinstance(result, Exception):
            logger.error(f"Bot node error: {result}")
    
    # Update state with aggregated signals
    state["signals"] = all_signals
    
    logger.info(f"Collected {len(all_signals)} bot signals")
    
    return state


def _is_stock_symbol(symbol: str) -> bool:
    """Check if symbol is a stock (basic heuristic)"""
    symbol_upper = symbol.upper()
    # Stock symbols typically end with .US or are common stock tickers
    stock_indicators = [".US", ".STOCK", "NYSE:", "NASDAQ:"]
    return any(indicator in symbol_upper for indicator in stock_indicators) or (
        len(symbol) <= 5 and symbol_upper.isalpha() and not _is_crypto_symbol(symbol)
    )


def _is_forex_symbol(symbol: str) -> bool:
    """Check if symbol is forex (basic heuristic)"""
    symbol_upper = symbol.upper()
    # Forex pairs typically have format like EURUSD, GBPUSD, etc.
    forex_indicators = [".FOREX", ".FX", "/"]
    return any(indicator in symbol_upper for indicator in forex_indicators) or (
        len(symbol) == 6 and symbol_upper.isalpha() and not _is_crypto_symbol(symbol)
    )


def _is_crypto_symbol(symbol: str) -> bool:
    """Check if symbol is crypto (basic heuristic)"""
    symbol_upper = symbol.upper()
    # Crypto symbols typically contain BTC, ETH, USDT, etc.
    crypto_indicators = [
        "BTC", "ETH", "USDT", "USDC", "BNB", "SOL", "ADA", "DOT",
        "-USD", "-USDT", ".CC", ".CRYPTO", "CRYPTO:"
    ]
    return any(indicator in symbol_upper for indicator in crypto_indicators)


def _prepare_market_data(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare market data for bot consumption.
    
    Bots expect market_data in format: {"data": [{"open": ..., "close": ..., ...}]}
    If market_data is in a different format, convert it.
    
    Args:
        market_data: Market data from state (may be in various formats)
    
    Returns:
        Market data in bot-expected format
    """
    # If already in correct format, return as-is
    if "data" in market_data and isinstance(market_data["data"], list):
        return market_data
    
    # If market_data has direct OHLCV fields, create a single data point
    if "close" in market_data or "price" in market_data:
        close = market_data.get("close") or market_data.get("price")
        return {
            "data": [{
                "open": market_data.get("open", close),
                "high": market_data.get("high", close),
                "low": market_data.get("low", close),
                "close": close,
                "volume": market_data.get("volume", 0)
            }]
        }
    
    # If no recognizable format, return empty data (bot will return HOLD)
    logger.warning(f"Unknown market_data format: {list(market_data.keys())}")
    return {"data": []}

