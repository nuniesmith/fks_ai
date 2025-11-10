"""
Consensus Node for Multi-Agent Trading System

Aggregates signals from multiple bots and applies BTC priority rules.
"""

from typing import Dict, Any, List
from loguru import logger

from agents.state import AgentState


async def consensus_node(state: AgentState) -> AgentState:
    """
    Generate consensus signal from multiple bot signals.
    
    Applies BTC priority rules (50-60% allocation) and aggregates signals.
    
    Args:
        state: Current agent state with bot signals
    
    Returns:
        Updated state with consensus signal
    """
    try:
        signals = state.get("signals", [])
        symbol = state.get("symbol", "")
        
        if not signals:
            logger.warning(f"No signals available for consensus: {symbol}")
            state["consensus_signal"] = {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "No bot signals available"
            }
            return state
        
        # Separate signals by bot type
        stock_signals = [s for s in signals if s.get("bot") == "StockBot"]
        forex_signals = [s for s in signals if s.get("bot") == "ForexBot"]
        crypto_signals = [s for s in signals if s.get("bot") == "CryptoBot"]
        
        # Check for BTC priority
        btc_signals = [s for s in crypto_signals if s.get("btc_priority", False)]
        
        # Apply BTC priority (50-60% weight for BTC signals)
        if btc_signals:
            btc_weight = 0.55  # 55% weight for BTC
            other_weight = 0.45  # 45% for other signals
            
            # Calculate weighted consensus
            consensus = _calculate_weighted_consensus(
                signals,
                btc_signals=btc_signals,
                btc_weight=btc_weight,
                other_weight=other_weight
            )
        else:
            # Standard consensus (equal weight for all signals)
            consensus = _calculate_standard_consensus(signals)
        
        # Add consensus to state
        state["consensus_signal"] = consensus
        state["bot_signals"] = {
            "stock": stock_signals,
            "forex": forex_signals,
            "crypto": crypto_signals,
            "btc": btc_signals
        }
        
        logger.info(
            f"Consensus for {symbol}: {consensus.get('signal')} "
            f"(confidence: {consensus.get('confidence', 0):.2f})"
        )
        
        return state
    except Exception as e:
        logger.error(f"Error in consensus node: {e}", exc_info=True)
        # Return default consensus on error
        state["consensus_signal"] = {
            "signal": "HOLD",
            "confidence": 0.0,
            "reason": f"Consensus error: {str(e)}"
        }
        return state


def _calculate_weighted_consensus(
    signals: List[Dict[str, Any]],
    btc_signals: List[Dict[str, Any]],
    btc_weight: float = 0.55,
    other_weight: float = 0.45
) -> Dict[str, Any]:
    """Calculate weighted consensus with BTC priority"""
    if not signals:
        return {"signal": "HOLD", "confidence": 0.0, "reason": "No signals"}
    
    # Separate BTC and other signals
    other_signals = [s for s in signals if s not in btc_signals]
    
    # Calculate weighted scores for each action
    buy_score = 0.0
    sell_score = 0.0
    hold_score = 0.0
    
    # Process BTC signals (higher weight)
    for signal in btc_signals:
        confidence = signal.get("confidence", 0.0)
        signal_type = signal.get("signal", "HOLD")
        weight = btc_weight / len(btc_signals) if btc_signals else 0.0
        
        if signal_type == "BUY":
            buy_score += confidence * weight
        elif signal_type == "SELL":
            sell_score += confidence * weight
        else:
            hold_score += confidence * weight
    
    # Process other signals (lower weight)
    for signal in other_signals:
        confidence = signal.get("confidence", 0.0)
        signal_type = signal.get("signal", "HOLD")
        weight = other_weight / len(other_signals) if other_signals else 0.0
        
        if signal_type == "BUY":
            buy_score += confidence * weight
        elif signal_type == "SELL":
            sell_score += confidence * weight
        else:
            hold_score += confidence * weight
    
    # Determine consensus signal
    if buy_score > sell_score and buy_score > hold_score and buy_score > 0.5:
        consensus_signal = "BUY"
        consensus_confidence = min(buy_score, 1.0)
        # Use BTC signal prices if available, otherwise use first signal
        btc_signal = btc_signals[0] if btc_signals else signals[0]
        entry_price = btc_signal.get("entry_price") if btc_signal.get("signal") == "BUY" else None
        stop_loss = btc_signal.get("stop_loss") if btc_signal.get("signal") == "BUY" else None
        take_profit = btc_signal.get("take_profit") if btc_signal.get("signal") == "BUY" else None
    elif sell_score > buy_score and sell_score > hold_score and sell_score > 0.5:
        consensus_signal = "SELL"
        consensus_confidence = min(sell_score, 1.0)
        entry_price = stop_loss = take_profit = None
    else:
        consensus_signal = "HOLD"
        consensus_confidence = max(hold_score, 0.5)
        entry_price = stop_loss = take_profit = None
    
    # Build consensus result
    consensus = {
        "signal": consensus_signal,
        "confidence": consensus_confidence,
        "reason": f"BTC-weighted consensus (BTC: {len(btc_signals)}, Other: {len(other_signals)})",
        "scores": {
            "buy": buy_score,
            "sell": sell_score,
            "hold": hold_score
        }
    }
    
    # Add prices if available
    if entry_price:
        consensus["entry_price"] = entry_price
        consensus["stop_loss"] = stop_loss
        consensus["take_profit"] = take_profit
    
    return consensus


def _calculate_standard_consensus(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate standard consensus (equal weight for all signals)"""
    if not signals:
        return {"signal": "HOLD", "confidence": 0.0, "reason": "No signals"}
    
    # Count signals by type
    buy_count = sum(1 for s in signals if s.get("signal") == "BUY")
    sell_count = sum(1 for s in signals if s.get("signal") == "SELL")
    hold_count = sum(1 for s in signals if s.get("signal") == "HOLD")
    
    # Calculate average confidence for each type
    buy_confidence = sum(
        s.get("confidence", 0.0) for s in signals if s.get("signal") == "BUY"
    ) / buy_count if buy_count > 0 else 0.0
    
    sell_confidence = sum(
        s.get("confidence", 0.0) for s in signals if s.get("signal") == "SELL"
    ) / sell_count if sell_count > 0 else 0.0
    
    hold_confidence = sum(
        s.get("confidence", 0.0) for s in signals if s.get("signal") == "HOLD"
    ) / hold_count if hold_count > 0 else 0.0
    
    # Determine consensus
    if buy_count > sell_count and buy_count > hold_count and buy_confidence > 0.5:
        consensus_signal = "BUY"
        consensus_confidence = buy_confidence
        # Use first BUY signal for prices
        buy_signal = next((s for s in signals if s.get("signal") == "BUY"), None)
        entry_price = buy_signal.get("entry_price") if buy_signal else None
        stop_loss = buy_signal.get("stop_loss") if buy_signal else None
        take_profit = buy_signal.get("take_profit") if buy_signal else None
    elif sell_count > buy_count and sell_count > hold_count and sell_confidence > 0.5:
        consensus_signal = "SELL"
        consensus_confidence = sell_confidence
        entry_price = stop_loss = take_profit = None
    else:
        consensus_signal = "HOLD"
        consensus_confidence = max(hold_confidence, 0.5)
        entry_price = stop_loss = take_profit = None
    
    # Build consensus result
    consensus = {
        "signal": consensus_signal,
        "confidence": consensus_confidence,
        "reason": f"Standard consensus (BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count})",
        "scores": {
            "buy": buy_confidence,
            "sell": sell_confidence,
            "hold": hold_confidence
        }
    }
    
    # Add prices if available
    if entry_price:
        consensus["entry_price"] = entry_price
        consensus["stop_loss"] = stop_loss
        consensus["take_profit"] = take_profit
    
    return consensus

