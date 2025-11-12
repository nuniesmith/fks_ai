"""
Graph Nodes for Multi-Agent Trading System

Implements the execution logic for each node in the StateGraph.
Enhanced with confidence threshold validation for quality control.
"""

import asyncio
from typing import Any, Dict

from agents.analysts import analyze_macro, analyze_risk, analyze_sentiment, analyze_technical
from agents.base import DEFAULT_CONFIDENCE_THRESHOLD, validate_confidence_threshold
from agents.debaters import generate_bear_case, generate_bull_case, synthesize_debate
from agents.state import AgentState
from memory import TradingMemory


async def run_analysts(state: AgentState) -> AgentState:
    """
    Run all 4 analyst agents in parallel.
    Only includes analyst insights that meet confidence threshold.

    Args:
        state: Current agent state with market_data

    Returns:
        Updated state with high-confidence analyst insights in messages
    """
    # Extract data for each analyst
    market_data = state['market_data']
    symbol = state['symbol']

    # Prepare analyst-specific data
    technical_data = {
        'symbol': symbol,
        'close': market_data.get('close'),
        'rsi': market_data.get('rsi'),
        'macd': market_data.get('macd'),
        'bb_upper': market_data.get('bb_upper'),
        'bb_lower': market_data.get('bb_lower'),
        'volume': market_data.get('volume'),
        'atr': market_data.get('atr')
    }

    sentiment_data = {
        'symbol': symbol,
        'fear_greed_index': market_data.get('fear_greed_index', 50),
        'social_volume': market_data.get('social_volume', 'medium'),
        'news_sentiment': market_data.get('news_sentiment', 'neutral'),
        'whale_activity': market_data.get('whale_activity', 'unknown')
    }

    macro_data = {
        'symbol': symbol,
        'cpi_yoy': market_data.get('cpi_yoy'),
        'fed_funds_rate': market_data.get('fed_funds_rate'),
        'dxy': market_data.get('dxy'),
        'spx_correlation': market_data.get('spx_correlation'),
        'gold_correlation': market_data.get('gold_correlation')
    }

    risk_data = {
        'symbol': symbol,
        'entry_price': market_data.get('close'),
        'direction': 'LONG',  # Will be determined by debate
        'confidence': 0.5,  # Initial placeholder
        'account_size': market_data.get('account_size', 100000),
        'current_positions': market_data.get('current_positions', 0),
        'volatility': market_data.get('volatility', 0.02),
        'current_drawdown': market_data.get('current_drawdown', 0)
    }

    # Run all analysts in parallel
    results = await asyncio.gather(
        analyze_technical(technical_data),
        analyze_sentiment(sentiment_data),
        analyze_macro(macro_data),
        analyze_risk(risk_data)
    )

    # Validate confidence thresholds and add only high-confidence results
    min_confidence = DEFAULT_CONFIDENCE_THRESHOLD
    high_confidence_results = []
    skipped_analysts = []

    for result in results:
        analysis_text = result['analysis']
        validation = validate_confidence_threshold(analysis_text, min_confidence)

        if validation['meets_threshold']:
            # Add to messages
            state['messages'].append({
                'role': result['agent'],
                'content': analysis_text,
                'confidence': validation['confidence']
            })
            high_confidence_results.append(result['agent'])
        else:
            # Log skipped analyst
            skipped_analysts.append({
                'agent': result['agent'],
                'reason': validation['reason'],
                'confidence': validation.get('confidence')
            })

    # Add metadata about skipped analysts
    if skipped_analysts:
        state['messages'].append({
            'role': 'system',
            'content': f"Skipped {len(skipped_analysts)} low-confidence analysts: {', '.join([a['agent'] for a in skipped_analysts])}",
            'skipped_details': skipped_analysts
        })

    return state


async def debate_node(state: AgentState) -> AgentState:
    """
    Run Bull vs Bear debate based on analyst insights.

    Args:
        state: State with analyst messages

    Returns:
        Updated state with bull/bear debates
    """
    # Extract analyst insights
    analyst_insights = [
        msg['content'] for msg in state['messages']
        if msg['role'] in ['technical_analyst', 'sentiment_analyst', 'macro_analyst', 'risk_analyst']
    ]

    # Run bull and bear in parallel
    bull_result, bear_result = await asyncio.gather(
        generate_bull_case(analyst_insights),
        generate_bear_case(analyst_insights)
    )

    # Store debates
    state['debates'] = [
        bull_result['argument'],
        bear_result['argument']
    ]

    # Add to messages
    state['messages'].append({
        'role': 'bull',
        'content': bull_result['argument']
    })
    state['messages'].append({
        'role': 'bear',
        'content': bear_result['argument']
    })

    return state


async def manager_decision_node(state: AgentState) -> AgentState:
    """
    Manager synthesizes debate into final decision.
    Extracts and validates confidence score from decision.

    Args:
        state: State with bull/bear debates

    Returns:
        Updated state with final_decision and validated confidence
    """
    if len(state['debates']) < 2:
        raise ValueError("Need both bull and bear arguments")

    bull_argument = state['debates'][0]
    bear_argument = state['debates'][1]

    # Synthesize debate
    result = await synthesize_debate(
        bull_argument=bull_argument,
        bear_argument=bear_argument,
        market_regime=state.get('regime', 'unknown'),
        additional_context=state.get('market_data', {})
    )

    decision_text = result['decision']

    # Extract and validate confidence
    validation = validate_confidence_threshold(decision_text, DEFAULT_CONFIDENCE_THRESHOLD)

    # Store validated confidence in state
    state['confidence'] = validation.get('confidence', 0.0)

    # Store decision
    state['final_decision'] = {
        'decision': decision_text,
        'inputs': result['inputs'],
        'confidence': state['confidence'],
        'meets_threshold': validation['meets_threshold'],
        'validation': validation
    }

    # Add to messages
    state['messages'].append({
        'role': 'manager',
        'content': decision_text,
        'confidence': state['confidence']
    })

    return state


async def reflection_node(state: AgentState) -> AgentState:
    """
    Store decision in ChromaDB and retrieve similar past insights.

    Args:
        state: State with final_decision

    Returns:
        Updated state with memory context
    """
    memory = TradingMemory()

    # Create insight text
    decision = state['final_decision']
    insight_text = f"""
    Symbol: {state['symbol']}
    Decision: {decision}
    Debates: {state['debates']}
    Timestamp: {state['timestamp']}
    """

    # Store in memory
    memory.add_insight(
        text=insight_text,
        metadata={
            'symbol': state['symbol'],
            'timestamp': state['timestamp'],
            'regime': state.get('regime', 'unknown'),
            'confidence': state.get('confidence', 0.5)
        }
    )

    # Query similar decisions
    similar = memory.query_similar(
        query=f"Trading {state['symbol']} in {state.get('regime', 'unknown')} regime",
        n_results=3
    )

    # Add to state
    state['memory'] = [s['text'] for s in similar]

    return state


def should_execute_trade(state: AgentState) -> str:
    """
    Conditional edge: Decide if we should execute trade.
    Uses validated confidence from manager decision node.

    Args:
        state: State with final_decision and validated confidence

    Returns:
        'execute' or 'skip'

    Examples:
        High confidence BUY: returns 'execute'
        Low confidence BUY: returns 'skip'
        HOLD decision: returns 'skip'
        INSUFFICIENT CONFIDENCE: returns 'skip'
    """
    decision = state.get('final_decision', {})
    decision_text = decision.get('decision', '').upper()

    # Check if manager reported insufficient confidence
    if 'INSUFFICIENT CONFIDENCE' in decision_text:
        return 'skip'

    # Check if decision is BUY or SELL (not HOLD)
    if 'BUY' in decision_text or 'SELL' in decision_text:
        # Use validated confidence from manager decision
        confidence = state.get('confidence', 0.0)

        # Check if meets threshold (stored in final_decision validation)
        meets_threshold = decision.get('meets_threshold', False)

        if meets_threshold and confidence > DEFAULT_CONFIDENCE_THRESHOLD:
            return 'execute'

    return 'skip'
