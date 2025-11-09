"""
Multi-Agent Trading Graph

LangGraph orchestration connecting analysts, debaters, and decision-making.
"""

from agents.state import AgentState
from langgraph.graph import END, StateGraph

from .nodes import debate_node, manager_decision_node, reflection_node, run_analysts, should_execute_trade


def build_trading_graph():
    """
    Construct the multi-agent trading StateGraph.

    Graph flow:
    1. Analysts (parallel) → Technical, Sentiment, Macro, Risk
    2. Debate → Bull vs Bear adversarial arguments
    3. Manager → Synthesize debate into decision
    4. Conditional → Check if confidence > threshold
    5. Reflect → Store in memory, query similar decisions

    Returns:
        Compiled LangGraph

    Example:
        >>> graph = build_trading_graph()
        >>> result = await graph.ainvoke(initial_state)
    """
    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analysts", run_analysts)
    workflow.add_node("debate", debate_node)
    workflow.add_node("manager", manager_decision_node)
    workflow.add_node("reflect", reflection_node)

    # Add edges
    workflow.add_edge("analysts", "debate")
    workflow.add_edge("debate", "manager")

    # Conditional edge: Execute trade or skip
    workflow.add_conditional_edges(
        "manager",
        should_execute_trade,
        {
            "execute": "reflect",
            "skip": END
        }
    )

    workflow.add_edge("reflect", END)

    # Set entry point
    workflow.set_entry_point("analysts")

    # Compile
    return workflow.compile()


# Export compiled graph
trading_graph = build_trading_graph()


async def analyze_symbol(
    symbol: str,
    market_data: dict,
    regime: str = "unknown"
) -> dict:
    """
    Convenience function to analyze a symbol through the full graph.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        market_data: Market data dict with OHLCV and indicators
        regime: Market regime (bull/bear/sideways)

    Returns:
        Final state with decision

    Example:
        >>> result = await analyze_symbol(
        ...     symbol="BTCUSDT",
        ...     market_data={"close": 67500, "rsi": 65, ...},
        ...     regime="bull"
        ... )
        >>> print(result['final_decision'])
    """
    from agents.state import create_initial_state

    # Create initial state
    initial_state = create_initial_state(symbol, market_data)
    initial_state['regime'] = regime

    # Run graph
    final_state = await trading_graph.ainvoke(initial_state)

    return final_state
