"""
Multi-Agent Trading Graph

LangGraph orchestration connecting analysts, debaters, bots, and decision-making.
"""

from agents.state import AgentState
from langgraph.graph import END, StateGraph

from .nodes import debate_node, manager_decision_node, reflection_node, run_analysts, should_execute_trade
from .bot_nodes import run_bots_parallel
from .consensus_node import consensus_node


def build_trading_graph(include_bots: bool = True):
    """
    Construct the multi-agent trading StateGraph.

    Graph flow:
    1. Analysts (parallel) → Technical, Sentiment, Macro, Risk
    2. Bots (parallel) → StockBot, ForexBot, CryptoBot (optional)
    3. Debate → Bull vs Bear adversarial arguments
    4. Consensus → Aggregate bot signals (if bots enabled)
    5. Manager → Synthesize debate into decision
    6. Conditional → Check if confidence > threshold
    7. Reflect → Store in memory, query similar decisions

    Args:
        include_bots: Whether to include trading bots in the workflow

    Returns:
        Compiled LangGraph

    Example:
        >>> graph = build_trading_graph(include_bots=True)
        >>> result = await graph.ainvoke(initial_state)
    """
    # Create graph
    workflow = StateGraph(AgentState)

    # Add core nodes
    workflow.add_node("analysts", run_analysts)
    workflow.add_node("debate", debate_node)
    workflow.add_node("manager", manager_decision_node)
    workflow.add_node("reflect", reflection_node)

    # Add bot nodes if enabled
    if include_bots:
        workflow.add_node("bots", run_bots_parallel)
        workflow.add_node("consensus", consensus_node)
        
        # Bot-enabled flow: analysts → bots → consensus → debate → manager
        workflow.add_edge("analysts", "bots")
        workflow.add_edge("bots", "consensus")
        workflow.add_edge("consensus", "debate")
    else:
        # Standard flow: analysts → debate → manager
        workflow.add_edge("analysts", "debate")

    # Add edges (common for both flows)
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


# Export compiled graphs (with and without bots)
trading_graph = build_trading_graph(include_bots=True)
trading_graph_no_bots = build_trading_graph(include_bots=False)


async def analyze_symbol(
    symbol: str,
    market_data: dict,
    regime: str = "unknown",
    include_bots: bool = True
) -> dict:
    """
    Convenience function to analyze a symbol through the full graph.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        market_data: Market data dict with OHLCV and indicators
        regime: Market regime (bull/bear/sideways)
        include_bots: Whether to include trading bots in the workflow

    Returns:
        Final state with decision

    Example:
        >>> result = await analyze_symbol(
        ...     symbol="BTCUSDT",
        ...     market_data={"close": 67500, "rsi": 65, ...},
        ...     regime="bull",
        ...     include_bots=True
        ... )
        >>> print(result['final_decision'])
    """
    from agents.state import create_initial_state

    # Create initial state
    initial_state = create_initial_state(symbol, market_data)
    initial_state['regime'] = regime

    # Select appropriate graph
    if include_bots:
        graph_to_use = trading_graph
    else:
        graph_to_use = trading_graph_no_bots

    # Run graph
    final_state = await graph_to_use.ainvoke(initial_state)

    return final_state
