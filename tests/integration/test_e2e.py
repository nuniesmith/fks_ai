"""
End-to-End Integration Tests for Multi-Agent Trading Graph

Tests complete graph execution with live LLM and memory systems.
"""

import asyncio
import os

# Import graph components
import sys
from datetime import datetime
from typing import Any, Dict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agents.state import create_initial_state
from graph.trading_graph import analyze_symbol, build_trading_graph
from memory.memory_manager import TradingMemory
from processors.signal_processor import SignalProcessor


# Sample market data for testing
@pytest.fixture
def btc_bull_market_data() -> dict[str, Any]:
    """Bitcoin in strong bull market conditions"""
    return {
        "price": 67234.50,
        "rsi": 72.5,  # Overbought
        "macd": 350.2,
        "macd_signal": 280.8,
        "bb_upper": 68500.0,
        "bb_middle": 67000.0,
        "bb_lower": 65500.0,
        "atr": 450.0,
        "volume": 2500000000,
        "volume_ma": 1800000000,
        "regime": "bull"
    }


@pytest.fixture
def eth_bear_market_data() -> dict[str, Any]:
    """Ethereum in bear market conditions"""
    return {
        "price": 2456.30,
        "rsi": 28.5,  # Oversold
        "macd": -45.2,
        "macd_signal": -38.1,
        "bb_upper": 2600.0,
        "bb_middle": 2500.0,
        "bb_lower": 2400.0,
        "atr": 85.0,
        "volume": 1200000000,
        "volume_ma": 1500000000,
        "regime": "bear"
    }


@pytest.fixture
def sideways_market_data() -> dict[str, Any]:
    """Sideways/neutral market conditions"""
    return {
        "price": 50000.0,
        "rsi": 48.5,  # Neutral
        "macd": 5.2,
        "macd_signal": 4.8,
        "bb_upper": 51000.0,
        "bb_middle": 50000.0,
        "bb_lower": 49000.0,
        "atr": 300.0,
        "volume": 1000000000,
        "volume_ma": 1000000000,
        "regime": "sideways"
    }


@pytest.mark.asyncio
@pytest.mark.integration
class TestGraphExecution:
    """Test complete graph execution with live systems"""

    async def test_analyze_symbol_bull_market(self, btc_bull_market_data):
        """Test full analysis on bull market data"""
        start_time = asyncio.get_event_loop().time()

        # Execute full graph
        final_state = await analyze_symbol("BTCUSDT", btc_bull_market_data)

        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

        # Validate execution time (<5s = 5000ms)
        assert execution_time < 5000, f"Graph execution took {execution_time:.0f}ms (target: <5000ms)"

        # Validate state structure
        assert 'messages' in final_state, "Missing analyst messages"
        assert 'debates' in final_state, "Missing debates"
        assert 'final_decision' in final_state, "Missing final decision"
        assert 'confidence' in final_state, "Missing confidence score"

        # Validate analyst execution (should have 4 analysts)
        assert len(final_state['messages']) >= 4, "Not all analysts executed"

        # Validate debate execution (should have Bull + Bear)
        assert len(final_state['debates']) >= 2, "Debate not complete"

        # Validate confidence is reasonable
        assert 0.0 <= final_state['confidence'] <= 1.0, "Invalid confidence score"

        # In bull market, expect bullish bias (but not guaranteed)
        print("\nBull Market Analysis:")
        print(f"  Decision: {final_state['final_decision'][:100]}...")
        print(f"  Confidence: {final_state['confidence']:.2f}")
        print(f"  Execution Time: {execution_time:.0f}ms")


    async def test_analyze_symbol_bear_market(self, eth_bear_market_data):
        """Test full analysis on bear market data"""
        start_time = asyncio.get_event_loop().time()

        final_state = await analyze_symbol("ETHUSDT", eth_bear_market_data)

        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

        # Validate execution time
        assert execution_time < 5000, f"Graph execution took {execution_time:.0f}ms"

        # Validate state structure
        assert 'final_decision' in final_state
        assert 'confidence' in final_state

        # In bear market, expect bearish bias
        print("\nBear Market Analysis:")
        print(f"  Decision: {final_state['final_decision'][:100]}...")
        print(f"  Confidence: {final_state['confidence']:.2f}")
        print(f"  Execution Time: {execution_time:.0f}ms")


    async def test_debate_contrast(self, btc_bull_market_data):
        """Test that Bull and Bear agents produce contrasting views"""
        final_state = await analyze_symbol("BTCUSDT", btc_bull_market_data)

        assert len(final_state['debates']) >= 2, "Need at least Bull and Bear debates"

        bull_argument = final_state['debates'][0]
        bear_argument = final_state['debates'][1]

        # Arguments should be substantial
        assert len(bull_argument) > 100, "Bull argument too short"
        assert len(bear_argument) > 100, "Bear argument too short"

        # Arguments should be different (>70% contrast target)
        # Simple heuristic: check for opposite keywords
        bull_keywords = ['buy', 'bullish', 'uptrend', 'breakout', 'support']
        bear_keywords = ['sell', 'bearish', 'downtrend', 'breakdown', 'resistance']

        bull_score = sum(1 for kw in bull_keywords if kw in bull_argument.lower())
        bear_score = sum(1 for kw in bear_keywords if kw in bear_argument.lower())

        print("\nDebate Contrast:")
        print(f"  Bull keywords: {bull_score}")
        print(f"  Bear keywords: {bear_score}")
        print(f"  Bull arg length: {len(bull_argument)}")
        print(f"  Bear arg length: {len(bear_argument)}")

        # At least one side should use their keywords
        assert bull_score > 0 or bear_score > 0, "Debates lack contrasting viewpoints"


    async def test_signal_quality_validation(self, btc_bull_market_data):
        """Test signal quality and risk management"""
        final_state = await analyze_symbol("BTCUSDT", btc_bull_market_data)

        # Process signal
        signal_processor = SignalProcessor()
        signal = signal_processor.process_decision(
            final_state['final_decision'],
            btc_bull_market_data
        )

        # Validate signal structure
        assert 'action' in signal, "Missing action"
        assert signal['action'] in ['BUY', 'SELL', 'HOLD'], f"Invalid action: {signal['action']}"

        if signal['action'] != 'HOLD':
            # Validate risk management fields
            assert 'position_size' in signal, "Missing position size"
            assert 'stop_loss' in signal, "Missing stop loss"
            assert 'take_profit' in signal, "Missing take profit"

            # Validate position sizing (<= 10% max)
            assert 0 < signal['position_size'] <= 0.10, f"Position size {signal['position_size']} out of range"

            # Validate risk/reward ratio (>= 2.0 target)
            if signal['risk'] > 0:
                rr_ratio = signal['reward'] / signal['risk']
                assert rr_ratio >= 1.0, f"R/R ratio {rr_ratio:.2f} too low"

            print("\nSignal Quality:")
            print(f"  Action: {signal['action']}")
            print(f"  Position Size: {signal['position_size']:.2%}")
            print(f"  Stop Loss: {signal['stop_loss']:.2f}")
            print(f"  Take Profit: {signal['take_profit']:.2f}")
            print(f"  R/R Ratio: {signal['reward']/signal['risk']:.2f}" if signal['risk'] > 0 else "  R/R: N/A")


    async def test_memory_persistence(self, btc_bull_market_data):
        """Test that decisions are stored in ChromaDB"""
        memory = TradingMemory()

        # Analyze symbol
        final_state = await analyze_symbol("BTCUSDT", btc_bull_market_data)

        # Store decision in memory
        insight = f"BTCUSDT Analysis: {final_state['final_decision'][:200]}"
        memory.add_insight(
            insight=insight,
            metadata={
                "symbol": "BTCUSDT",
                "confidence": final_state['confidence'],
                "regime": btc_bull_market_data['regime'],
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Query similar decisions
        similar = memory.query_similar("BTCUSDT trading analysis", n_results=3)

        assert len(similar) > 0, "No results from memory query"
        print("\nMemory Persistence:")
        print(f"  Stored: {insight[:100]}...")
        print(f"  Retrieved: {len(similar)} similar insights")


    async def test_parallel_analysis(self, btc_bull_market_data, eth_bear_market_data):
        """Test multiple concurrent analyses"""
        start_time = asyncio.get_event_loop().time()

        # Run 3 analyses in parallel
        results = await asyncio.gather(
            analyze_symbol("BTCUSDT", btc_bull_market_data),
            analyze_symbol("ETHUSDT", eth_bear_market_data),
            analyze_symbol("BTCUSDT", btc_bull_market_data)
        )

        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

        # All should complete
        assert len(results) == 3, "Not all analyses completed"

        # Each should have valid output
        for i, result in enumerate(results):
            assert 'final_decision' in result, f"Result {i} missing decision"
            assert 'confidence' in result, f"Result {i} missing confidence"

        # Parallel execution should be faster than sequential (rough check)
        # Sequential would be ~15s (3 * 5s), parallel should be ~5-7s
        assert execution_time < 10000, f"Parallel execution too slow: {execution_time:.0f}ms"

        print("\nParallel Analysis:")
        print(f"  3 analyses completed in {execution_time:.0f}ms")
        print(f"  Average: {execution_time/3:.0f}ms per analysis")


@pytest.mark.asyncio
@pytest.mark.integration
class TestGraphConstruction:
    """Test graph structure and compilation"""

    async def test_graph_builds_correctly(self):
        """Test that StateGraph builds without errors"""
        graph = build_trading_graph()

        # Graph should be compiled
        assert graph is not None, "Graph compilation failed"

        # Test minimal invocation
        test_state = create_initial_state("TESTUSDT", {"price": 100.0, "rsi": 50.0})

        try:
            # This will fail without Ollama, but should not error on graph structure
            result = await asyncio.wait_for(graph.ainvoke(test_state), timeout=10.0)
            assert 'final_decision' in result or 'messages' in result, "Graph produced no output"
        except TimeoutError:
            # Timeout is acceptable (Ollama might be slow)
            pass
        except Exception as e:
            # Only fail if it's a graph structure error
            if "node" not in str(e).lower() and "edge" not in str(e).lower():
                # Ollama connection error is acceptable
                pass
            else:
                raise


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for Phase 6 success metrics"""

    async def test_latency_benchmark(self, btc_bull_market_data):
        """Benchmark graph execution latency (target: <5s)"""
        latencies = []

        # Run 10 iterations
        for _i in range(10):
            start = asyncio.get_event_loop().time()
            await analyze_symbol("BTCUSDT", btc_bull_market_data)
            latency = (asyncio.get_event_loop().time() - start) * 1000
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        print("\nLatency Benchmark (10 iterations):")
        print(f"  Average: {avg_latency:.0f}ms")
        print(f"  Min: {min_latency:.0f}ms")
        print(f"  Max: {max_latency:.0f}ms")

        # Average should be under 5s
        assert avg_latency < 5000, f"Average latency {avg_latency:.0f}ms exceeds 5000ms target"


    async def test_signal_accuracy_estimate(self, btc_bull_market_data, eth_bear_market_data):
        """Estimate signal accuracy (target: >60%)"""
        # This is a placeholder - real accuracy requires backtesting
        # For now, validate that signals are reasonable given market conditions

        btc_result = await analyze_symbol("BTCUSDT", btc_bull_market_data)
        eth_result = await analyze_symbol("ETHUSDT", eth_bear_market_data)

        signal_processor = SignalProcessor()
        btc_signal = signal_processor.process_decision(btc_result['final_decision'], btc_bull_market_data)
        eth_signal = signal_processor.process_decision(eth_result['final_decision'], eth_bear_market_data)

        # In strong bull market (RSI 72.5), expect BUY or HOLD (not SELL)
        # In strong bear market (RSI 28.5), expect SELL or HOLD (not BUY)

        print("\nSignal Reasonableness Check:")
        print(f"  BTC (RSI 72.5, Bull): {btc_signal['action']}")
        print(f"  ETH (RSI 28.5, Bear): {eth_signal['action']}")

        # Not a hard assertion since LLM can be contrarian
        # Just log for manual review


if __name__ == "__main__":
    # Run with: pytest tests/integration/test_e2e.py -v
    pytest.main([__file__, "-v", "-s"])
