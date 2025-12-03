"""
Bitcoin Workflow Tests

Comprehensive tests for the Bitcoin-focused trading workflow including:
- Signal generation accuracy
- Dynamic risk calculations
- On-chain analysis
- Performance timing benchmarks
"""

import pytest
import asyncio
import time
from datetime import datetime
from typing import Dict, Any

from src.workflows import (
    BitcoinWorkflow,
    get_workflow,
    SignalQuality,
    SignalDirection,
    DynamicRisk,
    ConfidenceLevel,
    TimingMetrics,
)
from src.workflows.bitcoin import BTCTrendState, OnChainBTCMetrics, BTCSignalContext


class TestBitcoinWorkflowBasics:
    """Basic workflow initialization and structure tests."""
    
    def test_workflow_registration(self):
        """Test Bitcoin workflow is registered."""
        workflow = get_workflow("bitcoin")
        assert workflow is not None
        assert workflow.market_type == "bitcoin"
    
    def test_workflow_initialization(self):
        """Test workflow initializes with correct defaults."""
        workflow = BitcoinWorkflow()
        assert workflow.base_risk_pct == 2.0
        assert workflow.max_position_pct == 10.0
        assert len(workflow.watchlist) > 0
        assert "BTCUSDT" in workflow.watchlist
    
    def test_custom_initialization(self):
        """Test workflow with custom parameters."""
        workflow = BitcoinWorkflow(base_risk_pct=1.5, max_position_pct=8.0)
        assert workflow.base_risk_pct == 1.5
        assert workflow.max_position_pct == 8.0


class TestDynamicRisk:
    """Tests for dynamic risk calculation based on confidence."""
    
    def test_very_high_confidence(self):
        """Test risk calculation for very high confidence (85%+)."""
        workflow = BitcoinWorkflow()
        risk = workflow.calculate_dynamic_risk(0.90)
        
        assert risk.confidence_level == ConfidenceLevel.VERY_HIGH
        assert risk.position_multiplier == 1.0
        assert risk.recommended_action == "Full position"
        assert risk.max_position_pct == workflow.max_position_pct
    
    def test_high_confidence(self):
        """Test risk calculation for high confidence (75-84%)."""
        workflow = BitcoinWorkflow()
        risk = workflow.calculate_dynamic_risk(0.80)
        
        assert risk.confidence_level == ConfidenceLevel.HIGH
        assert risk.position_multiplier == 0.75
        assert "Reduced" in risk.recommended_action
    
    def test_medium_confidence(self):
        """Test risk calculation for medium confidence (65-74%)."""
        workflow = BitcoinWorkflow()
        risk = workflow.calculate_dynamic_risk(0.70)
        
        assert risk.confidence_level == ConfidenceLevel.MEDIUM
        assert risk.position_multiplier == 0.5
        assert "Half" in risk.recommended_action
    
    def test_low_confidence(self):
        """Test risk calculation for low confidence (55-64%)."""
        workflow = BitcoinWorkflow()
        risk = workflow.calculate_dynamic_risk(0.60)
        
        assert risk.confidence_level == ConfidenceLevel.LOW
        assert risk.position_multiplier == 0.25
        assert "Quarter" in risk.recommended_action
    
    def test_very_low_confidence(self):
        """Test risk calculation for very low confidence (<55%)."""
        workflow = BitcoinWorkflow()
        risk = workflow.calculate_dynamic_risk(0.40)
        
        assert risk.confidence_level == ConfidenceLevel.VERY_LOW
        assert risk.position_multiplier == 0.0
        assert "Paper trade" in risk.recommended_action
    
    def test_risk_serialization(self):
        """Test DynamicRisk serializes correctly."""
        workflow = BitcoinWorkflow()
        risk = workflow.calculate_dynamic_risk(0.85)
        
        data = risk.to_dict()
        assert "confidence_level" in data
        assert "base_risk_pct" in data
        assert "adjusted_risk_pct" in data
        assert "position_multiplier" in data
        assert "max_position_pct" in data
        assert "recommended_action" in data


class TestTrendClassification:
    """Tests for Bitcoin trend state classification."""
    
    def test_strong_bull(self):
        """Test strong bull classification."""
        workflow = BitcoinWorkflow()
        state = workflow.classify_trend("up", "up", "up")
        assert state == BTCTrendState.STRONG_BULL
    
    def test_bull(self):
        """Test bull classification."""
        workflow = BitcoinWorkflow()
        state = workflow.classify_trend("up", "up", "sideways")
        assert state == BTCTrendState.BULL
    
    def test_weak_bull(self):
        """Test weak bull classification."""
        workflow = BitcoinWorkflow()
        state = workflow.classify_trend("up", "sideways", "down")
        assert state == BTCTrendState.WEAK_BULL
    
    def test_strong_bear(self):
        """Test strong bear classification."""
        workflow = BitcoinWorkflow()
        state = workflow.classify_trend("down", "down", "down")
        assert state == BTCTrendState.STRONG_BEAR
    
    def test_bear(self):
        """Test bear classification."""
        workflow = BitcoinWorkflow()
        state = workflow.classify_trend("down", "down", "sideways")
        assert state == BTCTrendState.BEAR
    
    def test_neutral(self):
        """Test neutral classification."""
        workflow = BitcoinWorkflow()
        state = workflow.classify_trend("sideways", "sideways", "sideways")
        assert state == BTCTrendState.NEUTRAL


class TestOnChainMetrics:
    """Tests for on-chain metrics analysis."""
    
    def test_bullish_signals(self):
        """Test counting bullish on-chain signals."""
        metrics = OnChainBTCMetrics(
            exchange_netflow_btc=-2000,  # Bullish
            whale_accumulation_score=0.5,  # Bullish
            miner_outflow_btc=50,  # Bullish (miners holding)
            hash_rate_change_pct=3.0,  # Bullish
            futures_funding_rate=0.005,  # Bullish (not overheated)
        )
        
        assert metrics.bullish_signals() >= 4
        assert metrics.bearish_signals() == 0
    
    def test_bearish_signals(self):
        """Test counting bearish on-chain signals."""
        metrics = OnChainBTCMetrics(
            exchange_netflow_btc=10000,  # Bearish
            whale_accumulation_score=-0.5,  # Bearish
            miner_outflow_btc=2000,  # Bearish
            hash_rate_change_pct=-10,  # Bearish
            futures_funding_rate=0.1,  # Bearish (overheated)
        )
        
        assert metrics.bearish_signals() >= 4
        assert metrics.bullish_signals() == 0
    
    def test_net_score_calculation(self):
        """Test net on-chain score calculation."""
        # Bullish scenario
        bullish = OnChainBTCMetrics(
            exchange_netflow_btc=-5000,
            whale_accumulation_score=0.8,
            futures_funding_rate=0.005,
        )
        assert bullish.net_score() > 0
        
        # Bearish scenario
        bearish = OnChainBTCMetrics(
            exchange_netflow_btc=10000,
            whale_accumulation_score=-0.8,
            futures_funding_rate=0.1,
            miner_outflow_btc=2000,  # Ensure bearish signal
            hash_rate_change_pct=-10,  # Ensure bearish signal
        )
        assert bearish.net_score() < 0
        
        # Neutral scenario - set values to avoid triggering signals
        neutral = OnChainBTCMetrics(
            exchange_netflow_btc=0,  # Between -1000 and 5000
            whale_accumulation_score=0,  # Between -0.3 and 0.3
            miner_outflow_btc=500,  # Between 100 and 1000
            hash_rate_change_pct=-2,  # Between -5 and 0
            futures_funding_rate=0.02,  # Between 0.01 and 0.05
        )
        assert neutral.net_score() == 0


class TestMarketRegimeDetection:
    """Tests for market regime detection."""
    
    def test_volatile_regime(self):
        """Test detection of volatile market."""
        workflow = BitcoinWorkflow()
        regime = workflow.detect_market_regime(5.0, 2.5, 8.0)
        assert regime == "volatile"
    
    def test_ranging_regime(self):
        """Test detection of ranging market."""
        workflow = BitcoinWorkflow()
        regime = workflow.detect_market_regime(1.5, 0.6, 2.0)
        assert regime == "ranging"
    
    def test_trending_regime(self):
        """Test detection of trending market."""
        workflow = BitcoinWorkflow()
        regime = workflow.detect_market_regime(2.5, 1.2, 5.0)
        assert regime == "trending"


class TestSignalGeneration:
    """Tests for Bitcoin signal generation."""
    
    @pytest.mark.asyncio
    async def test_generates_long_signal_in_uptrend(self):
        """Test long signal generation in uptrend."""
        workflow = BitcoinWorkflow()
        data = await workflow.fetch_market_data(["BTCUSDT"])
        candidates = await workflow.identify_candidates(data)
        
        # Should generate at least one candidate in strong uptrend
        assert len(candidates) > 0
        
        # Check signal properties
        candidate = candidates[0]
        assert candidate.symbol == "BTCUSDT"
        assert candidate.direction == SignalDirection.LONG
        assert candidate.confidence > 0
        assert candidate.entry_price > 0
        assert candidate.stop_loss < candidate.entry_price
        assert candidate.take_profit > candidate.entry_price
    
    @pytest.mark.asyncio
    async def test_signal_has_dynamic_risk(self):
        """Test that signals include dynamic risk metadata."""
        workflow = BitcoinWorkflow()
        data = await workflow.fetch_market_data(["BTCUSDT"])
        candidates = await workflow.identify_candidates(data)
        
        assert len(candidates) > 0
        candidate = candidates[0]
        
        assert "dynamic_risk" in candidate.metadata
        assert "confidence_level" in candidate.metadata["dynamic_risk"]
        assert "position_multiplier" in candidate.metadata["dynamic_risk"]
    
    @pytest.mark.asyncio
    async def test_signal_has_btc_context(self):
        """Test that signals include BTC-specific context."""
        workflow = BitcoinWorkflow()
        data = await workflow.fetch_market_data(["BTCUSDT"])
        candidates = await workflow.identify_candidates(data)
        
        assert len(candidates) > 0
        candidate = candidates[0]
        
        assert "btc_context" in candidate.metadata
        context = candidate.metadata["btc_context"]
        assert "trend_state" in context
        assert "market_regime" in context
        assert "support_levels" in context
        assert "resistance_levels" in context
    
    @pytest.mark.asyncio
    async def test_risk_reward_ratio(self):
        """Test that signals have acceptable R:R ratio."""
        workflow = BitcoinWorkflow()
        data = await workflow.fetch_market_data(["BTCUSDT"])
        candidates = await workflow.identify_candidates(data)
        
        for candidate in candidates:
            # Should be at least 1.5:1 R:R
            assert candidate.risk_reward_ratio >= 1.5, f"R:R too low: {candidate.risk_reward_ratio}"


class TestValidation:
    """Tests for signal validation."""
    
    @pytest.mark.asyncio
    async def test_validation_passes_for_valid_signal(self):
        """Test validation passes for properly formed signal."""
        workflow = BitcoinWorkflow()
        data = await workflow.fetch_market_data(["BTCUSDT"])
        candidates = await workflow.identify_candidates(data)
        
        assert len(candidates) > 0
        candidate = candidates[0]
        
        is_valid, quality, warnings = await workflow.validate_candidate(
            candidate, data["BTCUSDT"], 10000.0
        )
        
        # With mock data showing strong uptrend, should pass
        assert is_valid
        assert quality != SignalQuality.REJECTED
    
    @pytest.mark.asyncio
    async def test_quality_mapping(self):
        """Test that quality is mapped correctly from confidence."""
        workflow = BitcoinWorkflow()
        
        # Test quality boundaries
        high_conf_risk = workflow.calculate_dynamic_risk(0.90)
        assert high_conf_risk.confidence_level == ConfidenceLevel.VERY_HIGH
        
        med_conf_risk = workflow.calculate_dynamic_risk(0.70)
        assert med_conf_risk.confidence_level == ConfidenceLevel.MEDIUM


class TestTimingPerformance:
    """Performance timing tests to ensure speed requirements."""
    
    @pytest.mark.asyncio
    async def test_data_fetch_under_threshold(self):
        """Test data fetching completes within threshold."""
        workflow = BitcoinWorkflow()
        
        start = time.perf_counter()
        await workflow.fetch_market_data(["BTCUSDT"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Should complete under 100ms for mock data
        assert elapsed_ms < 100, f"Data fetch took {elapsed_ms:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_candidate_identification_under_threshold(self):
        """Test candidate identification completes within threshold."""
        workflow = BitcoinWorkflow()
        data = await workflow.fetch_market_data(["BTCUSDT"])
        
        start = time.perf_counter()
        await workflow.identify_candidates(data)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Should complete under 50ms
        assert elapsed_ms < 50, f"Candidate identification took {elapsed_ms:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_validation_under_threshold(self):
        """Test validation completes within threshold."""
        workflow = BitcoinWorkflow()
        data = await workflow.fetch_market_data(["BTCUSDT"])
        candidates = await workflow.identify_candidates(data)
        
        if candidates:
            start = time.perf_counter()
            await workflow.validate_candidate(candidates[0], data["BTCUSDT"], 10000.0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Should complete under 50ms
            assert elapsed_ms < 50, f"Validation took {elapsed_ms:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_full_workflow_under_threshold(self):
        """Test full workflow completes within threshold."""
        workflow = BitcoinWorkflow()
        
        start = time.perf_counter()
        result = await workflow.run(account_balance=10000.0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Full workflow should complete under 200ms for mock data
        assert elapsed_ms < 200, f"Full workflow took {elapsed_ms:.2f}ms"
        
        # Check timing metrics are captured
        assert "timing" in result.metadata
        timing = result.metadata["timing"]
        assert timing["total_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_timing_metrics_structure(self):
        """Test that timing metrics are properly structured."""
        workflow = BitcoinWorkflow()
        result = await workflow.run(account_balance=10000.0)
        
        timing = result.metadata.get("timing", {})
        expected_keys = [
            "data_fetch_ms",
            "technical_analysis_ms",
            "on_chain_analysis_ms",
            "candidate_identification_ms",
            "validation_ms",
            "risk_calculation_ms",
            "total_ms",
        ]
        
        for key in expected_keys:
            assert key in timing, f"Missing timing key: {key}"
            assert isinstance(timing[key], (int, float)), f"{key} should be numeric"


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_execution(self):
        """Test complete workflow execution."""
        workflow = BitcoinWorkflow()
        result = await workflow.run(account_balance=10000.0)
        
        assert result.market_type == "bitcoin"
        assert result.symbols_scanned > 0
        assert result.execution_time_ms > 0
        assert result.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_workflow_result_serialization(self):
        """Test workflow result can be serialized."""
        workflow = BitcoinWorkflow()
        result = await workflow.run(account_balance=10000.0)
        
        data = result.to_dict()
        assert "market_type" in data
        assert "signals_generated" in data
        assert "execution_time_ms" in data
        assert "metadata" in data
    
    @pytest.mark.asyncio
    async def test_multiple_runs_consistent(self):
        """Test multiple workflow runs produce consistent results."""
        workflow = BitcoinWorkflow()
        
        results = []
        for _ in range(3):
            result = await workflow.run(account_balance=10000.0)
            results.append(result)
        
        # All runs should find the same number of candidates (with mock data)
        candidates_counts = [r.candidates_found for r in results]
        assert len(set(candidates_counts)) == 1, "Inconsistent candidate counts"


class TestEdgeCases:
    """Edge case tests."""
    
    @pytest.mark.asyncio
    async def test_empty_symbol_list(self):
        """Test workflow handles empty symbol list."""
        workflow = BitcoinWorkflow()
        data = await workflow.fetch_market_data([])
        
        assert data == {}
    
    @pytest.mark.asyncio
    async def test_low_account_balance(self):
        """Test workflow handles low account balance."""
        workflow = BitcoinWorkflow()
        result = await workflow.run(account_balance=100.0)
        
        # Should still complete without error
        assert result is not None
    
    def test_confidence_boundary_values(self):
        """Test confidence level boundaries."""
        workflow = BitcoinWorkflow()
        
        # Test exact boundaries
        assert workflow.get_confidence_level(0.85) == ConfidenceLevel.VERY_HIGH
        assert workflow.get_confidence_level(0.84999) == ConfidenceLevel.HIGH
        assert workflow.get_confidence_level(0.75) == ConfidenceLevel.HIGH
        assert workflow.get_confidence_level(0.74999) == ConfidenceLevel.MEDIUM
        assert workflow.get_confidence_level(0.65) == ConfidenceLevel.MEDIUM
        assert workflow.get_confidence_level(0.64999) == ConfidenceLevel.LOW
        assert workflow.get_confidence_level(0.55) == ConfidenceLevel.LOW
        assert workflow.get_confidence_level(0.54999) == ConfidenceLevel.VERY_LOW
    
    def test_days_to_halving(self):
        """Test halving calculation."""
        workflow = BitcoinWorkflow()
        days = workflow.days_to_halving()
        
        # Should be positive (next halving is in future)
        assert days >= 0


# Benchmark tests for performance tracking
class TestBenchmarks:
    """Benchmark tests for tracking performance over time."""
    
    @pytest.mark.asyncio
    async def test_benchmark_single_symbol(self):
        """Benchmark single symbol processing."""
        workflow = BitcoinWorkflow()
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            await workflow.run(symbols=["BTCUSDT"], account_balance=10000.0)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\nBenchmark - Single Symbol:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Min: {min_time:.2f}ms")
        print(f"  Max: {max_time:.2f}ms")
        
        # Average should be under 100ms
        assert avg_time < 100, f"Average too slow: {avg_time:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_benchmark_all_symbols(self):
        """Benchmark all symbols processing."""
        workflow = BitcoinWorkflow()
        
        times = []
        for _ in range(5):
            start = time.perf_counter()
            await workflow.run(account_balance=10000.0)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        
        print(f"\nBenchmark - All Symbols:")
        print(f"  Average: {avg_time:.2f}ms")
        
        # Should complete under 200ms
        assert avg_time < 200, f"Average too slow: {avg_time:.2f}ms"
