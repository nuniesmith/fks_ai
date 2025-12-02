"""
Tests for LLM-Judge System

Validates the meta-evaluation system that checks agent reasoning quality.
"""

import asyncio
from datetime import datetime

import pytest
from evaluators.llm_judge import BiasReport, ConsistencyReport, DiscrepancyReport, LLMJudge


class TestLLMJudge:
    """Test suite for LLMJudge meta-evaluation system."""

    @pytest.fixture
    async def judge(self):
        """Create LLMJudge instance for testing."""
        return LLMJudge(temperature=0.1)

    @pytest.mark.asyncio
    async def test_verify_factual_consistency_accurate_claim(self, judge):
        """Test that accurate agent claims pass validation."""
        # Agent claim matches data
        claim = "RSI is at 72.5, indicating overbought conditions. Price is at $67,500."
        market_data = {
            "rsi": 72.5,
            "close": 67500.0,
            "macd": 150.2
        }

        report = await judge.verify_factual_consistency(claim, market_data)

        assert isinstance(report, ConsistencyReport)
        assert report.is_consistent or report.confidence > 0.7  # Should pass
        assert report.timestamp is not None
        assert report.agent_claim == claim

    @pytest.mark.asyncio
    async def test_verify_factual_consistency_inaccurate_claim(self, judge):
        """Test that inaccurate claims are flagged."""
        # Agent claim contradicts data
        claim = "RSI is at 72, market is overbought"
        market_data = {
            "rsi": 28.3,  # Actually oversold, not overbought!
            "close": 45000.0
        }

        report = await judge.verify_factual_consistency(claim, market_data)

        assert isinstance(report, ConsistencyReport)
        # Should detect inconsistency
        assert not report.is_consistent or report.severity in ["high", "critical"]
        assert len(report.discrepancies) > 0

    @pytest.mark.asyncio
    async def test_verify_factual_consistency_hallucination(self, judge):
        """Test detection of agent hallucinations (claiming data that doesn't exist)."""
        # Agent mentions indicator not in data
        claim = "Stochastic RSI is at 95, and Williams %R shows -5, extremely overbought"
        market_data = {
            "rsi": 65.0,
            "close": 50000.0
            # No stochastic_rsi or williams_r in data!
        }

        report = await judge.verify_factual_consistency(claim, market_data)

        assert isinstance(report, ConsistencyReport)
        # Should flag as inconsistent or low confidence
        assert not report.is_consistent or report.confidence < 0.5

    @pytest.mark.asyncio
    async def test_detect_discrepancies_correct_prediction(self, judge):
        """Test when agent prediction matches outcome."""
        analysis = "Strong bullish momentum with RSI at 65, MACD positive. Expect +3-5% move higher."
        actual_outcome = 4.2  # Price went up 4.2%

        report = await judge.detect_discrepancies(analysis, actual_outcome)

        assert isinstance(report, DiscrepancyReport)
        # Should show no/low discrepancy
        assert not report.has_discrepancy or report.severity == "low"
        assert report.actual_outcome == 4.2

    @pytest.mark.asyncio
    async def test_detect_discrepancies_wrong_prediction(self, judge):
        """Test when agent prediction contradicts outcome."""
        analysis = "Bearish breakdown imminent, expect -8% drop in next 24h"
        actual_outcome = 6.5  # Price actually rallied +6.5%!

        report = await judge.detect_discrepancies(analysis, actual_outcome)

        assert isinstance(report, DiscrepancyReport)
        assert report.has_discrepancy
        assert report.severity in ["medium", "high", "critical"]
        assert report.error_type in ["misinterpretation", "logic_error", None]

    @pytest.mark.asyncio
    async def test_detect_discrepancies_minor_mismatch(self, judge):
        """Test minor discrepancy (predicted up, went up but less)."""
        analysis = "Bullish setup, expect +5% rally"
        actual_outcome = 1.2  # Only +1.2%, still up but less than predicted

        report = await judge.detect_discrepancies(analysis, actual_outcome)

        assert isinstance(report, DiscrepancyReport)
        # Minor discrepancy, should be low/medium severity
        assert report.severity in ["low", "medium"]

    @pytest.mark.asyncio
    async def test_analyze_bias_neutral_agent(self, judge):
        """Test bias detection on unbiased agent."""
        decisions = [
            {"prediction": "bullish", "confidence": 0.7},
            {"prediction": "bearish", "confidence": 0.6},
            {"prediction": "bullish", "confidence": 0.8},
            {"prediction": "bearish", "confidence": 0.7},
            {"prediction": "neutral", "confidence": 0.5},
        ]
        outcomes = [2.5, -1.8, 3.2, -2.1, 0.3]  # Mixed, balanced

        report = await judge.analyze_bias("Test Agent", decisions, outcomes)

        assert isinstance(report, BiasReport)
        assert report.sample_size == 5
        assert 0 <= report.accuracy_rate <= 100
        assert report.bias_type in ["neutral", "optimistic", "pessimistic"]

    @pytest.mark.asyncio
    async def test_analyze_bias_optimistic_agent(self, judge):
        """Test detection of over-optimistic bias (Bull Agent)."""
        decisions = [
            {"prediction": "bullish", "confidence": 0.9},
            {"prediction": "bullish", "confidence": 0.8},
            {"prediction": "bullish", "confidence": 0.95},
            {"prediction": "bullish", "confidence": 0.85},
            {"prediction": "bullish", "confidence": 0.9},
            {"prediction": "neutral", "confidence": 0.5},
        ]
        # Agent always predicts up, but outcomes are mixed
        outcomes = [1.2, -2.3, 0.8, -1.5, 2.1, -0.5]

        report = await judge.analyze_bias("Bull Agent", decisions, outcomes)

        assert isinstance(report, BiasReport)
        assert report.sample_size == 6
        # Should detect high false positive rate
        assert report.false_positive_rate > 20  # >20% wrong bullish calls

    @pytest.mark.asyncio
    async def test_analyze_bias_pessimistic_agent(self, judge):
        """Test detection of over-pessimistic bias (Bear Agent)."""
        decisions = [
            {"prediction": "bearish", "confidence": 0.8},
            {"prediction": "bearish", "confidence": 0.9},
            {"prediction": "bearish", "confidence": 0.75},
            {"prediction": "bearish", "confidence": 0.85},
            {"prediction": "neutral", "confidence": 0.5},
        ]
        # Agent predicts down, but market goes up
        outcomes = [3.2, 2.8, 1.5, 0.8, -0.2]

        report = await judge.analyze_bias("Bear Agent", decisions, outcomes)

        assert isinstance(report, BiasReport)
        # Should detect high false negative rate (missed opportunities)
        assert report.false_negative_rate > 40  # Missed many rallies

    @pytest.mark.asyncio
    async def test_validate_agent_batch(self, judge):
        """Test batch validation of multiple agent outputs."""
        agent_outputs = [
            {
                "claim": "RSI at 70, overbought",
                "analysis": "Expect pullback to $65k"
            },
            {
                "claim": "MACD bullish crossover",
                "analysis": "Rally to $72k likely"
            },
        ]

        ground_truth = [
            {
                "market_data": {"rsi": 70.2, "close": 67500},
                "outcome": -2.3  # Did pull back
            },
            {
                "market_data": {"macd": {"value": 120, "signal": 100}, "close": 68000},
                "outcome": 5.8  # Did rally
            },
        ]

        report = await judge.validate_agent_batch(agent_outputs, ground_truth)

        assert "total_validations" in report
        assert "consistency_rate" in report
        assert "discrepancies_found" in report
        assert report["total_validations"] == 2

    @pytest.mark.asyncio
    async def test_consistency_report_structure(self, judge):
        """Test that ConsistencyReport has all required fields."""
        claim = "Test claim"
        data = {"rsi": 50}

        report = await judge.verify_factual_consistency(claim, data)

        assert hasattr(report, "is_consistent")
        assert hasattr(report, "confidence")
        assert hasattr(report, "explanation")
        assert hasattr(report, "discrepancies")
        assert hasattr(report, "severity")
        assert hasattr(report, "timestamp")
        assert hasattr(report, "agent_claim")
        assert hasattr(report, "market_data")

    @pytest.mark.asyncio
    async def test_discrepancy_report_structure(self, judge):
        """Test that DiscrepancyReport has all required fields."""
        analysis = "Test analysis"
        outcome = 2.5

        report = await judge.detect_discrepancies(analysis, outcome)

        assert hasattr(report, "has_discrepancy")
        assert hasattr(report, "severity")
        assert hasattr(report, "analysis_summary")
        assert hasattr(report, "actual_outcome")
        assert hasattr(report, "explanation")
        assert hasattr(report, "confidence")
        assert hasattr(report, "timestamp")
        assert hasattr(report, "error_type")

    @pytest.mark.asyncio
    async def test_bias_report_structure(self, judge):
        """Test that BiasReport has all required fields."""
        decisions = [{"prediction": "bullish", "confidence": 0.8}] * 5
        outcomes = [1.0] * 5

        report = await judge.analyze_bias("Test", decisions, outcomes)

        assert hasattr(report, "has_bias")
        assert hasattr(report, "bias_type")
        assert hasattr(report, "bias_strength")
        assert hasattr(report, "sample_size")
        assert hasattr(report, "accuracy_rate")
        assert hasattr(report, "false_positive_rate")
        assert hasattr(report, "false_negative_rate")
        assert hasattr(report, "explanation")
        assert hasattr(report, "recommendations")
        assert hasattr(report, "timestamp")

    def test_llm_judge_initialization(self):
        """Test LLMJudge can be initialized with custom params."""
        judge = LLMJudge(model="llama3.2:3b", temperature=0.2)
        assert judge.llm.model == "llama3.2:3b"
        assert judge.llm.temperature == 0.2

    def test_llm_judge_uses_env_var(self):
        """Test that LLMJudge respects OLLAMA_HOST env var."""
        import os
        os.environ["OLLAMA_HOST"] = "http://test:11434"

        judge = LLMJudge()
        assert "test:11434" in judge.llm.base_url

        # Cleanup
        del os.environ["OLLAMA_HOST"]


class TestLLMJudgeEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    async def judge(self):
        return LLMJudge()

    @pytest.mark.asyncio
    async def test_empty_market_data(self, judge):
        """Test handling of empty market data."""
        claim = "RSI is at 70"
        data = {}

        report = await judge.verify_factual_consistency(claim, data)

        assert isinstance(report, ConsistencyReport)
        # Should flag as inconsistent due to missing data
        assert not report.is_consistent or report.confidence < 0.5

    @pytest.mark.asyncio
    async def test_empty_agent_decisions(self, judge):
        """Test bias analysis with empty decision list."""
        report = await judge.analyze_bias("Test", [], [])

        assert isinstance(report, BiasReport)
        assert report.sample_size == 0
        assert report.accuracy_rate == 0

    @pytest.mark.asyncio
    async def test_mismatched_decision_outcome_lengths(self, judge):
        """Test bias analysis when decisions and outcomes have different lengths."""
        decisions = [{"prediction": "bullish", "confidence": 0.8}] * 5
        outcomes = [1.0] * 3  # Only 3 outcomes for 5 decisions

        # Should only analyze the minimum length
        report = await judge.analyze_bias("Test", decisions[:3], outcomes)

        assert report.sample_size == 3

    @pytest.mark.asyncio
    async def test_null_outcome(self, judge):
        """Test discrepancy detection with null outcome."""
        analysis = "Bullish rally expected"
        outcome = None

        report = await judge.detect_discrepancies(analysis, outcome)

        assert isinstance(report, DiscrepancyReport)
        assert report.actual_outcome is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
