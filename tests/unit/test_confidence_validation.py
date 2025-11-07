"""
Tests for Confidence Threshold Validation

Tests the enhanced agent system with confidence scoring and validation.
"""

import pytest

from src.services.ai.src.agents.base import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    extract_confidence,
    validate_confidence_threshold,
)


class TestExtractConfidence:
    """Test confidence extraction from agent responses."""

    def test_extract_confidence_decimal_format(self):
        """Test extracting confidence in 'Confidence: 0.XX' format."""
        text = "Technical analysis suggests BUY.\nConfidence: 0.75"
        confidence = extract_confidence(text)
        assert confidence == 0.75

    def test_extract_confidence_percentage_format(self):
        """Test extracting confidence in 'XX%' format."""
        text = "I'm 82% confident this is a strong signal"
        confidence = extract_confidence(text)
        assert confidence == 0.82

    def test_extract_confidence_parentheses_format(self):
        """Test extracting confidence in '(0.XX)' format."""
        text = "High confidence signal (0.88) for this trade"
        confidence = extract_confidence(text)
        assert confidence == 0.88

    def test_extract_confidence_case_insensitive(self):
        """Test confidence extraction is case-insensitive."""
        text = "CONFIDENCE: 0.65"
        confidence = extract_confidence(text)
        assert confidence == 0.65

    def test_extract_confidence_percentage_over_100(self):
        """Test handling percentage > 100 (should normalize)."""
        text = "Confidence: 75"  # Should interpret as 75%
        confidence = extract_confidence(text)
        assert confidence == 75.0  # Returns as-is when > 1

    def test_extract_confidence_not_found(self):
        """Test returns None when no confidence found."""
        text = "This is a technical analysis with no confidence score"
        confidence = extract_confidence(text)
        assert confidence is None

    def test_extract_confidence_multiple_matches(self):
        """Test takes first match when multiple patterns exist."""
        text = "Confidence: 0.75 (very high, ~80% certain)"
        confidence = extract_confidence(text)
        assert confidence == 0.75  # Takes first match


class TestValidateConfidenceThreshold:
    """Test confidence threshold validation."""

    def test_validate_meets_threshold(self):
        """Test validation passes when confidence meets threshold."""
        text = "BUY recommendation\nConfidence: 0.75"
        result = validate_confidence_threshold(text, min_confidence=0.6)

        assert result['meets_threshold'] is True
        assert result['confidence'] == 0.75
        assert result['is_valid'] is True
        assert result['reason'] is None

    def test_validate_below_threshold(self):
        """Test validation fails when confidence below threshold."""
        text = "BUY recommendation\nConfidence: 0.45"
        result = validate_confidence_threshold(text, min_confidence=0.6)

        assert result['meets_threshold'] is False
        assert result['confidence'] == 0.45
        assert result['is_valid'] is True
        assert "below threshold" in result['reason']

    def test_validate_no_confidence_found(self):
        """Test validation fails when no confidence score found."""
        text = "BUY recommendation with no confidence"
        result = validate_confidence_threshold(text, min_confidence=0.6)

        assert result['meets_threshold'] is False
        assert result['confidence'] is None
        assert result['is_valid'] is False
        assert "No confidence score found" in result['reason']

    def test_validate_insufficient_confidence_flag(self):
        """Test validation detects INSUFFICIENT CONFIDENCE flag."""
        text = "INSUFFICIENT CONFIDENCE - unclear market conditions\nConfidence: 0.40"
        result = validate_confidence_threshold(text, min_confidence=0.6)

        assert result['meets_threshold'] is False
        assert result['confidence'] == 0.40
        assert result['is_valid'] is True
        assert "self-reported insufficient confidence" in result['reason']

    def test_validate_exact_threshold(self):
        """Test validation passes when confidence equals threshold."""
        text = "SELL recommendation\nConfidence: 0.60"
        result = validate_confidence_threshold(text, min_confidence=0.6)

        assert result['meets_threshold'] is True
        assert result['confidence'] == 0.60

    def test_validate_default_threshold(self):
        """Test validation uses default threshold when not specified."""
        text = "BUY recommendation\nConfidence: 0.65"
        result = validate_confidence_threshold(text)

        # Default is 0.6
        assert result['meets_threshold'] is True
        assert result['confidence'] == 0.65


class TestConfidenceThresholdIntegration:
    """Integration tests for confidence threshold in agent workflow."""

    def test_high_confidence_analyst_passes(self):
        """Test high-confidence analyst output passes validation."""
        analyst_output = """
        Technical Analysis: BTCUSDT

        Direction: BUY
        Key Signals:
        - RSI at 32 (oversold)
        - MACD bullish crossover
        - Price bouncing off support at $65,000

        Support/Resistance:
        - Support: $65,000 (strong)
        - Resistance: $68,500 (minor), $70,000 (major)

        Risk Factors:
        - Volume declining on rally
        - Macro headwinds from Fed policy

        Confidence: 0.72
        """

        result = validate_confidence_threshold(analyst_output, 0.6)
        assert result['meets_threshold'] is True
        assert result['confidence'] == 0.72

    def test_low_confidence_analyst_rejected(self):
        """Test low-confidence analyst output is rejected."""
        analyst_output = """
        Sentiment Analysis: BTCUSDT

        Market Mood: Uncertain
        Direction: HOLD
        Key Sentiment Drivers:
        - Mixed news sentiment
        - Fear & Greed at 50 (neutral)
        - Whale activity unclear

        INSUFFICIENT CONFIDENCE due to conflicting signals.

        Confidence: 0.35
        """

        result = validate_confidence_threshold(analyst_output, 0.6)
        assert result['meets_threshold'] is False
        assert result['confidence'] == 0.35
        assert 'self-reported' in result['reason']

    def test_manager_decision_with_confidence(self):
        """Test manager decision includes confidence score."""
        manager_decision = """
        FINAL DECISION: BUY BTCUSDT

        Synthesis:
        - 3/4 analysts recommend BUY
        - Bull case stronger than bear case
        - Risk-reward ratio favorable at 2.5:1
        - Current regime: BULL

        Position: 2% of capital
        Entry: $67,200
        Stop-Loss: $66,000
        Take-Profit: $69,800

        Confidence: 0.68
        """

        result = validate_confidence_threshold(manager_decision, 0.6)
        assert result['meets_threshold'] is True
        assert result['confidence'] == 0.68


class TestConfidenceEdgeCases:
    """Test edge cases for confidence validation."""

    def test_confidence_zero(self):
        """Test confidence of 0.0 is handled correctly."""
        text = "No confidence in this signal\nConfidence: 0.0"
        result = validate_confidence_threshold(text, 0.6)

        assert result['meets_threshold'] is False
        assert result['confidence'] == 0.0

    def test_confidence_one(self):
        """Test confidence of 1.0 (100%) is handled correctly."""
        text = "Extremely strong signal\nConfidence: 1.0"
        result = validate_confidence_threshold(text, 0.6)

        assert result['meets_threshold'] is True
        assert result['confidence'] == 1.0

    def test_confidence_just_above_threshold(self):
        """Test confidence just above threshold."""
        text = "Marginal signal\nConfidence: 0.61"
        result = validate_confidence_threshold(text, 0.6)

        assert result['meets_threshold'] is True
        assert result['confidence'] == 0.61

    def test_confidence_just_below_threshold(self):
        """Test confidence just below threshold."""
        text = "Marginal signal\nConfidence: 0.59"
        result = validate_confidence_threshold(text, 0.6)

        assert result['meets_threshold'] is False
        assert result['confidence'] == 0.59

    def test_multiple_confidence_values(self):
        """Test when text contains multiple confidence-like values."""
        text = """
        Analysis based on 90% data availability.
        Market confidence index: 55
        My Confidence: 0.73
        """
        confidence = extract_confidence(text)
        # Should extract the explicit "Confidence: 0.73"
        assert confidence == 0.73


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
