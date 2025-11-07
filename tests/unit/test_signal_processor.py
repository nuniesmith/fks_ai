"""
Unit tests for SignalProcessor

Tests decision parsing, position sizing, risk management
"""
from datetime import datetime

import pytest


def test_signal_processor_initialization():
    """Test SignalProcessor with default parameters"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    assert processor.max_position_size == 0.10  # 10%
    assert processor.default_risk_per_trade == 0.02  # 2%
    assert processor.min_risk_reward_ratio == 2.0
    assert processor.atr_stop_multiplier == 2.0
    assert processor.atr_target_multiplier == 3.0


def test_signal_processor_custom_params():
    """Test SignalProcessor with custom risk parameters"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor(
        max_position_size=0.05,
        default_risk_per_trade=0.01,
        min_risk_reward_ratio=3.0
    )

    assert processor.max_position_size == 0.05
    assert processor.default_risk_per_trade == 0.01
    assert processor.min_risk_reward_ratio == 3.0


def test_parse_action_buy():
    """Test parsing BUY action from text"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    texts = [
        "RECOMMENDATION: BUY",
        "Action: BUY BTCUSDT",
        "Final decision: BUY with 70% confidence",
        "buy signal detected"
    ]

    for text in texts:
        action = processor._parse_action(text)
        assert action == 'BUY', f"Failed to parse BUY from: {text}"


def test_parse_action_sell():
    """Test parsing SELL action from text"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    texts = [
        "RECOMMENDATION: SELL",
        "Action: SELL ETHUSDT",
        "Final decision: SELL with 65% confidence",
        "sell signal triggered"
    ]

    for text in texts:
        action = processor._parse_action(text)
        assert action == 'SELL', f"Failed to parse SELL from: {text}"


def test_parse_action_hold():
    """Test parsing HOLD action from text"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    texts = [
        "RECOMMENDATION: HOLD",
        "Action: HOLD position",
        "Final decision: HOLD, wait for clarity",
        "No clear signal, hold"
    ]

    for text in texts:
        action = processor._parse_action(text)
        assert action == 'HOLD', f"Failed to parse HOLD from: {text}"


def test_parse_confidence_percentage():
    """Test parsing confidence as percentage"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    test_cases = [
        ("Confidence: 75%", 0.75),
        ("CONFIDENCE: 62%", 0.62),
        ("confidence of 80%", 0.80),
        ("50% confidence level", 0.50)
    ]

    for text, expected in test_cases:
        confidence = processor._parse_confidence(text)
        assert confidence == expected, f"Failed: {text} -> {confidence} (expected {expected})"


def test_parse_confidence_decimal():
    """Test parsing confidence as decimal"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    test_cases = [
        ("Confidence: 0.75", 0.75),
        ("confidence of 0.62", 0.62),
        ("Confidence 0.8", 0.80)
    ]

    for text, expected in test_cases:
        confidence = processor._parse_confidence(text)
        assert confidence == expected


def test_parse_confidence_default():
    """Test default confidence when not found"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    confidence = processor._parse_confidence("No confidence mentioned")
    assert confidence == 0.5  # Default


def test_calculate_position_size_basic():
    """Test position size calculation"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor(
        max_position_size=0.10,
        default_risk_per_trade=0.02
    )

    # 100k account, 2% risk, 70% confidence
    size = processor._calculate_position_size(
        account_size=100000,
        confidence=0.70
    )

    # Expected: 100k * 0.02 * 0.70 = 1,400
    assert size == 1400.0


def test_calculate_position_size_capped():
    """Test position size is capped at max"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor(max_position_size=0.10)

    # Very high confidence should cap at 10%
    size = processor._calculate_position_size(
        account_size=100000,
        confidence=1.0  # 100% confidence
    )

    max_size = 100000 * 0.10
    assert size <= max_size


def test_calculate_stop_loss_explicit():
    """Test stop-loss from explicit value in decision"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    decision = "BUY at 68200, STOP: 67400"

    stop = processor._calculate_stop_loss(
        decision=decision,
        entry_price=68200.0,
        market_data={'technical': {'atr': 400.0}}
    )

    assert stop == 67400.0


def test_calculate_stop_loss_atr_based():
    """Test ATR-based stop-loss calculation"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor(atr_stop_multiplier=2.0)

    decision = "BUY signal"
    entry = 68200.0
    atr = 400.0

    stop = processor._calculate_stop_loss(
        decision=decision,
        entry_price=entry,
        market_data={'technical': {'atr': atr}}
    )

    # Expected: 68200 - (2 * 400) = 67400
    expected = entry - (2.0 * atr)
    assert stop == expected


def test_calculate_stop_loss_fixed_percentage():
    """Test fixed 2% stop-loss when no ATR"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    decision = "BUY signal"
    entry = 68200.0

    stop = processor._calculate_stop_loss(
        decision=decision,
        entry_price=entry,
        market_data={}  # No ATR
    )

    # Expected: 68200 * 0.98 = 66836
    expected = entry * 0.98
    assert stop == expected


def test_calculate_take_profit_explicit():
    """Test take-profit from explicit value"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    decision = "BUY at 68200, TARGET: 70000"

    target = processor._calculate_take_profit(
        decision=decision,
        entry_price=68200.0,
        stop_loss=67400.0
    )

    assert target == 70000.0


def test_calculate_take_profit_risk_reward():
    """Test risk/reward based take-profit"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor(min_risk_reward_ratio=2.0)

    decision = "BUY signal"
    entry = 68200.0
    stop = 67400.0

    target = processor._calculate_take_profit(
        decision=decision,
        entry_price=entry,
        stop_loss=stop
    )

    # Risk = 68200 - 67400 = 800
    # Reward = 800 * 2.0 = 1600
    # Target = 68200 + 1600 = 69800
    expected = entry + ((entry - stop) * 2.0)
    assert target == expected


def test_validate_risk_reward_pass():
    """Test risk/reward validation passes"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor(min_risk_reward_ratio=2.0)

    is_valid = processor._validate_risk_reward(
        entry_price=68200.0,
        stop_loss=67400.0,
        take_profit=69800.0
    )

    # Risk = 800, Reward = 1600, R/R = 2.0
    assert is_valid is True


def test_validate_risk_reward_fail():
    """Test risk/reward validation fails"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor(min_risk_reward_ratio=2.0)

    is_valid = processor._validate_risk_reward(
        entry_price=68200.0,
        stop_loss=67400.0,
        take_profit=68600.0  # Only 0.5:1 R/R
    )

    assert is_valid is False


def test_process_buy_signal_complete(sample_manager_decision, sample_market_data):
    """Test full BUY signal processing"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    signal = processor.process(
        decision=sample_manager_decision,
        symbol='BTCUSDT',
        account_size=100000,
        market_data=sample_market_data
    )

    assert signal['action'] == 'BUY'
    assert signal['symbol'] == 'BTCUSDT'
    assert 0.0 < signal['confidence'] <= 1.0
    assert signal['entry_price'] > 0
    assert signal['stop_loss'] < signal['entry_price']
    assert signal['take_profit'] > signal['entry_price']
    assert signal['position_size'] > 0
    assert signal['risk_reward_ratio'] >= 2.0
    assert 'timestamp' in signal


def test_process_hold_signal():
    """Test HOLD signal processing"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    decision = "HOLD position, wait for better entry"

    signal = processor.process(
        decision=decision,
        symbol='BTCUSDT',
        account_size=100000,
        market_data={}
    )

    assert signal['action'] == 'HOLD'
    assert signal['position_size'] == 0
    assert signal['entry_price'] is None
    assert signal['stop_loss'] is None
    assert signal['take_profit'] is None


def test_batch_process_multiple_decisions(sample_bull_argument, sample_bear_argument):
    """Test batch processing multiple decisions"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    decisions = [
        {
            'decision': sample_bull_argument,
            'symbol': 'BTCUSDT',
            'account_size': 100000,
            'market_data': {'technical': {'atr': 400.0}}
        },
        {
            'decision': sample_bear_argument,
            'symbol': 'ETHUSDT',
            'account_size': 100000,
            'market_data': {'technical': {'atr': 50.0}}
        }
    ]

    signals = processor.batch_process(decisions)

    assert len(signals) == 2
    assert signals[0]['symbol'] == 'BTCUSDT'
    assert signals[1]['symbol'] == 'ETHUSDT'


def test_process_with_missing_market_data():
    """Test processing with minimal market data"""
    from src.services.ai.src.processors.signal_processor import SignalProcessor

    processor = SignalProcessor()

    decision = "BUY BTCUSDT with 70% confidence"

    signal = processor.process(
        decision=decision,
        symbol='BTCUSDT',
        account_size=100000,
        market_data={}  # No technical data
    )

    # Should still work with defaults
    assert signal['action'] == 'BUY'
    assert signal['stop_loss'] is not None
    assert signal['take_profit'] is not None
