"""
Signal Processor

Converts agent decisions into executable trading signals with position sizing,
stop-loss, and take-profit levels.
"""

import re
from datetime import datetime
from typing import Dict, Optional


class SignalProcessor:
    """
    Transforms agent decisions into standardized trading signals.

    Features:
        - Position sizing based on confidence and risk parameters
        - Stop-loss calculation (ATR-based or fixed %)
        - Take-profit targets (risk/reward ratio)
        - Signal validation and sanitization
    """

    def __init__(
        self,
        max_position_size: float = 0.10,  # 10% max per trade
        default_risk_per_trade: float = 0.02,  # 2% risk default
        min_risk_reward_ratio: float = 2.0,  # Min 2:1 R/R
        atr_multiplier_stop: float = 2.0,  # 2x ATR for stop-loss
        atr_multiplier_target: float = 4.0  # 4x ATR for take-profit
    ):
        """
        Initialize signal processor with risk parameters.

        Args:
            max_position_size: Maximum position size as % of capital (0-1)
            default_risk_per_trade: Default risk per trade as % (0-1)
            min_risk_reward_ratio: Minimum risk/reward ratio
            atr_multiplier_stop: ATR multiplier for stop-loss
            atr_multiplier_target: ATR multiplier for take-profit
        """
        self.max_position_size = max_position_size
        self.default_risk = default_risk_per_trade
        self.min_rr_ratio = min_risk_reward_ratio
        self.atr_stop_mult = atr_multiplier_stop
        self.atr_target_mult = atr_multiplier_target

    def process(
        self,
        agent_decision: dict,
        market_data: dict,
        account_size: float = 100000
    ) -> Optional[dict]:
        """
        Convert agent decision to executable signal.

        Args:
            agent_decision: Manager's final decision
            market_data: Current market data (price, ATR, etc.)
            account_size: Account size in dollars

        Returns:
            Trading signal dict or None if invalid

        Example:
            >>> processor = SignalProcessor()
            >>> signal = processor.process(
            ...     agent_decision={'decision': 'BUY', 'confidence': 0.75},
            ...     market_data={'close': 67500, 'atr': 1200},
            ...     account_size=100000
            ... )
        """
        # Extract decision text
        decision_text = agent_decision.get('decision', '')

        # Parse action (BUY/SELL/HOLD)
        action = self._parse_action(decision_text)
        if action == 'HOLD':
            return None  # No signal for HOLD

        # Parse confidence
        confidence = self._parse_confidence(decision_text)

        # Get entry price
        entry_price = market_data.get('close')
        if not entry_price:
            return None

        # Calculate position size
        position_size = self._calculate_position_size(
            confidence=confidence,
            account_size=account_size
        )

        # Calculate stop-loss
        stop_loss = self._calculate_stop_loss(
            entry_price=entry_price,
            action=action,
            atr=market_data.get('atr'),
            decision_text=decision_text
        )

        # Calculate take-profit
        take_profit = self._calculate_take_profit(
            entry_price=entry_price,
            stop_loss=stop_loss,
            action=action,
            atr=market_data.get('atr')
        )

        # Validate risk/reward
        if not self._validate_risk_reward(entry_price, stop_loss, take_profit, action):
            return None  # Risk/reward too poor

        # Build signal
        signal = {
            'symbol': market_data.get('symbol', 'UNKNOWN'),
            'action': action,
            'entry_price': entry_price,
            'position_size': position_size,
            'position_size_pct': position_size / account_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'risk_dollars': abs(entry_price - stop_loss) * (position_size / entry_price),
            'reward_dollars': abs(take_profit - entry_price) * (position_size / entry_price),
            'risk_reward_ratio': self._calc_rr_ratio(entry_price, stop_loss, take_profit, action),
            'reasoning': decision_text,
            'timestamp': datetime.now().isoformat(),
            'account_size': account_size
        }

        return signal

    def _parse_action(self, decision_text: str) -> str:
        """Extract BUY/SELL/HOLD from decision text."""
        text_upper = decision_text.upper()

        if 'BUY' in text_upper or 'LONG' in text_upper:
            return 'BUY'
        elif 'SELL' in text_upper or 'SHORT' in text_upper:
            return 'SELL'
        else:
            return 'HOLD'

    def _parse_confidence(self, decision_text: str) -> float:
        """Extract confidence score from decision text."""
        # Look for patterns like "Confidence: 0.75" or "75% confidence"
        patterns = [
            r'confidence[:\s]+(\d+\.?\d*)',
            r'(\d+)%\s+confidence',
            r'confidence[:\s]+(\d+)%'
        ]

        for pattern in patterns:
            match = re.search(pattern, decision_text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                # Normalize to 0-1
                if value > 1:
                    value = value / 100
                return max(0.0, min(1.0, value))

        # Default confidence
        return 0.5

    def _calculate_position_size(self, confidence: float, account_size: float) -> float:
        """
        Calculate position size based on confidence.

        Formula: position_size = account_size * risk_per_trade * confidence
        Capped at max_position_size
        """
        risk_adjusted = self.default_risk * confidence
        position_pct = min(risk_adjusted, self.max_position_size)
        return account_size * position_pct

    def _calculate_stop_loss(
        self,
        entry_price: float,
        action: str,
        atr: Optional[float],
        decision_text: str
    ) -> float:
        """
        Calculate stop-loss level.

        Priority:
        1. Explicit stop in decision text
        2. ATR-based (2x ATR)
        3. Fixed 2% default
        """
        # Try to parse explicit stop from decision
        stop_patterns = [
            r'stop[-\s]?loss[:\s]+\$?(\d+\.?\d*)',
            r'stop[:\s]+\$?(\d+\.?\d*)',
            r'sl[:\s]+\$?(\d+\.?\d*)'
        ]

        for pattern in stop_patterns:
            match = re.search(pattern, decision_text, re.IGNORECASE)
            if match:
                return float(match.group(1))

        # ATR-based stop
        if atr:
            if action == 'BUY':
                return entry_price - (atr * self.atr_stop_mult)
            else:  # SELL
                return entry_price + (atr * self.atr_stop_mult)

        # Fixed % stop (2% default)
        if action == 'BUY':
            return entry_price * 0.98
        else:  # SELL
            return entry_price * 1.02

    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        action: str,
        atr: Optional[float]
    ) -> float:
        """
        Calculate take-profit target.

        Uses minimum risk/reward ratio (default 2:1)
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * self.min_rr_ratio

        if action == 'BUY':
            return entry_price + reward
        else:  # SELL
            return entry_price - reward

    def _validate_risk_reward(
        self,
        entry: float,
        stop: float,
        target: float,
        action: str
    ) -> bool:
        """Validate risk/reward ratio meets minimum threshold."""
        rr_ratio = self._calc_rr_ratio(entry, stop, target, action)
        return rr_ratio >= self.min_rr_ratio

    def _calc_rr_ratio(
        self,
        entry: float,
        stop: float,
        target: float,
        action: str
    ) -> float:
        """Calculate risk/reward ratio."""
        risk = abs(entry - stop)
        reward = abs(target - entry)

        if risk == 0:
            return 0.0

        return reward / risk

    def batch_process(
        self,
        decisions: list,
        market_data_list: list,
        account_size: float = 100000
    ) -> list:
        """
        Process multiple decisions in batch.

        Args:
            decisions: List of agent decisions
            market_data_list: List of market data dicts
            account_size: Account size

        Returns:
            List of signals (excluding None/HOLD)
        """
        signals = []

        for decision, market_data in zip(decisions, market_data_list, strict=False):
            signal = self.process(decision, market_data, account_size)
            if signal:
                signals.append(signal)

        return signals
