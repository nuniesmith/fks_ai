"""
Ground Truth Validation Framework

Compares agent predictions against optimal hindsight trades to measure
actual prediction quality using historical data.

Architecture:
    - Fetches agent predictions from ChromaDB memory
    - Calculates optimal trades using TimescaleDB price data
    - Compares predictions vs optimal trades
    - Generates confusion matrices and performance metrics

Key Metrics:
    - Accuracy: % of correct predictions
    - Precision: % of buy signals that were correct
    - Recall: % of opportunities captured
    - F1 Score: Harmonic mean of precision/recall
    - Missed Opportunities: % of optimal trades not predicted
    - False Signals: % of incorrect predictions

Usage:
    >>> validator = GroundTruthValidator()
    >>>
    >>> # Validate agent predictions vs reality
    >>> report = await validator.validate_agent(
    ...     agent_name="Bull",
    ...     symbol="BTCUSDT",
    ...     start_date=datetime(2024, 1, 1),
    ...     end_date=datetime(2024, 12, 31),
    ...     timeframe="1d"
    ... )
    >>>
    >>> print(f"Accuracy: {report.accuracy:.1%}")
    >>> print(f"F1 Score: {report.f1_score:.2f}")
    >>> print(f"Missed Opportunities: {report.missed_opportunities}")
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class PredictionType(Enum):
    """Agent prediction types"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class TradeOutcome(Enum):
    """Actual market outcomes"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass
class AgentPrediction:
    """
    Historical agent prediction from memory.

    Attributes:
        timestamp: When prediction was made
        agent_name: Name of agent (Bull, Bear, Manager, etc.)
        symbol: Trading pair (BTCUSDT, ETHUSD, etc.)
        prediction: Predicted direction (bullish/bearish/neutral)
        confidence: Confidence score (0-1)
        reasoning: Agent's explanation
        timeframe: Prediction horizon (1h, 4h, 1d)
        price_at_prediction: Market price when prediction made
    """
    timestamp: datetime
    agent_name: str
    symbol: str
    prediction: PredictionType
    confidence: float
    reasoning: str
    timeframe: str
    price_at_prediction: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimalTrade:
    """
    Optimal trade calculated with perfect hindsight.

    Attributes:
        entry_time: When trade should have started
        exit_time: When trade should have closed
        direction: Long or short
        entry_price: Entry price
        exit_price: Exit price
        profit_percent: Profit/loss percentage
        max_profit_percent: Maximum profit available in period
        slippage_percent: Estimated slippage cost
        fee_percent: Trading fees
        net_profit_percent: Profit after costs
    """
    entry_time: datetime
    exit_time: datetime
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    profit_percent: float
    max_profit_percent: float
    slippage_percent: float = 0.1  # 0.1% default
    fee_percent: float = 0.1  # 0.1% default

    @property
    def net_profit_percent(self) -> float:
        """Profit after slippage and fees"""
        return self.profit_percent - self.slippage_percent - (2 * self.fee_percent)


@dataclass
class ValidationResult:
    """
    Ground truth validation result comparing predictions to reality.

    Attributes:
        agent_name: Name of agent validated
        symbol: Trading pair
        start_date: Validation period start
        end_date: Validation period end
        timeframe: Prediction timeframe
        total_predictions: Number of predictions made
        correct_predictions: Number of correct predictions
        accuracy: Percentage of correct predictions
        precision: Precision score (TP / (TP + FP))
        recall: Recall score (TP / (TP + FN))
        f1_score: F1 score (harmonic mean)
        confusion_matrix: 2D array [[TN, FP], [FN, TP]]
        missed_opportunities: Optimal trades not predicted
        false_signals: Incorrect predictions
        avg_confidence_correct: Average confidence on correct predictions
        avg_confidence_incorrect: Average confidence on wrong predictions
        profit_if_followed: Profit % if all predictions were followed
        profit_if_optimal: Profit % if optimal trades were made
        efficiency_ratio: actual_profit / optimal_profit
    """
    agent_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    timeframe: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    missed_opportunities: int
    false_signals: int
    avg_confidence_correct: float
    avg_confidence_incorrect: float
    profit_if_followed: float
    profit_if_optimal: float
    efficiency_ratio: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "agent_name": self.agent_name,
            "symbol": self.symbol,
            "period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
                "timeframe": self.timeframe
            },
            "predictions": {
                "total": self.total_predictions,
                "correct": self.correct_predictions,
                "accuracy": round(self.accuracy, 4)
            },
            "metrics": {
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1_score": round(self.f1_score, 4)
            },
            "confusion_matrix": {
                "true_negatives": int(self.confusion_matrix[0, 0]),
                "false_positives": int(self.confusion_matrix[0, 1]),
                "false_negatives": int(self.confusion_matrix[1, 0]),
                "true_positives": int(self.confusion_matrix[1, 1])
            },
            "errors": {
                "missed_opportunities": self.missed_opportunities,
                "false_signals": self.false_signals
            },
            "confidence": {
                "avg_when_correct": round(self.avg_confidence_correct, 4),
                "avg_when_wrong": round(self.avg_confidence_incorrect, 4)
            },
            "profitability": {
                "if_followed_agent": round(self.profit_if_followed, 2),
                "if_optimal_trades": round(self.profit_if_optimal, 2),
                "efficiency_ratio": round(self.efficiency_ratio, 4)
            },
            "timestamp": self.timestamp.isoformat()
        }


class GroundTruthValidator:
    """
    Validates agent predictions against optimal hindsight trades.

    Compares what agents predicted vs what perfect trader would have done.
    Generates confusion matrices, accuracy metrics, and profitability analysis.

    Example:
        >>> validator = GroundTruthValidator()
        >>>
        >>> # Validate Bull agent on BTC for 2024
        >>> report = await validator.validate_agent(
        ...     agent_name="Bull",
        ...     symbol="BTCUSDT",
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 12, 31),
        ...     timeframe="1d"
        ... )
        >>>
        >>> # Check if agent meets quality threshold
        >>> if report.accuracy > 0.7 and report.f1_score > 0.65:
        ...     print(f"✅ Agent passed validation")
        ... else:
        ...     print(f"❌ Agent needs improvement")
    """

    def __init__(
        self,
        min_confidence: float = 0.0,
        profit_threshold: float = 1.0,  # Minimum 1% profit to count as opportunity
        slippage: float = 0.1,
        fees: float = 0.1
    ):
        """
        Initialize ground truth validator.

        Args:
            min_confidence: Minimum confidence to include prediction (0-1)
            profit_threshold: Minimum profit % to count as trading opportunity
            slippage: Slippage percentage (default 0.1%)
            fees: Trading fee percentage per trade (default 0.1%)
        """
        self.min_confidence = min_confidence
        self.profit_threshold = profit_threshold
        self.slippage = slippage
        self.fees = fees

        # Will be initialized when needed
        self.memory = None
        self.db_connection = None

    async def validate_agent(
        self,
        agent_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> ValidationResult:
        """
        Validate agent predictions against optimal hindsight trades.

        Args:
            agent_name: Name of agent to validate (Bull, Bear, Manager, etc.)
            symbol: Trading pair (BTCUSDT, ETHUSD, etc.)
            start_date: Start of validation period
            end_date: End of validation period
            timeframe: Prediction horizon (1h, 4h, 1d)

        Returns:
            ValidationResult with accuracy, confusion matrix, profitability

        Example:
            >>> report = await validator.validate_agent(
            ...     agent_name="Bull",
            ...     symbol="BTCUSDT",
            ...     start_date=datetime(2024, 1, 1),
            ...     end_date=datetime(2024, 6, 30),
            ...     timeframe="1d"
            ... )
            >>> print(f"Accuracy: {report.accuracy:.1%}")
            >>> print(f"F1: {report.f1_score:.2f}")
        """
        # Step 1: Collect historical predictions
        predictions = await self._collect_historical_predictions(
            agent_name=agent_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )

        if not predictions:
            raise ValueError(f"No predictions found for {agent_name} on {symbol}")

        # Step 2: Calculate optimal trades with hindsight
        optimal_trades = await self._calculate_optimal_trades(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )

        # Step 3: Compare predictions to optimal trades
        comparison = await self._compare_predictions_to_optimal(
            predictions=predictions,
            optimal_trades=optimal_trades,
            timeframe=timeframe
        )

        # Step 4: Generate validation result with metrics
        result = self._generate_validation_result(
            agent_name=agent_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            predictions=predictions,
            comparison=comparison
        )

        return result

    async def _collect_historical_predictions(
        self,
        agent_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> list[AgentPrediction]:
        """
        Collect historical agent predictions from ChromaDB memory.

        Queries memory for all predictions made by agent in time period.
        Filters by minimum confidence threshold.

        Args:
            agent_name: Agent to query
            symbol: Trading pair
            start_date: Period start
            end_date: Period end
            timeframe: Prediction horizon

        Returns:
            List of AgentPrediction objects
        """
        # Initialize ChromaDB memory if needed
        if self.memory is None:
            from memory import TradingMemory
            self.memory = TradingMemory()

        predictions = []

        try:
            # Query ChromaDB for agent decisions in date range
            # Use semantic search with agent/symbol context
            query = f"{agent_name} {symbol} trading decision {timeframe}"

            # Get large result set (ChromaDB returns most similar first)
            results = self.memory.query_similar(
                query=query,
                n_results=1000  # Fetch up to 1000 predictions
            )

            # Parse results and filter by date range
            for result in results:
                metadata = result.get('metadata', {})

                # Skip if not matching our criteria
                if metadata.get('agent') != agent_name:
                    continue
                if metadata.get('symbol') != symbol:
                    continue
                if metadata.get('timeframe') != timeframe:
                    continue

                # Parse timestamp
                timestamp_str = metadata.get('timestamp')
                if not timestamp_str:
                    continue

                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                # Filter by date range
                if not (start_date <= timestamp <= end_date):
                    continue

                # Parse prediction type
                decision = metadata.get('decision', '').lower()
                if 'bull' in decision or 'buy' in decision or 'long' in decision:
                    prediction_type = PredictionType.BULLISH
                elif 'bear' in decision or 'sell' in decision or 'short' in decision:
                    prediction_type = PredictionType.BEARISH
                else:
                    prediction_type = PredictionType.NEUTRAL

                # Extract confidence (default to 0.5 if not found)
                confidence = float(metadata.get('confidence', 0.5))

                # Filter by minimum confidence
                if confidence < self.min_confidence:
                    continue

                # Create AgentPrediction object
                prediction = AgentPrediction(
                    timestamp=timestamp,
                    agent_name=agent_name,
                    symbol=symbol,
                    prediction=prediction_type,
                    confidence=confidence,
                    reasoning=result.get('document', ''),
                    timeframe=timeframe,
                    price_at_prediction=float(metadata.get('price', 0.0)),
                    metadata=metadata
                )

                predictions.append(prediction)

            # Sort by timestamp
            predictions.sort(key=lambda p: p.timestamp)

        except Exception as e:
            # Log error but don't fail completely
            print(f"Warning: Error querying ChromaDB: {e}")
            # Return empty list if query fails
            return []

        return predictions

    async def _calculate_optimal_trades(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> list[OptimalTrade]:
        """
        Calculate optimal trades with perfect hindsight.

        Uses historical price data to find best entry/exit points.
        Accounts for slippage and fees.

        Args:
            symbol: Trading pair
            start_date: Period start
            end_date: Period end
            timeframe: Trade duration

        Returns:
            List of OptimalTrade objects showing perfect trades
        """
        # Initialize database connection if needed
        if self.db_connection is None:
            # Import here to avoid circular dependency
            import os

            import psycopg2

            self.db_connection = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DB", "trading_db"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "")
            )

        optimal_trades = []

        try:
            # Parse timeframe to hours
            timeframe_hours = self._parse_timeframe_to_hours(timeframe)

            # Query OHLCV data from TimescaleDB
            # Assuming we have a 'market_data' hypertable
            query = """
                SELECT time, open, high, low, close, volume
                FROM market_data
                WHERE symbol = %s
                  AND time >= %s
                  AND time <= %s
                ORDER BY time ASC
            """

            cursor = self.db_connection.cursor()
            cursor.execute(query, (symbol, start_date, end_date))
            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                return []

            # Convert to list of OHLCV tuples
            ohlcv_data = [
                {
                    'time': row[0],
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4]),
                    'volume': float(row[5])
                }
                for row in rows
            ]

            # Find optimal trades by identifying significant price movements
            i = 0
            while i < len(ohlcv_data) - 1:
                entry_candle = ohlcv_data[i]
                entry_time = entry_candle['time']
                entry_price = entry_candle['close']

                # Look ahead for the timeframe duration
                exit_index = self._find_exit_index(i, len(ohlcv_data), timeframe_hours, ohlcv_data)

                if exit_index is None:
                    break

                # Calculate max profit in this window (both long and short)
                max_high = max(c['high'] for c in ohlcv_data[i:exit_index + 1])
                min_low = min(c['low'] for c in ohlcv_data[i:exit_index + 1])

                long_profit = ((max_high - entry_price) / entry_price) * 100
                short_profit = ((entry_price - min_low) / entry_price) * 100

                # Only create trade if profit exceeds threshold (after costs)
                if long_profit - self.slippage - (2 * self.fees) > self.profit_threshold:
                    # Find the candle with the high
                    exit_candle = max(
                        ohlcv_data[i:exit_index + 1],
                        key=lambda c: c['high']
                    )

                    optimal_trades.append(OptimalTrade(
                        entry_time=entry_time,
                        exit_time=exit_candle['time'],
                        direction="long",
                        entry_price=entry_price,
                        exit_price=exit_candle['high'],
                        profit_percent=long_profit,
                        max_profit_percent=long_profit,
                        slippage_percent=self.slippage,
                        fee_percent=self.fees
                    ))

                elif short_profit - self.slippage - (2 * self.fees) > self.profit_threshold:
                    # Find the candle with the low
                    exit_candle = min(
                        ohlcv_data[i:exit_index + 1],
                        key=lambda c: c['low']
                    )

                    optimal_trades.append(OptimalTrade(
                        entry_time=entry_time,
                        exit_time=exit_candle['time'],
                        direction="short",
                        entry_price=entry_price,
                        exit_price=exit_candle['low'],
                        profit_percent=short_profit,
                        max_profit_percent=short_profit,
                        slippage_percent=self.slippage,
                        fee_percent=self.fees
                    ))

                # Move to next potential entry (non-overlapping windows)
                i = exit_index + 1

        except Exception as e:
            print(f"Warning: Error calculating optimal trades: {e}")
            return []

        return optimal_trades

    def _parse_timeframe_to_hours(self, timeframe: str) -> int:
        """Convert timeframe string to hours"""
        timeframe = timeframe.lower()
        if 'h' in timeframe:
            return int(timeframe.replace('h', ''))
        elif 'd' in timeframe:
            return int(timeframe.replace('d', '')) * 24
        elif 'w' in timeframe:
            return int(timeframe.replace('w', '')) * 24 * 7
        else:
            return 24  # Default to 1 day

    def _find_exit_index(
        self,
        start_index: int,
        total_length: int,
        timeframe_hours: int,
        ohlcv_data: list[dict]
    ) -> Optional[int]:
        """Find the index of the candle at the exit time"""
        entry_time = ohlcv_data[start_index]['time']
        target_exit_time = entry_time + timedelta(hours=timeframe_hours)

        # Find closest candle to target exit time
        for i in range(start_index + 1, total_length):
            if ohlcv_data[i]['time'] >= target_exit_time:
                return i

        return None  # No exit found in range

    async def _compare_predictions_to_optimal(
        self,
        predictions: list[AgentPrediction],
        optimal_trades: list[OptimalTrade],
        timeframe: str
    ) -> dict[str, Any]:
        """
        Compare agent predictions to optimal trades.

        Matches predictions to optimal trades by timestamp proximity.
        Calculates true positives, false positives, false negatives.

        Args:
            predictions: Agent predictions
            optimal_trades: Optimal trades
            timeframe: Time window for matching

        Returns:
            Dictionary with comparison results
        """
        tp = fp = tn = fn = 0
        correct_predictions = []
        incorrect_predictions = []
        missed_opportunities = []

        # Create time window for matching (±30 minutes default)
        time_window = timedelta(minutes=30)

        # Match predictions to optimal trades
        matched_trades = set()  # Track which trades have been matched

        for prediction in predictions:
            pred_time = prediction.timestamp
            pred_type = prediction.prediction

            # Find matching optimal trade (if any)
            matching_trade = None
            min_time_diff = time_window.total_seconds()

            for i, trade in enumerate(optimal_trades):
                if i in matched_trades:
                    continue

                # Check if prediction time is close to trade entry
                time_diff = abs((trade.entry_time - pred_time).total_seconds())
                if time_diff <= min_time_diff:
                    min_time_diff = time_diff
                    matching_trade = (i, trade)

            if matching_trade:
                trade_idx, trade = matching_trade
                matched_trades.add(trade_idx)

                # Check if agent predicted correctly
                if trade.direction == "long":
                    # Optimal was to go long
                    if pred_type == PredictionType.BULLISH:
                        tp += 1
                        correct_predictions.append({
                            "prediction": prediction,
                            "optimal_trade": trade,
                            "outcome": "correct_bullish"
                        })
                    elif pred_type == PredictionType.BEARISH:
                        fn += 1
                        incorrect_predictions.append({
                            "prediction": prediction,
                            "optimal_trade": trade,
                            "outcome": "missed_long"
                        })
                    else:  # NEUTRAL
                        fn += 1
                        incorrect_predictions.append({
                            "prediction": prediction,
                            "optimal_trade": trade,
                            "outcome": "neutral_missed_long"
                        })

                elif trade.direction == "short":
                    # Optimal was to go short
                    if pred_type == PredictionType.BEARISH:
                        tn += 1
                        correct_predictions.append({
                            "prediction": prediction,
                            "optimal_trade": trade,
                            "outcome": "correct_bearish"
                        })
                    elif pred_type == PredictionType.BULLISH:
                        fp += 1
                        incorrect_predictions.append({
                            "prediction": prediction,
                            "optimal_trade": trade,
                            "outcome": "wrong_direction"
                        })
                    else:  # NEUTRAL
                        tn += 1
                        correct_predictions.append({
                            "prediction": prediction,
                            "optimal_trade": trade,
                            "outcome": "neutral_avoided_short"
                        })

            else:
                # No optimal trade found - market was sideways/unprofitable
                if pred_type == PredictionType.BULLISH:
                    fp += 1
                    incorrect_predictions.append({
                        "prediction": prediction,
                        "optimal_trade": None,
                        "outcome": "false_bullish_signal"
                    })
                elif pred_type == PredictionType.BEARISH:
                    fp += 1
                    incorrect_predictions.append({
                        "prediction": prediction,
                        "optimal_trade": None,
                        "outcome": "false_bearish_signal"
                    })
                else:  # NEUTRAL
                    tn += 1
                    correct_predictions.append({
                        "prediction": prediction,
                        "optimal_trade": None,
                        "outcome": "correct_neutral"
                    })

        # Find missed opportunities (optimal trades with no matching prediction)
        for i, trade in enumerate(optimal_trades):
            if i not in matched_trades:
                fn += 1
                missed_opportunities.append({
                    "optimal_trade": trade,
                    "reason": "no_prediction_made"
                })

        return {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "correct_predictions": correct_predictions,
            "incorrect_predictions": incorrect_predictions,
            "missed_opportunities": missed_opportunities
        }

    def _generate_validation_result(
        self,
        agent_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        predictions: list[AgentPrediction],
        comparison: dict[str, Any]
    ) -> ValidationResult:
        """
        Generate validation result from comparison data.

        Calculates accuracy, precision, recall, F1, confusion matrix.
        Computes profitability metrics.

        Args:
            agent_name: Agent name
            symbol: Trading pair
            start_date: Period start
            end_date: Period end
            timeframe: Timeframe
            predictions: All predictions
            comparison: Comparison results

        Returns:
            ValidationResult with all metrics
        """
        # Extract metrics from comparison
        tp = comparison["true_positives"]
        fp = comparison["false_positives"]
        tn = comparison["true_negatives"]
        fn = comparison["false_negatives"]

        total = tp + fp + tn + fn
        correct = tp + tn

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Build confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])

        # Calculate confidence averages
        correct_preds = comparison["correct_predictions"]
        incorrect_preds = comparison["incorrect_predictions"]

        avg_conf_correct = np.mean([p.confidence for p in correct_preds]) if correct_preds else 0.0
        avg_conf_incorrect = np.mean([p.confidence for p in incorrect_preds]) if incorrect_preds else 0.0

        # TODO: Calculate actual profitability
        profit_if_followed = 0.0
        profit_if_optimal = 0.0
        efficiency_ratio = 0.0

        return ValidationResult(
            agent_name=agent_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            total_predictions=len(predictions),
            correct_predictions=correct,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            missed_opportunities=fn,
            false_signals=fp,
            avg_confidence_correct=avg_conf_correct,
            avg_confidence_incorrect=avg_conf_incorrect,
            profit_if_followed=profit_if_followed,
            profit_if_optimal=profit_if_optimal,
            efficiency_ratio=efficiency_ratio
        )

    async def validate_multiple_agents(
        self,
        agent_names: list[str],
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> dict[str, ValidationResult]:
        """
        Validate multiple agents in parallel.

        Args:
            agent_names: List of agent names
            symbol: Trading pair
            start_date: Period start
            end_date: Period end
            timeframe: Prediction horizon

        Returns:
            Dictionary mapping agent names to ValidationResult

        Example:
            >>> results = await validator.validate_multiple_agents(
            ...     agent_names=["Bull", "Bear", "Manager"],
            ...     symbol="BTCUSDT",
            ...     start_date=datetime(2024, 1, 1),
            ...     end_date=datetime(2024, 12, 31)
            ... )
            >>>
            >>> for agent, result in results.items():
            ...     print(f"{agent}: {result.accuracy:.1%} accuracy")
        """
        tasks = [
            self.validate_agent(agent, symbol, start_date, end_date, timeframe)
            for agent in agent_names
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            agent: result
            for agent, result in zip(agent_names, results, strict=False)
            if not isinstance(result, Exception)
        }
