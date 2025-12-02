"""
LLM-Judge System for Agent Reasoning Validation

Meta-evaluation system that uses an LLM to validate agent reasoning quality,
detect hallucinations, identify discrepancies, and analyze systematic biases.

Architecture:
    - Uses Ollama llama3.2:3b as the "judge" LLM
    - Validates agent outputs against ground truth data
    - Detects factual inconsistencies and hallucinations
    - Identifies systematic biases in Bull/Bear agents

Usage:
    >>> judge = LLMJudge()
    >>>
    >>> # Verify factual consistency
    >>> report = await judge.verify_factual_consistency(
    ...     agent_claim="RSI is at 72, indicating overbought conditions",
    ...     market_data={"rsi": 45.2}
    ... )
    >>>
    >>> # Detect discrepancies
    >>> discrepancies = await judge.detect_discrepancies(
    ...     agent_analysis="Strong uptrend with bullish momentum",
    ...     actual_outcome=-5.2  # Price dropped 5.2%
    ... )
    >>>
    >>> # Analyze bias patterns
    >>> bias = await judge.analyze_bias(
    ...     agent_decisions=[...],  # Historical decisions
    ...     market_outcomes=[...]   # Actual price movements
    ... )
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


@dataclass
class ConsistencyReport:
    """
    Report on factual consistency between agent claim and market data.

    Attributes:
        is_consistent: Whether claim matches data
        confidence: Judge confidence (0-1)
        explanation: Detailed reasoning
        discrepancies: List of specific mismatches found
        severity: low/medium/high/critical
        timestamp: When validation occurred
    """
    is_consistent: bool
    confidence: float
    explanation: str
    discrepancies: list[str]
    severity: str  # low, medium, high, critical
    timestamp: datetime
    agent_claim: str
    market_data: dict[str, Any]


@dataclass
class DiscrepancyReport:
    """
    Report on discrepancies between agent analysis and actual outcomes.

    Attributes:
        has_discrepancy: Whether significant mismatch exists
        severity: low/medium/high/critical
        analysis_summary: Agent's prediction/analysis
        actual_outcome: What actually happened
        explanation: Why discrepancy occurred
        confidence: Judge confidence (0-1)
        timestamp: When analysis occurred
    """
    has_discrepancy: bool
    severity: str
    analysis_summary: str
    actual_outcome: Any
    explanation: str
    confidence: float
    timestamp: datetime
    error_type: Optional[str] = None  # hallucination, misinterpretation, logic_error


@dataclass
class BiasReport:
    """
    Report on systematic bias in agent predictions.

    Attributes:
        has_bias: Whether systematic bias detected
        bias_type: optimistic, pessimistic, neutral
        bias_strength: 0-1 (how strong the bias is)
        sample_size: Number of decisions analyzed
        accuracy_rate: Percentage of correct predictions
        false_positive_rate: Rate of incorrect bullish calls
        false_negative_rate: Rate of missed opportunities
        explanation: Detailed bias analysis
        recommendations: How to mitigate bias
        timestamp: When analysis occurred
    """
    has_bias: bool
    bias_type: str  # optimistic, pessimistic, neutral
    bias_strength: float  # 0-1
    sample_size: int
    accuracy_rate: float
    false_positive_rate: float
    false_negative_rate: float
    explanation: str
    recommendations: list[str]
    timestamp: datetime


class LLMJudge:
    """
    LLM-powered judge for validating agent reasoning and detecting issues.

    Uses a meta-LLM (judge) to evaluate other agents' outputs for:
    - Factual consistency with market data
    - Discrepancies between predictions and outcomes
    - Systematic biases in decision-making

    The judge LLM acts as a critical reviewer, cross-referencing agent
    claims against ground truth data and historical performance.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.1,  # Low temp for consistent evaluation
        base_url: Optional[str] = None
    ):
        """
        Initialize LLM-Judge system.

        Args:
            model: Ollama model name for judge LLM
            temperature: Sampling temperature (lower = more consistent)
            base_url: Ollama API URL (defaults to OLLAMA_HOST env var)
        """
        # Use environment variable or fallback
        if base_url is None:
            base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        # Initialize judge LLM with low temperature for consistency
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url
        )

        # Prompt templates for different validation tasks
        self._init_prompts()

    def _init_prompts(self):
        """Initialize prompt templates for different validation tasks."""

        # Factual consistency validation prompt
        self.consistency_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a rigorous fact-checker for trading analysis.

Your job is to verify if an agent's claim matches the actual market data.
Be precise and objective. Flag even minor inconsistencies.

Respond with JSON:
{{
    "is_consistent": true/false,
    "confidence": 0.0-1.0,
    "explanation": "detailed reasoning",
    "discrepancies": ["list", "of", "issues"],
    "severity": "low|medium|high|critical"
}}"""),
            ("human", """Agent Claim: {claim}

Actual Market Data: {data}

Verify if the claim is factually consistent with the data. Consider:
1. Numerical accuracy (values match within reason?)
2. Directional correctness (up/down/neutral aligned?)
3. Indicator interpretation (RSI/MACD/etc. read correctly?)
4. Logical consistency (does the conclusion follow from data?)

Be strict but fair. Output JSON only.""")
        ])

        # Discrepancy detection prompt
        self.discrepancy_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an analyst reviewing trading predictions vs outcomes.

Your job is to identify and explain discrepancies between what an agent
predicted would happen and what actually occurred.

Respond with JSON:
{{
    "has_discrepancy": true/false,
    "severity": "low|medium|high|critical",
    "explanation": "why prediction failed",
    "error_type": "hallucination|misinterpretation|logic_error|null",
    "confidence": 0.0-1.0
}}"""),
            ("human", """Agent Analysis: {analysis}

Actual Outcome: {outcome}

Compare the prediction to reality. Identify:
1. Did the prediction match the outcome?
2. If not, what type of error occurred?
3. Was this a minor miscalculation or major hallucination?
4. Could the agent have known better with the data available?

Output JSON only.""")
        ])

        # Bias analysis prompt
        self.bias_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a statistical analyst detecting systematic biases.

Your job is to analyze a series of trading decisions and outcomes to
detect if an agent has a systematic bias (over-optimism, over-pessimism).

Respond with JSON:
{{
    "has_bias": true/false,
    "bias_type": "optimistic|pessimistic|neutral",
    "bias_strength": 0.0-1.0,
    "explanation": "statistical evidence",
    "recommendations": ["mitigation", "strategies"]
}}"""),
            ("human", """Agent: {agent_name}
Sample Size: {sample_size} decisions

Decisions vs Outcomes:
{decision_outcome_pairs}

Statistics:
- Accuracy: {accuracy}%
- False Positives (wrong bullish): {false_pos}%
- False Negatives (missed opportunities): {false_neg}%

Analyze for systematic bias. Consider:
1. Is there a pattern of over-optimism or over-pessimism?
2. Are false positives/negatives skewed in one direction?
3. Does the agent consistently over/underestimate?

Output JSON only.""")
        ])

    async def verify_factual_consistency(
        self,
        agent_claim: str,
        market_data: dict[str, Any],
        agent_name: Optional[str] = None
    ) -> ConsistencyReport:
        """
        Verify if agent's claim is factually consistent with market data.

        Args:
            agent_claim: What the agent stated/claimed
            market_data: Actual market data (OHLCV, indicators, etc.)
            agent_name: Name of agent being validated (for logging)

        Returns:
            ConsistencyReport with validation results

        Example:
            >>> report = await judge.verify_factual_consistency(
            ...     agent_claim="RSI is at 72, overbought",
            ...     market_data={"rsi": 45.2, "close": 67500}
            ... )
            >>> assert not report.is_consistent
            >>> assert "RSI mismatch" in report.discrepancies
        """
        # Format market data for prompt
        data_str = "\n".join(f"- {k}: {v}" for k, v in market_data.items())

        # Invoke judge LLM
        response = await self.llm.ainvoke(
            self.consistency_prompt.format_messages(
                claim=agent_claim,
                data=data_str
            )
        )

        # Parse JSON response
        import json
        try:
            result = json.loads(response.content)

            return ConsistencyReport(
                is_consistent=result.get("is_consistent", False),
                confidence=result.get("confidence", 0.0),
                explanation=result.get("explanation", ""),
                discrepancies=result.get("discrepancies", []),
                severity=result.get("severity", "unknown"),
                timestamp=datetime.now(),
                agent_claim=agent_claim,
                market_data=market_data
            )
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            return ConsistencyReport(
                is_consistent=False,
                confidence=0.0,
                explanation=f"Judge LLM error: {response.content}",
                discrepancies=["Failed to parse judge response"],
                severity="critical",
                timestamp=datetime.now(),
                agent_claim=agent_claim,
                market_data=market_data
            )

    async def detect_discrepancies(
        self,
        agent_analysis: str,
        actual_outcome: Any,
        context: Optional[dict[str, Any]] = None
    ) -> DiscrepancyReport:
        """
        Detect discrepancies between agent analysis and actual outcomes.

        Args:
            agent_analysis: Agent's prediction/analysis text
            actual_outcome: What actually happened (price change, result, etc.)
            context: Additional context (timeframe, conditions, etc.)

        Returns:
            DiscrepancyReport with findings

        Example:
            >>> report = await judge.detect_discrepancies(
            ...     agent_analysis="Strong bullish momentum, expect +5% rally",
            ...     actual_outcome=-3.2  # Price dropped 3.2%
            ... )
            >>> assert report.has_discrepancy
            >>> assert report.severity == "high"
        """
        # Format outcome
        outcome_str = str(actual_outcome)
        if isinstance(actual_outcome, (int, float)):
            outcome_str = f"{actual_outcome:+.2f}%"

        # Invoke judge LLM
        response = await self.llm.ainvoke(
            self.discrepancy_prompt.format_messages(
                analysis=agent_analysis,
                outcome=outcome_str
            )
        )

        # Parse JSON response
        import json
        try:
            result = json.loads(response.content)

            return DiscrepancyReport(
                has_discrepancy=result.get("has_discrepancy", True),
                severity=result.get("severity", "unknown"),
                analysis_summary=agent_analysis,
                actual_outcome=actual_outcome,
                explanation=result.get("explanation", ""),
                confidence=result.get("confidence", 0.0),
                timestamp=datetime.now(),
                error_type=result.get("error_type")
            )
        except json.JSONDecodeError:
            return DiscrepancyReport(
                has_discrepancy=True,
                severity="critical",
                analysis_summary=agent_analysis,
                actual_outcome=actual_outcome,
                explanation=f"Judge LLM error: {response.content}",
                confidence=0.0,
                timestamp=datetime.now(),
                error_type="judge_error"
            )

    async def analyze_bias(
        self,
        agent_name: str,
        agent_decisions: list[dict[str, Any]],
        market_outcomes: list[float],
        threshold: float = 0.3  # Bias strength threshold
    ) -> BiasReport:
        """
        Analyze systematic bias in agent's decision-making patterns.

        Args:
            agent_name: Name of agent (e.g., "Bull Agent", "Technical Analyst")
            agent_decisions: List of agent decisions with predictions
            market_outcomes: Corresponding actual price movements (%)
            threshold: Bias strength threshold for flagging (0-1)

        Returns:
            BiasReport with bias analysis

        Example:
            >>> decisions = [
            ...     {"prediction": "bullish", "confidence": 0.8},
            ...     {"prediction": "bullish", "confidence": 0.9},
            ...     ...
            ... ]
            >>> outcomes = [2.5, -1.2, 3.1, -0.5, ...]  # Actual % changes
            >>>
            >>> report = await judge.analyze_bias("Bull Agent", decisions, outcomes)
            >>> if report.has_bias and report.bias_type == "optimistic":
            ...     print(f"Agent has {report.bias_strength:.1%} optimistic bias")
        """
        # Calculate statistics
        sample_size = len(agent_decisions)

        # Match predictions to outcomes
        correct = 0
        false_positives = 0  # Predicted up, went down
        false_negatives = 0  # Predicted down, went up

        decision_pairs = []
        for decision, outcome in zip(agent_decisions, market_outcomes, strict=False):
            pred = decision.get("prediction", "").lower()
            actual = "up" if outcome > 0 else "down" if outcome < 0 else "neutral"

            is_correct = (
                (pred in ["bullish", "buy"] and outcome > 0) or
                (pred in ["bearish", "sell"] and outcome < 0) or
                (pred in ["neutral", "hold"] and abs(outcome) < 1)
            )

            if is_correct:
                correct += 1
            elif pred in ["bullish", "buy"] and outcome < 0:
                false_positives += 1
            elif pred in ["bearish", "sell"] and outcome > 0:
                false_negatives += 1

            decision_pairs.append(
                f"Predicted: {pred} (conf: {decision.get('confidence', 0):.2f}) "
                f"→ Actual: {actual} ({outcome:+.2f}%)"
            )

        accuracy = (correct / sample_size * 100) if sample_size > 0 else 0
        false_pos_rate = (false_positives / sample_size * 100) if sample_size > 0 else 0
        false_neg_rate = (false_negatives / sample_size * 100) if sample_size > 0 else 0

        # Format for judge
        pairs_str = "\n".join(decision_pairs[:20])  # Limit to first 20 for context
        if len(decision_pairs) > 20:
            pairs_str += f"\n... and {len(decision_pairs) - 20} more"

        # Invoke judge LLM
        response = await self.llm.ainvoke(
            self.bias_prompt.format_messages(
                agent_name=agent_name,
                sample_size=sample_size,
                decision_outcome_pairs=pairs_str,
                accuracy=f"{accuracy:.1f}",
                false_pos=f"{false_pos_rate:.1f}",
                false_neg=f"{false_neg_rate:.1f}"
            )
        )

        # Parse JSON response
        import json
        try:
            result = json.loads(response.content)
            bias_strength = result.get("bias_strength", 0.0)

            return BiasReport(
                has_bias=bias_strength > threshold,
                bias_type=result.get("bias_type", "neutral"),
                bias_strength=bias_strength,
                sample_size=sample_size,
                accuracy_rate=accuracy,
                false_positive_rate=false_pos_rate,
                false_negative_rate=false_neg_rate,
                explanation=result.get("explanation", ""),
                recommendations=result.get("recommendations", []),
                timestamp=datetime.now()
            )
        except json.JSONDecodeError:
            return BiasReport(
                has_bias=False,
                bias_type="unknown",
                bias_strength=0.0,
                sample_size=sample_size,
                accuracy_rate=accuracy,
                false_positive_rate=false_pos_rate,
                false_negative_rate=false_neg_rate,
                explanation=f"Judge LLM error: {response.content}",
                recommendations=["Fix judge LLM response parsing"],
                timestamp=datetime.now()
            )

    async def validate_agent_batch(
        self,
        agent_outputs: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Batch validation of multiple agent outputs against ground truth.

        Useful for validating a series of agent predictions/analyses
        against actual market data and outcomes.

        Args:
            agent_outputs: List of agent output dicts
            ground_truth: Corresponding ground truth data

        Returns:
            Summary report with consistency/discrepancy statistics
        """
        consistency_results = []
        discrepancy_results = []

        for output, truth in zip(agent_outputs, ground_truth, strict=False):
            # Check factual consistency
            if "claim" in output and "market_data" in truth:
                consistency = await self.verify_factual_consistency(
                    agent_claim=output["claim"],
                    market_data=truth["market_data"]
                )
                consistency_results.append(consistency)

            # Check discrepancies
            if "analysis" in output and "outcome" in truth:
                discrepancy = await self.detect_discrepancies(
                    agent_analysis=output["analysis"],
                    actual_outcome=truth["outcome"]
                )
                discrepancy_results.append(discrepancy)

        # Aggregate results
        total = len(consistency_results)
        consistent = sum(1 for r in consistency_results if r.is_consistent)
        critical_issues = sum(
            1 for r in consistency_results if r.severity == "critical"
        )

        return {
            "total_validations": total,
            "consistent_count": consistent,
            "consistency_rate": (consistent / total * 100) if total > 0 else 0,
            "critical_issues": critical_issues,
            "discrepancies_found": sum(1 for r in discrepancy_results if r.has_discrepancy),
            "consistency_results": consistency_results,
            "discrepancy_results": discrepancy_results,
            "timestamp": datetime.now()
        }
