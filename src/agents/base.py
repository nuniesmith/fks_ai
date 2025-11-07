"""
Base Agent Factory for Multi-Agent Trading System

Creates Ollama-based agents with configurable prompts and parameters.
Supports confidence thresholds for quality control.
"""

import os
import re
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama


# Global confidence threshold (minimum for accepting agent outputs)
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


def create_agent(
    role: str,
    system_prompt: str,
    model: str = "llama3.2:3b",
    temperature: float = 0.7,
    base_url: Optional[str] = None,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Runnable:
    """
    Factory function for creating specialized trading agents.

    Args:
        role: Agent role identifier (e.g., "Technical Analyst", "Bull Agent")
        system_prompt: System prompt defining agent behavior and expertise
        model: Ollama model name (default: llama3.2:3b)
        temperature: Sampling temperature (0-1, higher = more creative)
        base_url: Ollama API base URL (defaults to OLLAMA_HOST env var or localhost)
        min_confidence: Minimum confidence threshold (0-1, default: 0.6)
                       Agents should self-evaluate and only provide high-confidence outputs

    Returns:
        Configured LangChain Runnable (prompt | LLM chain)

    Example:
        >>> technical_agent = create_agent(
        ...     role="Technical Analyst",
        ...     system_prompt="You are a technical analyst...",
        ...     temperature=0.5,
        ...     min_confidence=0.7
        ... )
        >>> response = await technical_agent.ainvoke({"input": "Analyze BTCUSDT"})
    """
    # Use environment variable or fallback to localhost
    if base_url is None:
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Initialize Ollama LLM
    llm = ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url
    )

    # Enhanced system prompt with confidence guidelines
    enhanced_prompt = f"""{system_prompt}

**CONFIDENCE THRESHOLD**: {min_confidence}
You MUST provide a confidence score (0-1) with your analysis.
- Only provide recommendations if your confidence >= {min_confidence}
- If confidence < {min_confidence}, state "INSUFFICIENT CONFIDENCE" and explain why
- Be honest about uncertainty - better to skip than make low-confidence trades
- Consider data quality, signal strength, and market conditions when assessing confidence

**Output Format**:
Always include a "Confidence: X.XX" line in your response (e.g., "Confidence: 0.75")
"""

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Role: {role}\n\n{enhanced_prompt}"),
        ("human", "{input}")
    ])

    # Return LangChain runnable (prompt | LLM)
    return prompt | llm


def extract_confidence(text: str) -> Optional[float]:
    """
    Extract confidence score from agent response.

    Args:
        text: Agent response text

    Returns:
        Confidence score (0-1) if found, None otherwise

    Examples:
        >>> extract_confidence("Confidence: 0.75")
        0.75
        >>> extract_confidence("75% confident")
        0.75
        >>> extract_confidence("High confidence (0.82)")
        0.82
    """
    # Pattern 1: "Confidence: 0.XX" or "Confidence: X.XX"
    pattern1 = r'Confidence:\s*(\d+\.?\d*)'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        conf = float(match.group(1))
        return conf if conf <= 1.0 else conf / 100

    # Pattern 2: "XX% confident/confidence"
    pattern2 = r'(\d+)%\s*confiden'
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100

    # Pattern 3: Decimal in parentheses "(0.XX)"
    pattern3 = r'\((\d+\.?\d*)\)'
    match = re.search(pattern3, text)
    if match:
        conf = float(match.group(1))
        if 0 <= conf <= 1:
            return conf

    return None


def validate_confidence_threshold(
    response_text: str,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Dict[str, Any]:
    """
    Validate agent response meets confidence threshold.

    Args:
        response_text: Agent's response
        min_confidence: Minimum acceptable confidence

    Returns:
        Dict with validation results:
        - meets_threshold: bool
        - confidence: float or None
        - is_valid: bool
        - reason: str (if invalid)

    Example:
        >>> result = validate_confidence_threshold("Analysis: BUY\nConfidence: 0.75", 0.6)
        >>> result['meets_threshold']
        True
    """
    confidence = extract_confidence(response_text)

    if confidence is None:
        return {
            "meets_threshold": False,
            "confidence": None,
            "is_valid": False,
            "reason": "No confidence score found in response"
        }

    if "INSUFFICIENT CONFIDENCE" in response_text.upper():
        return {
            "meets_threshold": False,
            "confidence": confidence,
            "is_valid": True,
            "reason": f"Agent self-reported insufficient confidence ({confidence:.2f})"
        }

    meets_threshold = confidence >= min_confidence

    return {
        "meets_threshold": meets_threshold,
        "confidence": confidence,
        "is_valid": True,
        "reason": None if meets_threshold else f"Confidence {confidence:.2f} below threshold {min_confidence:.2f}"
    }


def create_structured_agent(
    role: str,
    system_prompt: str,
    output_schema: Optional[dict] = None,
    model: str = "llama3.2:3b",
    temperature: float = 0.7,
    base_url: Optional[str] = None
) -> Runnable:
    """
    Create agent with structured output (JSON schema validation).

    Args:
        role: Agent role identifier
        system_prompt: System prompt
        output_schema: Pydantic model or dict schema for structured output
        model: Ollama model name
        temperature: Sampling temperature
        base_url: Ollama API base URL (defaults to OLLAMA_HOST env var or localhost)

    Returns:
        Configured agent with structured output parsing

    Note:
        Structured output requires Ollama >=0.3.0 and model support
    """
    # Use environment variable or fallback to localhost
    if base_url is None:
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    llm = ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
        format="json" if output_schema else None
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Role: {role}\n\n{system_prompt}\n\nOutput JSON only."),
        ("human", "{input}")
    ])

    return prompt | llm
