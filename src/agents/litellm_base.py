"""
LiteLLM-based Agent Factory for Multi-Agent Trading System

Creates model-agnostic agents using the UnifiedLLMClient.
Can use any provider: Ollama (local), Groq, Gemini, Cerebras, OpenAI.

Usage:
    from agents.litellm_base import create_litellm_agent, LiteLLMAgent
    
    # Create agent
    agent = create_litellm_agent(
        role="Technical Analyst",
        system_prompt="You analyze charts...",
        temperature=0.3
    )
    
    # Use agent
    response = await agent.ainvoke("Analyze BTCUSDT")
    
    # Or with structured output
    result = await agent.astructured_invoke("Analyze BTCUSDT", response_schema={...})
"""

import os
import re
import json
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from loguru import logger

# Import the unified LLM client
try:
    from llm import llm_client, UnifiedLLMClient, get_llm_config
    LITELLM_MODULE_AVAILABLE = True
except ImportError:
    LITELLM_MODULE_AVAILABLE = False
    logger.warning("LLM module not available, using fallback")


# Global confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class AgentConfig:
    """Configuration for a LiteLLM agent."""
    role: str
    system_prompt: str
    temperature: float = 0.7
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD
    max_tokens: int = 2048


class LiteLLMAgent:
    """
    Model-agnostic trading agent using LiteLLM.
    
    Supports any model that LiteLLM supports:
    - ollama/llama3.2 (local, free)
    - groq/llama-3.3-70b-versatile (fast, cheap)
    - gemini/gemini-1.5-pro (free, excellent reasoning)
    - openai/gpt-4o (best quality)
    """
    
    def __init__(
        self,
        role: str,
        system_prompt: str,
        temperature: float = 0.7,
        min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_tokens: int = 2048,
        client: Optional[UnifiedLLMClient] = None
    ):
        """
        Initialize a LiteLLM agent.
        
        Args:
            role: Agent role identifier
            system_prompt: System prompt defining agent behavior
            temperature: Sampling temperature (0-1)
            min_confidence: Minimum confidence threshold (0-1)
            max_tokens: Maximum output tokens
            client: Optional LLM client (uses singleton if not provided)
        """
        self.role = role
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.max_tokens = max_tokens
        self.client = client or llm_client
        
        # Build enhanced system prompt with confidence guidelines
        self.system_prompt = self._build_system_prompt(system_prompt)
        
        logger.debug(f"LiteLLMAgent created: role={role}, temp={temperature}")
    
    def _build_system_prompt(self, base_prompt: str) -> str:
        """Build enhanced system prompt with confidence guidelines."""
        return f"""Role: {self.role}

{base_prompt}

**CONFIDENCE THRESHOLD**: {self.min_confidence}
You MUST provide a confidence score (0-1) with your analysis.
- Only provide recommendations if your confidence >= {self.min_confidence}
- If confidence < {self.min_confidence}, state "INSUFFICIENT CONFIDENCE" and explain why
- Be honest about uncertainty - better to skip than make low-confidence trades
- Consider data quality, signal strength, and market conditions when assessing confidence

**Output Format**:
Always include a "Confidence: X.XX" line in your response (e.g., "Confidence: 0.75")
"""
    
    def invoke(self, input_text: str, **kwargs) -> str:
        """
        Synchronous agent invocation.
        
        Args:
            input_text: User input/query
            **kwargs: Additional LLM kwargs
        
        Returns:
            Agent response string
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
        
        return self.client.chat(
            messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens)
        )
    
    async def ainvoke(self, input_text: str, **kwargs) -> str:
        """
        Async agent invocation.
        
        Args:
            input_text: User input/query
            **kwargs: Additional LLM kwargs
        
        Returns:
            Agent response string
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
        
        return await self.client.achat(
            messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens)
        )
    
    async def astructured_invoke(
        self,
        input_text: str,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Async invocation with structured JSON output.
        
        Args:
            input_text: User input/query
            response_schema: Optional JSON schema for response
            **kwargs: Additional LLM kwargs
        
        Returns:
            Parsed JSON response dict
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
        
        return await self.client.astructured_chat(
            messages,
            response_schema=response_schema,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens)
        )
    
    def with_conversation(self, messages: List[Dict[str, str]]) -> "ConversationAgent":
        """
        Create a conversation-aware version of this agent.
        
        Args:
            messages: Existing conversation history
        
        Returns:
            ConversationAgent wrapping this agent
        """
        return ConversationAgent(self, messages)


class ConversationAgent:
    """Agent wrapper that maintains conversation history."""
    
    def __init__(self, agent: LiteLLMAgent, history: Optional[List[Dict[str, str]]] = None):
        self.agent = agent
        self.history = history or []
        
        # Ensure system message is first
        if not self.history or self.history[0].get("role") != "system":
            self.history.insert(0, {"role": "system", "content": agent.system_prompt})
    
    async def ainvoke(self, input_text: str, **kwargs) -> str:
        """Add to conversation and get response."""
        self.history.append({"role": "user", "content": input_text})
        
        response = await self.agent.client.achat(
            self.history,
            temperature=kwargs.get("temperature", self.agent.temperature),
            max_tokens=kwargs.get("max_tokens", self.agent.max_tokens)
        )
        
        self.history.append({"role": "assistant", "content": response})
        return response
    
    def clear_history(self):
        """Clear conversation history (keep system message)."""
        self.history = [self.history[0]] if self.history else []


def create_litellm_agent(
    role: str,
    system_prompt: str,
    temperature: float = 0.7,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_tokens: int = 2048,
    **kwargs
) -> LiteLLMAgent:
    """
    Factory function for creating LiteLLM-based trading agents.
    
    Args:
        role: Agent role identifier (e.g., "Technical Analyst", "Bull Agent")
        system_prompt: System prompt defining agent behavior
        temperature: Sampling temperature (0-1, default: 0.7)
        min_confidence: Minimum confidence threshold (0-1, default: 0.6)
        max_tokens: Maximum output tokens (default: 2048)
        **kwargs: Additional agent kwargs
    
    Returns:
        Configured LiteLLMAgent
    
    Example:
        >>> agent = create_litellm_agent(
        ...     role="Technical Analyst",
        ...     system_prompt="You analyze technical indicators...",
        ...     temperature=0.3
        ... )
        >>> response = await agent.ainvoke("Analyze BTCUSDT daily chart")
    """
    return LiteLLMAgent(
        role=role,
        system_prompt=system_prompt,
        temperature=temperature,
        min_confidence=min_confidence,
        max_tokens=max_tokens,
        **kwargs
    )


def extract_confidence(text: str) -> Optional[float]:
    """
    Extract confidence score from agent response.
    
    Args:
        text: Agent response text
    
    Returns:
        Confidence score (0-1) if found, None otherwise
    """
    # Try multiple patterns
    patterns = [
        r'[Cc]onfidence:\s*([0-9]*\.?[0-9]+)',
        r'"confidence":\s*([0-9]*\.?[0-9]+)',
        r'confidence["\s:]+([0-9]*\.?[0-9]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                confidence = float(match.group(1))
                if 0 <= confidence <= 1:
                    return confidence
                elif 0 <= confidence <= 100:
                    return confidence / 100  # Convert percentage
            except ValueError:
                continue
    
    return None


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from agent response.
    
    Handles:
    - Raw JSON
    - JSON in markdown code blocks
    - JSON mixed with text
    
    Args:
        text: Agent response text
    
    Returns:
        Parsed JSON dict or None
    """
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON object
    try:
        # Try full text first
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in text
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None
