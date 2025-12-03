"""
LiteLLM Unified Client for fks_ai

Provides a model-agnostic interface for LLM calls.
Supports: Ollama, Groq, Cerebras, Gemini, OpenAI, Anthropic

Usage:
    from llm import llm_client
    
    # Synchronous
    response = llm_client.chat([{"role": "user", "content": "Hello"}])
    
    # Async
    response = await llm_client.achat([{"role": "user", "content": "Hello"}])
    
    # Structured JSON output
    result = await llm_client.astructured_chat(messages, response_schema)

Environment Variables:
    LITELLM_MODEL: Model to use (default: "ollama/llama3.2")
    OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
    LLM_TEMPERATURE: Temperature (default: 0.1)
    LLM_MAX_TOKENS: Max tokens (default: 2048)
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from loguru import logger

# Try to import litellm, fall back to direct Ollama if not available
try:
    from litellm import completion, acompletion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("LiteLLM not installed. Falling back to direct Ollama calls.")

# Fallback to langchain_ollama if litellm not available
try:
    from langchain_ollama import ChatOllama
    LANGCHAIN_OLLAMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_OLLAMA_AVAILABLE = False

from .config import get_llm_config, LLMConfig


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class UnifiedLLMClient:
    """
    Model-agnostic LLM client using LiteLLM.
    
    Falls back to direct Ollama/LangChain if LiteLLM not installed.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM client.
        
        Args:
            config: Optional LLMConfig. Uses singleton if not provided.
        """
        self.config = config or get_llm_config()
        self._langchain_llm = None
        
        logger.info(
            f"UnifiedLLMClient initialized: model={self.config.model}, "
            f"provider={self.config.provider}, litellm={LITELLM_AVAILABLE}"
        )
    
    def _get_langchain_llm(self):
        """Get or create LangChain Ollama LLM (fallback)."""
        if self._langchain_llm is None and LANGCHAIN_OLLAMA_AVAILABLE:
            self._langchain_llm = ChatOllama(
                model=self.config.model_name,
                temperature=self.config.temperature,
                base_url=self.config.ollama_host,
            )
        return self._langchain_llm
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Synchronous chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Override config (temperature, max_tokens, etc.)
        
        Returns:
            Response content string
        """
        if LITELLM_AVAILABLE:
            return self._litellm_chat(messages, **kwargs)
        else:
            return self._fallback_chat(messages, **kwargs)
    
    async def achat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Async chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Override config (temperature, max_tokens, etc.)
        
        Returns:
            Response content string
        """
        if LITELLM_AVAILABLE:
            return await self._litellm_achat(messages, **kwargs)
        else:
            return await self._fallback_achat(messages, **kwargs)
    
    def _litellm_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """LiteLLM synchronous chat."""
        call_kwargs = self.config.to_litellm_kwargs()
        call_kwargs.update(kwargs)
        call_kwargs["messages"] = messages
        
        try:
            response = completion(**call_kwargs)
            content = response.choices[0].message.content
            
            # Log usage if available
            if hasattr(response, 'usage') and response.usage:
                logger.debug(
                    f"LLM usage: {response.usage.prompt_tokens} prompt, "
                    f"{response.usage.completion_tokens} completion"
                )
            
            return content
        except Exception as e:
            logger.error(f"LiteLLM error: {e}")
            raise
    
    async def _litellm_achat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """LiteLLM async chat."""
        call_kwargs = self.config.to_litellm_kwargs()
        call_kwargs.update(kwargs)
        call_kwargs["messages"] = messages
        
        try:
            response = await acompletion(**call_kwargs)
            content = response.choices[0].message.content
            
            # Log usage if available
            if hasattr(response, 'usage') and response.usage:
                logger.debug(
                    f"LLM usage: {response.usage.prompt_tokens} prompt, "
                    f"{response.usage.completion_tokens} completion"
                )
            
            return content
        except Exception as e:
            logger.error(f"LiteLLM async error: {e}")
            raise
    
    def _fallback_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Fallback to LangChain Ollama (sync)."""
        llm = self._get_langchain_llm()
        if llm is None:
            raise RuntimeError("No LLM backend available. Install litellm or langchain-ollama.")
        
        # Convert to LangChain format
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        
        response = llm.invoke(lc_messages)
        return response.content
    
    async def _fallback_achat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Fallback to LangChain Ollama (async)."""
        llm = self._get_langchain_llm()
        if llm is None:
            raise RuntimeError("No LLM backend available. Install litellm or langchain-ollama.")
        
        # Convert to LangChain format
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        
        response = await llm.ainvoke(lc_messages)
        return response.content
    
    async def astructured_chat(
        self,
        messages: List[Dict[str, str]],
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get structured JSON output from LLM.
        
        Args:
            messages: List of message dicts
            response_schema: Optional JSON schema for response format
            **kwargs: Additional LLM kwargs
        
        Returns:
            Parsed JSON dict
        """
        # Add JSON instruction to system message
        json_instruction = "\n\nYou MUST respond with valid JSON only. No other text."
        
        enhanced_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                enhanced_messages.append({
                    "role": "system",
                    "content": msg["content"] + json_instruction
                })
            else:
                enhanced_messages.append(msg)
        
        # If no system message, add one
        if not any(m.get("role") == "system" for m in enhanced_messages):
            enhanced_messages.insert(0, {
                "role": "system",
                "content": "You are a helpful assistant." + json_instruction
            })
        
        response = await self.achat(enhanced_messages, **kwargs)
        
        # Parse JSON from response
        try:
            # Try to extract JSON from response (handle markdown code blocks)
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nResponse: {response}")
            return {"error": "JSON parse failed", "raw_response": response}
    
    def complete(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs
    ) -> str:
        """
        Simple completion with just a prompt string.
        
        Args:
            prompt: User prompt string
            system_prompt: Optional system prompt
            **kwargs: Additional LLM kwargs
        
        Returns:
            Response content string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self.chat(messages, **kwargs)
    
    async def acomplete(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs
    ) -> str:
        """
        Async completion with just a prompt string.
        
        Args:
            prompt: User prompt string
            system_prompt: Optional system prompt
            **kwargs: Additional LLM kwargs
        
        Returns:
            Response content string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return await self.achat(messages, **kwargs)
    
    def switch_model(self, model: str) -> None:
        """
        Switch to a different model at runtime.
        
        Args:
            model: New model string (e.g., "groq/llama-3.3-70b-versatile")
        """
        old_model = self.config.model
        self.config.model = model
        self._langchain_llm = None  # Reset fallback LLM
        logger.info(f"Switched model: {old_model} → {model}")


# Singleton instance
llm_client = UnifiedLLMClient()


# Convenience functions
def chat(messages: List[Dict[str, str]], **kwargs) -> str:
    """Convenience function for sync chat."""
    return llm_client.chat(messages, **kwargs)


async def achat(messages: List[Dict[str, str]], **kwargs) -> str:
    """Convenience function for async chat."""
    return await llm_client.achat(messages, **kwargs)
