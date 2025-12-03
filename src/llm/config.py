"""
LLM Configuration for fks_ai

Centralizes all LLM-related configuration with sensible defaults.
Supports multiple providers through LiteLLM's unified interface.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    GROQ = "groq"
    GEMINI = "gemini"
    CEREBRAS = "cerebras"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    """
    LLM Configuration.
    
    Environment Variables:
        LITELLM_MODEL: Full model string (e.g., "ollama/llama3.2", "gemini/gemini-1.5-flash")
        OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
        LLM_TEMPERATURE: Sampling temperature (default: 0.1)
        LLM_MAX_TOKENS: Max output tokens (default: 2048)
        
        API Keys (for cloud providers):
        GOOGLE_API_KEY: For Gemini
        GROQ_API_KEY: For Groq
        CEREBRAS_API_KEY: For Cerebras
        OPENAI_API_KEY: For OpenAI
        ANTHROPIC_API_KEY: For Anthropic/Claude
    """
    
    # Model configuration
    model: str = field(default_factory=lambda: os.getenv("LITELLM_MODEL", "ollama/llama3.2:3b"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048")))
    
    # Ollama-specific (for local development)
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    
    # Timeout settings
    timeout: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT", "120")))
    
    # Retry settings
    max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3")))
    
    @property
    def provider(self) -> str:
        """Extract provider from model string (e.g., 'ollama' from 'ollama/llama3.2')."""
        if "/" in self.model:
            return self.model.split("/")[0]
        return "ollama"  # Default to Ollama for backwards compatibility
    
    @property
    def model_name(self) -> str:
        """Extract model name from model string (e.g., 'llama3.2' from 'ollama/llama3.2')."""
        if "/" in self.model:
            return self.model.split("/", 1)[1]
        return self.model
    
    def get_api_base(self) -> Optional[str]:
        """Get API base URL for provider."""
        if self.provider == "ollama":
            return self.ollama_host
        return None  # LiteLLM handles other providers automatically
    
    def to_litellm_kwargs(self) -> Dict[str, Any]:
        """Convert config to LiteLLM completion kwargs."""
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        
        # Add API base for Ollama
        if self.provider == "ollama":
            kwargs["api_base"] = self.ollama_host
        
        return kwargs


# Singleton config instance
_config: Optional[LLMConfig] = None


def get_llm_config() -> LLMConfig:
    """Get the LLM configuration singleton."""
    global _config
    if _config is None:
        _config = LLMConfig()
    return _config


# Provider-specific model recommendations
RECOMMENDED_MODELS = {
    # Local (free, development)
    "ollama/llama3.2": "Fast local model, good for development",
    "ollama/llama3.2:3b": "Smaller/faster variant",
    "ollama/qwen2.5:7b": "Good reasoning, Chinese/English",
    "ollama/mistral:7b": "Fast, good general purpose",
    
    # Groq (cheap, fast)
    "groq/llama-3.3-70b-versatile": "Best Groq model, very fast",
    "groq/llama-3.1-8b-instant": "Ultra-fast, good for simple tasks",
    "groq/mixtral-8x7b-32768": "Good context window",
    
    # Gemini (FREE via Google AI Studio)
    "gemini/gemini-1.5-flash": "Ultra-fast, free tier",
    "gemini/gemini-1.5-pro": "Best reasoning, free tier",
    "gemini/gemini-2.0-flash-exp": "Experimental, newest",
    
    # Cerebras (fastest inference)
    "cerebras/llama3.1-8b": "1800+ tokens/sec",
    "cerebras/llama3.1-70b": "Fast 70B inference",
    
    # OpenAI
    "openai/gpt-4o-mini": "Cheap, fast, good quality",
    "openai/gpt-4o": "Best OpenAI model",
    
    # Anthropic
    "anthropic/claude-3-5-sonnet-20241022": "Best for analysis",
    "anthropic/claude-3-haiku-20240307": "Fast, cheap",
}
