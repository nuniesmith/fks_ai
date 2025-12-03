"""
LLM Module - Model-agnostic LLM clients for fks_ai

Supports:
- Ollama (local, free) - default for development
- Groq (fast, cheap)
- Gemini (free via Google AI Studio)
- Cerebras (ultra-fast)
- OpenAI (GPT-4o, GPT-5)
"""

from .litellm_client import UnifiedLLMClient, llm_client
from .config import LLMConfig, get_llm_config, RECOMMENDED_MODELS

__all__ = [
    "UnifiedLLMClient",
    "llm_client",
    "LLMConfig",
    "get_llm_config",
    "RECOMMENDED_MODELS",
]
