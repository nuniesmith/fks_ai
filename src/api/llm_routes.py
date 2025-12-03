"""
LLM API Routes for fks_ai

Provides endpoints to:
- Test LLM connectivity
- List available models
- Switch models at runtime
- Get current configuration
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from loguru import logger

# Import LLM module
try:
    from llm import llm_client, get_llm_config, RECOMMENDED_MODELS
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM module not available")


router = APIRouter(prefix="/llm", tags=["llm"])


class ChatRequest(BaseModel):
    """Chat request model."""
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    content: str
    model: str
    provider: str


class SwitchModelRequest(BaseModel):
    """Model switch request."""
    model: str


class LLMConfigResponse(BaseModel):
    """LLM configuration response."""
    model: str
    provider: str
    temperature: float
    max_tokens: int
    ollama_host: str
    litellm_available: bool


class ModelInfo(BaseModel):
    """Model information."""
    model: str
    description: str
    provider: str


@router.get("/health")
async def llm_health():
    """Check LLM service health."""
    if not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM module not available")
    
    config = get_llm_config()
    
    # Quick connectivity test
    try:
        response = llm_client.chat([
            {"role": "user", "content": "Say 'OK' if you can hear me."}
        ], max_tokens=10)
        
        return {
            "status": "healthy",
            "model": config.model,
            "provider": config.provider,
            "test_response": response[:50]
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM not responding: {str(e)}"
        )


@router.get("/config", response_model=LLMConfigResponse)
async def get_config():
    """Get current LLM configuration."""
    if not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM module not available")
    
    config = get_llm_config()
    return LLMConfigResponse(
        model=config.model,
        provider=config.provider,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        ollama_host=config.ollama_host,
        litellm_available=True
    )


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List recommended models."""
    if not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM module not available")
    
    models = []
    for model, description in RECOMMENDED_MODELS.items():
        provider = model.split("/")[0] if "/" in model else "ollama"
        models.append(ModelInfo(
            model=model,
            description=description,
            provider=provider
        ))
    
    return models


@router.post("/switch", response_model=LLMConfigResponse)
async def switch_model(request: SwitchModelRequest):
    """
    Switch to a different model.
    
    Example models:
    - ollama/llama3.2 (local, free)
    - groq/llama-3.3-70b-versatile (fast, cheap)
    - gemini/gemini-1.5-flash (free)
    - gemini/gemini-1.5-pro (free, best reasoning)
    """
    if not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM module not available")
    
    old_model = llm_client.config.model
    llm_client.switch_model(request.model)
    
    logger.info(f"Model switched: {old_model} → {request.model}")
    
    config = get_llm_config()
    return LLMConfigResponse(
        model=config.model,
        provider=config.provider,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        ollama_host=config.ollama_host,
        litellm_available=True
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the LLM.
    
    Request body:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7,  // optional
        "max_tokens": 1024   // optional
    }
    """
    if not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM module not available")
    
    kwargs = {}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    
    try:
        response = await llm_client.achat(request.messages, **kwargs)
        config = get_llm_config()
        
        return ChatResponse(
            content=response,
            model=config.model,
            provider=config.provider
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-structured")
async def test_structured_output():
    """Test structured JSON output."""
    if not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM module not available")
    
    messages = [
        {"role": "system", "content": "You are a trading analyst. Output JSON only."},
        {"role": "user", "content": """Analyze BTC sentiment.
        
Output format:
{
    "symbol": "BTC",
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}"""}
    ]
    
    try:
        result = await llm_client.astructured_chat(messages)
        config = get_llm_config()
        
        return {
            "result": result,
            "model": config.model,
            "provider": config.provider
        }
    except Exception as e:
        logger.error(f"Structured output test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
