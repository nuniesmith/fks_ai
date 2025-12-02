"""
Standardized health check endpoints for FKS services.
Implements liveness, readiness, and health probes.

Includes Google AI API (Gemini) status with free tier limits.
Falls back to Ollama for local LLM inference if Google AI API unavailable.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()


def check_google_ai_connection():
    """Check if Google AI API (Gemini) is available and configured"""
    import os
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    
    if not api_key:
        return {
            "configured": False,
            "error": "GOOGLE_AI_API_KEY not set"
        }
    
    # Check if API key is valid (basic check)
    try:
        # Simple validation - API key format check
        if len(api_key) > 20:  # Basic validation
            return {
                "configured": True,
                "api_key_set": True,
                "model": os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash-exp"),
                "free_tier_limit": int(os.getenv("GEMINI_FREE_TIER_LIMIT", "1500")),
                "note": "Free tier: 1,500-10,000 prompts/day"
            }
        else:
            return {
                "configured": False,
                "error": "API key format invalid"
            }
    except Exception as e:
        return {
            "configured": False,
            "error": str(e)
        }

def check_ollama_connection():
    """Check if Ollama service is available and connected"""
    import os
    import httpx
    
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        # Try to connect to Ollama API
        response = httpx.get(f"{ollama_host}/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {
                "connected": True,
                "host": ollama_host,
                "models_available": len(models),
                "models": [m.get("name", "unknown") for m in models[:5]]  # First 5 models
            }
        else:
            return {"connected": False, "error": f"HTTP {response.status_code}"}
    except httpx.RequestError as e:
        return {"connected": False, "error": str(e)}
    except Exception as e:
        return {"connected": False, "error": str(e)}

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint - liveness probe.
    Returns basic service health status with Google AI API and Ollama status.
    """
    import os
    from utils.timezone import now_est_iso
    
    google_ai_status = check_google_ai_connection()
    ollama_status = check_ollama_connection()
    
    # Determine primary provider
    llm_provider = os.getenv("LLM_PROVIDER", "gemini" if google_ai_status.get("configured") else "ollama")
    
    return {
        "status": "healthy",
        "service": "fks_ai",
        "timestamp": now_est_iso(),
        "version": "1.0.0",
        "llm_provider": llm_provider,
        "google_ai": google_ai_status,
        "ollama": ollama_status
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint.
    Verifies service is ready to accept traffic.
    Checks critical dependencies.
    """
    # TODO: Add dependency checks (database, external services, etc.)
    dependencies_ready = True
    dependency_status = {}
    
    # Example: Check database connection
    # try:
    #     await check_database()
    #     dependency_status["database"] = "ready"
    # except Exception as e:
    #     dependencies_ready = False
    #     dependency_status["database"] = f"error: {str(e)}"
    
    if not dependencies_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    from utils.timezone import now_est_iso
    
    return {
        "status": "ready",
        "service": "fks_ai",
        "timestamp": now_est_iso(),
        "dependencies": dependency_status
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness probe endpoint.
    Simple check to verify process is alive.
    """
    from utils.timezone import now_est_iso
    
    return {
        "status": "alive",
        "service": "fks_ai",
        "timestamp": now_est_iso()
    }
