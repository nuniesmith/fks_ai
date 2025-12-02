"""
FKS AI - ML/RAG Service

Optimizes hardcoded baseline strategies using simulations to verify improvements.
Uses AI models to enhance trading signals and optimize strategy parameters.

AI Providers:
- Google AI API (Gemini): Free tier (1,500-10,000 prompts/day) - Primary
- Ollama: Local AI models - Fallback/Alternative
- Future: Additional AI providers as needed

Purpose:
- Optimize baseline strategies through simulation
- Verify improvements using backtesting/simulation
- Enhance trading signals with AI analysis
- Use free AI providers (Google AI API, Ollama) for cost-effective optimization
"""
import logging
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FKS AI - ML/RAG Service",
    description="Optimizes baseline strategies using simulations and AI models (Google AI API free tier, Ollama local). Verifies improvements through simulation.",
    version="1.0.0"
)

# Set up standardized Prometheus metrics
try:
    import sys
    # Try to import from fks_api framework (if available)
    api_framework_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'api', 'src')
    if os.path.exists(api_framework_path) and api_framework_path not in sys.path:
        sys.path.insert(0, api_framework_path)
    
    try:
        from framework.middleware.prometheus_metrics import setup_prometheus_metrics
        setup_prometheus_metrics(
            app,
            service_name="fks_ai",
            version="1.0.0",
            commit=os.getenv("GIT_COMMIT", os.getenv("COMMIT_SHA")),
            build_date=os.getenv("BUILD_DATE", os.getenv("BUILD_TIMESTAMP")),
            enable_http_metrics=True,
            enable_process_metrics=True,
        )
    except ImportError:
        # Fallback: use prometheus_client directly if framework not available
        from prometheus_client import make_asgi_app
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
except Exception as e:
    logger.warning(f"Could not set up Prometheus metrics: {e}")

# Try to include signal enhancement router (standalone, no dependencies)
SIGNAL_ENHANCEMENT_AVAILABLE = False
try:
    import sys
    import os
    # Add src to path for imports
    src_path = os.path.join(os.path.dirname(__file__), '..')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from api.routes.signal_enhancement import router as signal_enhancement_router
    app.include_router(signal_enhancement_router)
    logger.info("✅ Signal enhancement endpoint loaded")
    SIGNAL_ENHANCEMENT_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ Signal enhancement endpoint not available: {e}")
    import traceback
    logger.warning(traceback.format_exc())
    SIGNAL_ENHANCEMENT_AVAILABLE = False

# Try to include vision router (computer vision for chart pattern recognition)
VISION_AVAILABLE = False
try:
    from api.routes.vision import router as vision_router
    app.include_router(vision_router)
    logger.info("✅ Computer vision endpoints loaded")
    VISION_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ Computer vision endpoints not available: {e}")
    VISION_AVAILABLE = False

# Try to include advanced API routes if available (full multi-agent system)
ADVANCED_ROUTES_AVAILABLE = False
try:
    # Only import if all dependencies are available
    import httpx
    from api.routes.health import router as health_router
    app.include_router(health_router)
    
    # Try full routes if available (may fail if agents not configured)
    try:
        from api.routes import app as routes_app
        # routes_app is a FastAPI app, we need to get its routers differently
        # For now, just note it's available
        ADVANCED_ROUTES_AVAILABLE = True
        logger.info("✅ Advanced API routes available")
    except Exception:
        logger.info("ℹ️ Advanced routes exist but not all dependencies available")
        ADVANCED_ROUTES_AVAILABLE = False
except ImportError:
    logger.info("ℹ️ Advanced routes require additional dependencies")
    ADVANCED_ROUTES_AVAILABLE = False

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
        import httpx
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from utils.timezone import now_est_iso
    
    google_ai_status = check_google_ai_connection()
    ollama_status = check_ollama_connection()
    
    # Determine primary provider
    llm_provider = os.getenv("LLM_PROVIDER", "gemini" if google_ai_status.get("configured") else "ollama")
    
    return JSONResponse({
        "status": "healthy",
        "service": "fks_ai",
        "version": "1.0.0",
        "timestamp": now_est_iso(),
        "gpu_available": False,  # TODO: Check CUDA availability
        "llm_provider": llm_provider,
        "google_ai": google_ai_status,
        "ollama": ollama_status,
        "signal_enhancement_available": SIGNAL_ENHANCEMENT_AVAILABLE,
        "advanced_routes_available": ADVANCED_ROUTES_AVAILABLE,
        "vision_available": VISION_AVAILABLE,
        "endpoints": {
            "/health": "Service health check",
            "/ai/enhance-signal": "Signal enhancement (always available)" if SIGNAL_ENHANCEMENT_AVAILABLE else "Not available",
            "/ai/analyze": "Full multi-agent analysis" if ADVANCED_ROUTES_AVAILABLE else "Not available",
            "/ai/debate": "Bull/Bear debate" if ADVANCED_ROUTES_AVAILABLE else "Not available",
            "/ai/vision/render": "Render candlestick chart" if VISION_AVAILABLE else "Not available",
            "/ai/vision/detect-patterns": "Detect chart patterns" if VISION_AVAILABLE else "Not available"
        }
    })

@app.get("/regime")
async def detect_regime():
    """Detect current market regime (placeholder)"""
    return JSONResponse({
        "regime": "unknown",
        "confidence": 0.0,
        "note": "Regime detection (VAE/Transformer) will be implemented in Phase 2"
    })

@app.get("/generate-strategy")
async def generate_strategy():
    """Generate trading strategy with LLM (placeholder)"""
    return JSONResponse({
        "strategy": {},
        "note": "LLM strategy generation with Ollama will be implemented in Phase 3"
    })

@app.get("/train-regime-model")
async def train_regime_model():
    """Train regime detection model (placeholder)"""
    return JSONResponse({
        "status": "not_started",
        "note": "Model training pipeline will be implemented in Phase 2"
    })

@app.get("/embeddings")
async def generate_embeddings():
    """Generate embeddings for RAG (placeholder)"""
    return JSONResponse({
        "embeddings": [],
        "note": "RAG embeddings with sentence-transformers will be implemented in Phase 1"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
