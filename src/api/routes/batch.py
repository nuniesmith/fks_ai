"""
Batch Screening API Routes

Endpoints for batch screening 100+ tickers with JSON output.
Integrates with Discord webhooks for daily opportunity notifications.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

try:
    from agents.batch import (
        BatchScreener,
        BatchConfig,
        ScreeningResult,
        DataQuality,
        run_batch_screening,
    )
    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/batch", tags=["batch"])


# Request/Response Models
class BatchScreenRequest(BaseModel):
    """Request model for batch screening."""
    symbols: List[str] = Field(..., description="List of ticker symbols to screen")
    max_concurrent: int = Field(default=10, ge=1, le=50)
    top_n_results: int = Field(default=10, ge=1, le=100)
    include_failures: bool = Field(default=False)
    
    # Optional thesis gate overrides
    min_health_score: Optional[float] = Field(default=None, ge=0, le=100)
    min_growth_score: Optional[float] = Field(default=None, ge=0, le=100)
    max_pe_ratio: Optional[float] = Field(default=None, ge=0)
    min_liquidity_usd: Optional[float] = Field(default=None, ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
                "max_concurrent": 10,
                "top_n_results": 5,
                "include_failures": False,
            }
        }


class BatchScreenResponse(BaseModel):
    """Response model for batch screening."""
    timestamp: str
    summary: Dict[str, Any]
    top_opportunities: List[Dict[str, Any]]
    failed_opportunities: List[Dict[str, Any]]
    errors: List[Dict[str, str]]


class DiscordWebhookConfig(BaseModel):
    """Configuration for Discord webhook notifications."""
    webhook_url: str = Field(..., description="Discord webhook URL")
    mention_role: Optional[str] = Field(default=None, description="Role ID to mention")
    include_failed: bool = Field(default=False)


class ScheduleScreeningRequest(BaseModel):
    """Request to schedule daily screening."""
    symbols: List[str] = Field(..., description="List of ticker symbols")
    schedule_time: str = Field(default="08:00", description="Time to run (HH:MM UTC)")
    discord_webhook_url: Optional[str] = Field(default=None)
    enabled: bool = Field(default=True)


# Endpoints
@router.post("/screen", response_model=BatchScreenResponse)
async def batch_screen(request: BatchScreenRequest):
    """
    Screen multiple symbols and return ranked opportunities.
    
    This runs the full AI analysis pipeline on each symbol:
    1. Parallel data gathering (market, fundamentals, sentiment, news)
    2. Bull/Bear debate with strict role prompts
    3. Manager synthesis with thesis gates
    4. Risk team position sizing
    
    Returns ranked opportunities in JSON format.
    """
    if not BATCH_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Batch screening module not available"
        )
    
    # Build config from request
    config = BatchConfig(
        max_concurrent=request.max_concurrent,
        top_n_results=request.top_n_results,
        include_failures=request.include_failures,
    )
    
    if request.min_health_score is not None:
        config.min_health_score = request.min_health_score
    if request.min_growth_score is not None:
        config.min_growth_score = request.min_growth_score
    if request.max_pe_ratio is not None:
        config.max_pe_ratio = request.max_pe_ratio
    if request.min_liquidity_usd is not None:
        config.min_liquidity_usd = request.min_liquidity_usd
    
    try:
        result = await run_batch_screening(request.symbols, config)
        result_dict = result.to_dict()
        
        return BatchScreenResponse(
            timestamp=result_dict["timestamp"],
            summary=result_dict["summary"],
            top_opportunities=result_dict["top_opportunities"],
            failed_opportunities=result_dict["failed_opportunities"],
            errors=result_dict["errors"],
        )
    except Exception as e:
        logger.error(f"Batch screening failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screen/discord")
async def batch_screen_with_discord(
    request: BatchScreenRequest,
    discord_config: DiscordWebhookConfig,
    background_tasks: BackgroundTasks,
):
    """
    Screen symbols and send results to Discord webhook.
    
    Runs screening in foreground and sends Discord notification in background.
    """
    if not BATCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Batch screening module not available"
        )
    
    # Build config
    config = BatchConfig(
        max_concurrent=request.max_concurrent,
        top_n_results=request.top_n_results,
        include_failures=request.include_failures,
    )
    
    try:
        result = await run_batch_screening(request.symbols, config)
        
        # Send to Discord in background
        background_tasks.add_task(
            send_discord_notification,
            result,
            discord_config.webhook_url,
            discord_config.mention_role,
        )
        
        return {
            "status": "completed",
            "discord_notification": "queued",
            "summary": result.to_dict()["summary"],
        }
    except Exception as e:
        logger.error(f"Batch screening failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watchlist/analyze")
async def analyze_watchlist(
    background_tasks: BackgroundTasks,
    discord_webhook_url: Optional[str] = None,
):
    """
    Analyze the user's enabled watchlist from fks_app.
    
    Fetches enabled assets from fks_app and runs full screening.
    Optionally sends results to Discord.
    """
    if not BATCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Batch screening module not available"
        )
    
    try:
        # Fetch enabled symbols from fks_app
        symbols = await fetch_watchlist_symbols()
        
        if not symbols:
            return {
                "status": "no_symbols",
                "message": "No enabled assets in watchlist",
            }
        
        # Run screening
        config = BatchConfig(top_n_results=10)
        result = await run_batch_screening(symbols, config)
        
        # Send to Discord if configured
        if discord_webhook_url:
            background_tasks.add_task(
                send_discord_notification,
                result,
                discord_webhook_url,
            )
        
        return {
            "status": "completed",
            "symbols_analyzed": len(symbols),
            "discord_notification": "queued" if discord_webhook_url else "not_configured",
            "result": result.to_dict(),
        }
    except Exception as e:
        logger.error(f"Watchlist analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def batch_status():
    """Get batch screening module status."""
    return {
        "batch_available": BATCH_AVAILABLE,
        "discord_configured": bool(os.environ.get("DISCORD_WEBHOOK_URL")),
        "default_config": {
            "max_concurrent": 10,
            "min_health_score": 50.0,
            "min_growth_score": 50.0,
            "max_pe_ratio": 18.0,
            "min_liquidity_usd": 100_000.0,
        }
    }


# Helper Functions
async def fetch_watchlist_symbols() -> List[str]:
    """Fetch enabled symbols from fks_app."""
    import httpx
    
    fks_app_url = os.environ.get("FKS_APP_URL", "http://fks_app:8000")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{fks_app_url}/api/assets/enabled",
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                return [asset["symbol"] for asset in data.get("assets", [])]
            else:
                logger.warning(f"Failed to fetch watchlist: {response.status_code}")
                return []
    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}")
        return []


async def send_discord_notification(
    result: "ScreeningResult",
    webhook_url: str,
    mention_role: Optional[str] = None,
):
    """Send screening results to Discord webhook."""
    import httpx
    
    try:
        message = result.to_discord_message()
        
        if mention_role:
            message = f"<@&{mention_role}>\n\n{message}"
        
        payload = {
            "content": message,
            "username": "FKS Trading Bot",
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                json=payload,
                timeout=10.0
            )
            if response.status_code not in (200, 204):
                logger.error(f"Discord webhook failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to send Discord notification: {e}")
