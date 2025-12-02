"""
Computer Vision API Routes

Endpoints for chart rendering and pattern recognition.
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import httpx

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/vision", tags=["computer-vision"])

# Import vision modules (with error handling)
try:
    from vision.chart_renderer import ChartRenderer
    from vision.chart_patterns import ChartPatternRecognizer
    VISION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vision modules not available: {e}")
    VISION_AVAILABLE = False
    ChartRenderer = None
    ChartPatternRecognizer = None


class OHLCVData(BaseModel):
    """OHLCV data point"""
    timestamp: int = Field(..., description="Unix timestamp")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: Optional[float] = Field(None, description="Volume")


class ChartRenderRequest(BaseModel):
    """Request for chart rendering"""
    ohlcv_data: List[OHLCVData] = Field(..., description="OHLCV data points")
    symbol: str = Field(..., description="Trading symbol")
    interval: str = Field(default="1h", description="Time interval")
    show_volume: bool = Field(default=True, description="Show volume subplot")
    chart_type: str = Field(default="candle", description="Chart type (candle, line, ohlc)")


class ChartRenderResponse(BaseModel):
    """Response from chart rendering"""
    success: bool
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_size: Optional[Dict[str, int]] = Field(None, description="Image dimensions")
    error: Optional[str] = None


class PatternDetectionRequest(BaseModel):
    """Request for pattern detection"""
    symbol: str = Field(..., description="Trading symbol")
    interval: str = Field(default="1h", description="Time interval")
    limit: int = Field(default=100, description="Number of candles")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Detection confidence threshold")


class PatternDetectionResponse(BaseModel):
    """Response from pattern detection"""
    symbol: str
    signal: str = Field(..., description="Trading signal (BUY, SELL, HOLD)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence")
    patterns_detected: List[Dict[str, Any]] = Field(default_factory=list)
    pattern_count: int
    bullish_patterns: int
    bearish_patterns: int
    analysis: str


@router.get("/health")
async def vision_health():
    """Check if vision module is available"""
    return {
        "available": VISION_AVAILABLE,
        "modules": {
            "chart_renderer": ChartRenderer is not None,
            "chart_patterns": ChartPatternRecognizer is not None
        }
    }


@router.post("/render", response_model=ChartRenderResponse)
async def render_chart(request: ChartRenderRequest):
    """
    Render OHLCV data as candlestick chart image
    
    Returns base64-encoded PNG image.
    """
    if not VISION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Vision module not available. Install dependencies: torchvision, ultralytics, mplfinance, opencv-python"
        )
    
    try:
        renderer = ChartRenderer()
        
        # Convert Pydantic models to dicts
        ohlcv_list = [candle.dict() for candle in request.ohlcv_data]
        
        # Render chart
        img = renderer.render_chart(
            ohlcv_data=ohlcv_list,
            symbol=request.symbol,
            interval=request.interval,
            show_volume=request.show_volume,
            chart_type=request.chart_type
        )
        
        # Convert to base64
        import base64
        import io
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
        return ChartRenderResponse(
            success=True,
            image_base64=img_base64,
            image_size={"width": img.size[0], "height": img.size[1]}
        )
        
    except Exception as e:
        logger.error(f"Chart rendering failed: {e}", exc_info=True)
        return ChartRenderResponse(
            success=False,
            error=str(e)
        )


@router.get("/render/{symbol}", response_model=ChartRenderResponse)
async def render_chart_from_symbol(
    symbol: str,
    interval: str = Query("1h", description="Time interval"),
    limit: int = Query(100, description="Number of candles"),
    data_service_url: str = Query("http://fks_data:8003", description="fks_data service URL")
):
    """
    Fetch OHLCV data from fks_data and render chart
    
    Returns base64-encoded PNG image.
    """
    if not VISION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Vision module not available"
        )
    
    try:
        renderer = ChartRenderer()
        
        # Fetch data from fks_data
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{data_service_url}/api/v1/data/ohlcv",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit,
                    "use_cache": True
                }
            )
            response.raise_for_status()
            data = response.json()
            
            ohlcv_list = data.get("data", [])
            if not ohlcv_list:
                raise HTTPException(
                    status_code=404,
                    detail=f"No OHLCV data found for {symbol}"
                )
        
        # Render chart
        img = renderer.render_chart(
            ohlcv_data=ohlcv_list,
            symbol=symbol,
            interval=interval
        )
        
        # Convert to base64
        import base64
        import io
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
        return ChartRenderResponse(
            success=True,
            image_base64=img_base64,
            image_size={"width": img.size[0], "height": img.size[1]}
        )
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to fetch data from fks_data: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch data from fks_data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Chart rendering failed: {e}", exc_info=True)
        return ChartRenderResponse(
            success=False,
            error=str(e)
        )


@router.post("/detect-patterns", response_model=PatternDetectionResponse)
async def detect_patterns(request: PatternDetectionRequest):
    """
    Detect chart patterns in candlestick chart
    
    Fetches OHLCV data, renders chart, and detects patterns using YOLOv8.
    """
    if not VISION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Vision module not available"
        )
    
    try:
        renderer = ChartRenderer()
        recognizer = ChartPatternRecognizer(
            confidence_threshold=request.confidence_threshold
        )
        recognizer.load_model()
        
        # Fetch and render chart
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "http://fks_data:8003/api/v1/data/ohlcv",
                params={
                    "symbol": request.symbol,
                    "interval": request.interval,
                    "limit": request.limit,
                    "use_cache": True
                }
            )
            response.raise_for_status()
            data = response.json()
            
            ohlcv_list = data.get("data", [])
            if not ohlcv_list:
                raise HTTPException(
                    status_code=404,
                    detail=f"No OHLCV data found for {request.symbol}"
                )
        
        # Render chart
        chart_img = renderer.render_chart(
            ohlcv_data=ohlcv_list,
            symbol=request.symbol,
            interval=request.interval
        )
        
        # Detect patterns
        analysis = recognizer.analyze_chart(chart_img, symbol=request.symbol)
        
        return PatternDetectionResponse(**analysis)
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to fetch data from fks_data: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch data from fks_data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Pattern detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pattern detection failed: {str(e)}"
        )
