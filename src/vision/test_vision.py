"""
Test script for Computer Vision functionality

Tests:
1. YOLOv8 model loading
2. Candlestick chart rendering
3. Chart image generation from OHLCV data
"""

import asyncio
import logging
from pathlib import Path
from chart_renderer import ChartRenderer
from chart_patterns import ChartPatternRecognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_yolov8_loading():
    """Test YOLOv8 model loading"""
    logger.info("=" * 60)
    logger.info("Test 1: YOLOv8 Model Loading")
    logger.info("=" * 60)
    
    try:
        recognizer = ChartPatternRecognizer(
            model_name="yolov8n.pt",  # Nano model for testing
            confidence_threshold=0.25
        )
        
        logger.info("Loading YOLOv8 model...")
        recognizer.load_model()
        
        logger.info("✅ YOLOv8 model loaded successfully")
        logger.info(f"   Model: {recognizer.model_name}")
        logger.info(f"   Confidence threshold: {recognizer.confidence_threshold}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ YOLOv8 model loading failed: {e}", exc_info=True)
        return False


def test_candlestick_rendering():
    """Test candlestick chart rendering with sample data"""
    logger.info("=" * 60)
    logger.info("Test 2: Candlestick Chart Rendering")
    logger.info("=" * 60)
    
    try:
        # Create sample OHLCV data
        import time
        base_time = int(time.time()) - (100 * 3600)  # 100 hours ago
        
        sample_ohlcv = []
        base_price = 50000.0
        
        for i in range(100):
            # Generate sample candlestick data
            open_price = base_price + (i * 10)
            close_price = open_price + (50 if i % 2 == 0 else -30)
            high_price = max(open_price, close_price) + 20
            low_price = min(open_price, close_price) - 20
            volume = 1000.0 + (i * 10)
            
            sample_ohlcv.append({
                "timestamp": base_time + (i * 3600),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })
        
        # Create renderer
        renderer = ChartRenderer(style="yahoo", figsize=(10, 6))
        
        logger.info(f"Rendering chart with {len(sample_ohlcv)} candles...")
        img = renderer.render_chart(
            ohlcv_data=sample_ohlcv,
            symbol="BTCUSDT",
            interval="1h",
            show_volume=True
        )
        
        logger.info(f"✅ Chart rendered successfully")
        logger.info(f"   Image size: {img.size}")
        logger.info(f"   Image mode: {img.mode}")
        
        # Save test image
        test_output = Path("/tmp/test_chart.png")
        img.save(test_output)
        logger.info(f"   Saved to: {test_output}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Candlestick rendering failed: {e}", exc_info=True)
        return False


async def test_chart_from_fks_data():
    """Test chart generation from fks_data service"""
    logger.info("=" * 60)
    logger.info("Test 3: Chart Generation from fks_data")
    logger.info("=" * 60)
    
    try:
        renderer = ChartRenderer()
        
        # Test with BTCUSDT
        symbol = "BTCUSDT"
        interval = "1h"
        limit = 100
        
        logger.info(f"Fetching OHLCV data for {symbol} from fks_data...")
        logger.info(f"   Interval: {interval}, Limit: {limit}")
        
        img = await renderer.render_chart_from_fks_data(
            symbol=symbol,
            interval=interval,
            limit=limit,
            data_service_url="http://fks_data:8003"
        )
        
        logger.info(f"✅ Chart generated from fks_data")
        logger.info(f"   Image size: {img.size}")
        
        # Save test image
        test_output = Path(f"/tmp/test_chart_{symbol.lower()}.png")
        img.save(test_output)
        logger.info(f"   Saved to: {test_output}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Chart generation from fks_data failed: {e}", exc_info=True)
        logger.warning("   This is expected if fks_data service is not available")
        return False


async def test_pattern_detection():
    """Test pattern detection on rendered chart"""
    logger.info("=" * 60)
    logger.info("Test 4: Pattern Detection")
    logger.info("=" * 60)
    
    try:
        # First render a chart
        renderer = ChartRenderer()
        
        # Create sample data
        import time
        base_time = int(time.time()) - (50 * 3600)
        sample_ohlcv = []
        base_price = 50000.0
        
        for i in range(50):
            open_price = base_price + (i * 10)
            close_price = open_price + (50 if i % 2 == 0 else -30)
            high_price = max(open_price, close_price) + 20
            low_price = min(open_price, close_price) - 20
            volume = 1000.0 + (i * 10)
            
            sample_ohlcv.append({
                "timestamp": base_time + (i * 3600),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })
        
        logger.info("Rendering chart for pattern detection...")
        chart_img = renderer.render_chart(
            ohlcv_data=sample_ohlcv,
            symbol="TEST",
            interval="1h"
        )
        
        # Load pattern recognizer
        recognizer = ChartPatternRecognizer()
        recognizer.load_model()
        
        logger.info("Detecting patterns in chart...")
        result = recognizer.analyze_chart(chart_img, symbol="TEST")
        
        logger.info(f"✅ Pattern detection completed")
        logger.info(f"   Signal: {result['signal']}")
        logger.info(f"   Confidence: {result['confidence']:.2f}")
        logger.info(f"   Patterns detected: {result['pattern_count']}")
        logger.info(f"   Bullish patterns: {result['bullish_patterns']}")
        logger.info(f"   Bearish patterns: {result['bearish_patterns']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pattern detection failed: {e}", exc_info=True)
        return False


async def main():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("Computer Vision Module Tests")
    logger.info("=" * 60 + "\n")
    
    results = {}
    
    # Test 1: YOLOv8 loading
    results["yolov8_loading"] = await test_yolov8_loading()
    
    # Test 2: Candlestick rendering
    results["candlestick_rendering"] = test_candlestick_rendering()
    
    # Test 3: Chart from fks_data (may fail if service unavailable)
    results["chart_from_fks_data"] = await test_chart_from_fks_data()
    
    # Test 4: Pattern detection
    results["pattern_detection"] = await test_pattern_detection()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '⚠️ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
