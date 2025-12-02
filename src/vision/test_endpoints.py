"""
Test script for Vision API endpoints

Tests the vision endpoints after dependencies are installed.
"""

import asyncio
import logging
import httpx
import base64
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VISION_SERVICE_URL = "http://localhost:8007"  # Adjust if needed


async def test_health():
    """Test vision health endpoint"""
    logger.info("=" * 60)
    logger.info("Test 1: Vision Health Check")
    logger.info("=" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{VISION_SERVICE_URL}/ai/vision/health")
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Vision health check passed")
                logger.info(f"   Available: {data.get('available', False)}")
                logger.info(f"   Modules: {data.get('modules', {})}")
                return True
            else:
                logger.error(f"❌ Health check failed: HTTP {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False


async def test_render_chart():
    """Test chart rendering endpoint"""
    logger.info("=" * 60)
    logger.info("Test 2: Chart Rendering")
    logger.info("=" * 60)
    
    try:
        import time
        base_time = int(time.time()) - (100 * 3600)
        
        # Create sample OHLCV data
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
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{VISION_SERVICE_URL}/ai/vision/render",
                json={
                    "ohlcv_data": sample_ohlcv,
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "show_volume": True,
                    "chart_type": "candle"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info("✅ Chart rendering successful")
                    logger.info(f"   Image size: {data.get('image_size', {})}")
                    
                    # Save image
                    if data.get("image_base64"):
                        img_bytes = base64.b64decode(data["image_base64"])
                        output_path = Path("/tmp/test_chart_api.png")
                        output_path.write_bytes(img_bytes)
                        logger.info(f"   Saved to: {output_path}")
                    
                    return True
                else:
                    logger.error(f"❌ Chart rendering failed: {data.get('error')}")
                    return False
            else:
                logger.error(f"❌ Chart rendering failed: HTTP {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Chart rendering failed: {e}", exc_info=True)
        return False


async def test_render_from_symbol():
    """Test chart rendering from symbol (fetches from fks_data)"""
    logger.info("=" * 60)
    logger.info("Test 3: Chart Rendering from Symbol")
    logger.info("=" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{VISION_SERVICE_URL}/ai/vision/render/BTCUSDT",
                params={
                    "interval": "1h",
                    "limit": 100
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info("✅ Chart rendering from symbol successful")
                    logger.info(f"   Image size: {data.get('image_size', {})}")
                    
                    # Save image
                    if data.get("image_base64"):
                        img_bytes = base64.b64decode(data["image_base64"])
                        output_path = Path("/tmp/test_chart_symbol_api.png")
                        output_path.write_bytes(img_bytes)
                        logger.info(f"   Saved to: {output_path}")
                    
                    return True
                else:
                    logger.error(f"❌ Chart rendering failed: {data.get('error')}")
                    return False
            else:
                logger.warning(f"⚠️ Chart rendering failed: HTTP {response.status_code}")
                logger.warning("   This is expected if fks_data service is not available")
                return False
                
    except Exception as e:
        logger.warning(f"⚠️ Chart rendering failed: {e}")
        logger.warning("   This is expected if fks_data service is not available")
        return False


async def test_pattern_detection():
    """Test pattern detection endpoint"""
    logger.info("=" * 60)
    logger.info("Test 4: Pattern Detection")
    logger.info("=" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{VISION_SERVICE_URL}/ai/vision/detect-patterns",
                json={
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "limit": 100,
                    "confidence_threshold": 0.25
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Pattern detection successful")
                logger.info(f"   Signal: {data.get('signal')}")
                logger.info(f"   Confidence: {data.get('confidence', 0):.2f}")
                logger.info(f"   Patterns detected: {data.get('pattern_count', 0)}")
                logger.info(f"   Bullish patterns: {data.get('bullish_patterns', 0)}")
                logger.info(f"   Bearish patterns: {data.get('bearish_patterns', 0)}")
                
                if data.get("patterns_detected"):
                    logger.info("   Pattern details:")
                    for pattern in data["patterns_detected"][:5]:  # Show first 5
                        logger.info(f"     - {pattern.get('pattern')}: {pattern.get('confidence', 0):.2f}")
                
                return True
            else:
                logger.warning(f"⚠️ Pattern detection failed: HTTP {response.status_code}")
                logger.warning("   This is expected if fks_data service is not available")
                return False
                
    except Exception as e:
        logger.warning(f"⚠️ Pattern detection failed: {e}")
        logger.warning("   This is expected if fks_data service is not available")
        return False


async def main():
    """Run all endpoint tests"""
    logger.info("\n" + "=" * 60)
    logger.info("Vision API Endpoint Tests")
    logger.info("=" * 60 + "\n")
    
    results = {}
    
    # Test 1: Health check
    results["health"] = await test_health()
    
    # Test 2: Chart rendering
    results["render"] = await test_render_chart()
    
    # Test 3: Chart rendering from symbol
    results["render_symbol"] = await test_render_from_symbol()
    
    # Test 4: Pattern detection
    results["pattern_detection"] = await test_pattern_detection()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    critical_tests = ["health", "render"]
    critical_passed = all(results.get(test, False) for test in critical_tests)
    
    logger.info(f"\nCritical tests: {'✅ PASSED' if critical_passed else '❌ FAILED'}")
    logger.info("Note: Tests requiring fks_data may fail if service is not available")
    
    return critical_passed


if __name__ == "__main__":
    asyncio.run(main())
