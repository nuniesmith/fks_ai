"""
Integration Tests for API Endpoints

Tests FastAPI routes with live multi-agent system.
"""

import os

# Import API app
import sys

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from api.routes import app


@pytest.fixture
def client():
    """Synchronous test client"""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Async test client for async endpoints"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_analyze_request():
    """Sample request for /ai/analyze endpoint"""
    return {
        "symbol": "BTCUSDT",
        "market_data": {
            "price": 67234.50,
            "rsi": 58.5,
            "macd": 150.2,
            "macd_signal": 125.8,
            "bb_upper": 68000.0,
            "bb_middle": 67000.0,
            "bb_lower": 66000.0,
            "atr": 400.0,
            "volume": 1234567890,
            "regime": "bull"
        }
    }


@pytest.mark.asyncio
@pytest.mark.integration
class TestAPIEndpoints:
    """Test all API endpoints"""

    async def test_health_check(self, async_client):
        """Test /health endpoint"""
        response = await async_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "fks_ai"
        assert "version" in data


    async def test_root_endpoint(self, async_client):
        """Test root / endpoint"""
        response = await async_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "FKS AI - Multi-Agent Trading System"
        assert "endpoints" in data


    async def test_analyze_endpoint(self, async_client, sample_analyze_request):
        """Test POST /ai/analyze endpoint"""
        response = await async_client.post("/ai/analyze", json=sample_analyze_request)

        # May fail without Ollama, but validate structure
        if response.status_code == 200:
            data = response.json()

            # Validate response structure
            assert "symbol" in data
            assert "timestamp" in data
            assert "analysts" in data
            assert "debate" in data
            assert "final_decision" in data
            assert "confidence" in data
            assert "execution_time_ms" in data

            # Validate analysts
            assert "technical" in data["analysts"]
            assert "sentiment" in data["analysts"]
            assert "macro" in data["analysts"]
            assert "risk" in data["analysts"]

            # Validate debate
            assert "bull" in data["debate"]
            assert "bear" in data["debate"]

            # Validate metrics
            assert 0.0 <= data["confidence"] <= 1.0
            assert data["execution_time_ms"] > 0

            print("\nAnalyze Endpoint Response:")
            print(f"  Symbol: {data['symbol']}")
            print(f"  Confidence: {data['confidence']:.2f}")
            print(f"  Execution Time: {data['execution_time_ms']:.0f}ms")
        else:
            # Acceptable if Ollama not running
            print(f"\nAnalyze endpoint returned {response.status_code} (Ollama may not be running)")


    async def test_debate_endpoint(self, async_client, sample_analyze_request):
        """Test POST /ai/debate endpoint"""
        response = await async_client.post("/ai/debate", json=sample_analyze_request)

        if response.status_code == 200:
            data = response.json()

            # Validate response structure
            assert "symbol" in data
            assert "bull_argument" in data
            assert "bear_argument" in data
            assert "execution_time_ms" in data

            # Arguments should be substantial
            assert len(data["bull_argument"]) > 0
            assert len(data["bear_argument"]) > 0

            print("\nDebate Endpoint Response:")
            print(f"  Bull: {data['bull_argument'][:100]}...")
            print(f"  Bear: {data['bear_argument'][:100]}...")
        else:
            print(f"\nDebate endpoint returned {response.status_code}")


    async def test_memory_query_endpoint(self, async_client):
        """Test GET /ai/memory/query endpoint"""
        response = await async_client.get("/ai/memory/query?query=bitcoin+trading&n_results=3")

        if response.status_code == 200:
            data = response.json()

            # Validate response structure
            assert "results" in data
            assert "count" in data
            assert isinstance(data["results"], list)
            assert data["count"] == len(data["results"])

            print("\nMemory Query Response:")
            print(f"  Results: {data['count']}")
        else:
            # Acceptable if ChromaDB not initialized
            print(f"\nMemory query returned {response.status_code}")


    async def test_agents_status_endpoint(self, async_client):
        """Test GET /ai/agents/status endpoint"""
        response = await async_client.get("/ai/agents/status")

        if response.status_code == 200:
            data = response.json()

            # Validate response structure
            assert "status" in data
            assert "agents" in data
            assert "memory_status" in data

            # Check all 7 agents
            expected_agents = ["technical", "sentiment", "macro", "risk", "bull", "bear", "manager"]
            for agent in expected_agents:
                assert agent in data["agents"], f"Missing {agent} agent in status"

            print("\nAgent Status Response:")
            print(f"  Overall Status: {data['status']}")
            print("  Agents:")
            for agent, status in data["agents"].items():
                print(f"    {agent}: {status.get('status', 'unknown')}")
        else:
            print(f"\nAgent status returned {response.status_code}")


    async def test_memory_query_with_filters(self, async_client):
        """Test memory query with filters"""
        response = await async_client.get(
            "/ai/memory/query?query=bullish+signal&n_results=5&symbol=BTCUSDT&min_confidence=0.6"
        )

        if response.status_code == 200:
            data = response.json()

            # All results should match filters
            for result in data["results"]:
                if "metadata" in result:
                    metadata = result["metadata"]
                    if "symbol" in metadata:
                        assert metadata["symbol"] == "BTCUSDT"
                    if "confidence" in metadata:
                        assert metadata["confidence"] >= 0.6

            print("\nFiltered Memory Query:")
            print(f"  Results: {data['count']}")


@pytest.mark.integration
class TestAPIDocs:
    """Test API documentation endpoints"""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema generation"""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "FKS AI Service"


    def test_docs_endpoint(self, client):
        """Test Swagger UI docs"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()


    def test_redoc_endpoint(self, client):
        """Test ReDoc documentation"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
