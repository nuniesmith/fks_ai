"""
API Routes for Multi-Agent Trading System

Endpoints:
- POST /ai/analyze - Full multi-agent analysis with trading signals
- POST /ai/debate - Bull/Bear debate only (no final decision)
- GET /ai/memory/query - Query similar past decisions
- GET /ai/agents/status - Health check for all agents
"""

import asyncio
import logging
import os

# Import Phase 6 components
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.analysts.macro import macro_analyst
from agents.analysts.risk import risk_analyst
from agents.analysts.sentiment import sentiment_analyst
from agents.analysts.technical import technical_analyst
from agents.debaters.bear import bear_agent
from agents.debaters.bull import bull_agent
from agents.debaters.manager import manager_agent
from agents.state import create_initial_state
from evaluators.llm_judge import LLMJudge
from graph.trading_graph import analyze_symbol, trading_graph
from memory import TradingMemory
from processors.signal_processor import SignalProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="FKS AI Service",
    description="Multi-Agent Trading Intelligence API",
    version="1.0.0"
)

# Initialize global components
try:
    trading_memory = TradingMemory()
    signal_processor = SignalProcessor()
    llm_judge = LLMJudge()
    logger.info("Initialized TradingMemory, SignalProcessor, and LLMJudge")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    trading_memory = None
    signal_processor = None


# Request/Response Models
class AnalyzeRequest(BaseModel):
    """Request model for full multi-agent analysis"""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    market_data: dict[str, Any] = Field(..., description="OHLCV and indicators")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class DebateRequest(BaseModel):
    """Request model for Bull/Bear debate"""
    symbol: str = Field(..., description="Trading symbol")
    market_data: dict[str, Any] = Field(..., description="OHLCV and indicators")


class MemoryQueryRequest(BaseModel):
    """Request model for memory similarity search"""
    query: str = Field(..., description="Search query text")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results")
    filter_metadata: Optional[dict[str, Any]] = Field(default=None, description="Filter by metadata")


class AnalyzeResponse(BaseModel):
    """Response model for analysis endpoint"""
    symbol: str
    timestamp: datetime
    analysts: dict[str, str]
    debate: dict[str, str]
    final_decision: str
    trading_signal: dict[str, Any]
    confidence: float
    regime: str
    execution_time_ms: float


class DebateResponse(BaseModel):
    """Response model for debate endpoint"""
    symbol: str
    timestamp: datetime
    bull_argument: str
    bear_argument: str
    execution_time_ms: float


class MemoryQueryResponse(BaseModel):
    """Response model for memory query endpoint"""
    results: list[dict[str, Any]]
    count: int


class AgentStatusResponse(BaseModel):
    """Response model for agent status endpoint"""
    status: str
    agents: dict[str, dict[str, Any]]
    memory_status: dict[str, Any]
    uptime_ms: float


# LLM-Judge Request/Response Models (Phase 7.2)

class ConsistencyCheckRequest(BaseModel):
    """Request for factual consistency validation"""
    agent_name: str = Field(..., description="Name of agent to validate")
    agent_claim: str = Field(..., description="Agent's claim to verify")
    market_data: dict[str, Any] = Field(..., description="Ground truth market data")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "Technical",
                "agent_claim": "BTC price is 67234 with RSI at 58.5, indicating neutral momentum",
                "market_data": {
                    "symbol": "BTCUSDT",
                    "price": 67234.50,
                    "rsi": 58.5,
                    "timestamp": "2025-10-31T12:00:00Z"
                }
            }
        }


class DiscrepancyCheckRequest(BaseModel):
    """Request for prediction discrepancy detection"""
    agent_analysis: str = Field(..., description="Agent's prediction/analysis")
    actual_outcome: float = Field(..., description="Actual price change (%)")
    context: Optional[dict[str, Any]] = Field(default=None, description="Additional context")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_analysis": "Strong bullish momentum expected, RSI showing oversold recovery",
                "actual_outcome": -5.2,
                "context": {
                    "timeframe": "24h",
                    "symbol": "BTCUSDT"
                }
            }
        }


class BiasCheckRequest(BaseModel):
    """Request for systematic bias analysis"""
    agent_name: str = Field(..., description="Name of agent to analyze")
    agent_decisions: list[dict[str, Any]] = Field(..., description="Historical decisions")
    market_outcomes: list[float] = Field(..., description="Actual price changes (%)")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "Bull",
                "agent_decisions": [
                    {"prediction": "bullish", "confidence": 0.8},
                    {"prediction": "bullish", "confidence": 0.9},
                    {"prediction": "bullish", "confidence": 0.7}
                ],
                "market_outcomes": [-2.0, 3.5, -1.5]
            }
        }


class ConsistencyCheckResponse(BaseModel):
    """Response for consistency validation"""
    is_consistent: bool
    confidence: float
    severity: str
    explanation: str
    discrepancies: list[str]
    agent_claim: str
    timestamp: datetime


class DiscrepancyCheckResponse(BaseModel):
    """Response for discrepancy detection"""
    has_discrepancy: bool
    severity: str
    error_type: Optional[str]
    explanation: str
    confidence: float
    timestamp: datetime


class BiasCheckResponse(BaseModel):
    """Response for bias analysis"""
    has_bias: bool
    bias_type: str
    bias_strength: float
    sample_size: int
    accuracy_rate: float
    false_positive_rate: float
    false_negative_rate: float
    explanation: str
    recommendations: list[str]
    timestamp: datetime


# API Endpoints

@app.post("/ai/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Full multi-agent analysis with trading signals.

    Executes complete StateGraph:
    1. Technical, Sentiment, Macro, Risk analysts
    2. Bull/Bear debate
    3. Manager synthesis
    4. Signal generation with risk management
    5. Reflection and memory storage

    Returns:
    - Analyst insights
    - Debate arguments
    - Final decision
    - Executable trading signal
    - Confidence score
    """
    start_time = asyncio.get_event_loop().time()

    try:
        # Execute full graph analysis
        logger.info(f"Starting analysis for {request.symbol}")
        final_state = await analyze_symbol(request.symbol, request.market_data)

        # Extract results
        analysts_output = {
            "technical": final_state.get('messages', [{}])[0].get('content', '') if final_state.get('messages') else '',
            "sentiment": final_state.get('messages', [{}])[1].get('content', '') if len(final_state.get('messages', [])) > 1 else '',
            "macro": final_state.get('messages', [{}])[2].get('content', '') if len(final_state.get('messages', [])) > 2 else '',
            "risk": final_state.get('messages', [{}])[3].get('content', '') if len(final_state.get('messages', [])) > 3 else ''
        }

        debate_output = {
            "bull": final_state.get('debates', [''])[0] if final_state.get('debates') else '',
            "bear": final_state.get('debates', ['', ''])[1] if len(final_state.get('debates', [])) > 1 else ''
        }

        # Process signal if signal_processor available
        trading_signal = {}
        if signal_processor and final_state.get('final_decision'):
            trading_signal = signal_processor.process_decision(
                final_state['final_decision'],
                request.market_data
            )

        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

        return AnalyzeResponse(
            symbol=request.symbol,
            timestamp=datetime.utcnow(),
            analysts=analysts_output,
            debate=debate_output,
            final_decision=final_state.get('final_decision', ''),
            trading_signal=trading_signal,
            confidence=final_state.get('confidence', 0.5),
            regime=final_state.get('regime', 'unknown'),
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Analysis failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/ai/debate", response_model=DebateResponse)
async def debate(request: DebateRequest):
    """
    Bull/Bear debate only (no manager synthesis).

    Runs adversarial debate between Bull and Bear agents without
    final decision. Useful for exploring contrasting viewpoints.

    Returns:
    - Bull agent's optimistic argument
    - Bear agent's pessimistic argument
    - Execution time
    """
    start_time = asyncio.get_event_loop().time()

    try:
        # Create minimal state for debate
        state = create_initial_state(request.symbol, request.market_data)

        # Run Bull and Bear agents in parallel
        logger.info(f"Starting debate for {request.symbol}")
        bull_result, bear_result = await asyncio.gather(
            bull_agent.ainvoke(state),
            bear_agent.ainvoke(state)
        )

        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

        return DebateResponse(
            symbol=request.symbol,
            timestamp=datetime.utcnow(),
            bull_argument=bull_result.get('content', '') if isinstance(bull_result, dict) else str(bull_result),
            bear_argument=bear_result.get('content', '') if isinstance(bear_result, dict) else str(bear_result),
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Debate failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Debate failed: {str(e)}")


@app.get("/ai/memory/query", response_model=MemoryQueryResponse)
async def query_memory(
    query: str = Query(..., description="Search query text"),
    n_results: int = Query(default=5, ge=1, le=20, description="Number of results"),
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Minimum confidence")
):
    """
    Query similar past decisions from ChromaDB memory.

    Performs semantic similarity search across historical agent decisions.
    Supports filtering by symbol and confidence threshold.

    Returns:
    - List of similar decisions with metadata
    - Result count
    """
    if not trading_memory:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    try:
        # Build filter metadata
        filter_metadata = {}
        if symbol:
            filter_metadata['symbol'] = symbol
        if min_confidence is not None:
            filter_metadata['confidence'] = {'$gte': min_confidence}

        # Query memory
        logger.info(f"Querying memory: query='{query}', n={n_results}, filters={filter_metadata}")
        results = trading_memory.query_similar(
            query=query,
            n_results=n_results,
            filter_metadata=filter_metadata if filter_metadata else None
        )

        return MemoryQueryResponse(
            results=results,
            count=len(results)
        )

    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Memory query failed: {str(e)}")


@app.get("/ai/agents/status", response_model=AgentStatusResponse)
async def agent_status():
    """
    Health check for all agents and system components.

    Tests connectivity to:
    - Ollama LLM service
    - ChromaDB memory
    - All 7 agents (4 analysts + 3 debaters)

    Returns:
    - Overall status (healthy/degraded/unhealthy)
    - Individual agent status
    - Memory system status
    - Uptime
    """
    start_time = asyncio.get_event_loop().time()

    agents_status = {}
    overall_status = "healthy"

    # Test all agents with proper input format
    # Agents expect {"input": "..."} not full AgentState
    test_prompts = {
        "technical": "Quick health check: Analyze BTCUSDT at $50000 with RSI 50",
        "sentiment": "Quick health check: Assess market sentiment for BTCUSDT",
        "macro": "Quick health check: Evaluate macro conditions for crypto",
        "risk": "Quick health check: Calculate position size for BTCUSDT",
        "bull": "Quick health check: Provide bullish case for BTCUSDT",
        "bear": "Quick health check: Provide bearish case for BTCUSDT",
        "manager": "Quick health check: Synthesize a trading decision"
    }

    # Test analyst agents
    analyst_agents = {
        "technical": technical_analyst,
        "sentiment": sentiment_analyst,
        "macro": macro_analyst,
        "risk": risk_analyst
    }

    for name, agent in analyst_agents.items():
        try:
            result = await asyncio.wait_for(
                agent.ainvoke({"input": test_prompts[name]}),
                timeout=15.0  # Increased timeout for LLM inference
            )
            agents_status[name] = {"status": "healthy", "response_type": type(result).__name__}
        except TimeoutError:
            agents_status[name] = {"status": "timeout", "error": "Response timeout"}
            overall_status = "degraded"
        except Exception as e:
            agents_status[name] = {"status": "unhealthy", "error": str(e)}
            overall_status = "unhealthy"

    # Test debate agents
    debate_agents = {
        "bull": bull_agent,
        "bear": bear_agent,
        "manager": manager_agent
    }

    for name, agent in debate_agents.items():
        try:
            result = await asyncio.wait_for(
                agent.ainvoke({"input": test_prompts[name]}),
                timeout=15.0  # Increased timeout for LLM inference
            )
            agents_status[name] = {"status": "healthy", "response_type": type(result).__name__}
        except TimeoutError:
            agents_status[name] = {"status": "timeout", "error": "Response timeout"}
            overall_status = "degraded"
        except Exception as e:
            agents_status[name] = {"status": "unhealthy", "error": str(e)}
            overall_status = "unhealthy"

    # Test memory system
    memory_status = {}
    if trading_memory:
        try:
            # Try to query memory
            trading_memory.query_similar("test", n_results=1)
            memory_status = {
                "status": "healthy",
                "collections": "trading_decisions",
                "can_query": True
            }
        except Exception as e:
            memory_status = {
                "status": "unhealthy",
                "error": str(e)
            }
            overall_status = "unhealthy"
    else:
        memory_status = {"status": "not_initialized"}
        overall_status = "degraded"

    uptime = (asyncio.get_event_loop().time() - start_time) * 1000

    return AgentStatusResponse(
        status=overall_status,
        agents=agents_status,
        memory_status=memory_status,
        uptime_ms=uptime
    )


# LLM-Judge Endpoints (Phase 7.2)

@app.post("/ai/judge/consistency", response_model=ConsistencyCheckResponse)
async def check_consistency(request: ConsistencyCheckRequest):
    """
    Validate factual consistency between agent claim and market data.

    Uses meta-LLM to verify if agent's claims match ground truth data.
    Detects hallucinations, numerical mismatches, and logical inconsistencies.

    Returns:
    - is_consistent: Boolean validation result
    - confidence: Judge confidence (0-1)
    - severity: low/medium/high/critical
    - discrepancies: List of specific issues found
    """
    try:
        logger.info(f"Checking consistency for {request.agent_name}")

        report = await llm_judge.verify_factual_consistency(
            agent_name=request.agent_name,
            agent_claim=request.agent_claim,
            market_data=request.market_data
        )

        return ConsistencyCheckResponse(
            is_consistent=report.is_consistent,
            confidence=report.confidence,
            severity=report.severity,
            explanation=report.explanation,
            discrepancies=report.discrepancies,
            agent_claim=request.agent_claim,
            timestamp=report.timestamp
        )
    except Exception as e:
        logger.error(f"Consistency check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/judge/discrepancy", response_model=DiscrepancyCheckResponse)
async def check_discrepancy(request: DiscrepancyCheckRequest):
    """
    Detect discrepancies between agent analysis and actual outcomes.

    Compares agent predictions against real market movements to identify:
    - Hallucinations (data not in reality)
    - Misinterpretations (wrong conclusion from correct data)
    - Logic errors (faulty reasoning)

    Returns:
    - has_discrepancy: Whether significant mismatch exists
    - severity: Impact level (low/medium/high/critical)
    - error_type: Classification of the error
    - explanation: Detailed analysis
    """
    try:
        logger.info(f"Checking discrepancy for analysis: {request.agent_analysis[:50]}...")

        report = await llm_judge.detect_discrepancies(
            agent_analysis=request.agent_analysis,
            actual_outcome=request.actual_outcome,
            context=request.context
        )

        return DiscrepancyCheckResponse(
            has_discrepancy=report.has_discrepancy,
            severity=report.severity,
            error_type=report.error_type,
            explanation=report.explanation,
            confidence=report.confidence,
            timestamp=report.timestamp
        )
    except Exception as e:
        logger.error(f"Discrepancy check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/judge/bias", response_model=BiasCheckResponse)
async def check_bias(request: BiasCheckRequest):
    """
    Analyze systematic bias in agent decision-making patterns.

    Examines historical decisions vs outcomes to detect:
    - Over-optimism (bull bias)
    - Over-pessimism (bear bias)
    - False positive/negative patterns

    Returns:
    - has_bias: Whether systematic bias detected
    - bias_type: optimistic/pessimistic/neutral
    - bias_strength: 0-1 (how strong)
    - accuracy_rate: Percentage of correct predictions
    - recommendations: Mitigation strategies
    """
    try:
        logger.info(f"Checking bias for {request.agent_name} ({len(request.agent_decisions)} decisions)")

        report = await llm_judge.analyze_bias(
            agent_name=request.agent_name,
            agent_decisions=request.agent_decisions,
            market_outcomes=request.market_outcomes
        )

        return BiasCheckResponse(
            has_bias=report.has_bias,
            bias_type=report.bias_type,
            bias_strength=report.bias_strength,
            sample_size=report.sample_size,
            accuracy_rate=report.accuracy_rate,
            false_positive_rate=report.false_positive_rate,
            false_negative_rate=report.false_negative_rate,
            explanation=report.explanation,
            recommendations=report.recommendations,
            timestamp=report.timestamp
        )
    except Exception as e:
        logger.error(f"Bias check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Ground Truth Validation Endpoints (Phase 7.3)
# ============================================================================

from evaluators.ground_truth import GroundTruthValidator, ValidationResult


class GroundTruthRequest(BaseModel):
    """Request model for ground truth validation"""
    agent_name: str = Field(..., description="Name of agent to validate")
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    start_date: str = Field(..., description="Start date (ISO format: YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (ISO format: YYYY-MM-DD)")
    timeframe: str = Field(default="1h", description="Timeframe (e.g., 1h, 4h, 1d)")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "technical_analyst",
                "symbol": "BTCUSDT",
                "start_date": "2024-10-01",
                "end_date": "2024-10-31",
                "timeframe": "1h"
            }
        }


class GroundTruthResponse(BaseModel):
    """Response model for ground truth validation"""
    agent_name: str
    symbol: str
    start_date: str
    end_date: str
    timeframe: str
    total_predictions: int
    total_optimal_trades: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list[list[int]]
    agent_total_profit_percent: float
    optimal_total_profit_percent: float
    efficiency_ratio: float
    correct_predictions: int
    incorrect_predictions: int
    missed_opportunities: int
    avg_confidence_correct: float
    avg_confidence_incorrect: float
    prediction_distribution: dict[str, int]


# Initialize ground truth validator (global instance)
try:
    ground_truth_validator = GroundTruthValidator(
        min_confidence=0.6,
        profit_threshold=2.0,
        slippage_percent=0.1,
        fee_percent=0.1
    )
    logger.info("Initialized GroundTruthValidator")
except Exception as e:
    logger.error(f"Failed to initialize GroundTruthValidator: {e}")
    ground_truth_validator = None


@app.post("/ai/validate/ground-truth", response_model=GroundTruthResponse)
async def validate_ground_truth(request: GroundTruthRequest):
    """
    Validate agent predictions against optimal trades (perfect hindsight).

    Compares historical agent predictions from ChromaDB to optimal trades
    calculated from TimescaleDB price data. Provides comprehensive metrics:
    - Accuracy, Precision, Recall, F1 Score
    - Confusion Matrix (TP, FP, TN, FN)
    - Profitability comparison (agent vs optimal)
    - Efficiency ratio (how well agent captured max profit)

    Example:
        POST /ai/validate/ground-truth
        {
            "agent_name": "technical_analyst",
            "symbol": "BTCUSDT",
            "start_date": "2024-10-01",
            "end_date": "2024-10-31",
            "timeframe": "1h"
        }
    """
    if ground_truth_validator is None:
        raise HTTPException(
            status_code=503,
            detail="GroundTruthValidator not initialized"
        )

    try:
        # Parse dates
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)

        # Validate date range
        if start_date >= end_date:
            raise HTTPException(
                status_code=400,
                detail="start_date must be before end_date"
            )

        # Run validation
        logger.info(f"Running ground truth validation for {request.agent_name} on {request.symbol}")
        result = await ground_truth_validator.validate_agent(
            agent_name=request.agent_name,
            symbol=request.symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=request.timeframe
        )

        # Convert to response model
        return GroundTruthResponse(
            agent_name=result.agent_name,
            symbol=result.symbol,
            start_date=result.start_date.isoformat(),
            end_date=result.end_date.isoformat(),
            timeframe=result.timeframe,
            total_predictions=result.total_predictions,
            total_optimal_trades=result.total_optimal_trades,
            true_positives=result.true_positives,
            false_positives=result.false_positives,
            true_negatives=result.true_negatives,
            false_negatives=result.false_negatives,
            accuracy=result.accuracy,
            precision=result.precision,
            recall=result.recall,
            f1_score=result.f1_score,
            confusion_matrix=result.confusion_matrix,
            agent_total_profit_percent=result.agent_total_profit_percent,
            optimal_total_profit_percent=result.optimal_total_profit_percent,
            efficiency_ratio=result.efficiency_ratio,
            correct_predictions=result.correct_predictions,
            incorrect_predictions=result.incorrect_predictions,
            missed_opportunities=result.missed_opportunities,
            avg_confidence_correct=result.avg_confidence_correct,
            avg_confidence_incorrect=result.avg_confidence_incorrect,
            prediction_distribution=result.prediction_distribution
        )

    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Ground truth validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Simple health check endpoint for Docker/Kubernetes"""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "fks_ai",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.get("/")
async def root():
    """API root with documentation links"""
    return JSONResponse(
        content={
            "service": "FKS AI - Multi-Agent Trading System",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "endpoints": {
                "analyze": "POST /ai/analyze - Full multi-agent analysis",
                "debate": "POST /ai/debate - Bull/Bear debate only",
                "memory": "GET /ai/memory/query - Query past decisions",
                "status": "GET /ai/agents/status - Agent health check",
                "judge_consistency": "POST /ai/judge/consistency - Validate factual accuracy",
                "judge_discrepancy": "POST /ai/judge/discrepancy - Detect prediction errors",
                "judge_bias": "POST /ai/judge/bias - Analyze systematic bias",
                "ground_truth": "POST /ai/validate/ground-truth - Backtest predictions vs optimal trades"
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
