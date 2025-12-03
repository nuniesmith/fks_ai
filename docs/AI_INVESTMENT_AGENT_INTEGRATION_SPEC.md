# AI Investment Agent Integration Specification

**Source**: https://github.com/rgoerwit/ai-investment-agent  
**Target**: fks_ai service  
**Last Updated**: 2025-12-03  
**Status**: Technical Spec Complete

---

## Executive Summary

This document details the integration of the `ai-investment-agent` multi-agent workflow into the existing fks_ai LangGraph system. The goal is to adopt their superior prompt engineering, scoring gates, and parallel data gathering while leveraging FKS's existing data layer, execution backends, and infrastructure.

---

## Architecture Comparison

### ai-investment-agent Workflow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        PARALLEL DATA GATHERING                           │
├──────────────┬──────────────┬──────────────┬────────────────────────────┤
│ Market       │ Sentiment    │ News         │ Fundamentals               │
│ Analyst      │ Analyst      │ Analyst      │ Analyst                    │
│ (Technical)  │ (Social)     │ (News)       │ (Financials)               │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬─────────────────────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                              │
                    ┌─────────▼─────────┐
                    │    BULL/BEAR      │
                    │    DEBATE (2x)    │◀──── Strict Role Commitment
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ RESEARCH MANAGER  │──── Hard Scoring Gates
                    │ (Investment Plan) │      (Health≥50%, Growth≥50%)
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │     TRADER        │──── Entry/Exit/Size
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐           ┌────▼────┐          ┌────▼────┐
   │  RISKY  │           │  SAFE   │          │ NEUTRAL │
   │ ANALYST │           │ ANALYST │          │ ANALYST │
   └────┬────┘           └────┬────┘          └────┬────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │PORTFOLIO MANAGER  │──── Final Decision + Position Size
                    └───────────────────┘
```

### Current fks_ai Workflow

```
┌────────────────────────────────────────────────────┐
│                fks_data Single Call                │
└────────────────────────┬───────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  BULL/BEAR DEBATE   │  (Less strict prompts)
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │      MANAGER        │  (No hard gates)
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Signal Output     │
              └─────────────────────┘
```

---

## Integration Points

### 1. Parallel Data Gathering Agents

**Current FKS**: Single `fks_data` call returns everything.

**Target**: Create 4 parallel analyst nodes calling fks_data:

| Agent | fks_data Endpoint | Purpose |
|-------|-------------------|---------|
| `market_analyst` | `/data/technical/{symbol}` | RSI, MACD, Support/Resistance, Liquidity |
| `sentiment_analyst` | `/data/sentiment/{symbol}` | Social media, StockTwits-equivalent |
| `news_analyst` | `/data/news/{symbol}` | Recent news, press releases |
| `fundamentals_analyst` | `/data/fundamentals/{symbol}` | P/E, ROE, Health Score, Growth Score |

**FKS Data Service Mapping**:

```python
# In fks_ai/agents/data_gathering.py

async def market_analyst_tool(symbol: str) -> dict:
    """Call fks_data for technical analysis."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{FKS_DATA_URL}/data/technical/{symbol}",
            params={"indicators": "rsi,macd,bollinger,volume"}
        )
        return response.json()

async def fundamentals_analyst_tool(symbol: str) -> dict:
    """Call fks_data for fundamental metrics."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{FKS_DATA_URL}/data/fundamentals/{symbol}"
        )
        return response.json()

# Similar for sentiment_analyst_tool, news_analyst_tool
```

### 2. Bull/Bear Debate Prompts

**Current FKS Prompts**: Generic debate structure.

**Target**: Adopt ai-investment-agent strict prompts.

**Key Changes to Adopt**:

```python
# Bull Researcher - Strict Role Commitment
BULL_RESEARCHER_PROMPT = """
You are a BULL RESEARCHER in a multi-agent trading system.
You are optimistic but data-driven.

## THESIS COMPLIANCE CRITERIA

Your role is to advocate aggressively for BUY opportunities that align with:

**Quantitative Requirements**:
- Financial health ≥7/12 (preferably ≥8/12)
- Growth score ≥3/6 (preferably ≥4/6)
- P/E ≤18 OR (P/E 18-25 with PEG ≤1.2)
- Liquidity >$250k daily average

**OUTPUT STRUCTURE**:

**THESIS COMPLIANCE** (Lead with this):
✓ Financial Health: [X]/12 (≥7 required)
✓ Growth Score: [Y]/6 (≥3 required)
✓ P/E: [Z] (≤18 or ≤25 with PEG≤1.2)

**BULL CASE SUMMARY**:
[2-3 strongest arguments with data]

**COUNTER TO BEAR CONCERNS**:
[Direct responses to expected bear arguments]

**CONVICTION**: [High/Medium/Low]

Keep concise (300-800 words).
"""

# Bear Researcher - Focus on Hard Fails
BEAR_RESEARCHER_PROMPT = """
You are a BEAR RESEARCHER in a multi-agent trading system.
You are cautious and risk-aware. Prioritize protecting capital.

## THESIS COMPLIANCE CRITERIA (Your Focus)

Focus on identifying violations:

**Quantitative Hard Fails**:
- Financial health <7/12 (below minimum)
- Growth score <3/6 (below minimum)
- P/E >18 without PEG ≤1.2 (overvalued)
- P/E >25 (always overvalued, no exceptions)
- Liquidity <$100k daily (insufficient)

**Qualitative Risks**:
- Jurisdiction risks
- Structural challenges
- Cyclical peaks
- Execution risks

**OUTPUT STRUCTURE**:

**THESIS VIOLATIONS**:
✗ [List specific violations with numbers]

**BEAR CASE SUMMARY**:
[2-3 strongest risk factors]

**KEY RISKS**:
1. [Risk with evidence]
2. [Risk with evidence]

**CONVICTION**: [High/Medium/Low]

Keep concise (300-800 words).
"""
```

### 3. Research Manager with Hard Gates

**Current FKS**: Simple consensus, no hard gates.

**Target**: Add scoring gates that reject garbage early.

```python
# In fks_ai/agents/research_manager.py

class ResearchManagerAgent:
    """Research manager with hard scoring gates."""
    
    HARD_FAIL_THRESHOLDS = {
        "financial_health_min": 50,  # 50% = 6/12
        "growth_score_min": 50,      # 50% = 3/6
        "liquidity_min_usd": 100_000,
        "pe_max": 25,
    }
    
    async def evaluate(self, state: AgentState) -> dict:
        fundamentals = state.get("fundamentals_report", {})
        
        # Extract scores
        health = fundamentals.get("health_score", 0)
        growth = fundamentals.get("growth_score", 0)
        pe = fundamentals.get("pe_ratio")
        peg = fundamentals.get("peg_ratio")
        liquidity = fundamentals.get("avg_daily_volume_usd", 0)
        
        # Hard fail checks
        hard_fails = []
        
        if health < self.HARD_FAIL_THRESHOLDS["financial_health_min"]:
            hard_fails.append(f"Health {health}% < 50% minimum")
        
        if growth < self.HARD_FAIL_THRESHOLDS["growth_score_min"]:
            hard_fails.append(f"Growth {growth}% < 50% minimum")
        
        if pe and pe > 25:
            hard_fails.append(f"P/E {pe} > 25 (hard fail)")
        elif pe and pe > 18 and (not peg or peg > 1.2):
            hard_fails.append(f"P/E {pe} > 18 without PEG ≤ 1.2")
        
        if liquidity < self.HARD_FAIL_THRESHOLDS["liquidity_min_usd"]:
            hard_fails.append(f"Liquidity ${liquidity:,.0f} < $100k minimum")
        
        if hard_fails:
            return {
                "recommendation": "SKIP",
                "hard_fails": hard_fails,
                "conviction_score": 0,
                "data_quality": "FAIL"
            }
        
        # Continue with full analysis if passed
        return await self._full_evaluation(state)
```

### 4. Triple Risk Analyst (Risky/Safe/Neutral)

**Current FKS**: No separate risk perspectives.

**Target**: Add 3 parallel risk agents for position sizing.

```python
# In fks_ai/agents/risk_analysis.py

class RiskAnalysisTeam:
    """Three-perspective risk analysis for position sizing."""
    
    POSITION_SIZE_PROFILES = {
        "aggressive": {"max_pct": 5.0, "risk_multiplier": 1.5},
        "neutral": {"max_pct": 3.0, "risk_multiplier": 1.0},
        "conservative": {"max_pct": 1.5, "risk_multiplier": 0.5},
    }
    
    async def analyze(self, trader_plan: dict) -> dict:
        """Run 3 parallel risk analysts and synthesize."""
        
        risky_task = self._risky_perspective(trader_plan)
        safe_task = self._safe_perspective(trader_plan)
        neutral_task = self._neutral_perspective(trader_plan)
        
        results = await asyncio.gather(risky_task, safe_task, neutral_task)
        
        return {
            "risky_position_pct": results[0]["position_pct"],
            "safe_position_pct": results[1]["position_pct"],
            "neutral_position_pct": results[2]["position_pct"],
            "recommended_position_pct": results[2]["position_pct"],  # Default to neutral
            "risk_perspectives": {
                "risky": results[0]["rationale"],
                "safe": results[1]["rationale"],
                "neutral": results[2]["rationale"],
            }
        }
```

### 5. Structured JSON Output

**Current FKS**: Markdown/text output.

**Target**: Add JSON mode for batch screening.

```python
# In fks_ai/api/trading_analysis.py

from pydantic import BaseModel
from typing import List, Optional

class TradingSignalOutput(BaseModel):
    """Structured output for batch screening."""
    
    symbol: str
    conviction_score: int  # 0-100
    health_score: int      # 0-100
    growth_score: int      # 0-100
    
    position_size_aggressive: float  # Percentage of portfolio
    position_size_neutral: float
    position_size_conservative: float
    
    data_quality: str  # "GOOD", "MARGINAL", "POOR"
    hard_fails: List[str]
    
    recommendation: str  # "BUY", "HOLD", "SELL", "SKIP"
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    
    bull_summary: str
    bear_summary: str
    reasoning_trace_url: Optional[str]  # Link to full debate

@router.post("/ai/trading/batch-analyze")
async def batch_analyze(symbols: List[str]) -> List[TradingSignalOutput]:
    """
    Analyze multiple symbols and return structured JSON.
    Used for daily opportunity ranking.
    """
    results = []
    
    for symbol in symbols:
        try:
            result = await run_full_analysis(symbol)
            results.append(TradingSignalOutput(
                symbol=symbol,
                conviction_score=result.conviction_score,
                health_score=result.health_score,
                growth_score=result.growth_score,
                position_size_aggressive=result.risk_analysis["risky_position_pct"],
                position_size_neutral=result.risk_analysis["neutral_position_pct"],
                position_size_conservative=result.risk_analysis["safe_position_pct"],
                data_quality=result.data_quality,
                hard_fails=result.hard_fails,
                recommendation=result.recommendation,
                entry_price=result.entry_price,
                stop_loss=result.stop_loss,
                take_profit=result.take_profit,
                bull_summary=result.bull_summary[:500],
                bear_summary=result.bear_summary[:500],
                reasoning_trace_url=f"/traces/{result.trace_id}"
            ))
        except Exception as e:
            results.append(TradingSignalOutput(
                symbol=symbol,
                conviction_score=0,
                health_score=0,
                growth_score=0,
                position_size_aggressive=0,
                position_size_neutral=0,
                position_size_conservative=0,
                data_quality="POOR",
                hard_fails=[str(e)],
                recommendation="SKIP",
                bull_summary="",
                bear_summary=""
            ))
    
    # Sort by conviction_score descending
    results.sort(key=lambda x: x.conviction_score, reverse=True)
    return results
```

---

## Implementation Plan

### Day 1-2: Setup & Exploration (DONE)

- [x] Clone ai-investment-agent repo
- [x] Study LangGraph workflow structure (`graph.py`)
- [x] Document agent prompts (`prompts.py`)
- [x] Map data tools to fks_data (`toolkit.py`)
- [x] Create this technical spec

### Day 3: Parallel Data Agents

Create `/services/ai/src/agents/data_gathering/`:

```
data_gathering/
├── __init__.py
├── market_analyst.py     # Technical analysis via fks_data
├── sentiment_analyst.py  # Social sentiment via fks_data
├── news_analyst.py       # News via fks_data
├── fundamentals_analyst.py  # Financials via fks_data
└── tools.py              # Shared tool definitions
```

### Day 4: Bull/Bear Debate Upgrade

Update `/services/ai/src/agents/debate/`:

```python
# Existing files to modify:
# - bull_agent.py  -> Add strict role commitment prompts
# - bear_agent.py  -> Add hard fail focus prompts
# - manager.py     -> Add scoring gates
```

### Day 5: Risk Team & JSON Output

Create `/services/ai/src/agents/risk/`:

```
risk/
├── __init__.py
├── risky_analyst.py
├── safe_analyst.py
├── neutral_analyst.py
└── portfolio_manager.py
```

Update API endpoints for structured JSON output.

---

## FKS Data Service Requirements

The following fks_data endpoints are needed (verify they exist):

| Endpoint | Required Fields |
|----------|-----------------|
| `GET /data/technical/{symbol}` | `rsi`, `macd`, `bollinger`, `support_levels`, `resistance_levels`, `avg_volume_30d` |
| `GET /data/fundamentals/{symbol}` | `pe_ratio`, `peg_ratio`, `roe`, `roa`, `current_ratio`, `debt_equity`, `revenue_growth`, `earnings_growth`, `health_score`, `growth_score` |
| `GET /data/sentiment/{symbol}` | `social_score`, `news_sentiment`, `analyst_coverage`, `retail_flow` |
| `GET /data/news/{symbol}` | `articles[]`, `press_releases[]`, `summary` |

If any endpoints are missing, add to fks_data backlog.

---

## Discord Integration

Daily output format (8 AM IST via cron):

```
🚨 DAILY OPPORTUNITIES - {date}

1. {symbol} ({company_name}) - Conviction: {score}/100 {stars}
   Health: {health}% | Growth: {growth}% | Liquidity: ${liquidity}
   Position: {aggressive}% (Aggressive) / {neutral}% (Neutral)
   Data Quality: {quality}
   → {reasoning_url}

2. ...

❌ Skipped {count} symbols (Data Quality: POOR)
⚠️ {count} symbols below threshold
```

Webhook: `POST https://discord.com/api/webhooks/{id}/{token}`

---

## Files Modified/Created

### New Files

```
services/ai/src/agents/data_gathering/__init__.py
services/ai/src/agents/data_gathering/market_analyst.py
services/ai/src/agents/data_gathering/sentiment_analyst.py
services/ai/src/agents/data_gathering/news_analyst.py
services/ai/src/agents/data_gathering/fundamentals_analyst.py
services/ai/src/agents/risk/__init__.py
services/ai/src/agents/risk/risky_analyst.py
services/ai/src/agents/risk/safe_analyst.py
services/ai/src/agents/risk/neutral_analyst.py
services/ai/src/agents/risk/portfolio_manager.py
services/ai/src/models/trading_signal.py
```

### Modified Files

```
services/ai/src/agents/debate/bull_agent.py  (prompt upgrade)
services/ai/src/agents/debate/bear_agent.py  (prompt upgrade)
services/ai/src/agents/debate/manager.py     (add hard gates)
services/ai/src/graphs/trading_graph.py      (add parallel nodes, risk team)
services/ai/src/api/trading_analysis.py      (add batch-analyze endpoint)
```

---

## Testing Strategy

1. **Unit Tests**: Each agent in isolation with mock data
2. **Integration Test**: Full workflow with test symbol (AAPL)
3. **Comparison Test**: Same symbol through old vs new workflow
4. **Batch Test**: 10 symbols through batch-analyze endpoint
5. **Discord Test**: Webhook integration with test channel

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| fks_data missing endpoints | Create placeholder tools, add to backlog |
| LLM rate limits | Use existing LiteLLM rate limiter, queue parallel calls |
| Memory contamination | Use ticker-specific ChromaDB collections (per their fix) |
| Prompt regression | Version prompts, A/B test before rollout |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Analysis time per symbol | < 60 seconds |
| Batch analysis (10 symbols) | < 5 minutes |
| Hard fail rejection rate | 30-50% (good = rejecting noise) |
| Conviction score distribution | Normal around 50-70 |
| Discord message delivery | < 5 seconds from analysis complete |

---

## References

- AI Investment Agent: https://github.com/rgoerwit/ai-investment-agent
- LangGraph Patterns: https://langchain-ai.github.io/langgraph/
- FKS AI Service: `/services/ai/`
- FKS Data Service: `/services/data_ingestion/`
