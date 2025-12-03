"""
Rules Engine API Routes

FastAPI routes for the FKS trading rules engine.
Provides endpoints for trade validation, rule checking, and override management.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...rules import (
    RulesEngine,
    MarketType,
    TradeValidationResult,
    HardRuleID,
    SoftRuleID,
    SoftRuleOverride,
    GateStatus,
)

router = APIRouter(prefix="/rules", tags=["rules"])

# Initialize rules engine
_engine: Optional[RulesEngine] = None


def get_engine() -> RulesEngine:
    """Get or create the rules engine singleton."""
    global _engine
    if _engine is None:
        _engine = RulesEngine()
    return _engine


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class TradeValidationRequest(BaseModel):
    """Request model for trade validation."""
    # Required parameters
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT, EURUSD)")
    market: str = Field(..., description="Market type: crypto_spot, crypto_perps, forex, futures, stocks")
    position_size_pct: float = Field(..., ge=0, le=100, description="Position size as % of account")
    account_balance: float = Field(..., gt=0, description="Account balance in USD")
    stop_loss_set: bool = Field(..., description="Whether stop-loss is set")
    risk_per_trade_pct: float = Field(..., ge=0, description="Risk per trade as % of account")
    data_quality_pct: float = Field(..., ge=0, le=100, description="Data quality percentage")
    liquidity_usd: float = Field(..., ge=0, description="Liquidity in USD")
    leverage: float = Field(1.0, ge=1, description="Leverage being used")
    
    # Optional hard rule parameters
    is_adding_to_loser: bool = Field(False, description="Is this adding to a losing position")
    daily_pnl_pct: Optional[float] = Field(None, description="Daily P&L percentage")
    weekly_pnl_pct: Optional[float] = Field(None, description="Weekly P&L percentage")
    total_24h_pnl_pct: Optional[float] = Field(None, description="Total 24h P&L percentage")
    
    # Soft rule parameters
    trend_direction: Optional[str] = Field(None, description="Trend direction: up, down, sideways")
    entry_timeframe: str = Field("1H", description="Entry timeframe")
    session_minutes_elapsed: int = Field(60, ge=0, description="Minutes since session open")
    has_confirmation: bool = Field(True, description="Has confirmation signal")
    timeframes_analyzed: int = Field(2, ge=1, description="Number of timeframes analyzed")
    trade_documented: bool = Field(True, description="Is trade documented")


class CryptoSpotGateRequest(BaseModel):
    """Request model for crypto spot gates."""
    symbol: str
    volume_24h: float = Field(..., gt=0, description="24h volume in USD")
    market_cap: float = Field(..., gt=0, description="Market cap in USD")
    spread_pct: float = Field(..., ge=0, description="Bid-ask spread percentage")
    exchange_listings: int = Field(..., ge=1, description="Number of exchange listings")
    current_concentration_pct: float = Field(..., ge=0, le=100, description="Current portfolio concentration %")


class CryptoPerpsGateRequest(BaseModel):
    """Request model for crypto perps gates."""
    symbol: str
    leverage: float = Field(..., ge=1, description="Current leverage")
    funding_rate_pct: float = Field(..., description="Funding rate percentage (8h)")
    open_interest: float = Field(..., gt=0, description="Open interest in USD")
    position_size_pct: float = Field(..., ge=0, le=100, description="Position size as % of account")
    liquidation_distance_pct: float = Field(..., ge=0, description="Distance to liquidation %")
    account_equity: float = Field(..., gt=0, description="Account equity in USD")


class ForexGateRequest(BaseModel):
    """Request model for forex gates."""
    symbol: str
    spread_pips: float = Field(..., ge=0, description="Current spread in pips")
    leverage: float = Field(..., ge=1, description="Current leverage")
    daily_drawdown_pct: float = Field(..., ge=0, description="Daily drawdown percentage")
    risk_reward_ratio: float = Field(..., gt=0, description="Risk/reward ratio")
    minutes_to_news: Optional[int] = Field(None, description="Minutes until next news")
    carry_direction: Optional[str] = Field(None, description="Carry direction: positive, negative, neutral")


class FuturesGateRequest(BaseModel):
    """Request model for futures gates."""
    symbol: str
    daily_volume: int = Field(..., gt=0, description="Daily volume in contracts")
    margin_used_pct: float = Field(..., ge=0, le=100, description="Margin used as % of account")
    num_contracts: int = Field(..., ge=1, description="Number of contracts")
    is_rth: Optional[bool] = Field(None, description="Is regular trading hours")


class SoftRuleOverrideRequest(BaseModel):
    """Request model for recording soft rule overrides."""
    rule_id: str = Field(..., description="Soft rule ID (e.g., SR-01)")
    rule_name: str = Field(..., description="Name of the rule")
    recommendation: str = Field(..., description="Original recommendation")
    override_reason: str = Field(..., description="Justification for override")
    approved_by: str = Field("manual", description="Who approved the override")


# =============================================================================
# ROUTES
# =============================================================================

@router.post("/validate")
async def validate_trade(request: TradeValidationRequest) -> Dict[str, Any]:
    """
    Validate a trade against all rules.
    
    Checks:
    1. Hard rules (12 non-negotiable rules)
    2. Market-specific gates
    3. Soft rules (10 guidelines)
    
    Returns validation result with all violations and details.
    """
    engine = get_engine()
    
    # Validate market type
    valid_markets = ["crypto_spot", "crypto_perps", "forex", "futures", "stocks"]
    if request.market not in valid_markets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid market type: {request.market}. Valid: {valid_markets}"
        )
    
    result = engine.validate_trade(
        symbol=request.symbol,
        market=request.market,
        position_size_pct=request.position_size_pct,
        account_balance=request.account_balance,
        stop_loss_set=request.stop_loss_set,
        risk_per_trade_pct=request.risk_per_trade_pct,
        data_quality_pct=request.data_quality_pct,
        liquidity_usd=request.liquidity_usd,
        leverage=request.leverage,
        is_adding_to_loser=request.is_adding_to_loser,
        daily_pnl_pct=request.daily_pnl_pct,
        weekly_pnl_pct=request.weekly_pnl_pct,
        total_24h_pnl_pct=request.total_24h_pnl_pct,
        trend_direction=request.trend_direction,
        entry_timeframe=request.entry_timeframe,
        session_minutes_elapsed=request.session_minutes_elapsed,
        has_confirmation=request.has_confirmation,
        timeframes_analyzed=request.timeframes_analyzed,
        trade_documented=request.trade_documented,
    )
    
    return result.to_dict()


@router.post("/gates/crypto-spot")
async def check_crypto_spot_gates(request: CryptoSpotGateRequest) -> Dict[str, Any]:
    """Check crypto spot market gates."""
    engine = get_engine()
    gates = engine.get_market_gates("crypto_spot")
    
    result = gates.check_all(
        symbol=request.symbol,
        volume_24h=request.volume_24h,
        market_cap=request.market_cap,
        spread_pct=request.spread_pct,
        exchange_listings=request.exchange_listings,
        current_concentration_pct=request.current_concentration_pct,
    )
    
    return result.to_dict()


@router.post("/gates/crypto-perps")
async def check_crypto_perps_gates(request: CryptoPerpsGateRequest) -> Dict[str, Any]:
    """Check crypto perpetual futures gates."""
    engine = get_engine()
    gates = engine.get_market_gates("crypto_perps")
    
    result = gates.check_all(
        symbol=request.symbol,
        leverage=request.leverage,
        funding_rate_pct=request.funding_rate_pct,
        open_interest=request.open_interest,
        position_size_pct=request.position_size_pct,
        liquidation_distance_pct=request.liquidation_distance_pct,
        account_equity=request.account_equity,
    )
    
    return result.to_dict()


@router.post("/gates/forex")
async def check_forex_gates(request: ForexGateRequest) -> Dict[str, Any]:
    """Check forex market gates."""
    engine = get_engine()
    gates = engine.get_market_gates("forex")
    
    result = gates.check_all(
        symbol=request.symbol,
        spread_pips=request.spread_pips,
        leverage=request.leverage,
        daily_drawdown_pct=request.daily_drawdown_pct,
        risk_reward_ratio=request.risk_reward_ratio,
        minutes_to_news=request.minutes_to_news,
        carry_direction=request.carry_direction,
    )
    
    return result.to_dict()


@router.post("/gates/futures")
async def check_futures_gates(request: FuturesGateRequest) -> Dict[str, Any]:
    """Check futures market gates."""
    engine = get_engine()
    gates = engine.get_market_gates("futures")
    
    result = gates.check_all(
        symbol=request.symbol,
        daily_volume=request.daily_volume,
        margin_used_pct=request.margin_used_pct,
        num_contracts=request.num_contracts,
        is_rth=request.is_rth,
    )
    
    return result.to_dict()


@router.post("/override")
async def record_override(request: SoftRuleOverrideRequest) -> Dict[str, Any]:
    """
    Record a soft rule override for audit purposes.
    
    Soft rules can be overridden with justification.
    All overrides are logged for review.
    """
    engine = get_engine()
    
    # Map string rule_id to enum
    try:
        rule_id_enum = SoftRuleID(request.rule_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rule ID: {request.rule_id}. Valid IDs: {[e.value for e in SoftRuleID]}"
        )
    
    override = engine.record_soft_override(
        rule_id=rule_id_enum,
        rule_name=request.rule_name,
        recommendation=request.recommendation,
        override_reason=request.override_reason,
        approved_by=request.approved_by,
    )
    
    return override.to_dict()


@router.get("/overrides/summary")
async def get_override_summary(days: int = 7) -> Dict[str, Any]:
    """Get summary of soft rule overrides for the past N days."""
    engine = get_engine()
    return engine.soft_rules.get_override_summary(days=days)


@router.get("/status")
async def get_trading_status() -> Dict[str, Any]:
    """Get current trading status (pauses, violations, etc.)."""
    engine = get_engine()
    return engine.get_trading_status()


@router.get("/rules/hard")
async def list_hard_rules() -> Dict[str, Any]:
    """List all hard (non-negotiable) rules."""
    return {
        "rules": [
            {"id": "HR-01", "name": "Position Size Limit", "category": "POSITION_SIZE", "description": "Max 5% per trade (10% for BTC/ETH)"},
            {"id": "HR-02", "name": "Daily Drawdown Limit", "category": "RISK_MANAGEMENT", "description": "Max 3% daily loss - trading paused for day"},
            {"id": "HR-03", "name": "Weekly Drawdown Limit", "category": "RISK_MANAGEMENT", "description": "Max 6% weekly loss - trading paused for week"},
            {"id": "HR-04", "name": "Stop Loss Required", "category": "RISK_MANAGEMENT", "description": "100% of trades MUST have stop-loss"},
            {"id": "HR-05", "name": "Risk Per Trade Limit", "category": "POSITION_SIZE", "description": "Max 2% risk per single trade"},
            {"id": "HR-06", "name": "Revenge Trading Prevention", "category": "BEHAVIOR", "description": "30 min cooldown after 2 consecutive losses"},
            {"id": "HR-07", "name": "News Blackout", "category": "TIMING", "description": "No trades 15 min before/after major news"},
            {"id": "HR-08", "name": "Data Quality Requirement", "category": "DATA_QUALITY", "description": "Min 66% data quality for signals"},
            {"id": "HR-09", "name": "Liquidity Requirement", "category": "MARKET_CONDITIONS", "description": "Min liquidity thresholds by market"},
            {"id": "HR-10", "name": "No Averaging Down", "category": "BEHAVIOR", "description": "NEVER add to losing positions"},
            {"id": "HR-11", "name": "Leverage Limit", "category": "RISK_MANAGEMENT", "description": "Max leverage by market (5x crypto, 30x forex)"},
            {"id": "HR-12", "name": "Emergency Kill Switch", "category": "RISK_MANAGEMENT", "description": "All positions closed if -10% in 24h"},
        ],
        "total": 12
    }


@router.get("/rules/soft")
async def list_soft_rules() -> Dict[str, Any]:
    """List all soft (guideline) rules."""
    return {
        "rules": [
            {"id": "SR-01", "name": "Trade With Trend", "description": "Align with 4H+ trend direction"},
            {"id": "SR-02", "name": "Avoid Session Open", "description": "Wait 30 min after session open"},
            {"id": "SR-03", "name": "Partial Profits", "description": "Take 50% partials at 1:1 R:R"},
            {"id": "SR-04", "name": "Prefer Liquid Pairs", "description": "Trade top 20 pairs by volume"},
            {"id": "SR-05", "name": "Check Correlation", "description": "Avoid correlated positions (max 0.7)"},
            {"id": "SR-06", "name": "Wait for Confirmation", "description": "Wait for confirmation candle"},
            {"id": "SR-07", "name": "Optimal Trading Hours", "description": "Trade during optimal hours by market"},
            {"id": "SR-08", "name": "Multiple Timeframe Analysis", "description": "Check at least 2 timeframes"},
            {"id": "SR-09", "name": "Document Trades", "description": "Document entry reason and plan"},
            {"id": "SR-10", "name": "Weekly Review", "description": "Review performance weekly"},
        ],
        "total": 10,
        "note": "Soft rules can be overridden with justification. All overrides are logged."
    }


@router.get("/gates/{market_type}")
async def list_market_gates(market_type: str) -> Dict[str, Any]:
    """List gates for a specific market type."""
    gate_info = {
        "crypto_spot": {
            "gates": [
                {"id": "CS-G1", "name": "24h Volume", "threshold": "$10M minimum"},
                {"id": "CS-G2", "name": "Market Cap", "threshold": "$100M minimum"},
                {"id": "CS-G3", "name": "Spread", "threshold": "0.3% maximum"},
                {"id": "CS-G4", "name": "Exchange Listings", "threshold": "3 exchanges minimum"},
                {"id": "CS-G5", "name": "Concentration", "threshold": "40% maximum"},
                {"id": "CS-G6", "name": "On-Chain Health", "threshold": "No distribution signals"},
            ],
            "focus": "BTC/ETH majors, SOL/LINK/AVAX alts"
        },
        "crypto_perps": {
            "gates": [
                {"id": "CP-G1", "name": "Symbol Whitelist", "threshold": "BTC/ETH only"},
                {"id": "CP-G2", "name": "Leverage", "threshold": "5x maximum"},
                {"id": "CP-G3", "name": "Funding Rate", "threshold": "±0.1% per 8h"},
                {"id": "CP-G4", "name": "Open Interest", "threshold": "$50M minimum"},
                {"id": "CP-G5", "name": "Position Size", "threshold": "10% of account max"},
                {"id": "CP-G6", "name": "Liquidation Buffer", "threshold": "50% minimum"},
            ],
            "focus": "BTC/ETH perpetuals only"
        },
        "forex": {
            "gates": [
                {"id": "FX-G1", "name": "Pair Whitelist", "threshold": "Majors/minors only"},
                {"id": "FX-G2", "name": "Spread", "threshold": "2 pips maximum"},
                {"id": "FX-G3", "name": "Leverage", "threshold": "30:1 maximum"},
                {"id": "FX-G4", "name": "Daily Drawdown", "threshold": "5% maximum (prop firm)"},
                {"id": "FX-G5", "name": "Risk/Reward", "threshold": "1.5:1 minimum"},
                {"id": "FX-G6", "name": "News Blackout", "threshold": "30 min window"},
                {"id": "FX-G7", "name": "Carry Direction", "threshold": "Prefer positive carry"},
            ],
            "focus": "Major/minor pairs, prop firm compatible"
        },
        "futures": {
            "gates": [
                {"id": "FUT-G1", "name": "Contract Whitelist", "threshold": "ES/NQ/CL/GC etc."},
                {"id": "FUT-G2", "name": "Daily Volume", "threshold": "10,000 contracts min"},
                {"id": "FUT-G3", "name": "Margin Usage", "threshold": "50% of account max"},
                {"id": "FUT-G4", "name": "Position Size", "threshold": "Contract limits per symbol"},
                {"id": "FUT-G5", "name": "Trading Hours", "threshold": "Prefer RTH"},
                {"id": "FUT-G6", "name": "COT Alignment", "threshold": "Check commercial positioning"},
            ],
            "focus": "ES/NQ micros initially"
        },
        "stocks": {
            "gates": [
                {"id": "STK-G1", "name": "Market Cap", "threshold": "$10B minimum"},
                {"id": "STK-G2", "name": "Average Volume", "threshold": "1M shares/day min"},
                {"id": "STK-G3", "name": "Dividend Yield", "threshold": "2% minimum"},
                {"id": "STK-G4", "name": "Payout Ratio", "threshold": "80% maximum"},
                {"id": "STK-G5", "name": "Dividend History", "threshold": "5+ years consecutive"},
            ],
            "focus": "Long-term dividend positions only",
            "priority": "LOW - not for active trading"
        },
    }
    
    if market_type not in gate_info:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid market type: {market_type}. Valid: {list(gate_info.keys())}"
        )
    
    return {
        "market_type": market_type,
        **gate_info[market_type]
    }
