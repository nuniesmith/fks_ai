"""
Risk Analyst Agent

Specializes in risk management, position sizing, VaR, and drawdown protection.
"""

from agents.base import create_agent

RISK_ANALYST_PROMPT = """You are a Risk Analyst specializing in portfolio risk management,
position sizing, Value at Risk (VaR), maximum drawdown protection, and tail risk hedging.

Your expertise includes:
1. Value at Risk (VaR) calculation and Monte Carlo simulation
2. Maximum Drawdown (MDD) analysis and protection strategies
3. Position sizing using Kelly Criterion, fixed fractional, and volatility-based methods
4. Tail risk hedging (black swan protection)
5. Correlation risk and portfolio diversification
6. Stop-loss and take-profit optimization
7. Risk-adjusted performance (Sharpe, Sortino, Calmar ratios)

Given trading signals and portfolio data, provide:
- **Risk Level**: Low, Medium, High, Extreme
- **Recommended Position Size**: % of capital to allocate
- **Stop-Loss**: Specific price level or % from entry
- **Take-Profit**: Target price for profit-taking
- **Max Risk Per Trade**: Maximum acceptable loss in $
- **VaR Estimate**: 95% confidence 1-day VaR
- **Risk Warnings**: Specific risks that could cause outsized losses

Your job is to PROTECT CAPITAL. Be conservative. If risk is too high, recommend reducing size or passing on the trade."""

# Create risk analyst agent
risk_analyst = create_agent(
    role="Risk Analyst",
    system_prompt=RISK_ANALYST_PROMPT,
    temperature=0.2  # Very low temperature for consistent risk assessment
)


async def analyze_risk(risk_data: dict) -> dict:
    """
    Run risk analysis on trading signal and portfolio.

    Args:
        risk_data: Dict with signal, portfolio metrics, volatility

    Returns:
        Dict with risk analysis and position sizing

    Example:
        >>> risk_data = {
        ...     "symbol": "BTCUSDT",
        ...     "entry_price": 67500,
        ...     "direction": "LONG",
        ...     "confidence": 0.75,
        ...     "account_size": 100000,
        ...     "current_positions": 3,
        ...     "volatility": 0.035,
        ...     "current_drawdown": -0.08
        ... }
        >>> result = await analyze_risk(risk_data)
    """
    # Format risk data for prompt
    prompt = f"""Analyze risk for potential trade in {risk_data.get('symbol', 'UNKNOWN')}:

Trade Signal:
- Direction: {risk_data.get('direction', 'N/A')}
- Entry Price: ${risk_data.get('entry_price', 'N/A')}
- Signal Confidence: {risk_data.get('confidence', 'N/A')}

Portfolio Status:
- Account Size: ${risk_data.get('account_size', 'N/A')}
- Current Positions: {risk_data.get('current_positions', 'N/A')}
- Current Drawdown: {risk_data.get('current_drawdown', 'N/A')}%
- Available Margin: ${risk_data.get('available_margin', 'N/A')}

Market Conditions:
- Volatility (1d): {risk_data.get('volatility', 'N/A')}
- ATR: ${risk_data.get('atr', 'N/A')}
- Max Drawdown (30d): {risk_data.get('max_drawdown_30d', 'N/A')}%
- Correlation to Portfolio: {risk_data.get('correlation', 'N/A')}

Risk Parameters:
- Max Risk Per Trade: {risk_data.get('max_risk_per_trade', '2')}%
- Max Portfolio Exposure: {risk_data.get('max_exposure', '20')}%
- Stop-Loss Method: {risk_data.get('stop_method', 'ATR-based')}

Additional Context:
{risk_data.get('additional_context', 'None')}

Provide risk analysis with risk level, recommended position size (% of account),
stop-loss level, take-profit target, max $ risk, VaR estimate, and specific warnings."""

    response = await risk_analyst.ainvoke({"input": prompt})

    return {
        "agent": "risk_analyst",
        "symbol": risk_data.get('symbol'),
        "analysis": response.content,
        "raw_response": response
    }
