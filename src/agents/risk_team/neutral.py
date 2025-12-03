"""
Neutral Analyst - Balanced Risk Perspective

Seeks optimal risk-adjusted position sizing.
Based on ai-investment-agent neutral_analyst.
"""

from typing import Any, Dict, Optional, List

try:
    from agents.litellm_base import LiteLLMAgent, create_litellm_agent
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


NEUTRAL_ANALYST_PROMPT = """You are the NEUTRAL ANALYST - the balanced voice in risk assessment.

Your role is to find the OPTIMAL RISK-ADJUSTED position size that balances opportunity and risk.

## YOUR PERSPECTIVE

You believe in:
- Optimal position sizing based on edge and risk
- Kelly Criterion principles (but fractional)
- Balancing upside capture with downside protection
- Data-driven sizing, not emotional

## POSITION SIZING FRAMEWORK

### Kelly-Inspired Approach
Position Size = (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win
Then apply 1/4 to 1/2 Kelly for safety margin

### Practical Guidelines

**Strong Edge** (>70% thesis compliance + favorable R:R):
- Recommend 4-6% position
- Full Kelly would suggest more, but we're conservative
- R:R > 3:1

**Moderate Edge** (60-70% compliance + decent R:R):
- Recommend 2-4% position
- Quarter Kelly appropriate
- R:R 2:1 to 3:1

**Marginal Edge** (50-60% compliance + acceptable R:R):
- Recommend 1-2% position
- Small position to participate
- R:R > 1.5:1

**No Edge** (<50% compliance):
- Recommend 0% (skip)
- Don't trade without edge

## FACTORS IN YOUR CALCULATION

1. **Thesis Compliance %**: Higher = larger position
2. **Risk/Reward Ratio**: Higher = larger position
3. **Liquidity**: Lower = smaller position (exit risk)
4. **Volatility**: Higher = smaller position (more uncertainty)
5. **Data Quality**: Lower = smaller position (less confidence)
6. **Correlation**: Higher to existing positions = smaller position

## OUTPUT FORMAT

**NEUTRAL ANALYST ASSESSMENT for [SYMBOL]**

**POSITION RECOMMENDATION**: [X]% of capital (risk-adjusted optimal)

**SIZING CALCULATION**:
- Thesis Compliance: {X}% → Base size: [Y]%
- Risk/Reward Adjustment: {ratio}:1 → Adjustment: [+/-Z]%
- Liquidity Adjustment: ${X}k daily → Adjustment: [+/-A]%
- Volatility Adjustment: [High/Med/Low] → Adjustment: [+/-B]%
- Final Recommended Size: [Total]%

**RISK/REWARD ANALYSIS**:
- Expected Upside: [X]% gain
- Expected Downside: [Y]% loss
- Probability-Weighted Return: [Z]%
- Risk/Reward Ratio: [A]:1

**POSITION PARAMETERS**:
- Entry: [Price/Approach]
- Stop-Loss: [%] (risk-based, not arbitrary)
- Target 1: [%] gain (1R)
- Target 2: [%] gain (2R)
- Target 3: [%] gain (3R)

**BALANCING RISKY vs SAFE VIEWS**:
- Risky Analyst likely recommends: [Higher %] because [reason]
- Safe Analyst likely recommends: [Lower %] because [reason]
- My balanced view: [X]% because [rationale]

**KEY CONSIDERATIONS**:
1. [Factor 1 and its impact]
2. [Factor 2 and its impact]
3. [Factor 3 and its impact]

**CONFIDENCE IN SIZING**: [High/Medium/Low]
- [Why this is the right size for this opportunity]

Remember: You are the BALANCED voice.
Find the OPTIMAL position, not the biggest or smallest.
Position sizing is as important as trade selection."""


class NeutralAnalystAgent:
    """Neutral Analyst for balanced position sizing."""
    
    def __init__(self, temperature: float = 0.3):
        self.temperature = temperature
        self._agent: Optional[LiteLLMAgent] = None
    
    @property
    def agent(self) -> LiteLLMAgent:
        """Lazy-load the LLM agent."""
        if self._agent is None:
            if LITELLM_AVAILABLE:
                self._agent = create_litellm_agent(
                    role="Neutral Analyst",
                    system_prompt=NEUTRAL_ANALYST_PROMPT,
                    temperature=self.temperature,
                    min_confidence=0.5
                )
            else:
                raise RuntimeError("LiteLLM module not available")
        return self._agent
    
    async def analyze(
        self,
        symbol: str,
        manager_decision: Dict[str, Any],
        data_gathering_result: Dict[str, Any] = None,
        debate_result: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Generate balanced position sizing recommendation."""
        prompt = self._format_prompt(symbol, manager_decision, data_gathering_result, debate_result)
        
        try:
            response = await self.agent.ainvoke(prompt)
            position_pct = self._extract_position_pct(response)
            
            return {
                "agent": "neutral_analyst",
                "symbol": symbol,
                "report": response,
                "position_pct": position_pct,
                "risk_factors": self._extract_risks(response),
                "stop_loss_pct": self._extract_stop_loss(response),
                "take_profit_pct": self._extract_take_profit(response),
                "risk_reward_ratio": self._extract_rr_ratio(response),
            }
        except Exception as e:
            return {
                "agent": "neutral_analyst",
                "symbol": symbol,
                "report": f"Error: {str(e)}",
                "position_pct": 0,
                "risk_factors": ["analysis_error"],
                "stop_loss_pct": 5.0,
                "take_profit_pct": 15.0,
                "risk_reward_ratio": 3.0,
            }
    
    def _format_prompt(
        self,
        symbol: str,
        manager_decision: Dict,
        data_result: Dict = None,
        debate: Dict = None
    ) -> str:
        """Format context for neutral analyst."""
        gates = manager_decision.get("thesis_gates", {})
        
        return f"""Analyze position sizing for {symbol} from BALANCED perspective:

**MANAGER DECISION**: {manager_decision.get("decision", "N/A")}

**THESIS GATES**:
- Health: {gates.get("health_score_pct", "N/A")}% (Pass: {gates.get("health_pass", "N/A")})
- Growth: {gates.get("growth_score_pct", "N/A")}% (Pass: {gates.get("growth_pass", "N/A")})
- Valuation: P/E {gates.get("pe_ratio", "N/A")} (Pass: {gates.get("valuation_pass", "N/A")})
- Liquidity: ${gates.get("daily_liquidity_usd", 0):,.0f} (Pass: {gates.get("liquidity_pass", "N/A")})
- All Gates Pass: {gates.get("all_pass", False)}
- Overall Compliance: {gates.get("compliance_pct", 0)}%

Provide your BALANCED position sizing recommendation.
Consider both the RISKY view (maximize) and SAFE view (preserve capital).
Find the OPTIMAL risk-adjusted position size."""
    
    def _extract_position_pct(self, response: str) -> float:
        """Extract position percentage from response."""
        import re
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*of\s*capital',
            r'recommend.*?(\d+(?:\.\d+)?)\s*%',
            r'final.*?(\d+(?:\.\d+)?)\s*%',
            r'position.*?(\d+(?:\.\d+)?)\s*%',
        ]
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return float(match.group(1))
        return 3.0  # Default balanced position
    
    def _extract_stop_loss(self, response: str) -> float:
        """Extract stop-loss percentage."""
        import re
        patterns = [
            r'stop[- ]?loss.*?(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%.*?stop',
        ]
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return float(match.group(1))
        return 5.0
    
    def _extract_take_profit(self, response: str) -> float:
        """Extract take-profit percentage."""
        import re
        patterns = [
            r'target.*?(\d+(?:\.\d+)?)\s*%\s*gain',
            r'take[- ]?profit.*?(\d+(?:\.\d+)?)\s*%',
            r'upside.*?(\d+(?:\.\d+)?)\s*%',
        ]
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return float(match.group(1))
        return 15.0
    
    def _extract_rr_ratio(self, response: str) -> float:
        """Extract risk/reward ratio."""
        import re
        patterns = [
            r'(\d+(?:\.\d+)?)\s*:\s*1\s*(?:r/?r|risk)',
            r'risk[/ ]?reward.*?(\d+(?:\.\d+)?)\s*:',
            r'r\s*[/:]\s*r.*?(\d+(?:\.\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return float(match.group(1))
        return 3.0
    
    def _extract_risks(self, response: str) -> List[str]:
        """Extract identified risks."""
        risks = []
        risk_keywords = {
            "liquidity": "liquidity_risk",
            "volatility": "high_volatility",
            "correlation": "correlation_risk",
            "quality": "data_quality",
            "uncertainty": "uncertainty",
        }
        for keyword, risk in risk_keywords.items():
            if keyword in response.lower():
                risks.append(risk)
        return risks


# Singleton instance
neutral_analyst = NeutralAnalystAgent()


async def generate_neutral_assessment(
    symbol: str,
    manager_decision: Dict[str, Any],
    data_gathering_result: Dict[str, Any] = None,
    debate_result: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Generate neutral analyst assessment."""
    return await neutral_analyst.analyze(
        symbol, manager_decision, data_gathering_result, debate_result
    )
