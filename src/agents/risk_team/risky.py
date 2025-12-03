"""
Risky Analyst - Aggressive Risk Perspective

Advocates for MAXIMIZING position size on high-conviction opportunities.
Based on ai-investment-agent risky_analyst.
"""

from typing import Any, Dict, Optional, List

try:
    from agents.litellm_base import LiteLLMAgent, create_litellm_agent
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


RISKY_ANALYST_PROMPT = """You are the RISKY ANALYST - the aggressive voice in risk assessment.

Your role is to advocate for MAXIMIZING position size when the opportunity is compelling.

## YOUR PERSPECTIVE

You believe in:
- Sizing appropriately for high-conviction opportunities
- Taking calculated risks for asymmetric returns
- Capturing full upside on thesis-compliant names
- Not leaving money on the table when edge is clear

## POSITION SIZING PHILOSOPHY

**High Conviction** (thesis passes all gates + strong catalysts):
- Recommend 6-10% initial position
- "This is the kind of opportunity we wait for"
- Risk/reward strongly favors upside

**Standard Conviction** (thesis passes + decent setup):
- Recommend 4-6% initial position
- "Solid opportunity worth meaningful allocation"
- Risk/reward favorable

**Lower Conviction** (thesis passes but concerns exist):
- Recommend 2-4% initial position
- "Worth participating but size appropriately"
- Risk/reward acceptable

## ARGUMENTS YOU MAKE

1. "The thesis gates all pass - we should be aggressive"
2. "Asymmetric risk/reward favors sizing up"
3. "Undiscovered opportunities require conviction"
4. "The market is underpricing [specific catalyst]"
5. "Technical setup supports aggressive entry"

## OUTPUT FORMAT

**RISKY ANALYST ASSESSMENT for [SYMBOL]**

**POSITION RECOMMENDATION**: [X]% of capital

**CONVICTION JUSTIFICATION**:
- Thesis Compliance: [Summary of gates passing]
- Asymmetric Opportunity: [Why upside > downside]
- Catalyst Timeline: [Near-term catalysts]

**WHY SIZE UP**:
1. [Aggressive argument 1]
2. [Aggressive argument 2]
3. [Aggressive argument 3]

**COUNTER TO CONSERVATIVE VIEW**:
[Why the safe analyst is being too cautious]

**ENTRY STRATEGY**:
- Entry: Immediate or scaled over [X] days
- Sizing: Start with [Y]%, add to [Z]% on confirmation

**RISKS ACKNOWLEDGED** (but manageable):
- [Risk 1]: [Why it's acceptable]
- [Risk 2]: [Why it's acceptable]

**STOP-LOSS**: [%] below entry (accept this as cost of opportunity)

Remember: You advocate for MAXIMIZING participation when opportunity is real.
Be aggressive but data-driven. Don't be reckless."""


class RiskyAnalystAgent:
    """Risky Analyst for aggressive position sizing."""
    
    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature
        self._agent: Optional[LiteLLMAgent] = None
    
    @property
    def agent(self) -> LiteLLMAgent:
        """Lazy-load the LLM agent."""
        if self._agent is None:
            if LITELLM_AVAILABLE:
                self._agent = create_litellm_agent(
                    role="Risky Analyst",
                    system_prompt=RISKY_ANALYST_PROMPT,
                    temperature=self.temperature,
                    min_confidence=0.4
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
        """Generate aggressive position sizing recommendation."""
        prompt = self._format_prompt(symbol, manager_decision, data_gathering_result, debate_result)
        
        try:
            response = await self.agent.ainvoke(prompt)
            position_pct = self._extract_position_pct(response)
            
            return {
                "agent": "risky_analyst",
                "symbol": symbol,
                "report": response,
                "position_pct": position_pct,
                "risk_factors": self._extract_risks(response),
            }
        except Exception as e:
            return {
                "agent": "risky_analyst",
                "symbol": symbol,
                "report": f"Error: {str(e)}",
                "position_pct": 0,
                "risk_factors": ["analysis_error"],
            }
    
    def _format_prompt(
        self,
        symbol: str,
        manager_decision: Dict,
        data_result: Dict = None,
        debate: Dict = None
    ) -> str:
        """Format context for risky analyst."""
        gates = manager_decision.get("thesis_gates", {})
        
        return f"""Analyze position sizing for {symbol}:

**MANAGER DECISION**: {manager_decision.get("decision", "N/A")}

**THESIS GATES**:
- Health: {gates.get("health_score_pct", "N/A")}% (Pass: {gates.get("health_pass", "N/A")})
- Growth: {gates.get("growth_score_pct", "N/A")}% (Pass: {gates.get("growth_pass", "N/A")})
- Valuation: P/E {gates.get("pe_ratio", "N/A")} (Pass: {gates.get("valuation_pass", "N/A")})
- Liquidity: ${gates.get("daily_liquidity_usd", 0):,.0f} (Pass: {gates.get("liquidity_pass", "N/A")})
- All Gates Pass: {gates.get("all_pass", False)}
- Compliance: {gates.get("compliance_pct", 0)}%

**CONTEXT**:
{debate.get("manager_decision", {}).get("decision", "") if debate else ""}

Provide your AGGRESSIVE position sizing recommendation."""
    
    def _extract_position_pct(self, response: str) -> float:
        """Extract position percentage from response."""
        import re
        # Look for patterns like "8% of capital" or "recommend 6-8%"
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*of\s*capital',
            r'recommend\s*(\d+(?:\.\d+)?)\s*%',
            r'position.*?(\d+(?:\.\d+)?)\s*%',
        ]
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return float(match.group(1))
        return 6.0  # Default aggressive position
    
    def _extract_risks(self, response: str) -> List[str]:
        """Extract acknowledged risks from response."""
        risks = []
        if "volatility" in response.lower():
            risks.append("high_volatility")
        if "liquidity" in response.lower():
            risks.append("liquidity_concern")
        if "jurisdiction" in response.lower():
            risks.append("jurisdiction_risk")
        return risks


# Singleton instance
risky_analyst = RiskyAnalystAgent()


async def generate_risky_assessment(
    symbol: str,
    manager_decision: Dict[str, Any],
    data_gathering_result: Dict[str, Any] = None,
    debate_result: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Generate risky analyst assessment."""
    return await risky_analyst.analyze(
        symbol, manager_decision, data_gathering_result, debate_result
    )
