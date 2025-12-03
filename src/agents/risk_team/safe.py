"""
Safe Analyst - Conservative Risk Perspective

Advocates for PROTECTING CAPITAL over maximizing returns.
Based on ai-investment-agent safe_analyst.
"""

from typing import Any, Dict, Optional, List

try:
    from agents.litellm_base import LiteLLMAgent, create_litellm_agent
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


SAFE_ANALYST_PROMPT = """You are the SAFE ANALYST - the conservative voice in risk assessment.

Your role is to PROTECT CAPITAL by advocating for smaller position sizes and identifying hidden risks.

## YOUR PERSPECTIVE

You believe in:
- Preserving capital above all else
- Assuming things will go wrong
- Sizing for worst-case scenarios
- Living to trade another day

## POSITION SIZING PHILOSOPHY

**Even High Conviction** (thesis passes all gates):
- Recommend 3-5% maximum
- "Even good trades can go wrong"
- Always leave room for error

**Standard Conviction** (thesis passes + decent setup):
- Recommend 2-3% initial position
- "Size for survival, not home runs"
- Risk/reward must account for tail risks

**Lower Conviction** (thesis passes but concerns exist):
- Recommend 1-2% or AVOID
- "If there's doubt, there's no doubt"
- Skip marginal opportunities

## RED FLAGS YOU LOOK FOR

1. **Liquidity Risk**: Low volume = can't exit when needed
2. **Jurisdiction Risk**: Authoritarian regimes, capital controls
3. **Concentration Risk**: Too much in one position/sector
4. **Volatility Risk**: ATR > 5% = size down
5. **Data Quality Risk**: Unreliable fundamentals = reduce conviction
6. **Cyclical Peak Risk**: Industry at top of cycle
7. **Execution Risk**: Wide spreads, poor fills

## OUTPUT FORMAT

**SAFE ANALYST ASSESSMENT for [SYMBOL]**

**POSITION RECOMMENDATION**: [X]% of capital (conservative)

**CAPITAL PRESERVATION RATIONALE**:
- Maximum acceptable loss: [X]% of capital
- Position sized to limit loss to: [Y]%
- Assumes stop-loss may gap through

**RED FLAGS IDENTIFIED**:
1. [Risk 1]: [Impact on sizing]
2. [Risk 2]: [Impact on sizing]
3. [Risk 3]: [Impact on sizing]

**WHY SIZE DOWN**:
1. [Conservative argument 1]
2. [Conservative argument 2]
3. [Conservative argument 3]

**COUNTER TO AGGRESSIVE VIEW**:
[Why the risky analyst is being too aggressive]

**WORST-CASE SCENARIO**:
- What could go wrong: [Description]
- Potential loss: [%]
- Why this matters: [Impact on portfolio]

**PROTECTIVE MEASURES**:
- Stop-loss: [%] (tight)
- Position limit: [X]% max even if thesis improves
- Exit triggers: [List warning signs]

**IF FORCED TO TAKE POSITION**:
- Start with: [Small %]
- Scale up only if: [Conditions]
- Never exceed: [Max %]

Remember: You advocate for CAPITAL PRESERVATION.
The goal is to survive, not to get rich quick.
One large loss can wipe out many small gains."""


class SafeAnalystAgent:
    """Safe Analyst for conservative position sizing."""
    
    def __init__(self, temperature: float = 0.3):
        self.temperature = temperature
        self._agent: Optional[LiteLLMAgent] = None
    
    @property
    def agent(self) -> LiteLLMAgent:
        """Lazy-load the LLM agent."""
        if self._agent is None:
            if LITELLM_AVAILABLE:
                self._agent = create_litellm_agent(
                    role="Safe Analyst",
                    system_prompt=SAFE_ANALYST_PROMPT,
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
        """Generate conservative position sizing recommendation."""
        prompt = self._format_prompt(symbol, manager_decision, data_gathering_result, debate_result)
        
        try:
            response = await self.agent.ainvoke(prompt)
            position_pct = self._extract_position_pct(response)
            
            return {
                "agent": "safe_analyst",
                "symbol": symbol,
                "report": response,
                "position_pct": position_pct,
                "risk_factors": self._extract_risks(response),
                "stop_loss_pct": self._extract_stop_loss(response),
            }
        except Exception as e:
            return {
                "agent": "safe_analyst",
                "symbol": symbol,
                "report": f"Error: {str(e)}",
                "position_pct": 0,
                "risk_factors": ["analysis_error"],
                "stop_loss_pct": 3.0,
            }
    
    def _format_prompt(
        self,
        symbol: str,
        manager_decision: Dict,
        data_result: Dict = None,
        debate: Dict = None
    ) -> str:
        """Format context for safe analyst."""
        gates = manager_decision.get("thesis_gates", {})
        
        # Extract bear concerns if available
        bear_concerns = ""
        if debate:
            bear_r1 = debate.get("bear_round_1", {}).get("argument", "")
            bear_r2 = debate.get("bear_round_2", {}).get("argument", "")
            if bear_r1 or bear_r2:
                bear_concerns = f"""
**BEAR CONCERNS** (from debate):
{bear_r1[:500] if bear_r1 else ""}
{bear_r2[:500] if bear_r2 else ""}
"""
        
        return f"""Analyze position sizing for {symbol} from CONSERVATIVE perspective:

**MANAGER DECISION**: {manager_decision.get("decision", "N/A")}

**THESIS GATES**:
- Health: {gates.get("health_score_pct", "N/A")}% (Pass: {gates.get("health_pass", "N/A")})
- Growth: {gates.get("growth_score_pct", "N/A")}% (Pass: {gates.get("growth_pass", "N/A")})
- Valuation: P/E {gates.get("pe_ratio", "N/A")} (Pass: {gates.get("valuation_pass", "N/A")})
- Liquidity: ${gates.get("daily_liquidity_usd", 0):,.0f} (Pass: {gates.get("liquidity_pass", "N/A")})
- All Gates Pass: {gates.get("all_pass", False)}

{bear_concerns}

Provide your CONSERVATIVE position sizing recommendation.
Focus on CAPITAL PRESERVATION and identifying hidden risks."""
    
    def _extract_position_pct(self, response: str) -> float:
        """Extract position percentage from response."""
        import re
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*of\s*capital',
            r'recommend\s*(\d+(?:\.\d+)?)\s*%',
            r'position.*?(\d+(?:\.\d+)?)\s*%',
        ]
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                pct = float(match.group(1))
                # Safe analyst typically recommends lower
                return min(pct, 5.0)
        return 2.0  # Default conservative position
    
    def _extract_stop_loss(self, response: str) -> float:
        """Extract stop-loss percentage from response."""
        import re
        patterns = [
            r'stop[- ]?loss.*?(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%.*?stop',
        ]
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return float(match.group(1))
        return 5.0  # Default conservative stop
    
    def _extract_risks(self, response: str) -> List[str]:
        """Extract identified risks from response."""
        risks = []
        risk_keywords = {
            "liquidity": "liquidity_risk",
            "volatility": "high_volatility",
            "jurisdiction": "jurisdiction_risk",
            "cyclical": "cyclical_peak",
            "concentration": "concentration_risk",
            "gap": "gap_risk",
            "spread": "wide_spreads",
            "quality": "data_quality",
        }
        for keyword, risk in risk_keywords.items():
            if keyword in response.lower():
                risks.append(risk)
        return risks


# Singleton instance
safe_analyst = SafeAnalystAgent()


async def generate_safe_assessment(
    symbol: str,
    manager_decision: Dict[str, Any],
    data_gathering_result: Dict[str, Any] = None,
    debate_result: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Generate safe analyst assessment."""
    return await safe_analyst.analyze(
        symbol, manager_decision, data_gathering_result, debate_result
    )
