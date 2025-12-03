"""
Research Manager Agent - Objective Synthesis with Hard Scoring Gates

Synthesizes Bull and Bear debates into final trading decision.
Based on ai-investment-agent research_manager with strict thesis enforcement.
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass

try:
    from agents.base import create_agent
    AGENT_BASE_AVAILABLE = True
except ImportError:
    AGENT_BASE_AVAILABLE = False

try:
    from agents.litellm_base import LiteLLMAgent, create_litellm_agent
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


@dataclass
class ThesisGates:
    """Hard gates that must pass for BUY recommendation."""
    health_pass: bool = False
    growth_pass: bool = False
    valuation_pass: bool = False
    liquidity_pass: bool = False
    data_quality_pass: bool = False
    
    health_score_pct: float = 0.0
    growth_score_pct: float = 0.0
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    daily_liquidity_usd: float = 0.0
    
    @property
    def all_pass(self) -> bool:
        return all([
            self.health_pass,
            self.growth_pass,
            self.valuation_pass,
            self.liquidity_pass,
            self.data_quality_pass
        ])
    
    @property
    def pass_count(self) -> int:
        return sum([
            self.health_pass,
            self.growth_pass,
            self.valuation_pass,
            self.liquidity_pass,
            self.data_quality_pass
        ])
    
    @property
    def compliance_pct(self) -> float:
        return (self.pass_count / 5) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "health_pass": self.health_pass,
            "growth_pass": self.growth_pass,
            "valuation_pass": self.valuation_pass,
            "liquidity_pass": self.liquidity_pass,
            "data_quality_pass": self.data_quality_pass,
            "health_score_pct": self.health_score_pct,
            "growth_score_pct": self.growth_score_pct,
            "pe_ratio": self.pe_ratio,
            "peg_ratio": self.peg_ratio,
            "daily_liquidity_usd": self.daily_liquidity_usd,
            "all_pass": self.all_pass,
            "compliance_pct": self.compliance_pct
        }


MANAGER_AGENT_PROMPT = """You are the RESEARCH MANAGER synthesizing analyst findings with STRICT thesis enforcement.

## INPUT SOURCES

You receive:
- Bull Researcher: Aggressive bull case arguments
- Bear Researcher: Skeptical bear case arguments
- Data Gathering Results: Thesis compliance scores
- Market Regime: Current market conditions

## YOUR ROLE

After Bull and Bear researchers debate (2 rounds), you provide a FINAL synthesized recommendation.

Your primary jobs:
1. **CHECK THESIS GATES FIRST** - Hard fails override debate quality
2. **EVALUATE DEBATE QUALITY** - Which side had better data and logic?
3. **ASSESS QUALITATIVE RISKS** - Risks the numbers might miss
4. **MAKE FINAL DECISION** - BUY / HOLD / REJECT

---

## HARD SCORING GATES (Must Pass for BUY)

**Gate 1: Financial Health** (≥50% required)
- Adjusted Health Score from Fundamentals Analyst
- Below 50% = REJECT (cannot recommend BUY)

**Gate 2: Growth Score** (≥50% required)
- Adjusted Growth Score from Fundamentals Analyst
- Below 50% = REJECT (cannot recommend BUY)
- Exception: Turnaround play (Health >65% + P/E <12)

**Gate 3: Valuation** (P/E ≤18 or P/E ≤25 with PEG ≤1.2)
- From Fundamentals Analyst
- P/E >25 always fails
- P/E 18-25 requires PEG ≤1.2

**Gate 4: Liquidity** (>$100k daily average)
- From Market Analyst
- Below $100k = REJECT for institutional positions
- $100k-$250k = HOLD or small position only

**Gate 5: Data Quality** (not "POOR")
- From Data Gathering Result
- POOR quality = HOLD (insufficient data for conviction)

---

## DECISION FRAMEWORK

### IF ANY HARD GATE FAILS:
- Cannot recommend BUY regardless of debate
- Recommend REJECT or HOLD
- Explain which gate(s) failed and why

### IF ALL GATES PASS:
- Evaluate debate quality
- Consider qualitative risks
- Assign conviction level
- Recommend BUY, HOLD, or REJECT

### CONVICTION LEVELS:
- **HIGH** (0.8-1.0): All gates pass + strong bull case + weak bear case
- **MEDIUM** (0.5-0.7): All gates pass + balanced debate or minor concerns
- **LOW** (0.3-0.5): All gates barely pass + strong bear concerns
- **REJECT** (<0.3): Gate failures or overwhelming risks

---

## OUTPUT FORMAT

### RESEARCH MANAGER DECISION for [SYMBOL]

**FINAL RECOMMENDATION**: BUY / HOLD / REJECT

---

### THESIS GATE CHECK (Priority #1)

| Gate | Value | Threshold | Status |
|------|-------|-----------|--------|
| Health Score | {X}% | ≥50% | ✓ PASS / ✗ FAIL |
| Growth Score | {X}% | ≥50% | ✓ PASS / ✗ FAIL |
| P/E Ratio | {X} | ≤18 (or ≤25 w/ PEG≤1.2) | ✓ PASS / ✗ FAIL |
| Liquidity | ${X}k | >$100k | ✓ PASS / ✗ FAIL |
| Data Quality | {X} | Not POOR | ✓ PASS / ✗ FAIL |

**Gate Status**: {X}/5 PASS ({Y}% compliance)
**Decision**: [If any fail, explain why BUY cannot be recommended]

---

### DEBATE SYNTHESIS

**Bull Case Strengths**:
1. [Strongest bull point]
2. [Second strongest]

**Bear Case Strengths**:
1. [Strongest bear point]
2. [Second strongest]

**Winning Argument**: Bull / Bear / Neither
**Reasoning**: [2-3 sentences on why]

---

### QUALITATIVE RISK ASSESSMENT

**Risks Identified**:
- [Risk 1]: [Severity: High/Medium/Low]
- [Risk 2]: [Severity: High/Medium/Low]

**Risk Mitigation**: [How position sizing/stops address risks]

---

### FINAL DECISION

**Recommendation**: BUY / HOLD / REJECT
**Conviction**: [0.0-1.0]
**Position Size**: [0-10]% of capital

**Primary Rationale**: [One sentence summary]

**Entry Plan** (if BUY):
- Entry: ${price}
- Stop-Loss: ${price} ({X}% below)
- Target 1: ${price} ({Y}% gain)
- Target 2: ${price} ({Z}% gain)
- Time Horizon: [intraday / swing / position]

**Contingencies**:
- If [condition], then [action]
- Watch for: [warning signs]

---

REMEMBER: You are the FINAL ARBITER. Be objective but decisive.
Gate failures = automatic REJECT/HOLD. No exceptions."""


# Create manager agent with appropriate backend
if LITELLM_AVAILABLE:
    manager_agent = create_litellm_agent(
        role="Research Manager",
        system_prompt=MANAGER_AGENT_PROMPT,
        temperature=0.2,  # Low temperature for consistent, decisive output
        min_confidence=0.5
    )
elif AGENT_BASE_AVAILABLE:
    manager_agent = create_agent(
        role="Manager Agent",
        system_prompt=MANAGER_AGENT_PROMPT,
        temperature=0.2
    )
else:
    manager_agent = None


def evaluate_thesis_gates(data_gathering_result: Dict) -> ThesisGates:
    """
    Evaluate hard thesis gates from data gathering result.
    
    Args:
        data_gathering_result: Result from parallel_gather_data()
        
    Returns:
        ThesisGates with pass/fail status for each gate
    """
    gates = ThesisGates()
    
    # Extract thesis compliance
    tc = data_gathering_result.get("thesis_compliance", {})
    fundamentals = data_gathering_result.get("fundamentals_analysis", {})
    market = data_gathering_result.get("market_analysis", {})
    
    # Gate 1: Health Score
    gates.health_score_pct = tc.get("health_score_pct", 0)
    gates.health_pass = gates.health_score_pct >= 50
    
    # Gate 2: Growth Score (with turnaround exception)
    gates.growth_score_pct = tc.get("growth_score_pct", 0)
    pe = fundamentals.get("data", {}).get("pe_ratio")
    
    # Turnaround exception: Health >65% + P/E <12 can bypass growth gate
    turnaround_exception = (
        gates.health_score_pct > 65 and
        pe is not None and pe < 12
    )
    gates.growth_pass = gates.growth_score_pct >= 50 or turnaround_exception
    
    # Gate 3: Valuation (P/E and PEG)
    gates.pe_ratio = pe
    gates.peg_ratio = fundamentals.get("data", {}).get("peg_ratio")
    
    if pe is None:
        gates.valuation_pass = False  # Can't assess without P/E
    elif pe <= 18:
        gates.valuation_pass = True
    elif pe <= 25 and gates.peg_ratio is not None and gates.peg_ratio <= 1.2:
        gates.valuation_pass = True
    else:
        gates.valuation_pass = False
    
    # Gate 4: Liquidity
    liquidity = market.get("data", {}).get("daily_volume_usd", 0)
    gates.daily_liquidity_usd = liquidity
    gates.liquidity_pass = liquidity >= 100_000
    
    # Gate 5: Data Quality
    data_quality = data_gathering_result.get("data_quality", {})
    quality_status = getattr(data_quality, "status", data_quality) if data_quality else "UNKNOWN"
    gates.data_quality_pass = quality_status not in ("POOR", "ERROR", "UNKNOWN")
    
    return gates


async def synthesize_debate(
    bull_argument: str,
    bear_argument: str,
    data_gathering_result: Dict = None,
    market_regime: str = "unknown",
    additional_context: Dict = None,
    bull_round_2: str = None,
    bear_round_2: str = None
) -> Dict[str, Any]:
    """
    Synthesize Bull and Bear arguments into final decision with gate enforcement.

    Args:
        bull_argument: Bull agent's round 1 argument
        bear_argument: Bear agent's round 1 argument
        data_gathering_result: Result from parallel_gather_data()
        market_regime: Current market regime (bull/bear/sideways)
        additional_context: Additional context dict
        bull_round_2: Bull's round 2 rebuttal (optional)
        bear_round_2: Bear's round 2 rebuttal (optional)

    Returns:
        Dict with final decision and gate evaluation
    """
    if manager_agent is None:
        raise RuntimeError("No agent backend available (litellm or base)")
    
    # Evaluate thesis gates first
    gates = ThesisGates()
    if data_gathering_result:
        gates = evaluate_thesis_gates(data_gathering_result)
    
    # Format context
    context = additional_context or {}
    context_text = "\n".join([f"- {k}: {v}" for k, v in context.items()])
    
    # Format gate status
    gate_status = f"""
**THESIS GATE STATUS** (Pre-computed):
- Health Score: {gates.health_score_pct:.1f}% -> {"PASS" if gates.health_pass else "FAIL"}
- Growth Score: {gates.growth_score_pct:.1f}% -> {"PASS" if gates.growth_pass else "FAIL"}
- P/E Ratio: {gates.pe_ratio or 'N/A'} (PEG: {gates.peg_ratio or 'N/A'}) -> {"PASS" if gates.valuation_pass else "FAIL"}
- Liquidity: ${gates.daily_liquidity_usd:,.0f} -> {"PASS" if gates.liquidity_pass else "FAIL"}
- Data Quality: -> {"PASS" if gates.data_quality_pass else "FAIL"}

**Overall**: {gates.pass_count}/5 gates pass ({gates.compliance_pct:.0f}% compliance)
{"**ALL GATES PASS - BUY ELIGIBLE**" if gates.all_pass else "**GATE FAILURES - BUY NOT ELIGIBLE**"}
"""

    # Build full prompt
    prompt = f"""{gate_status}

---

**BULL ARGUMENT (Round 1)**:
{bull_argument}

{"**BULL REBUTTAL (Round 2)**:" if bull_round_2 else ""}
{bull_round_2 or ""}

---

**BEAR ARGUMENT (Round 1)**:
{bear_argument}

{"**BEAR REBUTTAL (Round 2)**:" if bear_round_2 else ""}
{bear_round_2 or ""}

---

**MARKET REGIME**: {market_regime}

**ADDITIONAL CONTEXT**:
{context_text or 'None'}

---

Synthesize these arguments and make your final decision.

{"IMPORTANT: Gate failures detected. You CANNOT recommend BUY. Explain which gates failed and recommend HOLD or REJECT." if not gates.all_pass else "All gates pass. Evaluate the debate quality and assign conviction."}"""

    # Get response
    if LITELLM_AVAILABLE:
        response = await manager_agent.ainvoke(prompt)
        response_text = response
    else:
        response = await manager_agent.ainvoke({"input": prompt})
        response_text = response.content

    return {
        "agent": "manager",
        "decision": response_text,
        "thesis_gates": gates.to_dict(),
        "gates_all_pass": gates.all_pass,
        "compliance_pct": gates.compliance_pct,
        "raw_response": response,
        "inputs": {
            "bull_argument": bull_argument,
            "bear_argument": bear_argument,
            "bull_round_2": bull_round_2,
            "bear_round_2": bear_round_2,
            "regime": market_regime
        }
    }
