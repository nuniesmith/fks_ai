"""Debater agents module exports"""

from .bear import bear_agent, generate_bear_case
from .bull import bull_agent, generate_bull_case
from .manager import manager_agent, synthesize_debate

__all__ = [
    "bull_agent",
    "generate_bull_case",
    "bear_agent",
    "generate_bear_case",
    "manager_agent",
    "synthesize_debate"
]
