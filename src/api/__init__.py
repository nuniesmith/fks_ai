"""
FKS AI Service API

RESTful API endpoints for multi-agent trading system.
Provides access to analyst agents, debate system, memory queries, and signal generation.
"""

from .routes import app

__all__ = ['app']
