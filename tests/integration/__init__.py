"""
Integration Tests for Multi-Agent System

Tests end-to-end graph execution with live Ollama and ChromaDB.
These tests require:
- Ollama service running with llama3.2:3b model
- ChromaDB initialized
- All Phase 6 components functional

Run after container rebuild: docker-compose exec fks_ai pytest tests/integration/ -v
"""
