"""
ChromaDB Memory Manager for Trading Insights

Provides persistent memory for multi-agent system decisions and learnings.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings


class TradingMemory:
    """
    ChromaDB-based memory system for storing and retrieving trading insights.

    Features:
        - Semantic search for similar past decisions
        - Persistent storage across restarts
        - Metadata filtering (symbol, regime, action)
        - Automatic embedding generation
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        collection_name: str = "trading_insights"
    ):
        """
        Initialize ChromaDB client and collection.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of ChromaDB collection
        """
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Multi-agent trading decisions and insights"}
        )

    def add_insight(
        self,
        text: str,
        metadata: dict,
        insight_id: Optional[str] = None
    ) -> str:
        """
        Store trading insight with metadata.

        Args:
            text: Insight text (decision reasoning, agent debate, etc.)
            metadata: Metadata dict (symbol, action, confidence, regime, timestamp)
            insight_id: Optional custom ID (auto-generated if None)

        Returns:
            Insight ID

        Example:
            >>> memory.add_insight(
            ...     text="BTCUSDT LONG: Bull regime, RSI oversold, strong support",
            ...     metadata={
            ...         "symbol": "BTCUSDT",
            ...         "action": "BUY",
            ...         "confidence": 0.75,
            ...         "regime": "bull",
            ...         "timestamp": "2025-10-30T12:00:00"
            ...     }
            ... )
        """
        if insight_id is None:
            insight_id = f"insight_{len(self.collection.get()['ids']) + 1}"

        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[insight_id]
        )

        return insight_id

    def query_similar(
        self,
        query: str,
        n_results: int = 5,
        metadata_filter: Optional[dict] = None
    ) -> list[dict]:
        """
        Retrieve similar past insights using semantic search.

        Args:
            query: Query text for semantic similarity
            n_results: Number of results to return
            metadata_filter: Optional metadata filter (e.g., {"symbol": "BTCUSDT"})

        Returns:
            List of dicts with 'text' and 'metadata'

        Example:
            >>> results = memory.query_similar(
            ...     query="BTCUSDT long opportunity in bull market",
            ...     n_results=3,
            ...     metadata_filter={"symbol": "BTCUSDT"}
            ... )
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=metadata_filter
        )

        # Format results
        insights = []
        for i in range(len(results['ids'][0])):
            insights.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })

        return insights

    def get_by_id(self, insight_id: str) -> Optional[dict]:
        """
        Retrieve specific insight by ID.

        Args:
            insight_id: Insight ID

        Returns:
            Dict with 'text' and 'metadata' or None if not found
        """
        results = self.collection.get(ids=[insight_id])

        if not results['ids']:
            return None

        return {
            "id": results['ids'][0],
            "text": results['documents'][0],
            "metadata": results['metadatas'][0]
        }

    def get_all(self, limit: Optional[int] = None) -> list[dict]:
        """
        Get all insights (or limited subset).

        Args:
            limit: Maximum number of insights to return

        Returns:
            List of insight dicts
        """
        results = self.collection.get(limit=limit)

        insights = []
        for i in range(len(results['ids'])):
            insights.append({
                "id": results['ids'][i],
                "text": results['documents'][i],
                "metadata": results['metadatas'][i]
            })

        return insights

    def count(self) -> int:
        """Get total number of stored insights."""
        return self.collection.count()

    def clear(self):
        """Clear all insights from collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Multi-agent trading decisions and insights"}
        )
