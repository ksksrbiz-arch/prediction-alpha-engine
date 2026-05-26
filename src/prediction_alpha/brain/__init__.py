"""True Neutral Brain v2 integration layer for Prediction Alpha Engine.

This package turns the engine's high-value, agent-researched opportunities
into first-class citizens inside the user's sovereign knowledge graph + vector store.

Key components:
- BrainIngestionConfig
- TrueNeutralBrainIngestor (write path with dedup + embeddings)
- BrainRetriever (read path for the Brain and other agents)
"""

from prediction_alpha.brain.config import BrainIngestionConfig, build_brain_config
from prediction_alpha.brain.ingestor import TrueNeutralBrainIngestor
from prediction_alpha.brain.models import BrainOpportunity, WealthTrackRelevance
from prediction_alpha.brain.retriever import BrainRetriever

__all__ = [
    "BrainIngestionConfig",
    "build_brain_config",
    "TrueNeutralBrainIngestor",
    "BrainOpportunity",
    "WealthTrackRelevance",
    "BrainRetriever",
]
