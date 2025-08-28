#!/usr/bin/env python3
"""
Search module for hybrid search engine.
Provides kNN vector search functionality with semantic similarity.
"""

from .schema import (
    SearchSpec, SearchResult, SearchResponse, EmbeddingResult,
    Entity, MarketImpact, Language, EntityType,
    SearchFilters, SearchBoosts, TimeRange
)

# Always available imports
from .vector_query import VectorQueryBuilder

# Optional imports that require external dependencies
try:
    from .query_embedder import QueryEmbedder, get_query_embedder
    _EMBEDDER_AVAILABLE = True
except ImportError:
    _EMBEDDER_AVAILABLE = False

# Optional imports that require external dependencies
try:
    from .vector_search import VectorSearchEngine, create_vector_search_engine
    _VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    _VECTOR_SEARCH_AVAILABLE = False

__all__ = [
    # Schema classes (always available)
    'SearchSpec', 'SearchResult', 'SearchResponse', 'EmbeddingResult',
    'Entity', 'MarketImpact', 'Language', 'EntityType',
    'SearchFilters', 'SearchBoosts', 'TimeRange',
    
    # Core components (always available)
    'VectorQueryBuilder',
]

# Add optional components if available
if _EMBEDDER_AVAILABLE:
    __all__.extend(['QueryEmbedder', 'get_query_embedder'])

if _VECTOR_SEARCH_AVAILABLE:
    __all__.extend(['VectorSearchEngine', 'create_vector_search_engine'])