#!/usr/bin/env python3
"""
Search module for hybrid search engine.
Provides local hybrid BM25 + vector search functionality with investment-aware scoring.
"""

from .schema import (
    SearchSpec, SearchResult, SearchResponse, EmbeddingResult,
    Entity, MarketImpact, Language, EntityType,
    SearchFilters, SearchBoosts, TimeRange
)

from .local_search import LocalHybridSearchEngine
from .cluster import RealTimeResultClusterer, ClusteredSearchResponse
from .blend import ResultBlender

__all__ = [
    # Schema classes
    'SearchSpec', 'SearchResult', 'SearchResponse', 'EmbeddingResult',
    'Entity', 'MarketImpact', 'Language', 'EntityType',
    'SearchFilters', 'SearchBoosts', 'TimeRange',
    
    # Core components
    'LocalHybridSearchEngine', 
    'RealTimeResultClusterer', 'ClusteredSearchResponse',
    'ResultBlender',
]