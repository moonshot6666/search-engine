#!/usr/bin/env python3
"""
SearchSpec schema definition for hybrid search engine.
Pydantic models for LLM-to-search translation and query validation.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class EntityType(str, Enum):
    """Entity types supported by the search engine."""
    TOKEN = "token"
    PROJECT = "project"
    KOL = "kol"
    EVENT = "event"
    MACRO = "macro"


class MarketImpact(str, Enum):
    """Market impact sentiment values."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Language(str, Enum):
    """Supported languages."""
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    JA = "ja"
    KO = "ko"
    ZH = "zh"


class Entity(BaseModel):
    """Entity reference for filtering and boosting."""
    entity_type: EntityType
    entity_id: str
    name: Optional[str] = None
    symbol: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class TimeRange(BaseModel):
    """Time range filter for search queries."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    days_back: Optional[int] = Field(ge=1, le=365, default=None)
    
    @validator('days_back')
    def validate_days_back(cls, v, values):
        if v is not None and ('start' in values or 'end' in values):
            if values.get('start') or values.get('end'):
                raise ValueError("Cannot specify both days_back and start/end dates")
        return v


class SearchFilters(BaseModel):
    """Search filters for content filtering."""
    sentiment: Optional[List[MarketImpact]] = None
    events: Optional[List[str]] = None
    macros: Optional[List[str]] = None
    has_kols: Optional[bool] = None
    language: Optional[List[Language]] = None
    time_range: Optional[TimeRange] = None
    min_engagement: Optional[int] = Field(ge=0, default=None)
    min_authority: Optional[float] = Field(ge=0.0, le=1.0, default=None)


class SearchBoosts(BaseModel):
    """Boost weights for score blending (must sum to 1.0)."""
    recency: float = Field(ge=0.0, le=1.0, default=0.10)
    authority: float = Field(ge=0.0, le=1.0, default=0.05)
    engagement: float = Field(ge=0.0, le=1.0, default=0.05)
    
    @validator('engagement')
    def validate_boost_sum(cls, v, values):
        total = v + values.get('recency', 0.0) + values.get('authority', 0.0)
        if abs(total - 0.2) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Boost weights must sum to 0.2 (recency + authority + engagement), got {total}")
        return v


class SearchSpec(BaseModel):
    """
    Core search specification for hybrid search.
    
    This schema defines the interface between LLM query generation
    and the hybrid search engine execution.
    """
    # Primary search query
    query: str = Field(min_length=1, max_length=1000, description="Main search terms")
    
    # Entity filters
    entities: Dict[str, List[Entity]] = Field(
        default_factory=dict,
        description="Entity filters: must, should, must_not"
    )
    
    # Content filters
    filters: SearchFilters = Field(
        default_factory=SearchFilters,
        description="Content filtering criteria"
    )
    
    # Score boosting weights
    boosts: SearchBoosts = Field(
        default_factory=SearchBoosts,
        description="Score blending weights for ranking"
    )
    
    # Result size limit
    size: int = Field(ge=1, le=200, default=20, description="Number of results to return")
    
    # Search method weights (for hybrid search)
    bm25_weight: float = Field(ge=0.0, le=1.0, default=0.45)
    vector_weight: float = Field(ge=0.0, le=1.0, default=0.35)
    
    @validator('entities')
    def validate_entity_keys(cls, v):
        allowed_keys = {'must', 'should', 'must_not'}
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Invalid entity filter key: {key}. Must be one of {allowed_keys}")
        return v
    
    @validator('vector_weight')
    def validate_search_weights(cls, v, values):
        bm25_weight = values.get('bm25_weight', 0.45)
        boost_total = 0.2  # recency + authority + engagement
        total = v + bm25_weight + boost_total
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Search weights must sum to 1.0, got {total}")
        return v


class SearchResult(BaseModel):
    """Individual search result with metadata."""
    tweet_id: str
    content: str
    original_content: str
    created_at: datetime
    source_handle: str
    source_followers: int
    engagement_score: int
    authority_score: float
    entities: List[Entity]
    market_impact: MarketImpact
    language: Language
    
    # Search scores
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    recency_score: Optional[float] = None
    authority_boost: Optional[float] = None
    engagement_boost: Optional[float] = None
    final_score: float
    
    # Clustering metadata (if applied)
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None


class SearchResponse(BaseModel):
    """Complete search response with results and metadata."""
    query_spec: SearchSpec
    results: List[SearchResult]
    total_hits: int
    execution_time_ms: int
    
    # Score distribution statistics
    score_stats: Dict[str, float] = Field(default_factory=dict)
    
    # Clustering information (if applied)
    clusters: Optional[Dict[int, Dict[str, Any]]] = None
    
    # Query analysis
    query_embedding_time_ms: Optional[int] = None
    search_time_ms: Optional[int] = None
    post_processing_time_ms: Optional[int] = None


class EmbeddingResult(BaseModel):
    """Result of embedding generation for caching."""
    text: str
    embedding: List[float]
    embedding_model: str
    embedding_time_ms: int
    cache_hit: bool = False