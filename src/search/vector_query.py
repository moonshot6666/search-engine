#!/usr/bin/env python3
"""
Vector Query Builder for kNN search.
Converts SearchSpec to OpenSearch kNN queries with filters and boosting.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from .schema import SearchSpec, Entity, MarketImpact, Language, TimeRange


class VectorQueryBuilder:
    """
    Build OpenSearch kNN queries from SearchSpec.
    
    Generates kNN queries with cosine similarity search on embedding field,
    applying the same filters as BM25 queries for consistency.
    """
    
    def __init__(self, 
                 embedding_field: str = "embedding",
                 embedding_dimension: int = 1024,
                 index_name: str = "tweets"):
        """
        Initialize vector query builder.
        
        Args:
            embedding_field: Name of the embedding field in OpenSearch index
            embedding_dimension: Dimension of embedding vectors
            index_name: Default index name for queries
        """
        self.embedding_field = embedding_field
        self.embedding_dimension = embedding_dimension
        self.index_name = index_name
    
    def build_knn_query(self, 
                       search_spec: SearchSpec, 
                       query_embedding: List[float],
                       k: Optional[int] = None) -> Dict[str, Any]:
        """
        Build complete kNN query for OpenSearch.
        
        Args:
            search_spec: Search specification with filters and boosts
            query_embedding: Query embedding vector
            k: Number of nearest neighbors (defaults to search_spec.size)
            
        Returns:
            OpenSearch kNN query dictionary
        """
        if k is None:
            k = min(search_spec.size * 2, 200)  # Oversample for better results
        
        # Build the kNN query structure
        knn_query = {
            "size": search_spec.size,
            "knn": {
                self.embedding_field: {
                    "vector": query_embedding,
                    "k": k
                }
            },
            "_source": {
                "includes": [
                    "tweet_id", "clean_content", "original_content", "created_at_iso",
                    "source_handle", "source_followers", "engagement_score", "authority",
                    "entities", "market_impact", "language", "day_bucket",
                    "has_tokens", "has_projects", "has_kols", "has_events", "has_macros"
                ]
            }
        }
        
        # Add filters
        filters = self._build_filters(search_spec)
        if filters:
            knn_query["knn"][self.embedding_field]["filter"] = filters
        
        # Add post-processing for score boosting
        if self._needs_score_boosting(search_spec):
            knn_query["script_score"] = {
                "query": {
                    "knn": knn_query["knn"]
                },
                "script": self._build_score_script(search_spec)
            }
            # Remove the top-level kNN and move to script_score
            del knn_query["knn"]
        
        return knn_query
    
    def _build_filters(self, search_spec: SearchSpec) -> Optional[Dict[str, Any]]:
        """
        Build filter clauses from SearchSpec filters.
        
        Args:
            search_spec: Search specification
            
        Returns:
            OpenSearch filter query or None if no filters
        """
        filter_clauses = []
        
        # Entity filters
        entity_filters = self._build_entity_filters(search_spec.entities)
        if entity_filters:
            filter_clauses.extend(entity_filters)
        
        # Content filters
        content_filters = self._build_content_filters(search_spec.filters)
        if content_filters:
            filter_clauses.extend(content_filters)
        
        if not filter_clauses:
            return None
        
        if len(filter_clauses) == 1:
            return filter_clauses[0]
        
        return {
            "bool": {
                "must": filter_clauses
            }
        }
    
    def _build_entity_filters(self, entities: Dict[str, List[Entity]]) -> List[Dict[str, Any]]:
        """
        Build entity-based filters.
        
        Args:
            entities: Entity filters from SearchSpec
            
        Returns:
            List of filter clauses
        """
        filters = []
        
        for filter_type, entity_list in entities.items():
            if not entity_list:
                continue
            
            entity_clauses = []
            for entity in entity_list:
                # Build nested query for entity matching
                entity_clause = {
                    "nested": {
                        "path": "entities",
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"entities.entity_type": entity.entity_type.value}},
                                    {"term": {"entities.entity_id": entity.entity_id}}
                                ]
                            }
                        }
                    }
                }
                entity_clauses.append(entity_clause)
            
            # Combine entity clauses based on filter type
            if filter_type == "must":
                # All entities must be present
                filters.extend(entity_clauses)
            elif filter_type == "should":
                # At least one entity should be present
                if entity_clauses:
                    filters.append({
                        "bool": {
                            "should": entity_clauses,
                            "minimum_should_match": 1
                        }
                    })
            elif filter_type == "must_not":
                # None of these entities should be present
                for clause in entity_clauses:
                    filters.append({
                        "bool": {
                            "must_not": [clause]
                        }
                    })
        
        return filters
    
    def _build_content_filters(self, filters) -> List[Dict[str, Any]]:
        """
        Build content-based filters from SearchFilters.
        
        Args:
            filters: SearchFilters object
            
        Returns:
            List of filter clauses
        """
        filter_clauses = []
        
        # Market impact/sentiment filter
        if filters.sentiment:
            sentiment_values = [s.value for s in filters.sentiment]
            filter_clauses.append({
                "terms": {"market_impact": sentiment_values}
            })
        
        # Event filters (using boolean fields)
        if filters.events:
            for event in filters.events:
                if event in ["halving", "etf_approval", "hack", "regulation"]:
                    filter_clauses.append({
                        "term": {"has_events": True}
                    })
                    # Could add more specific event matching here
        
        # Macro filters
        if filters.macros:
            for macro in filters.macros:
                if macro in ["fed_rate", "inflation", "bank_crisis", "recession"]:
                    filter_clauses.append({
                        "term": {"has_macros": True}
                    })
        
        # KOL presence filter
        if filters.has_kols is not None:
            filter_clauses.append({
                "term": {"has_kols": filters.has_kols}
            })
        
        # Language filter
        if filters.language:
            language_values = [lang.value for lang in filters.language]
            filter_clauses.append({
                "terms": {"language": language_values}
            })
        
        # Time range filter
        if filters.time_range:
            time_filter = self._build_time_filter(filters.time_range)
            if time_filter:
                filter_clauses.append(time_filter)
        
        # Engagement threshold
        if filters.min_engagement is not None:
            filter_clauses.append({
                "range": {"engagement_score": {"gte": filters.min_engagement}}
            })
        
        # Authority threshold
        if filters.min_authority is not None:
            filter_clauses.append({
                "range": {"authority": {"gte": filters.min_authority}}
            })
        
        return filter_clauses
    
    def _build_time_filter(self, time_range: TimeRange) -> Optional[Dict[str, Any]]:
        """
        Build time range filter.
        
        Args:
            time_range: TimeRange specification
            
        Returns:
            Range filter for time or None
        """
        range_conditions = {}
        
        if time_range.days_back is not None:
            # Calculate start date from days_back
            start_date = datetime.utcnow() - timedelta(days=time_range.days_back)
            range_conditions["gte"] = start_date.isoformat()
        else:
            if time_range.start:
                range_conditions["gte"] = time_range.start.isoformat()
            if time_range.end:
                range_conditions["lte"] = time_range.end.isoformat()
        
        if not range_conditions:
            return None
        
        return {
            "range": {
                "created_at_iso": range_conditions
            }
        }
    
    def _needs_score_boosting(self, search_spec: SearchSpec) -> bool:
        """
        Check if score boosting is needed.
        
        Args:
            search_spec: Search specification
            
        Returns:
            True if boosting is required
        """
        boosts = search_spec.boosts
        return (boosts.recency != 0.0 or 
                boosts.authority != 0.0 or 
                boosts.engagement != 0.0)
    
    def _build_score_script(self, search_spec: SearchSpec) -> Dict[str, Any]:
        """
        Build script for score boosting.
        
        Args:
            search_spec: Search specification with boost weights
            
        Returns:
            Script score configuration
        """
        boosts = search_spec.boosts
        
        # Build script source
        script_parts = ["_score"]  # Start with kNN similarity score
        
        # Recency boost (exponential decay from current time)
        if boosts.recency > 0.0:
            script_parts.append(f"""
            + {boosts.recency} * Math.exp(-0.5 * Math.pow((System.currentTimeMillis() - doc['created_at_iso'].value.getMillis()) / (24 * 60 * 60 * 1000.0), 2))
            """.strip())
        
        # Authority boost (linear scaling)
        if boosts.authority > 0.0:
            script_parts.append(f"""
            + {boosts.authority} * (doc['authority'].empty ? 0.0 : doc['authority'].value)
            """.strip())
        
        # Engagement boost (log scaling)
        if boosts.engagement > 0.0:
            script_parts.append(f"""
            + {boosts.engagement} * Math.log(1 + (doc['engagement_score'].empty ? 0 : doc['engagement_score'].value))
            """.strip())
        
        script_source = " ".join(script_parts)
        
        return {
            "source": script_source,
            "lang": "painless"
        }
    
    def build_multi_knn_query(self, 
                             search_specs: List[SearchSpec],
                             query_embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Build multi-query kNN search for batch processing.
        
        Args:
            search_specs: List of search specifications
            query_embeddings: Corresponding query embeddings
            
        Returns:
            Multi-search query for OpenSearch
        """
        if len(search_specs) != len(query_embeddings):
            raise ValueError("Number of search specs must match number of embeddings")
        
        queries = []
        for spec, embedding in zip(search_specs, query_embeddings):
            query = self.build_knn_query(spec, embedding)
            queries.append({"index": self.index_name})
            queries.append(query)
        
        return {"queries": queries}
    
    def estimate_query_cost(self, search_spec: SearchSpec, query_embedding: List[float]) -> Dict[str, Any]:
        """
        Estimate computational cost of kNN query.
        
        Args:
            search_spec: Search specification
            query_embedding: Query embedding vector
            
        Returns:
            Cost estimation metrics
        """
        # Base kNN search cost
        k = min(search_spec.size * 2, 200)
        base_cost = k * self.embedding_dimension  # Vector similarity computations
        
        # Filter cost estimation
        filter_cost = 0
        if search_spec.entities:
            filter_cost += len([e for entities in search_spec.entities.values() for e in entities]) * 100
        
        if search_spec.filters.time_range:
            filter_cost += 50  # Time range filtering
        
        if search_spec.filters.sentiment:
            filter_cost += len(search_spec.filters.sentiment) * 10
        
        # Boosting cost
        boost_cost = 0
        if self._needs_score_boosting(search_spec):
            boost_cost = k * 20  # Script score evaluation
        
        return {
            "base_cost": base_cost,
            "filter_cost": filter_cost,
            "boost_cost": boost_cost,
            "total_cost": base_cost + filter_cost + boost_cost,
            "k_neighbors": k,
            "estimated_ms": (base_cost + filter_cost + boost_cost) / 1000.0  # Rough estimate
        }
    
    def validate_query(self, search_spec: SearchSpec, query_embedding: List[float]) -> List[str]:
        """
        Validate kNN query configuration.
        
        Args:
            search_spec: Search specification
            query_embedding: Query embedding vector
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        # Check embedding dimension
        if len(query_embedding) != self.embedding_dimension:
            warnings.append(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(query_embedding)}")
        
        # Check for empty embedding
        if not query_embedding or all(v == 0.0 for v in query_embedding):
            warnings.append("Query embedding is empty or all zeros")
        
        # Check k size
        k = min(search_spec.size * 2, 200)
        if k < search_spec.size:
            warnings.append(f"k ({k}) is less than requested size ({search_spec.size})")
        
        # Check for conflicting filters
        if search_spec.entities.get("must") and search_spec.entities.get("must_not"):
            must_ids = {e.entity_id for e in search_spec.entities["must"]}
            must_not_ids = {e.entity_id for e in search_spec.entities["must_not"]}
            conflicts = must_ids & must_not_ids
            if conflicts:
                warnings.append(f"Conflicting entity filters: {conflicts} in both must and must_not")
        
        # Check time range validity
        if search_spec.filters.time_range:
            tr = search_spec.filters.time_range
            if tr.start and tr.end and tr.start > tr.end:
                warnings.append("Time range start is after end")
        
        return warnings