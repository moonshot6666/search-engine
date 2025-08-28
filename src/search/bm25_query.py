"""
BM25 Query Builder for Hybrid Search Engine

Converts SearchSpec DSL to OpenSearch BM25 queries with entity filters,
time ranges, sentiment filtering, and phrase boosting. Handles nested
entity queries and complex boolean logic.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from .schema import SearchSpec, EntityType, MarketImpact, TimeRange, Entity

logger = logging.getLogger(__name__)


class BM25QueryBuilder:
    """Builds OpenSearch BM25 queries from SearchSpec objects."""
    
    def __init__(self):
        """Initialize the query builder."""
        self.entity_boost_weights = {
            EntityType.TOKEN: 2.0,
            EntityType.PROJECT: 1.8,
            EntityType.KOL: 1.5,
            EntityType.EVENT: 1.3,
            EntityType.MACRO: 1.2
        }
        
    def build_query(self, search_spec: SearchSpec) -> Dict[str, Any]:
        """
        Build complete OpenSearch BM25 query from SearchSpec.
        
        Args:
            search_spec: Validated search specification
            
        Returns:
            Complete OpenSearch query dictionary
        """
        query_body = {
            "query": self._build_bool_query(search_spec),
            "size": search_spec.size,
            "_source": {
                "excludes": ["embedding"]  # Exclude vector field from BM25-only results
            },
            "sort": self._build_sort_clause(search_spec),
            "highlight": self._build_highlight_config()
        }
        
        # Add aggregations if needed for faceted search
        query_body["aggs"] = self._build_aggregations()
        
        return query_body
        
    def _build_bool_query(self, search_spec: SearchSpec) -> Dict[str, Any]:
        """
        Build the main boolean query with all filters and boosts.
        
        Args:
            search_spec: Search specification
            
        Returns:
            Boolean query structure
        """
        bool_query = {
            "bool": {
                "must": [],
                "should": [],
                "must_not": [],
                "filter": []
            }
        }
        
        # Add main content query
        if search_spec.query.strip():
            content_query = self._build_content_query(search_spec.query)
            bool_query["bool"]["must"].append(content_query)
            
        # Add entity filters
        entity_clauses = self._build_entity_filters(search_spec.entities)
        if entity_clauses["must"]:
            bool_query["bool"]["must"].extend(entity_clauses["must"])
        if entity_clauses["should"]:
            bool_query["bool"]["should"].extend(entity_clauses["should"])
        if entity_clauses["must_not"]:
            bool_query["bool"]["must_not"].extend(entity_clauses["must_not"])
            
        # Add advanced filters
        filter_clauses = self._build_advanced_filters(search_spec.filters)
        if filter_clauses:
            bool_query["bool"]["filter"].extend(filter_clauses)
            
        # Set minimum should match for OR conditions
        if bool_query["bool"]["should"]:
            bool_query["bool"]["minimum_should_match"] = 1
            
        return bool_query
        
    def _build_content_query(self, query_text: str) -> Dict[str, Any]:
        """
        Build multi-match query for content search with field boosting.
        
        Args:
            query_text: Search query string
            
        Returns:
            Multi-match query structure
        """
        return {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "clean_content^2.0",      # Primary content field with boost
                    "clean_content.keyword^1.5",  # Exact phrase matching
                    "entities.name^1.8",      # Entity names
                    "entities.symbol^2.2"     # Entity symbols (highest boost)
                ],
                "type": "best_fields",
                "operator": "and",
                "fuzziness": "AUTO",
                "prefix_length": 2,
                "max_expansions": 50
            }
        }
        
    def _build_entity_filters(self, entity_filters) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build nested entity filters with confidence-based boosting.
        
        Args:
            entity_filters: EntityFilters object
            
        Returns:
            Dictionary with must/should/must_not entity clauses
        """
        clauses = {
            "must": [],
            "should": [], 
            "must_not": []
        }
        
        # Required entities (must match)
        for entity in entity_filters.must:
            nested_query = self._build_nested_entity_query(entity, required=True)
            clauses["must"].append(nested_query)
            
        # Preferred entities (should match, with boosting)
        for entity in entity_filters.should:
            nested_query = self._build_nested_entity_query(entity, required=False)
            clauses["should"].append(nested_query)
            
        # Excluded entities (must not match)
        for entity in entity_filters.must_not:
            nested_query = self._build_nested_entity_query(entity, required=True)
            clauses["must_not"].append(nested_query)
            
        return clauses
        
    def _build_nested_entity_query(self, entity: Entity, required: bool = False) -> Dict[str, Any]:
        """
        Build nested query for entity matching with confidence weighting.
        
        Args:
            entity: Entity specification
            required: Whether this is a required match
            
        Returns:
            Nested query structure
        """
        base_boost = self.entity_boost_weights.get(entity.entity_type, 1.0)
        confidence_boost = entity.confidence * base_boost
        
        nested_query = {
            "nested": {
                "path": "entities",
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"entities.entity_id": entity.entity_id}},
                            {"term": {"entities.entity_type": entity.entity_type.value}}
                        ]
                    }
                },
                "score_mode": "max"
            }
        }
        
        # Add confidence boosting for non-required matches
        if not required and confidence_boost != 1.0:
            nested_query = {
                "function_score": {
                    "query": nested_query,
                    "boost": confidence_boost,
                    "boost_mode": "multiply"
                }
            }
            
        return nested_query
        
    def _build_advanced_filters(self, filters) -> List[Dict[str, Any]]:
        """
        Build advanced filter clauses for sentiment, time, engagement, etc.
        
        Args:
            filters: Filters object
            
        Returns:
            List of filter clauses
        """
        filter_clauses = []
        
        # Market sentiment filter
        if filters.sentiment:
            sentiment_values = [s.value if hasattr(s, 'value') else s for s in filters.sentiment]
            filter_clauses.append({
                "terms": {"market_impact": sentiment_values}
            })
            
        # Event type filters
        if filters.events:
            event_filter = {
                "nested": {
                    "path": "entities",
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"entities.entity_type": "event"}},
                                {"terms": {"entities.entity_id": filters.events}}
                            ]
                        }
                    }
                }
            }
            filter_clauses.append(event_filter)
            
        # Macro event filters
        if filters.macros:
            macro_filter = {
                "nested": {
                    "path": "entities",
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"entities.entity_type": "macro"}},
                                {"terms": {"entities.entity_id": filters.macros}}
                            ]
                        }
                    }
                }
            }
            filter_clauses.append(macro_filter)
            
        # KOL presence filter
        if filters.has_kols is not None:
            filter_clauses.append({
                "term": {"has_kols": filters.has_kols}
            })
            
        # Language filter
        if filters.language:
            lang_values = [lang.value if hasattr(lang, 'value') else lang for lang in filters.language]
            filter_clauses.append({
                "terms": {"language": lang_values}
            })
            
        # Time range filter
        if filters.time_range:
            time_filter = self._build_time_range_filter(filters.time_range)
            filter_clauses.append(time_filter)
            
        # Engagement threshold filter
        if filters.min_engagement is not None:
            filter_clauses.append({
                "range": {"engagement_score": {"gte": filters.min_engagement}}
            })
            
        # Follower threshold filter
        if filters.min_followers is not None:
            filter_clauses.append({
                "range": {"source_followers": {"gte": filters.min_followers}}
            })
            
        return filter_clauses
        
    def _build_time_range_filter(self, time_range: TimeRange) -> Dict[str, Any]:
        """
        Build time range filter for temporal queries.
        
        Args:
            time_range: TimeRange specification
            
        Returns:
            Range filter for created_at_iso field
        """
        range_filter = {"range": {"created_at_iso": {}}}
        
        if time_range.start:
            # Ensure timezone awareness
            start_time = time_range.start
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            range_filter["range"]["created_at_iso"]["gte"] = start_time.isoformat()
            
        if time_range.end:
            # Ensure timezone awareness
            end_time = time_range.end
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            range_filter["range"]["created_at_iso"]["lte"] = end_time.isoformat()
            
        return range_filter
        
    def _build_sort_clause(self, search_spec: SearchSpec) -> List[Dict[str, Any]]:
        """
        Build sort clause with relevance score and boost factors.
        
        Args:
            search_spec: Search specification with boost weights
            
        Returns:
            Sort clause list
        """
        sort_clauses = []
        
        # Primary relevance score
        sort_clauses.append("_score")
        
        # Recency boost (weighted by boost config)
        if search_spec.boosts.recency > 0:
            sort_clauses.append({
                "created_at_iso": {
                    "order": "desc",
                    "missing": "_last"
                }
            })
            
        # Authority boost (follower count as proxy)
        if search_spec.boosts.authority > 0:
            sort_clauses.append({
                "source_followers": {
                    "order": "desc",
                    "missing": "_last"
                }
            })
            
        # Engagement boost
        if search_spec.boosts.engagement > 0:
            sort_clauses.append({
                "engagement_score": {
                    "order": "desc",
                    "missing": "_last"
                }
            })
            
        return sort_clauses
        
    def _build_highlight_config(self) -> Dict[str, Any]:
        """
        Build highlight configuration for search result snippets.
        
        Returns:
            Highlight configuration
        """
        return {
            "fields": {
                "clean_content": {
                    "fragment_size": 150,
                    "number_of_fragments": 3,
                    "highlight_query": {
                        "match_phrase_prefix": {}
                    }
                }
            },
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "encoder": "html"
        }
        
    def _build_aggregations(self) -> Dict[str, Any]:
        """
        Build aggregations for faceted search and analytics.
        
        Returns:
            Aggregation configuration
        """
        return {
            "market_impact": {
                "terms": {
                    "field": "market_impact",
                    "size": 10
                }
            },
            "entity_types": {
                "nested": {"path": "entities"},
                "aggs": {
                    "types": {
                        "terms": {
                            "field": "entities.entity_type",
                            "size": 10
                        }
                    }
                }
            },
            "languages": {
                "terms": {
                    "field": "language",
                    "size": 10
                }
            },
            "date_histogram": {
                "date_histogram": {
                    "field": "created_at_iso",
                    "calendar_interval": "1d",
                    "min_doc_count": 1
                }
            }
        }
        
    def build_phrase_query(self, phrase: str, field: str = "clean_content", boost: float = 1.0) -> Dict[str, Any]:
        """
        Build phrase query for exact phrase matching.
        
        Args:
            phrase: Exact phrase to match
            field: Target field
            boost: Query boost factor
            
        Returns:
            Match phrase query
        """
        return {
            "match_phrase": {
                field: {
                    "query": phrase,
                    "boost": boost,
                    "slop": 2  # Allow 2 words between phrase terms
                }
            }
        }
        
    def build_fuzzy_query(self, term: str, field: str = "clean_content", fuzziness: str = "AUTO") -> Dict[str, Any]:
        """
        Build fuzzy query for typo-tolerant matching.
        
        Args:
            term: Search term
            field: Target field
            fuzziness: Fuzziness setting
            
        Returns:
            Fuzzy query
        """
        return {
            "fuzzy": {
                field: {
                    "value": term,
                    "fuzziness": fuzziness,
                    "prefix_length": 1,
                    "max_expansions": 50
                }
            }
        }
        
    def explain_query(self, query: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of the query structure.
        
        Args:
            query: OpenSearch query dictionary
            
        Returns:
            Query explanation string
        """
        explanations = []
        
        if "query" in query and "bool" in query["query"]:
            bool_query = query["query"]["bool"]
            
            if bool_query.get("must"):
                explanations.append(f"MUST match: {len(bool_query['must'])} conditions")
                
            if bool_query.get("should"):
                explanations.append(f"SHOULD match: {len(bool_query['should'])} conditions")
                
            if bool_query.get("must_not"):
                explanations.append(f"MUST NOT match: {len(bool_query['must_not'])} conditions")
                
            if bool_query.get("filter"):
                explanations.append(f"Filtered by: {len(bool_query['filter'])} conditions")
                
        if "size" in query:
            explanations.append(f"Return {query['size']} results")
            
        return " | ".join(explanations) if explanations else "Empty query"