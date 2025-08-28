#!/usr/bin/env python3
"""
kNN Vector Search Engine for semantic similarity search.
Executes kNN queries against OpenSearch and processes results with score normalization.
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from opensearchpy import OpenSearch
from opensearchpy.exceptions import OpenSearchException

from .schema import SearchSpec, SearchResult, SearchResponse, Entity, MarketImpact, Language
from .vector_query import VectorQueryBuilder
from .query_embedder import QueryEmbedder, get_query_embedder


class VectorSearchEngine:
    """
    kNN vector search engine for semantic similarity search.
    
    Handles embedding generation, query execution, score normalization,
    and result formatting for vector-based search.
    """
    
    def __init__(self,
                 opensearch_client: Optional[OpenSearch] = None,
                 opensearch_config: Optional[Dict[str, Any]] = None,
                 index_name: str = "tweets",
                 embedding_field: str = "embedding",
                 embedding_dimension: int = 1024,
                 model_name: str = 'BAAI/bge-large-en-v1.5',
                 cache_dir: Optional[str] = None,
                 entities_config_path: Optional[str] = None):
        """
        Initialize vector search engine.
        
        Args:
            opensearch_client: OpenSearch client instance
            opensearch_config: OpenSearch connection config
            index_name: Name of the search index
            embedding_field: Name of embedding field in index
            embedding_dimension: Dimension of embedding vectors
            model_name: Embedding model name
            cache_dir: Directory for caches
            entities_config_path: Path to entities configuration
        """
        # Initialize OpenSearch client
        if opensearch_client:
            self.client = opensearch_client
        elif opensearch_config:
            self.client = OpenSearch(**opensearch_config)
        else:
            # Default local configuration
            self.client = OpenSearch([{'host': 'localhost', 'port': 9200}])
        
        self.index_name = index_name
        self.embedding_field = embedding_field
        self.embedding_dimension = embedding_dimension
        
        # Initialize query builder
        self.query_builder = VectorQueryBuilder(
            embedding_field=embedding_field,
            embedding_dimension=embedding_dimension,
            index_name=index_name
        )
        
        # Initialize query embedder
        self.embedder = get_query_embedder(
            model_name=model_name,
            cache_dir=cache_dir,
            entities_config_path=entities_config_path
        )
        
        print(f"Vector search engine initialized:")
        print(f"  Index: {index_name}")
        print(f"  Embedding field: {embedding_field}")
        print(f"  Embedding dimension: {embedding_dimension}")
        print(f"  Model: {model_name}")
    
    def search(self, search_spec: SearchSpec) -> SearchResponse:
        """
        Execute kNN vector search.
        
        Args:
            search_spec: Search specification
            
        Returns:
            SearchResponse with results and metadata
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            embed_start = time.time()
            embedding_result = self.embedder.embed_query(search_spec)
            embedding_time_ms = int((time.time() - embed_start) * 1000)
            
            # Build kNN query
            query = self.query_builder.build_knn_query(search_spec, embedding_result.embedding)
            
            # Validate query
            warnings = self.query_builder.validate_query(search_spec, embedding_result.embedding)
            if warnings:
                print(f"Query validation warnings: {warnings}")
            
            # Execute search
            search_start = time.time()
            response = self.client.search(
                index=self.index_name,
                body=query,
                timeout="30s"
            )
            search_time_ms = int((time.time() - search_start) * 1000)
            
            # Process results
            process_start = time.time()
            results = self._process_search_results(response, search_spec)
            
            # Normalize scores
            normalized_results = self._normalize_scores(results)
            process_time_ms = int((time.time() - process_start) * 1000)
            
            total_time_ms = int((time.time() - start_time) * 1000)
            
            # Calculate score statistics
            score_stats = self._calculate_score_stats(normalized_results)
            
            return SearchResponse(
                query_spec=search_spec,
                results=normalized_results,
                total_hits=response.get('hits', {}).get('total', {}).get('value', 0),
                execution_time_ms=total_time_ms,
                query_embedding_time_ms=embedding_time_ms,
                search_time_ms=search_time_ms,
                post_processing_time_ms=process_time_ms,
                score_stats=score_stats
            )
            
        except OpenSearchException as e:
            raise RuntimeError(f"OpenSearch error during vector search: {e}")
        except Exception as e:
            raise RuntimeError(f"Vector search failed: {e}")
    
    def batch_search(self, search_specs: List[SearchSpec]) -> List[SearchResponse]:
        """
        Execute multiple kNN searches efficiently.
        
        Args:
            search_specs: List of search specifications
            
        Returns:
            List of SearchResponse objects
        """
        if not search_specs:
            return []
        
        start_time = time.time()
        
        try:
            # Generate embeddings for all queries
            embed_start = time.time()
            embedding_results = self.embedder.embed_batch_queries(search_specs)
            embedding_time_ms = int((time.time() - embed_start) * 1000)
            
            # Execute searches individually (could be optimized with msearch)
            responses = []
            for spec, embed_result in zip(search_specs, embedding_results):
                query = self.query_builder.build_knn_query(spec, embed_result.embedding)
                
                search_start = time.time()
                response = self.client.search(
                    index=self.index_name,
                    body=query,
                    timeout="30s"
                )
                search_time_ms = int((time.time() - search_start) * 1000)
                
                # Process results
                results = self._process_search_results(response, spec)
                normalized_results = self._normalize_scores(results)
                score_stats = self._calculate_score_stats(normalized_results)
                
                responses.append(SearchResponse(
                    query_spec=spec,
                    results=normalized_results,
                    total_hits=response.get('hits', {}).get('total', {}).get('value', 0),
                    execution_time_ms=search_time_ms,
                    query_embedding_time_ms=embedding_time_ms // len(search_specs),
                    search_time_ms=search_time_ms,
                    post_processing_time_ms=0,  # Included in search time
                    score_stats=score_stats
                ))
            
            return responses
            
        except Exception as e:
            raise RuntimeError(f"Batch vector search failed: {e}")
    
    def _process_search_results(self, response: Dict[str, Any], search_spec: SearchSpec) -> List[SearchResult]:
        """
        Process OpenSearch response into SearchResult objects.
        
        Args:
            response: OpenSearch response
            search_spec: Original search specification
            
        Returns:
            List of SearchResult objects
        """
        results = []
        hits = response.get('hits', {}).get('hits', [])
        
        for hit in hits:
            source = hit.get('_source', {})
            
            # Parse entities
            entities = []
            for entity_data in source.get('entities', []):
                entities.append(Entity(
                    entity_type=entity_data.get('entity_type'),
                    entity_id=entity_data.get('entity_id'),
                    name=entity_data.get('name'),
                    symbol=entity_data.get('symbol'),
                    confidence=entity_data.get('confidence', 1.0)
                ))
            
            # Parse dates
            created_at_str = source.get('created_at_iso', '')
            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            
            # Parse market impact and language
            market_impact = MarketImpact(source.get('market_impact', 'neutral'))
            language = Language(source.get('language', 'en'))
            
            # Calculate authority score (normalize follower count)
            authority_score = min(1.0, source.get('source_followers', 0) / 1000000.0)  # Max 1M followers = 1.0
            
            result = SearchResult(
                tweet_id=source.get('tweet_id', ''),
                content=source.get('clean_content', ''),
                original_content=source.get('original_content', ''),
                created_at=created_at,
                source_handle=source.get('source_handle', ''),
                source_followers=source.get('source_followers', 0),
                engagement_score=source.get('engagement_score', 0),
                authority_score=authority_score,
                entities=entities,
                market_impact=market_impact,
                language=language,
                vector_score=hit.get('_score', 0.0),
                final_score=hit.get('_score', 0.0)  # Will be updated during normalization
            )
            
            results.append(result)
        
        return results
    
    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Normalize vector similarity scores.
        
        Args:
            results: List of search results
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return results
        
        # Extract vector scores
        vector_scores = [r.vector_score for r in results if r.vector_score is not None]
        
        if not vector_scores:
            return results
        
        # Apply z-score normalization to vector scores
        if len(vector_scores) > 1:
            mean_score = statistics.mean(vector_scores)
            std_score = statistics.stdev(vector_scores)
            
            if std_score > 0:
                for result in results:
                    if result.vector_score is not None:
                        # Z-score normalization
                        normalized = (result.vector_score - mean_score) / std_score
                        # Scale to [0, 1] range
                        result.vector_score = max(0.0, min(1.0, (normalized + 3) / 6))
                        result.final_score = result.vector_score
        else:
            # Single result - normalize to 1.0
            results[0].vector_score = 1.0
            results[0].final_score = 1.0
        
        # Calculate additional score components
        for result in results:
            # Recency score (exponential decay)
            days_old = (datetime.utcnow() - result.created_at).days
            result.recency_score = np.exp(-days_old / 30.0)  # 30-day half-life
            
            # Authority boost (already calculated)
            result.authority_boost = result.authority_score
            
            # Engagement boost (log scale)
            result.engagement_boost = np.log(1 + result.engagement_score) / np.log(1001)  # Max ~1000 engagement
        
        return results
    
    def _calculate_score_stats(self, results: List[SearchResult]) -> Dict[str, float]:
        """
        Calculate score distribution statistics.
        
        Args:
            results: List of search results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {}
        
        vector_scores = [r.vector_score for r in results if r.vector_score is not None]
        final_scores = [r.final_score for r in results]
        
        stats = {}
        
        if vector_scores:
            stats.update({
                'vector_score_min': min(vector_scores),
                'vector_score_max': max(vector_scores),
                'vector_score_mean': statistics.mean(vector_scores),
                'vector_score_std': statistics.stdev(vector_scores) if len(vector_scores) > 1 else 0.0
            })
        
        if final_scores:
            stats.update({
                'final_score_min': min(final_scores),
                'final_score_max': max(final_scores),
                'final_score_mean': statistics.mean(final_scores),
                'final_score_std': statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0
            })
        
        return stats
    
    def check_index_health(self) -> Dict[str, Any]:
        """
        Check the health of the search index.
        
        Returns:
            Index health information
        """
        try:
            # Check if index exists
            if not self.client.indices.exists(index=self.index_name):
                return {
                    'status': 'error',
                    'message': f'Index {self.index_name} does not exist'
                }
            
            # Get index stats
            stats = self.client.indices.stats(index=self.index_name)
            index_stats = stats['indices'][self.index_name]
            
            # Get mapping info
            mapping = self.client.indices.get_mapping(index=self.index_name)
            properties = mapping[self.index_name]['mappings'].get('properties', {})
            
            # Check embedding field
            embedding_info = properties.get(self.embedding_field, {})
            has_embedding = embedding_info.get('type') == 'knn_vector'
            embedding_dim = embedding_info.get('dimension', 0)
            
            return {
                'status': 'healthy' if has_embedding and embedding_dim == self.embedding_dimension else 'warning',
                'index_name': self.index_name,
                'document_count': index_stats['total']['docs']['count'],
                'index_size': index_stats['total']['store']['size_in_bytes'],
                'embedding_field_exists': has_embedding,
                'embedding_dimension': embedding_dim,
                'dimension_match': embedding_dim == self.embedding_dimension
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to check index health: {e}'
            }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get search engine statistics.
        
        Returns:
            Statistics about the search engine
        """
        embedding_stats = self.embedder.get_embedding_stats()
        index_health = self.check_index_health()
        
        return {
            'embedding_stats': embedding_stats,
            'index_health': index_health,
            'query_builder': {
                'embedding_field': self.query_builder.embedding_field,
                'embedding_dimension': self.query_builder.embedding_dimension,
                'index_name': self.query_builder.index_name
            }
        }
    
    def optimize_query(self, search_spec: SearchSpec) -> Tuple[SearchSpec, Dict[str, Any]]:
        """
        Optimize search query for better performance.
        
        Args:
            search_spec: Original search specification
            
        Returns:
            Tuple of optimized SearchSpec and optimization info
        """
        optimizations = []
        optimized_spec = search_spec.copy(deep=True)
        
        # Optimize result size
        if search_spec.size > 100:
            optimized_spec.size = 100
            optimizations.append("Limited result size to 100 for performance")
        
        # Optimize entity filters (convert should to must if many entities)
        if search_spec.entities.get('should'):
            should_entities = search_spec.entities['should']
            if len(should_entities) > 5:
                # Move most confident entities to must
                sorted_entities = sorted(should_entities, key=lambda e: e.confidence, reverse=True)
                optimized_spec.entities['must'] = optimized_spec.entities.get('must', []) + sorted_entities[:2]
                optimized_spec.entities['should'] = sorted_entities[2:]
                optimizations.append("Converted high-confidence should entities to must")
        
        # Add time filter if none exists (performance optimization)
        if not search_spec.filters.time_range:
            from .schema import TimeRange
            optimized_spec.filters.time_range = TimeRange(days_back=90)  # Last 90 days
            optimizations.append("Added 90-day time filter for performance")
        
        return optimized_spec, {
            'optimizations_applied': optimizations,
            'original_size': search_spec.size,
            'optimized_size': optimized_spec.size
        }


# Factory function for easy initialization
def create_vector_search_engine(opensearch_config: Optional[Dict[str, Any]] = None,
                               **kwargs) -> VectorSearchEngine:
    """
    Create vector search engine with default configuration.
    
    Args:
        opensearch_config: OpenSearch connection configuration
        **kwargs: Additional arguments for VectorSearchEngine
        
    Returns:
        VectorSearchEngine instance
    """
    if opensearch_config is None:
        opensearch_config = {
            'hosts': [{'host': 'localhost', 'port': 9200}],
            'use_ssl': False,
            'verify_certs': False,
            'timeout': 30
        }
    
    return VectorSearchEngine(
        opensearch_config=opensearch_config,
        **kwargs
    )