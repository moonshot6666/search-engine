"""
BM25 Search Engine for Hybrid Search System

Executes BM25 queries against OpenSearch, normalizes scores, and formats
results. Handles score blending preparation and provides detailed search
metadata for the hybrid search pipeline.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import RequestError, ConnectionError

from .schema import SearchSpec, SearchResult, SearchResponse
from .bm25_query import BM25QueryBuilder

logger = logging.getLogger(__name__)


class BM25SearchEngine:
    """BM25 search engine with score normalization and result formatting."""
    
    def __init__(self,
                 host: str = "localhost",
                 port: int = 9200,
                 use_ssl: bool = False,
                 verify_certs: bool = False,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 index_name: str = "tweets-hybrid",
                 timeout: int = 30):
        """
        Initialize BM25 search engine with OpenSearch connection.
        
        Args:
            host: OpenSearch host
            port: OpenSearch port
            use_ssl: Whether to use SSL
            verify_certs: Whether to verify SSL certificates
            username: Authentication username
            password: Authentication password
            index_name: Default index name
            timeout: Request timeout in seconds
        """
        auth = None
        if username and password:
            auth = (username, password)
            
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            connection_class=RequestsHttpConnection,
            timeout=timeout
        )
        
        self.index_name = index_name
        self.query_builder = BM25QueryBuilder()
        self.scoring_config = ScoringConfig()
        
    def search(self, 
               search_spec: SearchSpec,
               index_name: Optional[str] = None,
               explain: bool = False) -> SearchResponse:
        """
        Execute BM25 search against OpenSearch index.
        
        Args:
            search_spec: Validated search specification
            index_name: Override default index name
            explain: Whether to include score explanations
            
        Returns:
            Complete search response with normalized scores
        """
        start_time = time.time()
        index = index_name or self.index_name
        
        try:
            # Build OpenSearch query
            query = self.query_builder.build_query(search_spec)
            if explain:
                query["explain"] = True
                
            logger.debug(f"Executing BM25 query: {query}")
            
            # Execute search
            response = self.client.search(
                index=index,
                body=query,
                request_timeout=30
            )
            
            # Process results
            search_results, score_stats = self._process_search_results(
                response, search_spec
            )
            
            # Calculate query time
            query_time_ms = int((time.time() - start_time) * 1000)
            
            # Build response
            search_response = SearchResponse(
                results=search_results,
                total_hits=response["hits"]["total"]["value"],
                query_time_ms=query_time_ms,
                search_spec=search_spec,
                score_breakdown=score_stats,
                filters_applied=self._extract_applied_filters(query)
            )
            
            logger.info(f"BM25 search completed: {len(search_results)} results in {query_time_ms}ms")
            return search_response
            
        except ConnectionError as e:
            logger.error(f"OpenSearch connection error: {e}")
            raise RuntimeError(f"Search service unavailable: {e}")
            
        except RequestError as e:
            logger.error(f"OpenSearch request error: {e}")
            raise ValueError(f"Invalid search query: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected search error: {e}")
            raise RuntimeError(f"Search failed: {e}")
            
    def _process_search_results(self, 
                              response: Dict[str, Any],
                              search_spec: SearchSpec) -> Tuple[List[SearchResult], Dict[str, float]]:
        """
        Process OpenSearch response and normalize scores.
        
        Args:
            response: Raw OpenSearch response
            search_spec: Original search specification
            
        Returns:
            Tuple of (processed results, score statistics)
        """
        hits = response["hits"]["hits"]
        if not hits:
            return [], {}
            
        # Extract raw scores for normalization
        raw_scores = [hit["_score"] for hit in hits if hit["_score"] is not None]
        normalized_scores = self._normalize_scores(raw_scores)
        
        # Calculate score statistics
        score_stats = {
            "min_score": min(raw_scores) if raw_scores else 0.0,
            "max_score": max(raw_scores) if raw_scores else 0.0,
            "mean_score": np.mean(raw_scores) if raw_scores else 0.0,
            "std_score": np.std(raw_scores) if raw_scores else 0.0,
            "normalization_method": self.scoring_config.normalization_method
        }
        
        # Process individual results
        results = []
        for i, hit in enumerate(hits):
            try:
                result = self._convert_hit_to_result(
                    hit, 
                    normalized_scores[i] if i < len(normalized_scores) else 0.0
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process search hit: {e}")
                continue
                
        return results, score_stats
        
    def _convert_hit_to_result(self, hit: Dict[str, Any], normalized_score: float) -> SearchResult:
        """
        Convert OpenSearch hit to SearchResult object.
        
        Args:
            hit: OpenSearch hit document
            normalized_score: Normalized BM25 score
            
        Returns:
            SearchResult object
        """
        source = hit["_source"]
        
        # Parse entities
        entities = []
        for entity_data in source.get("entities", []):
            entities.append({
                "entity_type": entity_data.get("entity_type"),
                "entity_id": entity_data.get("entity_id"),
                "name": entity_data.get("name"),
                "symbol": entity_data.get("symbol"),
                "confidence": entity_data.get("confidence", 1.0)
            })
            
        return SearchResult(
            tweet_id=source["tweet_id"],
            clean_content=source["clean_content"],
            original_content=source["original_content"],
            created_at_iso=source["created_at_iso"],
            source_handle=source["source_handle"],
            source_followers=source["source_followers"],
            engagement_score=source["engagement_score"],
            market_impact=source["market_impact"],
            entities=entities,
            bm25_score=hit["_score"],
            vector_score=None,  # Not available in BM25-only search
            combined_score=normalized_score,
            day_bucket=source["day_bucket"],
            language=source["language"],
            has_tokens=source["has_tokens"],
            has_projects=source["has_projects"],
            has_kols=source["has_kols"],
            has_events=source["has_events"],
            has_macros=source["has_macros"]
        )
        
    def _normalize_scores(self, scores: List[float], method: str = None) -> List[float]:
        """
        Normalize BM25 scores using specified method.
        
        Args:
            scores: Raw BM25 scores
            method: Normalization method (z_score, min_max, or none)
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
            
        method = method or self.scoring_config.normalization_method
        scores_array = np.array(scores)
        
        if method == "z_score":
            # Z-score normalization
            mean_score = np.mean(scores_array)
            std_score = np.std(scores_array)
            
            if std_score == 0:
                return [0.5] * len(scores)  # All scores equal
                
            normalized = (scores_array - mean_score) / std_score
            # Convert to 0-1 range using sigmoid
            normalized = 1 / (1 + np.exp(-normalized))
            
        elif method == "min_max":
            # Min-max normalization
            min_score = np.min(scores_array)
            max_score = np.max(scores_array)
            
            if max_score == min_score:
                return [0.5] * len(scores)  # All scores equal
                
            normalized = (scores_array - min_score) / (max_score - min_score)
            
        elif method == "none":
            # No normalization
            normalized = scores_array
            
        else:
            logger.warning(f"Unknown normalization method: {method}")
            normalized = scores_array
            
        return normalized.tolist()
        
    def _extract_applied_filters(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract applied filters from query for response metadata.
        
        Args:
            query: OpenSearch query dictionary
            
        Returns:
            Dictionary of applied filters
        """
        applied_filters = {}
        
        if "query" in query and "bool" in query["query"]:
            bool_query = query["query"]["bool"]
            
            # Count filter types
            if bool_query.get("filter"):
                filter_count = len(bool_query["filter"])
                applied_filters["filter_count"] = filter_count
                
            if bool_query.get("must"):
                applied_filters["required_conditions"] = len(bool_query["must"])
                
            if bool_query.get("should"):
                applied_filters["optional_conditions"] = len(bool_query["should"])
                
            if bool_query.get("must_not"):
                applied_filters["excluded_conditions"] = len(bool_query["must_not"])
                
        if "sort" in query:
            applied_filters["custom_sorting"] = True
            
        if "highlight" in query:
            applied_filters["highlighting_enabled"] = True
            
        return applied_filters
        
    def get_search_stats(self, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get search index statistics and health information.
        
        Args:
            index_name: Index name to check
            
        Returns:
            Index statistics dictionary
        """
        index = index_name or self.index_name
        
        try:
            # Check if index exists
            if not self.client.indices.exists(index=index):
                return {"error": f"Index {index} does not exist"}
                
            # Get index stats
            stats = self.client.indices.stats(index=index)
            index_stats = stats["indices"][index]
            
            # Get index settings
            settings = self.client.indices.get_settings(index=index)
            index_settings = settings[index]["settings"]
            
            return {
                "index_name": index,
                "document_count": index_stats["total"]["docs"]["count"],
                "store_size_bytes": index_stats["total"]["store"]["size_in_bytes"],
                "primary_shards": index_stats["primaries"]["docs"]["count"],
                "total_shards": len(index_stats["shards"]),
                "refresh_time": index_stats["total"]["refresh"]["total_time_in_millis"],
                "search_time": index_stats["total"]["search"]["query_time_in_millis"],
                "search_count": index_stats["total"]["search"]["query_total"],
                "index_settings": {
                    "number_of_shards": index_settings["index"]["number_of_shards"],
                    "number_of_replicas": index_settings["index"]["number_of_replicas"]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {"error": str(e)}
            
    def validate_connection(self) -> bool:
        """
        Validate OpenSearch connection and index availability.
        
        Returns:
            True if connection is healthy
        """
        try:
            # Test cluster health
            health = self.client.cluster.health()
            if health["status"] not in ["green", "yellow"]:
                logger.warning(f"Cluster status: {health['status']}")
                
            # Test index existence
            if not self.client.indices.exists(index=self.index_name):
                logger.error(f"Index {self.index_name} does not exist")
                return False
                
            logger.info("OpenSearch connection validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"OpenSearch connection validation failed: {e}")
            return False
            
    def bulk_score_documents(self, 
                           documents: List[Dict[str, Any]], 
                           query_text: str) -> List[float]:
        """
        Score multiple documents against a query using BM25.
        
        Args:
            documents: List of documents to score
            query_text: Query string for scoring
            
        Returns:
            List of BM25 scores for each document
        """
        try:
            # Use explain API to get BM25 scores
            scores = []
            
            for doc in documents:
                doc_id = doc.get("tweet_id")
                if not doc_id:
                    scores.append(0.0)
                    continue
                    
                explain_query = {
                    "query": {
                        "match": {
                            "clean_content": query_text
                        }
                    }
                }
                
                try:
                    response = self.client.explain(
                        index=self.index_name,
                        id=doc_id,
                        body=explain_query
                    )
                    
                    if response.get("matched", False):
                        scores.append(response["explanation"]["value"])
                    else:
                        scores.append(0.0)
                        
                except Exception as e:
                    logger.warning(f"Failed to score document {doc_id}: {e}")
                    scores.append(0.0)
                    
            return scores
            
        except Exception as e:
            logger.error(f"Bulk scoring failed: {e}")
            return [0.0] * len(documents)


def main():
    """Command-line interface for BM25 search testing."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="BM25 Search Engine CLI")
    parser.add_argument("--host", default="localhost", help="OpenSearch host")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port")
    parser.add_argument("--index", default="tweets-hybrid", help="Index name")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--size", type=int, default=10, help="Number of results")
    parser.add_argument("--explain", action="store_true", help="Include score explanations")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize search engine
    engine = BM25SearchEngine(
        host=args.host,
        port=args.port,
        index_name=args.index
    )
    
    # Validate connection
    if not engine.validate_connection():
        print("Failed to connect to OpenSearch")
        return
        
    # Create search spec
    from .schema import SearchSpec
    search_spec = SearchSpec(
        query=args.query,
        size=args.size
    )
    
    try:
        # Execute search
        response = engine.search(search_spec, explain=args.explain)
        
        # Print results
        print(f"Found {response.total_hits} results in {response.query_time_ms}ms")
        print(f"Score stats: {response.score_breakdown}")
        print()
        
        for i, result in enumerate(response.results, 1):
            print(f"{i}. [{result.bm25_score:.3f}] {result.clean_content[:100]}...")
            print(f"   @{result.source_handle} | {result.created_at_iso} | {result.market_impact}")
            if result.entities:
                entities_str = ", ".join([e["name"] for e in result.entities])
                print(f"   Entities: {entities_str}")
            print()
            
    except Exception as e:
        print(f"Search failed: {e}")


if __name__ == "__main__":
    main()