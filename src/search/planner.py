#!/usr/bin/env python3
"""
Hybrid Search Query Planner - Coordinates BM25 + kNN search execution and blending.
Implements the core search architecture: SearchSpec -> BM25+kNN -> Score Blending.
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .schema import SearchSpec, SearchResult, SearchResponse
from .bm25_search import BM25SearchEngine
from .vector_search import VectorSearchEngine


@dataclass
class SearchPlan:
    """Execution plan for hybrid search."""
    bm25_query: Dict[str, Any]
    vector_query: Dict[str, Any] 
    score_weights: Dict[str, float]
    normalization_method: str = "z_score"


@dataclass
class BlendedScore:
    """Score breakdown for a search result."""
    bm25_score: float
    vector_score: float
    recency_score: float
    authority_score: float
    engagement_score: float
    final_score: float


class HybridSearchPlanner:
    """
    Coordinates BM25 + kNN search execution and score blending.
    Implements the core hybrid search architecture from CLAUDE.md.
    """
    
    def __init__(
        self,
        bm25_engine: BM25SearchEngine,
        vector_engine: VectorSearchEngine,
        default_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize hybrid search planner.
        
        Args:
            bm25_engine: BM25 keyword search engine
            vector_engine: kNN vector search engine  
            default_weights: Default score blending weights
        """
        self.bm25_engine = bm25_engine
        self.vector_engine = vector_engine
        
        # Default weights from CLAUDE.md: 45% BM25 + 35% kNN + 20% other factors
        self.default_weights = default_weights or {
            "bm25": 0.45,
            "vector": 0.35,
            "recency": 0.10,
            "authority": 0.05,
            "engagement": 0.05
        }
        
        # Recency decay parameter (λ ≈ 0.1-0.2 for news intents)
        self.recency_lambda = 0.12
    
    def create_search_plan(self, search_spec: SearchSpec) -> SearchPlan:
        """Create execution plan for hybrid search."""
        
        # Use custom boosts if provided, otherwise defaults
        weights = {}
        if search_spec.boosts:
            # Normalize boosts to ensure they sum to 1.0
            boost_sum = sum(search_spec.boosts.values())
            if boost_sum > 0:
                weights = {k: v / boost_sum for k, v in search_spec.boosts.items()}
            
            # Map SearchSpec boost names to internal weights
            score_weights = {
                "bm25": self.default_weights["bm25"],
                "vector": self.default_weights["vector"],  
                "recency": weights.get("recency", self.default_weights["recency"]),
                "authority": weights.get("authority", self.default_weights["authority"]),
                "engagement": weights.get("engagement", self.default_weights["engagement"])
            }
        else:
            score_weights = self.default_weights.copy()
        
        return SearchPlan(
            bm25_query={},  # Will be built by individual engines
            vector_query={},
            score_weights=score_weights,
            normalization_method="z_score"
        )
    
    async def execute_hybrid_search(self, search_spec: SearchSpec) -> SearchResponse:
        """
        Execute hybrid search with parallel BM25 + kNN queries.
        
        Args:
            search_spec: Search specification with query, filters, boosts
            
        Returns:
            Blended search results with score breakdown
        """
        start_time = time.time()
        plan = self.create_search_plan(search_spec)
        
        # Execute BM25 and kNN searches in parallel
        try:
            bm25_task = asyncio.create_task(self._execute_bm25_search(search_spec))
            vector_task = asyncio.create_task(self._execute_vector_search(search_spec))
            
            bm25_response, vector_response = await asyncio.gather(
                bm25_task, vector_task, return_exceptions=True
            )
            
            # Handle execution errors
            if isinstance(bm25_response, Exception):
                print(f"BM25 search error: {bm25_response}")
                bm25_response = SearchResponse(results=[], total_hits=0, query_time_ms=0)
            
            if isinstance(vector_response, Exception):
                print(f"Vector search error: {vector_response}")
                vector_response = SearchResponse(results=[], total_hits=0, query_time_ms=0)
                
        except Exception as e:
            print(f"Parallel search execution error: {e}")
            # Fallback to sequential execution
            bm25_response = await self._execute_bm25_search(search_spec)
            vector_response = await self._execute_vector_search(search_spec)
        
        # Blend results
        blended_results = self._blend_search_results(
            bm25_response.results,
            vector_response.results, 
            plan.score_weights,
            search_spec.size
        )
        
        # Calculate execution time
        total_time = int((time.time() - start_time) * 1000)
        
        # Build response with metadata
        response = SearchResponse(
            results=blended_results,
            total_hits=max(bm25_response.total_hits, vector_response.total_hits),
            query_time_ms=total_time,
            metadata={
                "hybrid_plan": {
                    "weights": plan.score_weights,
                    "bm25_hits": bm25_response.total_hits,
                    "vector_hits": vector_response.total_hits,
                    "bm25_time_ms": bm25_response.query_time_ms,
                    "vector_time_ms": vector_response.query_time_ms,
                }
            }
        )
        
        return response
    
    async def _execute_bm25_search(self, search_spec: SearchSpec) -> SearchResponse:
        """Execute BM25 search asynchronously."""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.bm25_engine.search, search_spec
            )
        except Exception as e:
            print(f"BM25 search execution error: {e}")
            return SearchResponse(results=[], total_hits=0, query_time_ms=0)
    
    async def _execute_vector_search(self, search_spec: SearchSpec) -> SearchResponse:
        """Execute vector search asynchronously.""" 
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.vector_engine.search, search_spec
            )
        except Exception as e:
            print(f"Vector search execution error: {e}")
            return SearchResponse(results=[], total_hits=0, query_time_ms=0)
    
    def _blend_search_results(
        self,
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
        weights: Dict[str, float],
        max_results: int
    ) -> List[SearchResult]:
        """
        Blend BM25 and vector search results with score normalization.
        
        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            weights: Score blending weights
            max_results: Maximum number of results to return
            
        Returns:
            Blended and ranked results
        """
        
        # Collect all unique documents
        doc_map = {}
        
        # Add BM25 results
        for result in bm25_results:
            doc_map[result.tweet_id] = {
                "result": result,
                "bm25_score": result.score,
                "vector_score": 0.0
            }
        
        # Add vector results
        for result in vector_results:
            if result.tweet_id in doc_map:
                doc_map[result.tweet_id]["vector_score"] = result.score
            else:
                doc_map[result.tweet_id] = {
                    "result": result,
                    "bm25_score": 0.0,
                    "vector_score": result.score
                }
        
        # Normalize scores
        bm25_scores = [doc["bm25_score"] for doc in doc_map.values()]
        vector_scores = [doc["vector_score"] for doc in doc_map.values()]
        
        bm25_normalized = self._normalize_scores(bm25_scores)
        vector_normalized = self._normalize_scores(vector_scores)
        
        # Calculate blended scores
        blended_results = []
        
        for i, (tweet_id, doc_info) in enumerate(doc_map.items()):
            result = doc_info["result"]
            
            # Get normalized scores
            norm_bm25 = bm25_normalized[i] if i < len(bm25_normalized) else 0.0
            norm_vector = vector_normalized[i] if i < len(vector_normalized) else 0.0
            
            # Calculate additional scoring factors
            recency_score = self._calculate_recency_score(result.created_at)
            authority_score = self._calculate_authority_score(result)
            engagement_score = self._calculate_engagement_score(result)
            
            # Blend final score
            final_score = (
                weights["bm25"] * norm_bm25 +
                weights["vector"] * norm_vector +
                weights["recency"] * recency_score +
                weights["authority"] * authority_score +
                weights["engagement"] * engagement_score
            )
            
            # Update result with blended score and metadata
            blended_result = SearchResult(
                tweet_id=result.tweet_id,
                content=result.content,
                clean_content=result.clean_content,
                score=final_score,
                created_at=result.created_at,
                source_handle=result.source_handle,
                engagement_score=result.engagement_score,
                entities=result.entities,
                market_impact=result.market_impact,
                metadata={
                    **result.metadata,
                    "score_breakdown": BlendedScore(
                        bm25_score=norm_bm25,
                        vector_score=norm_vector,
                        recency_score=recency_score,
                        authority_score=authority_score,
                        engagement_score=engagement_score,
                        final_score=final_score
                    ).__dict__
                }
            )
            
            blended_results.append(blended_result)
        
        # Sort by final score and return top results
        blended_results.sort(key=lambda x: x.score, reverse=True)
        return blended_results[:max_results]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores using z-score method."""
        if not scores or len(scores) <= 1:
            return scores
            
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        
        if std_score == 0:
            return [0.0] * len(scores)
        
        # Z-score normalization with sigmoid to [0, 1]
        z_scores = (scores_array - mean_score) / std_score
        normalized = 1.0 / (1.0 + np.exp(-z_scores))  # Sigmoid activation
        
        return normalized.tolist()
    
    def _calculate_recency_score(self, created_at: str) -> float:
        """Calculate recency score with exponential decay."""
        try:
            # Parse timestamp
            if isinstance(created_at, str):
                tweet_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                tweet_time = created_at
            
            # Calculate age in days
            age_days = (datetime.now(tweet_time.tzinfo) - tweet_time).total_seconds() / 86400
            
            # Exponential decay: exp(-λ * age_days)
            recency_score = np.exp(-self.recency_lambda * max(0, age_days))
            
            return float(recency_score)
            
        except Exception as e:
            print(f"Error calculating recency score: {e}")
            return 0.0
    
    def _calculate_authority_score(self, result: SearchResult) -> float:
        """Calculate authority score based on source metrics."""
        try:
            # Check if source has high follower count (authority signal)
            source_followers = getattr(result, 'source_followers', 0) or 0
            
            # KOL boost: if has_kols=True, +0.5 boost as per CLAUDE.md heuristics
            has_kols = any(
                entity.get('entity_type') == 'kol' 
                for entity in (result.entities or [])
            )
            
            kol_boost = 0.5 if has_kols else 0.0
            
            # Follower-based authority (logarithmic scaling)
            if source_followers > 0:
                follower_score = min(1.0, np.log10(source_followers) / 6.0)  # Scale log10(1M) = 1.0
            else:
                follower_score = 0.0
            
            return kol_boost + follower_score
            
        except Exception as e:
            print(f"Error calculating authority score: {e}")
            return 0.0
    
    def _calculate_engagement_score(self, result: SearchResult) -> float:
        """Calculate engagement score based on likes/retweets/replies."""
        try:
            engagement = result.engagement_score or 0
            
            if engagement <= 0:
                return 0.0
            
            # Logarithmic scaling for engagement
            engagement_score = min(1.0, np.log10(engagement + 1) / 4.0)  # Scale log10(10K) ≈ 1.0
            
            return float(engagement_score)
            
        except Exception as e:
            print(f"Error calculating engagement score: {e}")
            return 0.0


class HybridSearchManager:
    """High-level manager for hybrid search operations."""
    
    def __init__(self, opensearch_host: str = "localhost:9200", index_name: str = "tweets-hybrid"):
        """Initialize hybrid search manager."""
        self.bm25_engine = BM25SearchEngine(
            opensearch_host=opensearch_host,
            index_name=index_name
        )
        self.vector_engine = VectorSearchEngine(
            opensearch_host=opensearch_host,
            index_name=index_name
        )
        self.planner = HybridSearchPlanner(self.bm25_engine, self.vector_engine)
    
    async def search(self, search_spec: SearchSpec) -> SearchResponse:
        """Execute hybrid search."""
        return await self.planner.execute_hybrid_search(search_spec)
    
    def close(self):
        """Close search engines and connections."""
        self.bm25_engine.close()
        self.vector_engine.close()


async def main():
    """Test hybrid search functionality."""
    from .schema import Entity
    
    # Create test search spec
    search_spec = SearchSpec(
        query="bitcoin price prediction",
        entities={
            "must": [Entity(entity_id="btc", entity_type="token", name="Bitcoin")],
            "should": [Entity(entity_id="elon", entity_type="kol", name="Elon Musk")]
        },
        filters={
            "sentiment": ["bullish", "neutral"],
            "time_range": {"from": "now-7d", "to": "now"}
        },
        boosts={
            "recency": 0.4,
            "authority": 0.3, 
            "engagement": 0.3
        },
        size=10
    )
    
    # Execute hybrid search
    manager = HybridSearchManager()
    try:
        response = await manager.search(search_spec)
        
        print(f"Hybrid Search Results:")
        print(f"Total hits: {response.total_hits}")
        print(f"Query time: {response.query_time_ms}ms")
        print(f"Results returned: {len(response.results)}")
        
        for i, result in enumerate(response.results[:3]):
            print(f"\n{i+1}. {result.content[:100]}...")
            print(f"   Score: {result.score:.4f}")
            print(f"   Source: @{result.source_handle}")
            print(f"   Entities: {[e.get('name') for e in result.entities or []]}")
            
            # Show score breakdown
            if "score_breakdown" in result.metadata:
                breakdown = result.metadata["score_breakdown"]
                print(f"   BM25: {breakdown['bm25_score']:.3f} | "
                      f"Vector: {breakdown['vector_score']:.3f} | "
                      f"Recency: {breakdown['recency_score']:.3f}")
    
    finally:
        manager.close()


if __name__ == "__main__":
    asyncio.run(main())