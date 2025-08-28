#!/usr/bin/env python3
"""
FastAPI main application for hybrid search engine.
Provides REST API endpoints for search functionality.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import search components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from search.schema import SearchSpec, Entity, SearchResult, SearchResponse
from search.blend import ResultBlender
from search.local_search import LocalHybridSearchEngine
from search.cluster import ClusteredSearchResponse
from llm.spec_gen import LLMSearchSpecGenerator

app = FastAPI(
    title="Hybrid Search Engine",
    description="Financial/crypto content search with BM25 + vector similarity",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    """API request model for search queries."""
    query: str
    entities: Optional[Dict[str, List[Dict]]] = None
    filters: Optional[Dict[str, Any]] = None
    size: int = 20


class SearchAPIResponse(BaseModel):
    """API response model for search results."""
    query: str
    total_hits: int
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    query_time_ms: int


class NaturalQueryRequest(BaseModel):
    """Natural language query request."""
    query: str = Field(min_length=1, max_length=500, description="Natural language question")
    size: int = Field(ge=1, le=50, default=20, description="Number of results")


# Mock search engines for demo (replace with real implementations)
class MockHybridSearchEngine:
    """Mock hybrid search engine for demonstration."""
    
    def __init__(self):
        # Load sample data
        self.sample_data = self._load_sample_data()
        self.blender = ResultBlender()
    
    def _load_sample_data(self) -> List[Dict[str, Any]]:
        """Load sample tweet data from expanded dataset."""
        import json
        from pathlib import Path
        
        # Get the project root directory (search_engine/)
        current_dir = Path(__file__).parent.parent.parent
        sample_file = current_dir / "data" / "expanded_sample_tweets.jsonl"
        
        sample_data = []
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        tweet_data = json.loads(line.strip())
                        # Add final_score for compatibility (will be recalculated during search)
                        tweet_data["final_score"] = 0.5  # Default placeholder
                        sample_data.append(tweet_data)
            print(f"Loaded {len(sample_data)} sample tweets from {sample_file}")
        except FileNotFoundError:
            print(f"Sample file not found: {sample_file}")
            # Fallback to original hardcoded data
            sample_data = [
                {
                    "tweet_id": "1",
                    "content": "Bitcoin is pumping hard! New ATH incoming 🚀 #bitcoin",
                    "clean_content": "bitcoin is pumping hard new ath incoming bitcoin",
                    "original_content": "Bitcoin is pumping hard! New ATH incoming 🚀 #bitcoin",
                    "created_at": "2024-01-15T10:00:00Z",
                    "source_handle": "crypto_trader",
                    "source_followers": 50000,
                    "engagement_score": 180,
                    "entities": [{"entity_type": "token", "entity_id": "btc", "name": "Bitcoin"}],
                    "market_impact": "bullish",
                    "authority_score": 0.8,
                    "language": "en",
                    "final_score": 0.95
                }
            ]
        except Exception as e:
            print(f"Error loading sample data: {e}")
            # Fallback to minimal data
            sample_data = []
            
        return sample_data
    
    async def search(self, search_spec: SearchSpec) -> SearchResponse:
        """Execute mock hybrid search."""
        start_time = time.time()
        
        # Simulate search scoring
        query_lower = search_spec.query.lower()
        scored_results = []
        
        for doc in self.sample_data:
            # Mock BM25 scoring
            bm25_score = 0.0
            content_lower = doc["clean_content"]
            for term in query_lower.split():
                if term in content_lower:
                    bm25_score += 0.3
            
            # Mock vector similarity scoring  
            vector_score = 0.7 if any(term in content_lower for term in query_lower.split()) else 0.2
            
            # Mock entity matching boost
            entity_boost = 0.0
            # Only process if there are actual entities to match
            has_actual_entities = any(len(entity_list) > 0 for entity_list in search_spec.entities.values())
            if has_actual_entities:
                print(f"DEBUG: Processing entities for doc {doc['tweet_id']}")
                for entity_list in search_spec.entities.values():
                    for entity in entity_list:
                        if any(e["entity_id"] == entity.entity_id for e in doc["entities"]):
                            entity_boost = 0.4
                            break
                    if entity_boost > 0:
                        break
            
            # Calculate final score (45% BM25 + 35% vector + 20% other)
            final_score = (
                0.45 * min(bm25_score, 1.0) +
                0.35 * vector_score +
                0.10 * (1.0 - min(time.time() - 1642248000, 86400*7) / (86400*7)) +  # Recency
                0.05 * doc["authority_score"] +
                0.05 * min(doc["engagement_score"] / 1000, 1.0)
            )
            
            if final_score > 0.1:  # Minimum relevance threshold
                # Convert dict entities to Entity objects
                entity_objects = [
                    Entity(
                        entity_type=e["entity_type"],
                        entity_id=e["entity_id"],
                        name=e["name"],
                        confidence=1.0
                    ) for e in doc["entities"]
                ]
                
                result = SearchResult(
                    tweet_id=doc["tweet_id"],
                    content=doc["content"],
                    original_content=doc["original_content"],
                    created_at=doc["created_at"],
                    source_handle=doc["source_handle"],
                    source_followers=doc["source_followers"],
                    engagement_score=doc["engagement_score"],
                    entities=entity_objects,
                    market_impact=doc["market_impact"],
                    authority_score=doc["authority_score"],
                    language=doc["language"],
                    final_score=final_score,
                    bm25_score=bm25_score,
                    vector_score=vector_score
                )
                scored_results.append(result)
        
        # Sort by score and apply size limit
        scored_results.sort(key=lambda x: x.final_score, reverse=True)
        final_results = scored_results[:search_spec.size]
        
        # Apply blending and diversification (temporarily disabled for demo)
        # blended_results = self.blender.blend_and_diversify(final_results, search_spec.size)
        blended_results = final_results  # Use direct results for now
        
        query_time = int((time.time() - start_time) * 1000)
        
        return SearchResponse(
            query_spec=search_spec,
            results=blended_results,
            total_hits=len(scored_results),
            execution_time_ms=query_time
        )


# Global search engine instance - using LocalHybridSearchEngine for real search
search_engine = LocalHybridSearchEngine()
spec_generator = LLMSearchSpecGenerator()

# Load data on startup - try embeddings file first, fallback to original data
embeddings_file = Path(__file__).parent.parent.parent / "normalized" / "tweets_with_embeddings.jsonl"
original_file = Path(__file__).parent.parent.parent / "data" / "expanded_sample_tweets.jsonl"

if embeddings_file.exists() and search_engine.load_documents_from_file(str(embeddings_file)):
    print(f"✅ Loaded data with real embeddings from {embeddings_file.name}")
elif search_engine.load_documents_from_file(str(original_file)):
    print(f"✅ Loaded original sample data from {original_file.name}")
else:
    print(f"⚠️  Failed to load any sample data - check data files")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Hybrid Search Engine",
        "version": "1.0.0",
        "description": "Financial/crypto content search with BM25 + vector similarity",
        "endpoints": {
            "search": "/search",
            "clustered_search": "/search/clustered",
            "natural_language": "/ask",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "search_engine": "operational",
            "embedding_service": "operational",
            "result_blender": "operational"
        }
    }


@app.post("/search", response_model=SearchAPIResponse)
async def search_tweets(request: SearchRequest):
    """
    Execute hybrid search query.
    
    **Query Flow:**
    1. Parse request → SearchSpec
    2. Execute BM25 + kNN search
    3. Blend scores (45% BM25 + 35% vector + 20% boosts)
    4. Deduplicate and diversify results
    5. Return ranked results with metadata
    """
    try:
        # Convert API request to SearchSpec
        entities = {}
        if request.entities:
            entities = {
                key: [Entity(**entity_dict) for entity_dict in entity_list]
                for key, entity_list in request.entities.items()
            }
        
        search_spec = SearchSpec(
            query=request.query,
            entities=entities,
            filters=request.filters or {},
            size=min(request.size, 50)  # Cap at 50 for demo
        )
        
        # Execute search
        search_response = await search_engine.search(search_spec)
        
        # Convert to API response format
        results = []
        for result in search_response.results:
            results.append({
                "tweet_id": result.tweet_id,
                "content": result.content,
                "score": round(result.final_score, 4),
                "created_at": result.created_at,
                "source_handle": result.source_handle,
                "engagement_score": result.engagement_score,
                "entities": result.entities,
                "market_impact": result.market_impact,
                "bm25_score": result.bm25_score,
                "vector_score": result.vector_score
            })
        
        return SearchAPIResponse(
            query=request.query,
            total_hits=search_response.total_hits,
            results=results,
            metadata={
                "execution_time_ms": search_response.execution_time_ms,
                "score_stats": search_response.score_stats,
                "query_spec": search_response.query_spec.model_dump()
            },
            query_time_ms=search_response.execution_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search")
async def search_tweets_get(
    q: str = Query(..., description="Search query"),
    size: int = Query(20, ge=1, le=50, description="Number of results"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    entity_id: Optional[str] = Query(None, description="Filter by entity ID"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment")
):
    """
    GET endpoint for simple search queries.
    """
    # Build search request
    entities = {}
    if entity_type and entity_id:
        entities["must"] = [{"entity_type": entity_type, "entity_id": entity_id, "name": ""}]
    
    filters = {}
    if sentiment:
        filters["sentiment"] = [sentiment]
    
    request = SearchRequest(
        query=q,
        entities=entities if entities else None,
        filters=filters if filters else None,
        size=size
    )
    
    return await search_tweets(request)


@app.post("/ask", response_model=SearchAPIResponse)
async def ask_natural_language(request: NaturalQueryRequest):
    """
    Natural language search endpoint.
    
    **Easy to Use:** Just ask a question in plain English:
    - "Why is Bitcoin pumping?"  
    - "Latest Ethereum news"
    - "What does Elon think about crypto?"
    - "Fed rate impact on crypto prices"
    
    The system automatically:
    1. Extracts entities (BTC, ETH, etc.)
    2. Determines sentiment filters
    3. Applies appropriate boosts
    4. Executes hybrid search
    """
    try:
        start_time = time.time()
        
        # Generate SearchSpec from natural language
        search_spec = await spec_generator.generate_search_spec(request.query)
        search_spec.size = request.size  # Override size
        
        # Execute search using existing engine
        search_response = await search_engine.search(search_spec)
        
        # Convert to API response format  
        results = []
        for result in search_response.results:
            results.append({
                "tweet_id": result.tweet_id,
                "content": result.content,
                "score": round(result.final_score, 4),
                "created_at": result.created_at,
                "source_handle": result.source_handle,
                "engagement_score": result.engagement_score,
                "entities": result.entities,
                "market_impact": result.market_impact,
                "bm25_score": result.bm25_score,
                "vector_score": result.vector_score
            })
        
        total_time = int((time.time() - start_time) * 1000)
        
        return SearchAPIResponse(
            query=request.query,
            total_hits=search_response.total_hits,
            results=results,
            metadata={
                "execution_time_ms": search_response.execution_time_ms,
                "query_spec": search_response.query_spec.model_dump()
            },
            query_time_ms=total_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Natural language search failed: {str(e)}")


@app.post("/search/clustered")
async def search_tweets_clustered(request: SearchRequest):
    """
    Execute hybrid search with HDBSCAN clustering for thematic organization.
    
    **Returns grouped results:**
    - Groups similar tweets into thematic clusters
    - Each cluster has a theme name and keywords
    - Provides outliers that don't fit any cluster
    - Includes clustering confidence scores
    
    **Perfect for:**
    - Market analysis across multiple topics
    - Identifying trending themes in crypto news
    - Understanding different perspectives on events
    """
    try:
        # Convert API request to SearchSpec
        entities = {}
        if request.entities:
            entities = {
                key: [Entity(**entity_dict) for entity_dict in entity_list]
                for key, entity_list in request.entities.items()
            }
        
        search_spec = SearchSpec(
            query=request.query,
            entities=entities,
            filters=request.filters or {},
            size=min(request.size, 50)  # Cap at 50 for demo
        )
        
        # Execute clustered search
        clustered_response = await search_engine.search_with_clustering(search_spec)
        
        # Convert to API response format
        clusters = []
        for cluster in clustered_response.clusters:
            cluster_results = []
            for result in cluster.results:
                cluster_results.append({
                    "tweet_id": result.tweet_id,
                    "content": result.content,
                    "score": round(result.final_score, 4),
                    "created_at": result.created_at,
                    "source_handle": result.source_handle,
                    "engagement_score": result.engagement_score,
                    "entities": result.entities,
                    "market_impact": result.market_impact,
                    "bm25_score": result.bm25_score,
                    "vector_score": result.vector_score
                })
            
            clusters.append({
                "cluster_id": cluster.cluster_id,
                "theme": cluster.theme,
                "size": cluster.size,
                "keywords": cluster.keywords,
                "confidence": round(cluster.confidence, 3),
                "results": cluster_results
            })
        
        # Convert outliers
        outliers = []
        for result in clustered_response.outliers:
            outliers.append({
                "tweet_id": result.tweet_id,
                "content": result.content,
                "score": round(result.final_score, 4),
                "created_at": result.created_at,
                "source_handle": result.source_handle,
                "engagement_score": result.engagement_score,
                "entities": result.entities,
                "market_impact": result.market_impact,
                "bm25_score": result.bm25_score,
                "vector_score": result.vector_score
            })
        
        return {
            "query": request.query,
            "total_hits": len(clustered_response.results),
            "clusters": clusters,
            "outliers": outliers,
            "clustering_stats": {
                "total_clusters": clustered_response.total_clusters,
                "clustering_time_ms": clustered_response.clustering_time_ms,
                "outlier_count": len(clustered_response.outliers)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustered search failed: {str(e)}")


@app.get("/examples")
async def get_search_examples():
    """Get example search queries for testing."""
    return {
        "structured_search_examples": [
            {
                "query": "bitcoin pump bullish",
                "description": "Find bullish Bitcoin content",
                "url": "/search?q=bitcoin pump bullish&sentiment=bullish"
            },
            {
                "query": "ethereum hack selloff",
                "description": "Find bearish Ethereum news",
                "url": "/search?q=ethereum hack selloff&sentiment=bearish"
            },
            {
                "query": "solana network issues",
                "description": "Technical problems discussion",
                "url": "/search?q=solana network issues&entity_type=token&entity_id=sol"
            },
            {
                "query": "fed rate crypto impact",
                "description": "Macro economic factors",
                "url": "/search?q=fed rate crypto impact&entity_type=macro&entity_id=fed_rate"
            },
            {
                "query": "vitalik ethereum upgrade",
                "description": "KOL opinions on tech developments",
                "url": "/search?q=vitalik ethereum upgrade&entity_type=kol&entity_id=vitalik"
            }
        ],
        "natural_language_examples": [
            {
                "query": "Why is Bitcoin pumping?",
                "description": "Automatically detects BTC entity and bullish sentiment"
            },
            {
                "query": "What's the latest Ethereum news?",
                "description": "Finds ETH content with recency boost"
            },
            {
                "query": "Elon Musk thoughts on crypto?",
                "description": "KOL-focused search with authority boost"
            },
            {
                "query": "Fed rate decision impact on crypto",
                "description": "Macro analysis with authority and recency"
            },
            {
                "query": "Recent bearish Solana sentiment",
                "description": "Combines SOL entity, bearish filter, recency"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)