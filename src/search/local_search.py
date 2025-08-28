#!/usr/bin/env python3
"""
Local Hybrid Search Engine - In-memory implementation for testing.
Simulates OpenSearch behavior without requiring external dependencies.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import math
import asyncio
from collections import defaultdict

from .schema import SearchSpec, SearchResult, SearchResponse, Entity, EntityType, MarketImpact, Language
from .cluster import RealTimeResultClusterer, ClusteredSearchResponse


class LocalBM25Engine:
    """Local BM25 implementation for testing."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.term_frequencies = {}
        self.document_lengths = {}
        self.avg_doc_length = 0
        
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for BM25 search."""
        self.documents = documents
        total_length = 0
        
        # Calculate term frequencies and document lengths
        for i, doc in enumerate(documents):
            content = doc.get("clean_content", "").lower().split()
            self.document_lengths[i] = len(content)
            total_length += len(content)
            
            # Count term frequencies
            term_freq = defaultdict(int)
            for term in content:
                term_freq[term] += 1
            
            # Update global term frequencies
            for term, freq in term_freq.items():
                if term not in self.term_frequencies:
                    self.term_frequencies[term] = {}
                self.term_frequencies[term][i] = freq
        
        self.avg_doc_length = total_length / len(documents) if documents else 0
        
    def search(self, query: str, size: int = 20) -> List[tuple]:
        """Search using BM25 algorithm."""
        query_terms = query.lower().split()
        scores = {}
        
        for doc_id in range(len(self.documents)):
            score = 0
            doc_length = self.document_lengths.get(doc_id, 0)
            
            for term in query_terms:
                if term in self.term_frequencies:
                    # Term frequency in document
                    tf = self.term_frequencies[term].get(doc_id, 0)
                    
                    # Document frequency (number of documents containing term)
                    df = len(self.term_frequencies[term])
                    
                    # IDF calculation
                    idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5))
                    
                    # BM25 score calculation
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                    
                    score += idf * (numerator / denominator)
            
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score and return top results
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:size]


class LocalVectorEngine:
    """Local vector similarity engine for testing."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents with embeddings."""
        self.documents = documents
        self.embeddings = [doc.get("embedding", []) for doc in documents]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)
    
    def search(self, query_vector: List[float], size: int = 20) -> List[tuple]:
        """Search using vector similarity."""
        if not query_vector:
            return []
            
        scores = []
        for i, embedding in enumerate(self.embeddings):
            if embedding:
                similarity = self.cosine_similarity(query_vector, embedding)
                scores.append((i, similarity))
        
        # Sort by similarity and return top results
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:size]


class LocalHybridSearchEngine:
    """Local hybrid search engine combining BM25 and vector search."""
    
    def __init__(self):
        self.bm25_engine = LocalBM25Engine()
        self.vector_engine = LocalVectorEngine()
        self.documents = []
        self.clusterer = RealTimeResultClusterer(
            min_cluster_size=2,  # Smaller clusters for sample data
            min_samples=1,       # More permissive for small datasets
            cluster_selection_epsilon=0.3  # Better clustering for similar content
        )
        # Pre-load embedding model to avoid loading it on every search
        self.embedding_model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model once during initialization."""
        try:
            import sentence_transformers
            print("🔄 Loading sentence transformer model (one-time setup)...")
            self.embedding_model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("✅ Embedding model loaded successfully")
        except ImportError:
            print("⚠️  sentence-transformers not available, using hash-based embeddings")
            self.embedding_model = None
        except Exception as e:
            print(f"⚠️  Error loading embedding model: {e}")
            self.embedding_model = None
        
    def load_documents_from_file(self, file_path: str) -> bool:
        """Load documents from JSONL file."""
        try:
            documents = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line.strip())
                        documents.append(doc)
            
            self.documents = documents
            
            # Index documents in both engines
            self.bm25_engine.index_documents(documents)
            self.vector_engine.index_documents(documents)
            
            # Check if documents have real embeddings
            has_embeddings = any(doc.get("embedding") and len(doc["embedding"]) > 0 for doc in documents[:5])
            embedding_dim = len(documents[0].get("embedding", [])) if documents and documents[0].get("embedding") else 0
            
            print(f"Loaded and indexed {len(documents)} documents")
            if has_embeddings:
                print(f"✅ Using real embeddings (dimension: {embedding_dim})")
            else:
                print("⚠️  Using simple hash-based embeddings (fallback)")
            
            return True
            
        except Exception as e:
            print(f"Error loading documents: {e}")
            return False
    
    def generate_simple_embedding(self, text: str) -> List[float]:
        """Generate a simple hash-based embedding for testing."""
        # This is a very simple embedding generation for testing
        # In real implementation, we'd use BGE-large or similar model
        words = text.lower().split()
        embedding = [0.0] * 128  # Smaller dimension for speed
        
        for i, word in enumerate(words):
            hash_val = hash(word)
            for j in range(len(embedding)):
                embedding[j] += (hash_val >> j) & 1
        
        # Normalize
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
            
        return embedding
    
    async def search(self, search_spec: SearchSpec) -> SearchResponse:
        """Execute hybrid search."""
        start_time = time.time()
        
        # Check if documents have real embeddings
        has_real_embeddings = (self.documents and 
                              self.documents[0].get("embedding") and 
                              len(self.documents[0]["embedding"]) > 10)  # Real embeddings are much larger
        
        if has_real_embeddings and self.embedding_model:
            # Use pre-loaded embedding model to generate query embedding
            query_embedding = self.embedding_model.encode(search_spec.query).tolist()
        else:
            # Use simple hash-based embedding for testing or fallback
            query_embedding = self.generate_simple_embedding(search_spec.query)
        
        # Execute BM25 search
        bm25_results = self.bm25_engine.search(search_spec.query, search_spec.size * 2)
        
        # Execute vector search  
        vector_results = self.vector_engine.search(query_embedding, search_spec.size * 2)
        
        # Combine and blend results
        blended_results = self._blend_results(
            bm25_results, vector_results, search_spec
        )
        
        # Convert to SearchResult objects
        search_results = []
        for doc_id, final_score, scores in blended_results:
            if doc_id < len(self.documents):
                doc = self.documents[doc_id]
                
                # Convert entities to Entity objects
                entities = []
                for e in doc.get("entities", []):
                    entities.append(Entity(
                        entity_type=EntityType(e["entity_type"]),
                        entity_id=e["entity_id"],
                        name=e["name"],
                        confidence=1.0
                    ))
                
                result = SearchResult(
                    tweet_id=doc["tweet_id"],
                    content=doc["content"],
                    original_content=doc.get("original_content", doc["content"]),
                    created_at=doc["created_at"],
                    source_handle=doc["source_handle"],
                    source_followers=doc["source_followers"],
                    engagement_score=doc["engagement_score"],
                    entities=entities,
                    market_impact=MarketImpact(doc["market_impact"]),
                    authority_score=doc["authority_score"],
                    language=Language(doc["language"]),
                    final_score=final_score,
                    bm25_score=scores.get("bm25", 0.0),
                    vector_score=scores.get("vector", 0.0)
                )
                
                search_results.append(result)
        
        query_time = int((time.time() - start_time) * 1000)
        
        return SearchResponse(
            query_spec=search_spec,
            results=search_results,
            total_hits=len(search_results),
            execution_time_ms=query_time
        )
    
    async def search_with_clustering(self, search_spec: SearchSpec) -> ClusteredSearchResponse:
        """Execute hybrid search with clustering for thematic organization."""
        start_time = time.time()
        
        # First perform regular hybrid search
        search_response = await self.search(search_spec)
        
        # Then cluster the results
        clustered_response = self.clusterer.cluster_results(search_response.results)
        
        # Update timing to include clustering
        clustered_response.clustering_time_ms = int((time.time() - start_time) * 1000) - search_response.execution_time_ms
        
        return clustered_response
    
    def _blend_results(self, bm25_results: List[tuple], vector_results: List[tuple], 
                      search_spec: SearchSpec) -> List[tuple]:
        """Blend BM25 and vector search results."""
        # Combine results by document ID
        combined_scores = {}
        
        # Add BM25 scores
        max_bm25 = max([score for _, score in bm25_results], default=1.0)
        for doc_id, score in bm25_results:
            combined_scores[doc_id] = {
                "bm25": score / max_bm25,  # Normalize
                "vector": 0.0
            }
        
        # Add vector scores
        max_vector = max([score for _, score in vector_results], default=1.0)
        for doc_id, score in vector_results:
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {"bm25": 0.0, "vector": 0.0}
            combined_scores[doc_id]["vector"] = score / max_vector  # Normalize
        
        # Calculate final scores
        final_results = []
        for doc_id, scores in combined_scores.items():
            # Use default weights: 45% BM25 + 35% vector + 20% other
            final_score = (
                0.45 * scores["bm25"] +
                0.35 * scores["vector"] +
                0.20 * 0.5  # Placeholder for other factors
            )
            
            final_results.append((doc_id, final_score, scores))
        
        # Sort by final score and return top results
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:search_spec.size]


async def main():
    """Test the local hybrid search engine."""
    # Initialize engine
    engine = LocalHybridSearchEngine()
    
    # Load test data
    data_file = Path(__file__).parent.parent.parent / "data" / "expanded_sample_tweets.jsonl"
    
    if engine.load_documents_from_file(str(data_file)):
        # Test search
        search_spec = SearchSpec(
            query="bitcoin pumping",
            size=5
        )
        
        response = await engine.search(search_spec)
        
        print(f"\nSearch Results:")
        print(f"Query: {search_spec.query}")
        print(f"Total hits: {response.total_hits}")
        print(f"Execution time: {response.execution_time_ms}ms")
        
        for i, result in enumerate(response.results):
            print(f"\n{i+1}. {result.content[:100]}...")
            print(f"   Score: {result.final_score:.4f} (BM25: {result.bm25_score:.3f}, Vector: {result.vector_score:.3f})")
            print(f"   Source: @{result.source_handle}")


if __name__ == "__main__":
    asyncio.run(main())