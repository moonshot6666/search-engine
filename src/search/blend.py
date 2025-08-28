#!/usr/bin/env python3
"""
Score blending and result reranking for hybrid search.
Implements deduplication, diversity (MMR), and result post-processing.
"""

import hashlib
from typing import List, Dict, Any, Set, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

from .schema import SearchResult


@dataclass
class DiversityConfig:
    """Configuration for diversity algorithms."""
    jaccard_threshold: float = 0.95  # Near-duplicate threshold
    mmr_lambda: float = 0.3  # MMR diversity parameter (0=max diversity, 1=max relevance)
    max_per_source: int = 3  # Maximum results per source handle
    min_score_threshold: float = 0.1  # Minimum score to consider


class ResultBlender:
    """Handles result blending, deduplication, and diversification."""
    
    def __init__(self, config: Optional[DiversityConfig] = None):
        """Initialize result blender with configuration."""
        self.config = config or DiversityConfig()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def blend_and_diversify(
        self, 
        results: List[SearchResult],
        target_size: int = 50
    ) -> List[SearchResult]:
        """
        Apply complete blending pipeline: dedupe -> diversify -> rerank.
        
        Args:
            results: Input search results with blended scores
            target_size: Target number of results to return
            
        Returns:
            Processed and diversified results
        """
        if not results:
            return []
        
        # Step 1: Remove near-duplicates
        deduplicated = self.remove_near_duplicates(results)
        
        # Step 2: Apply source diversity (MMR)
        diversified = self.apply_source_diversity(deduplicated)
        
        # Step 3: Apply MMR diversification on content
        mmr_results = self.apply_mmr_diversification(diversified, target_size)
        
        # Step 4: Final ranking with score adjustments
        final_results = self.apply_final_ranking(mmr_results)
        
        return final_results[:target_size]
    
    def remove_near_duplicates(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove near-duplicate results using Jaccard similarity on shingles.
        
        Args:
            results: Input results to deduplicate
            
        Returns:
            Deduplicated results (keeps highest scoring of duplicates)
        """
        if len(results) <= 1:
            return results
        
        # Sort by score (descending) to keep best duplicates
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        kept_results = []
        content_hashes = set()
        
        for result in sorted_results:
            # Create content signature using shingles
            content_signature = self._create_content_signature(result.clean_content)
            
            # Check for near-duplicates
            is_duplicate = False
            for existing_hash in content_hashes:
                similarity = self._jaccard_similarity(content_signature, existing_hash)
                if similarity > self.config.jaccard_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept_results.append(result)
                content_hashes.add(frozenset(content_signature))
        
        return kept_results
    
    def apply_source_diversity(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Apply source diversity by limiting results per source handle.
        
        Args:
            results: Input results
            
        Returns:
            Source-diversified results
        """
        source_counts = {}
        diversified_results = []
        
        # Sort by score first
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        for result in sorted_results:
            source = result.source_handle or "unknown"
            current_count = source_counts.get(source, 0)
            
            if current_count < self.config.max_per_source:
                diversified_results.append(result)
                source_counts[source] = current_count + 1
        
        return diversified_results
    
    def apply_mmr_diversification(
        self, 
        results: List[SearchResult], 
        target_size: int
    ) -> List[SearchResult]:
        """
        Apply Maximal Marginal Relevance (MMR) for content diversity.
        
        Args:
            results: Input results
            target_size: Target number of results
            
        Returns:
            MMR-diversified results
        """
        if len(results) <= target_size:
            return results
        
        # Extract content for similarity calculation
        contents = [result.clean_content for result in results]
        
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(contents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
        except Exception as e:
            print(f"Error in TF-IDF calculation: {e}")
            # Fallback to score-based selection
            return sorted(results, key=lambda x: x.score, reverse=True)[:target_size]
        
        # MMR algorithm
        selected_indices = []
        remaining_indices = set(range(len(results)))
        
        # Start with highest scoring result
        best_idx = max(remaining_indices, key=lambda i: results[i].score)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # Iteratively select results balancing relevance and diversity
        while len(selected_indices) < target_size and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance score (normalized)
                relevance = results[idx].score
                
                # Diversity score (1 - max similarity to selected items)
                max_similarity = 0.0
                for selected_idx in selected_indices:
                    sim = similarity_matrix[idx][selected_idx]
                    max_similarity = max(max_similarity, sim)
                
                diversity = 1.0 - max_similarity
                
                # MMR score: λ * relevance + (1-λ) * diversity
                mmr_score = (
                    self.config.mmr_lambda * relevance + 
                    (1 - self.config.mmr_lambda) * diversity
                )
                mmr_scores.append((mmr_score, idx))
            
            # Select best MMR score
            if mmr_scores:
                _, best_idx = max(mmr_scores, key=lambda x: x[0])
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        return [results[i] for i in selected_indices]
    
    def apply_final_ranking(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Apply final ranking adjustments and quality filtering.
        
        Args:
            results: Input results to rank
            
        Returns:
            Final ranked results
        """
        # Filter by minimum score threshold
        filtered_results = [
            result for result in results 
            if result.score >= self.config.min_score_threshold
        ]
        
        # Apply final score adjustments based on result quality
        adjusted_results = []
        for result in filtered_results:
            adjusted_score = self._calculate_quality_adjusted_score(result)
            
            # Create new result with adjusted score
            adjusted_result = SearchResult(
                tweet_id=result.tweet_id,
                content=result.content,
                clean_content=result.clean_content,
                score=adjusted_score,
                created_at=result.created_at,
                source_handle=result.source_handle,
                engagement_score=result.engagement_score,
                entities=result.entities,
                market_impact=result.market_impact,
                metadata={
                    **result.metadata,
                    "original_score": result.score,
                    "quality_adjustment": adjusted_score - result.score
                }
            )
            adjusted_results.append(adjusted_result)
        
        # Sort by adjusted score
        return sorted(adjusted_results, key=lambda x: x.score, reverse=True)
    
    def _create_content_signature(self, content: str) -> Set[str]:
        """Create shingle-based signature for near-duplicate detection."""
        if not content:
            return set()
        
        # Create character 4-grams (shingles)
        content_clean = content.lower().replace(' ', '')
        shingles = set()
        
        for i in range(max(1, len(content_clean) - 3)):
            shingle = content_clean[i:i+4]
            shingles.add(shingle)
        
        return shingles
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_quality_adjusted_score(self, result: SearchResult) -> float:
        """Calculate quality-adjusted score based on content and metadata."""
        base_score = result.score
        
        # Content quality signals
        content_length = len(result.clean_content or "")
        
        # Length penalty for very short or very long content
        if content_length < 20:
            length_penalty = 0.8  # Short content penalty
        elif content_length > 280:
            length_penalty = 0.9  # Very long content slight penalty  
        else:
            length_penalty = 1.0
        
        # Entity richness bonus
        entity_count = len(result.entities or [])
        entity_bonus = 1.0 + min(0.1 * entity_count, 0.3)  # Up to 30% bonus
        
        # Market impact clarity bonus
        market_impact = result.market_impact or "neutral"
        clarity_bonus = 1.2 if market_impact in ["bullish", "bearish"] else 1.0
        
        # Calculate adjusted score
        adjusted_score = base_score * length_penalty * entity_bonus * clarity_bonus
        
        return float(adjusted_score)


def create_mini_embeddings(texts: List[str], max_features: int = 100) -> np.ndarray:
    """
    Create mini-embeddings using TF-IDF for fast similarity computation.
    
    Args:
        texts: List of text content
        max_features: Maximum TF-IDF features
        
    Returns:
        TF-IDF matrix for similarity computation
    """
    if not texts:
        return np.array([])
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray()
        
    except Exception as e:
        print(f"Error creating mini-embeddings: {e}")
        # Fallback to zero vectors
        return np.zeros((len(texts), max_features))


def calculate_result_diversity_metrics(results: List[SearchResult]) -> Dict[str, Any]:
    """
    Calculate diversity metrics for a result set.
    
    Args:
        results: List of search results
        
    Returns:
        Dictionary of diversity metrics
    """
    if not results:
        return {"diversity_score": 0.0, "source_diversity": 0.0, "entity_diversity": 0.0}
    
    # Source diversity
    sources = [result.source_handle for result in results if result.source_handle]
    unique_sources = len(set(sources))
    source_diversity = unique_sources / len(results) if results else 0.0
    
    # Entity diversity  
    all_entities = []
    for result in results:
        if result.entities:
            all_entities.extend([e.get('entity_id', '') for e in result.entities])
    
    unique_entities = len(set(all_entities))
    entity_diversity = unique_entities / max(len(all_entities), 1)
    
    # Content diversity (using mini-embeddings)
    contents = [result.clean_content or "" for result in results]
    mini_embeddings = create_mini_embeddings(contents)
    
    if mini_embeddings.size > 0:
        similarity_matrix = cosine_similarity(mini_embeddings)
        # Average pairwise similarity (lower = more diverse)
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        content_diversity = 1.0 - avg_similarity
    else:
        content_diversity = 0.0
    
    # Overall diversity score
    diversity_score = (source_diversity + entity_diversity + content_diversity) / 3
    
    return {
        "diversity_score": float(diversity_score),
        "source_diversity": float(source_diversity),
        "entity_diversity": float(entity_diversity), 
        "content_diversity": float(content_diversity),
        "unique_sources": unique_sources,
        "unique_entities": unique_entities,
        "total_results": len(results)
    }


# Example usage and testing
if __name__ == "__main__":
    # Create sample results for testing
    sample_results = [
        SearchResult(
            tweet_id="1",
            content="Bitcoin is pumping to new highs! $BTC",
            clean_content="bitcoin is pumping to new highs! $btc",
            score=0.95,
            created_at="2024-01-15T10:00:00Z",
            source_handle="crypto_trader",
            engagement_score=150,
            entities=[{"entity_type": "token", "entity_id": "btc", "name": "Bitcoin"}],
            market_impact="bullish"
        ),
        SearchResult(
            tweet_id="2", 
            content="BTC price surge continues as institutions buy more Bitcoin",
            clean_content="btc price surge continues as institutions buy more bitcoin",
            score=0.90,
            created_at="2024-01-15T10:05:00Z",
            source_handle="crypto_trader",  # Same source as above
            engagement_score=200,
            entities=[{"entity_type": "token", "entity_id": "btc", "name": "Bitcoin"}],
            market_impact="bullish"
        )
    ]
    
    # Test blending
    blender = ResultBlender()
    blended = blender.blend_and_diversify(sample_results, target_size=10)
    
    print(f"Original results: {len(sample_results)}")
    print(f"Blended results: {len(blended)}")
    
    # Test diversity metrics
    metrics = calculate_result_diversity_metrics(blended)
    print(f"Diversity metrics: {metrics}")