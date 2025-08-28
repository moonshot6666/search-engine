#!/usr/bin/env python3
"""
Real-time HDBSCAN clustering for search results.
Groups search results into thematic clusters based on content similarity.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .schema import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class SearchCluster:
    """Represents a cluster of search results with metadata."""
    cluster_id: int
    theme: str
    size: int
    results: List[SearchResult]
    keywords: List[str]
    confidence: float
    center_embedding: Optional[List[float]] = None


@dataclass
class ClusteredSearchResponse:
    """Search response with clustering information."""
    results: List[SearchResult]  # All results, flat
    clusters: List[SearchCluster]  # Organized clusters
    outliers: List[SearchResult]  # Results that don't fit clusters
    clustering_time_ms: int
    total_clusters: int


class RealTimeResultClusterer:
    """Real-time clustering of search results using HDBSCAN."""
    
    def __init__(self, 
                 min_cluster_size: int = 3,
                 min_samples: int = 2,
                 cluster_selection_epsilon: float = 0.1):
        """
        Initialize real-time clusterer.
        
        Args:
            min_cluster_size: Minimum results needed to form a cluster
            min_samples: Minimum density for core points
            cluster_selection_epsilon: Distance threshold for merging clusters
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples 
        self.cluster_selection_epsilon = cluster_selection_epsilon
        
        # TF-IDF vectorizer for content similarity when embeddings unavailable
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
    def cluster_results(self, results: List[SearchResult]) -> ClusteredSearchResponse:
        """
        Cluster search results in real-time.
        
        Args:
            results: List of search results to cluster
            
        Returns:
            Clustered search response with organized themes
        """
        import time
        start_time = time.time()
        
        # Skip clustering for small result sets
        if len(results) < self.min_cluster_size:
            return ClusteredSearchResponse(
                results=results,
                clusters=[],
                outliers=results,
                clustering_time_ms=int((time.time() - start_time) * 1000),
                total_clusters=0
            )
        
        # Extract features for clustering
        feature_matrix = self._extract_features(results)
        
        if feature_matrix is None or feature_matrix.shape[0] < self.min_cluster_size:
            return ClusteredSearchResponse(
                results=results,
                clusters=[],
                outliers=results,
                clustering_time_ms=int((time.time() - start_time) * 1000),
                total_clusters=0
            )
        
        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric='euclidean'
        )
        
        cluster_labels = clusterer.fit_predict(feature_matrix)
        
        # Organize results by clusters
        clusters, outliers = self._organize_clusters(results, cluster_labels, clusterer)
        
        clustering_time = int((time.time() - start_time) * 1000)
        
        return ClusteredSearchResponse(
            results=results,
            clusters=clusters,
            outliers=outliers,
            clustering_time_ms=clustering_time,
            total_clusters=len(clusters)
        )
    
    def _extract_features(self, results: List[SearchResult]) -> Optional[np.ndarray]:
        """Extract features for clustering from search results."""
        
        # Try to use embeddings if available
        embeddings = []
        for result in results:
            if hasattr(result, 'embedding') and result.embedding:
                embeddings.append(result.embedding)
        
        if len(embeddings) == len(results) and embeddings[0]:
            # Use pre-computed embeddings
            return np.array(embeddings)
        
        # Fallback to TF-IDF if no embeddings
        try:
            texts = []
            for result in results:
                # Use clean content or fallback to original content
                text = getattr(result, 'clean_content', result.content)
                if not text:
                    text = result.content
                texts.append(text)
            
            # Generate TF-IDF features
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            return tfidf_matrix.toarray()
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _organize_clusters(self, results: List[SearchResult], 
                          cluster_labels: np.ndarray, 
                          clusterer) -> Tuple[List[SearchCluster], List[SearchResult]]:
        """Organize results into clusters based on labels."""
        
        clusters = []
        outliers = []
        
        # Group results by cluster label
        cluster_groups = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Outlier
                outliers.append(results[i])
            else:
                if label not in cluster_groups:
                    cluster_groups[label] = []
                cluster_groups[label].append(results[i])
        
        # Create SearchCluster objects
        for cluster_id, cluster_results in cluster_groups.items():
            
            # Generate theme name and keywords
            theme, keywords = self._generate_cluster_theme(cluster_results)
            
            # Calculate cluster confidence (stability score)
            confidence = self._calculate_cluster_confidence(cluster_id, clusterer)
            
            cluster = SearchCluster(
                cluster_id=int(cluster_id),  # Convert numpy.int64 to int
                theme=theme,
                size=len(cluster_results),
                results=cluster_results,
                keywords=keywords,
                confidence=float(confidence)  # Convert numpy.float to float
            )
            
            clusters.append(cluster)
        
        # Sort clusters by size (largest first)
        clusters.sort(key=lambda c: c.size, reverse=True)
        
        return clusters, outliers
    
    def _generate_cluster_theme(self, cluster_results: List[SearchResult]) -> Tuple[str, List[str]]:
        """Generate a theme name and keywords for a cluster."""
        
        # Extract all content from cluster
        all_text = " ".join([result.content.lower() for result in cluster_results])
        
        # Count entity types and tokens mentioned
        entity_counts = {}
        token_mentions = []
        
        for result in cluster_results:
            if hasattr(result, 'entities') and result.entities:
                for entity in result.entities:
                    entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                    
                    if entity_type == 'token':
                        token_mentions.append(entity.name)
        
        # Generate theme based on dominant patterns
        theme_parts = []
        keywords = []
        
        # Most common entity type becomes part of theme
        if entity_counts:
            top_entity_type = max(entity_counts.items(), key=lambda x: x[1])
            if top_entity_type[1] >= 2:  # At least 2 mentions
                if top_entity_type[0] == 'token':
                    # Use specific token names
                    unique_tokens = list(set(token_mentions))[:2]  # Top 2 tokens
                    if unique_tokens:
                        theme_parts.append(" & ".join(unique_tokens))
                        keywords.extend([t.lower() for t in unique_tokens])
                else:
                    theme_parts.append(top_entity_type[0].title())
                    keywords.append(top_entity_type[0])
        
        # Add content-based keywords
        content_keywords = self._extract_content_keywords(all_text)
        keywords.extend(content_keywords[:3])  # Top 3 content keywords
        
        # Determine theme based on content patterns
        if any(word in all_text for word in ['hack', 'exploit', 'drain', 'vulnerability']):
            theme_parts.append("Security Issues")
        elif any(word in all_text for word in ['pump', 'rally', 'bull', 'ath', 'moon']):
            theme_parts.append("Price Surge")
        elif any(word in all_text for word in ['crash', 'dump', 'bear', 'selloff', 'drop']):
            theme_parts.append("Market Decline")
        elif any(word in all_text for word in ['launch', 'mainnet', 'upgrade', 'update']):
            theme_parts.append("Development News")
        elif any(word in all_text for word in ['regulation', 'government', 'legal', 'ban']):
            theme_parts.append("Regulatory News")
        else:
            theme_parts.append("General Discussion")
        
        # Combine theme parts
        if len(theme_parts) > 1:
            theme = f"{theme_parts[0]} - {theme_parts[1]}"
        elif theme_parts:
            theme = theme_parts[0]
        else:
            theme = "Mixed Content"
        
        return theme, list(set(keywords))  # Remove duplicates
    
    def _extract_content_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract top keywords from text content."""
        
        # Simple keyword extraction based on frequency
        # Filter out common words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = text.lower().split()
        word_counts = {}
        
        for word in words:
            # Remove punctuation and filter words
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) >= 3 and clean_word not in stop_words:
                word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:top_k]]
    
    def _calculate_cluster_confidence(self, cluster_id: int, clusterer) -> float:
        """Calculate confidence score for a cluster based on HDBSCAN stability."""
        
        try:
            # Use HDBSCAN's cluster persistence (stability) scores
            if hasattr(clusterer, 'cluster_persistence_') and clusterer.cluster_persistence_ is not None:
                if cluster_id < len(clusterer.cluster_persistence_):
                    return float(clusterer.cluster_persistence_[cluster_id])
            
            # Fallback to a reasonable default
            return 0.7
            
        except Exception:
            return 0.5  # Default confidence


async def test_clustering():
    """Test clustering functionality with sample data."""
    from pathlib import Path
    import json
    from .schema import Entity, EntityType, MarketImpact, Language
    
    # Load sample tweets
    data_file = Path(__file__).parent.parent.parent / "data" / "expanded_sample_tweets.jsonl"
    
    results = []
    with open(data_file, 'r') as f:
        for line in f:
            if line.strip():
                tweet = json.loads(line.strip())
                
                # Convert to SearchResult
                entities = [
                    Entity(
                        entity_type=EntityType(e["entity_type"]),
                        entity_id=e["entity_id"], 
                        name=e["name"],
                        confidence=1.0
                    ) for e in tweet.get("entities", [])
                ]
                
                result = SearchResult(
                    tweet_id=tweet["tweet_id"],
                    content=tweet["content"],
                    original_content=tweet.get("original_content", tweet["content"]),
                    created_at=tweet["created_at"],
                    source_handle=tweet["source_handle"],
                    source_followers=tweet["source_followers"],
                    engagement_score=tweet["engagement_score"],
                    entities=entities,
                    market_impact=MarketImpact(tweet["market_impact"]),
                    authority_score=tweet["authority_score"],
                    language=Language(tweet["language"]),
                    final_score=0.5
                )
                
                results.append(result)
                
                if len(results) >= 15:  # Test with subset
                    break
    
    # Test clustering with smaller parameters for sample data
    clusterer = RealTimeResultClusterer(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.3)
    clustered_response = clusterer.cluster_results(results)
    
    print(f"\n🔍 Clustering Results:")
    print(f"Total results: {len(results)}")
    print(f"Clusters found: {clustered_response.total_clusters}")
    print(f"Outliers: {len(clustered_response.outliers)}")
    print(f"Clustering time: {clustered_response.clustering_time_ms}ms")
    
    for cluster in clustered_response.clusters:
        print(f"\n📊 Cluster {cluster.cluster_id}: {cluster.theme}")
        print(f"   Size: {cluster.size} tweets")
        print(f"   Keywords: {cluster.keywords}")
        print(f"   Confidence: {cluster.confidence:.2f}")
        for result in cluster.results[:2]:  # Show first 2 tweets
            print(f"   - {result.content[:60]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_clustering())