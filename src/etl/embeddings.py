#!/usr/bin/env python3
"""
Embedding generation pipeline for semantic search.
Creates vector representations of normalized tweets for kNN similarity search.
"""

import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import jsonlines
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pickle


class EmbeddingGenerator:
    """Generates and caches embeddings for tweet content."""
    
    def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5', cache_dir: Optional[str] = None):
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model name for embeddings
            cache_dir: Directory to cache embeddings and model
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path('.cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(self.cache_dir / 'sentence_transformers')
        )
        
        # Get embedding dimension
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dimension}")
        
        # Initialize embedding cache
        self.embedding_cache_path = self.cache_dir / f'embeddings_{self._model_hash()}.pkl'
        self.embedding_cache = self._load_embedding_cache()
    
    def _model_hash(self) -> str:
        """Generate hash for model name to version cache."""
        return hashlib.md5(self.model_name.encode()).hexdigest()[:8]
    
    def _load_embedding_cache(self) -> Dict[str, np.ndarray]:
        """Load existing embedding cache."""
        if self.embedding_cache_path.exists():
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    cache = pickle.load(f)
                    print(f"Loaded {len(cache)} cached embeddings")
                    return cache
            except Exception as e:
                print(f"Failed to load embedding cache: {e}")
        
        return {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            print(f"Failed to save embedding cache: {e}")
    
    def _create_embedding_text(self, normalized_tweet: Dict[str, Any]) -> str:
        """
        Create text for embedding generation.
        Combines clean content with entity information for richer semantics.
        """
        content = normalized_tweet.get('clean_content', '')
        
        # Add entity context
        entities = normalized_tweet.get('entities', [])
        if entities:
            entity_names = [entity.get('name', '') for entity in entities if entity.get('name')]
            entity_text = ' | entities: ' + ', '.join(entity_names)
        else:
            entity_text = ''
        
        # Add market impact context
        market_impact = normalized_tweet.get('market_impact', 'neutral')
        impact_text = f' | impact: {market_impact}' if market_impact != 'neutral' else ''
        
        # Combine all parts
        full_text = content + entity_text + impact_text
        return full_text.strip()
    
    def _compute_content_hash(self, embedding_text: str) -> str:
        """Generate hash for content to enable caching."""
        return hashlib.sha256(embedding_text.encode('utf-8')).hexdigest()
    
    def generate_embedding(self, normalized_tweet: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Generate embedding for a single normalized tweet.
        
        Returns:
            Embedding vector as numpy array, or None if generation fails
        """
        try:
            # Create embedding text
            embedding_text = self._create_embedding_text(normalized_tweet)
            
            if not embedding_text.strip():
                return None
            
            # Check cache first
            content_hash = self._compute_content_hash(embedding_text)
            if content_hash in self.embedding_cache:
                return self.embedding_cache[content_hash]
            
            # Generate new embedding
            embedding = self.model.encode(
                embedding_text,
                normalize_embeddings=True,  # Cosine similarity ready
                show_progress_bar=False
            )
            
            # Convert to numpy array and cache
            embedding = np.array(embedding, dtype=np.float32)
            self.embedding_cache[content_hash] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding for tweet {normalized_tweet.get('tweet_id', 'unknown')}: {e}")
            return None
    
    def generate_batch_embeddings(self, normalized_tweets: List[Dict[str, Any]], batch_size: int = 32) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for a batch of tweets efficiently.
        
        Args:
            normalized_tweets: List of normalized tweet dictionaries
            batch_size: Batch size for model inference
            
        Returns:
            List of embedding arrays (same order as input)
        """
        embeddings = []
        
        # Prepare texts and check cache
        texts_to_embed = []
        cache_hits = []
        tweet_indices = []
        
        for i, tweet in enumerate(normalized_tweets):
            embedding_text = self._create_embedding_text(tweet)
            if not embedding_text.strip():
                embeddings.append(None)
                continue
                
            content_hash = self._compute_content_hash(embedding_text)
            if content_hash in self.embedding_cache:
                embeddings.append(self.embedding_cache[content_hash])
                cache_hits.append(i)
            else:
                texts_to_embed.append(embedding_text)
                tweet_indices.append(i)
                embeddings.append(None)  # Placeholder
        
        # Generate embeddings for cache misses
        if texts_to_embed:
            try:
                batch_embeddings = self.model.encode(
                    texts_to_embed,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                # Update results and cache
                for i, (tweet_idx, embedding) in enumerate(zip(tweet_indices, batch_embeddings)):
                    embedding = np.array(embedding, dtype=np.float32)
                    embeddings[tweet_idx] = embedding
                    
                    # Cache the embedding
                    embedding_text = self._create_embedding_text(normalized_tweets[tweet_idx])
                    content_hash = self._compute_content_hash(embedding_text)
                    self.embedding_cache[content_hash] = embedding
                    
            except Exception as e:
                print(f"Error in batch embedding generation: {e}")
                # Fill remaining with None
                for tweet_idx in tweet_indices:
                    if embeddings[tweet_idx] is None:
                        embeddings[tweet_idx] = None
        
        return embeddings
    
    def process_normalized_tweets(self, input_file: str, output_file: str, batch_size: int = 32) -> Dict[str, int]:
        """
        Process normalized tweets file and add embeddings.
        
        Args:
            input_file: Path to normalized tweets JSONL
            output_file: Path to output JSONL with embeddings
            batch_size: Batch size for processing
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_processed': 0,
            'embeddings_generated': 0,
            'embeddings_cached': 0,
            'errors': 0
        }
        
        with jsonlines.open(input_file, 'r') as reader, \
             jsonlines.open(output_file, 'w') as writer:
            
            batch = []
            
            for tweet in tqdm(reader, desc="Processing tweets for embeddings"):
                batch.append(tweet)
                
                if len(batch) >= batch_size:
                    self._process_batch(batch, writer, stats)
                    batch = []
            
            # Process remaining batch
            if batch:
                self._process_batch(batch, writer, stats)
        
        # Save updated cache
        self._save_embedding_cache()
        
        return stats
    
    def _process_batch(self, batch: List[Dict[str, Any]], writer: jsonlines.Writer, stats: Dict[str, int]):
        """Process a batch of tweets and write results."""
        # Generate embeddings for batch
        embeddings = self.generate_batch_embeddings(batch)
        
        # Update tweets with embeddings and write
        for tweet, embedding in zip(batch, embeddings):
            stats['total_processed'] += 1
            
            if embedding is not None:
                tweet['embedding'] = embedding.tolist()  # Convert to list for JSON serialization
                tweet['is_vectorized'] = True
                stats['embeddings_generated'] += 1
            else:
                tweet['is_vectorized'] = False
                stats['errors'] += 1
            
            writer.write(tweet)


def main():
    """CLI entry point for embedding generation."""
    parser = argparse.ArgumentParser(description='Generate embeddings for normalized tweets')
    parser.add_argument('--input', required=True, help='Input normalized tweets JSONL file')
    parser.add_argument('--output', help='Output JSONL file with embeddings (default: input_with_embeddings.jsonl)')
    parser.add_argument('--model', default='BAAI/bge-large-en-v1.5', 
                       help='Embedding model name (default: BAAI/bge-large-en-v1.5)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--cache-dir', default='.cache', help='Cache directory for model and embeddings')
    
    args = parser.parse_args()
    
    # Set output file if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_with_embeddings.jsonl")
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(
        model_name=args.model,
        cache_dir=args.cache_dir
    )
    
    # Process tweets
    print(f"Processing {args.input} -> {args.output}")
    stats = generator.process_normalized_tweets(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size
    )
    
    # Print statistics
    print(f"\nEmbedding generation complete:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Embeddings generated: {stats['embeddings_generated']}")
    print(f"  Cache hits: {stats['embeddings_cached']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Success rate: {stats['embeddings_generated'] / max(stats['total_processed'], 1) * 100:.1f}%")
    
    print(f"\nEmbedding dimension: {generator.embedding_dimension}")
    print(f"Cache size: {len(generator.embedding_cache)} embeddings")


if __name__ == '__main__':
    main()