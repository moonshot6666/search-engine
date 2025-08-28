#!/usr/bin/env python3
"""
Data indexing script for hybrid search engine.
Loads tweets with embeddings and indexes them into OpenSearch.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

from .setup_index import OpenSearchIndexManager
from .schema import IndexConfig, BM25Config

logger = logging.getLogger(__name__)


def load_tweets_with_embeddings(file_path: str) -> List[Dict[str, Any]]:
    """Load tweets with embeddings from JSONL file."""
    tweets = []
    
    print(f"Loading tweets from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    tweet = json.loads(line.strip())
                    
                    # Validate required fields
                    required_fields = ["tweet_id", "content", "clean_content", "embedding"]
                    for field in required_fields:
                        if field not in tweet:
                            print(f"Warning: Tweet {tweet.get('tweet_id', line_num)} missing field: {field}")
                            continue
                    
                    tweets.append(tweet)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
    
    print(f"Loaded {len(tweets)} tweets")
    return tweets


def prepare_document_for_indexing(tweet: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare tweet document for OpenSearch indexing."""
    
    # Create document matching the index mapping
    doc = {
        "tweet_id": tweet["tweet_id"],
        "clean_content": tweet["clean_content"],
        "original_content": tweet.get("content", tweet["clean_content"]),
        
        # Entity fields
        "entities": tweet.get("entities", []),
        
        # Market impact and sentiment
        "market_impact": tweet.get("market_impact", "neutral"),
        
        # Temporal fields
        "created_at_iso": tweet.get("created_at", "2024-01-01T00:00:00Z"),
        
        # Source and engagement metadata
        "source_handle": tweet.get("source_handle", "unknown"),
        "source_followers": tweet.get("source_followers", 0),
        "engagement_score": tweet.get("engagement_score", 0),
        "language": tweet.get("language", "en"),
        
        # Authority score
        "authority": tweet.get("authority_score", 0.5),
        
        # Vector embedding
        "embedding": tweet["embedding"],
        
        # Status flags
        "is_vectorized": True
    }
    
    # Add boolean flags for entities
    entity_types = [e.get("entity_type", "") for e in tweet.get("entities", [])]
    doc["has_tokens"] = "token" in entity_types
    doc["has_projects"] = "project" in entity_types  
    doc["has_kols"] = "kol" in entity_types
    doc["has_events"] = "event" in entity_types
    doc["has_macros"] = "macro" in entity_types
    
    return doc


def main():
    """Main function to index data into OpenSearch."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index tweet data with embeddings")
    parser.add_argument("--input", default="data/sample_tweets_with_embeddings.jsonl",
                       help="Input JSONL file with embeddings")
    parser.add_argument("--host", default="localhost", help="OpenSearch host")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port")
    parser.add_argument("--index", default="crypto-tweets-hybrid", help="Index name")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for indexing")
    parser.add_argument("--recreate", action="store_true", help="Recreate index if exists")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / args.input
    
    print(f"🚀 Starting data indexing")
    print(f"📁 Input: {input_file}")
    print(f"🔗 OpenSearch: {args.host}:{args.port}")
    print(f"📊 Index: {args.index}")
    
    # Initialize index manager
    manager = OpenSearchIndexManager(host=args.host, port=args.port)
    
    # Load tweets
    tweets = load_tweets_with_embeddings(str(input_file))
    
    if not tweets:
        print("❌ No tweets to index")
        return
    
    # Create index if needed
    if args.recreate or not manager.client.indices.exists(args.index):
        print(f"Creating index: {args.index}")
        config = IndexConfig(index_name=args.index)
        bm25_config = BM25Config()
        
        success = manager.create_index(args.index, config, bm25_config, args.recreate)
        if not success:
            print("❌ Failed to create index")
            return
    
    # Prepare documents
    print("Preparing documents for indexing...")
    documents = []
    for tweet in tweets:
        try:
            doc = prepare_document_for_indexing(tweet)
            documents.append(doc)
        except Exception as e:
            print(f"Error preparing document {tweet.get('tweet_id', 'unknown')}: {e}")
    
    print(f"Prepared {len(documents)} documents")
    
    # Index documents
    print(f"Indexing {len(documents)} documents...")
    start_time = time.time()
    
    result = manager.bulk_index_documents(
        index_name=args.index,
        documents=documents,
        batch_size=args.batch_size
    )
    
    index_time = time.time() - start_time
    
    # Show results
    print(f"\n✅ Indexing completed in {index_time:.2f}s")
    print(f"📊 Results:")
    print(f"   - Total documents: {result['total']}")
    print(f"   - Successful: {result['successful']}")
    print(f"   - Failed: {result['failed']}")
    
    if result['errors']:
        print(f"⚠️  Errors (showing first 3):")
        for error in result['errors'][:3]:
            print(f"   - {error}")
    
    # Test the index
    print(f"\n🧪 Testing indexed data...")
    
    # Simple search test
    try:
        test_response = manager.client.search(
            index=args.index,
            body={
                "query": {"match_all": {}},
                "size": 1
            }
        )
        
        total_docs = test_response["hits"]["total"]["value"]
        print(f"✅ Index contains {total_docs} searchable documents")
        
        if test_response["hits"]["hits"]:
            sample_doc = test_response["hits"]["hits"][0]["_source"]
            print(f"📝 Sample document fields: {list(sample_doc.keys())}")
            print(f"   - Content: {sample_doc.get('clean_content', '')[:50]}...")
            print(f"   - Entities: {len(sample_doc.get('entities', []))}")
            print(f"   - Embedding dimension: {len(sample_doc.get('embedding', []))}")
    
    except Exception as e:
        print(f"⚠️  Error testing index: {e}")
    
    print(f"\n🎉 Data indexing completed!")
    print(f"📝 Next steps:")
    print(f"   1. Test BM25 search: python -m src.search.test_bm25")
    print(f"   2. Test vector search: python -m src.search.test_vector") 
    print(f"   3. Test hybrid search: python -m src.search.test_hybrid")
    print(f"   4. Replace mock engine in main.py")


if __name__ == "__main__":
    main()