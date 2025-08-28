"""
OpenSearch Index Setup for Hybrid Search Engine

Creates and manages OpenSearch indices with BM25 + kNN vector mapping
based on the schema from CLAUDE.md. Supports both keyword search and
semantic vector search on normalized tweet data.
"""

import logging
from typing import Dict, Any, Optional

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import RequestError

from .schema import IndexConfig, BM25Config

logger = logging.getLogger(__name__)


class OpenSearchIndexManager:
    """Manages OpenSearch index creation and configuration."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 9200,
                 use_ssl: bool = False,
                 verify_certs: bool = False,
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Initialize OpenSearch client connection.
        
        Args:
            host: OpenSearch host
            port: OpenSearch port  
            use_ssl: Whether to use SSL
            verify_certs: Whether to verify SSL certificates
            username: Authentication username
            password: Authentication password
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
            timeout=30
        )
        
    def get_index_mapping(self, vector_dimension: int = 1024) -> Dict[str, Any]:
        """
        Get the index mapping for hybrid BM25 + kNN search.
        
        Based on the mapping specification from CLAUDE.md with support
        for both keyword search and vector similarity search.
        
        Args:
            vector_dimension: Dimension of vector embeddings
            
        Returns:
            OpenSearch index mapping configuration
        """
        return {
            "properties": {
                # Core content fields
                "tweet_id": {"type": "keyword"},
                "clean_content": {
                    "type": "text",
                    "analyzer": "english",
                    "fields": {
                        "keyword": {"type": "keyword", "ignore_above": 512}
                    }
                },
                "original_content": {
                    "type": "text", 
                    "analyzer": "standard",
                    "index": False
                },
                
                # Entity fields (nested for complex filtering)
                "entities": {
                    "type": "nested",
                    "properties": {
                        "entity_type": {"type": "keyword"},
                        "entity_id": {"type": "keyword"},
                        "name": {"type": "text", "analyzer": "english"},
                        "symbol": {"type": "keyword"},
                        "confidence": {"type": "float"}
                    }
                },
                
                # Market impact and sentiment
                "market_impact": {"type": "keyword"},
                "has_tokens": {"type": "boolean"},
                "has_projects": {"type": "boolean"},
                "has_kols": {"type": "boolean"},
                "has_events": {"type": "boolean"},
                "has_macros": {"type": "boolean"},
                
                # Temporal fields
                "created_at_iso": {"type": "date"},
                "day_bucket": {"type": "keyword"},
                
                # Source and engagement metadata
                "source_handle": {"type": "keyword"},
                "source_followers": {"type": "integer"},
                "engagement_score": {"type": "integer"},
                "language": {"type": "keyword"},
                
                # Vector field for kNN search
                "embedding": {
                    "type": "knn_vector",
                    "dimension": vector_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16
                        }
                    }
                },
                
                # Derived authority score
                "authority": {
                    "type": "float"
                },
                
                # Status flags
                "is_vectorized": {"type": "boolean"}
            }
        }
        
    def get_index_settings(self, config: IndexConfig, bm25_config: BM25Config) -> Dict[str, Any]:
        """
        Get index settings with BM25 configuration and analysis.
        
        Args:
            config: Index configuration
            bm25_config: BM25 algorithm parameters
            
        Returns:
            OpenSearch index settings
        """
        return {
            "number_of_shards": config.number_of_shards,
            "number_of_replicas": config.number_of_replicas,
            "max_result_window": config.max_result_window,
            
            # BM25 similarity configuration
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": bm25_config.k1,
                    "b": bm25_config.b
                }
            },
            
            # Analysis configuration
            "analysis": {
                "analyzer": {
                    "english": {
                        "type": "english",
                        "stopwords": "_english_"
                    },
                    "crypto_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "asciifolding",
                            "crypto_synonyms"
                        ]
                    }
                },
                "filter": {
                    "crypto_synonyms": {
                        "type": "synonym",
                        "synonyms": [
                            "btc,bitcoin",
                            "eth,ethereum",
                            "sol,solana",
                            "ada,cardano",
                            "avax,avalanche",
                            "defi,decentralized finance",
                            "nft,non-fungible token",
                            "dapp,decentralized application"
                        ]
                    }
                }
            },
            
            # kNN algorithm parameters
            "knn": True,
            "knn.algo_param.ef_search": 512
        }
        
    def create_index(self, 
                     index_name: str,
                     config: Optional[IndexConfig] = None,
                     bm25_config: Optional[BM25Config] = None,
                     force_recreate: bool = False) -> bool:
        """
        Create OpenSearch index with hybrid BM25 + kNN configuration.
        
        Args:
            index_name: Name of the index to create
            config: Index configuration (defaults to IndexConfig())
            bm25_config: BM25 parameters (defaults to BM25Config())
            force_recreate: Whether to delete existing index
            
        Returns:
            True if index was created successfully
        """
        if config is None:
            config = IndexConfig(index_name=index_name)
        if bm25_config is None:
            bm25_config = BM25Config()
            
        try:
            # Check if index already exists
            if self.client.indices.exists(index=index_name):
                if force_recreate:
                    logger.info(f"Deleting existing index: {index_name}")
                    self.client.indices.delete(index=index_name)
                else:
                    logger.warning(f"Index {index_name} already exists")
                    return False
                    
            # Create index with mapping and settings
            index_body = {
                "settings": self.get_index_settings(config, bm25_config),
                "mappings": self.get_index_mapping(config.vector_dimension)
            }
            
            logger.info(f"Creating index: {index_name}")
            response = self.client.indices.create(
                index=index_name,
                body=index_body
            )
            
            logger.info(f"Index created successfully: {response}")
            return True
            
        except RequestError as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating index {index_name}: {e}")
            return False
            
    def delete_index(self, index_name: str) -> bool:
        """
        Delete an OpenSearch index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if index was deleted successfully
        """
        try:
            if not self.client.indices.exists(index=index_name):
                logger.warning(f"Index {index_name} does not exist")
                return False
                
            response = self.client.indices.delete(index=index_name)
            logger.info(f"Index {index_name} deleted successfully: {response}")
            return True
            
        except RequestError as e:
            logger.error(f"Failed to delete index {index_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting index {index_name}: {e}")
            return False
            
    def get_index_info(self, index_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an OpenSearch index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Index information or None if not found
        """
        try:
            if not self.client.indices.exists(index=index_name):
                return None
                
            settings = self.client.indices.get_settings(index=index_name)
            mappings = self.client.indices.get_mapping(index=index_name)
            stats = self.client.indices.stats(index=index_name)
            
            return {
                "name": index_name,
                "settings": settings[index_name]["settings"],
                "mappings": mappings[index_name]["mappings"],
                "stats": stats["indices"][index_name]
            }
            
        except Exception as e:
            logger.error(f"Error getting index info for {index_name}: {e}")
            return None
            
    def bulk_index_documents(self, 
                           index_name: str,
                           documents: list,
                           batch_size: int = 1000) -> Dict[str, Any]:
        """
        Bulk index documents to OpenSearch.
        
        Args:
            index_name: Target index name
            documents: List of documents to index
            batch_size: Number of documents per batch
            
        Returns:
            Indexing statistics
        """
        from opensearchpy.helpers import bulk
        
        total_docs = len(documents)
        successful = 0
        failed = 0
        errors = []
        
        try:
            # Prepare documents for bulk indexing
            actions = []
            for doc in documents:
                action = {
                    "_index": index_name,
                    "_id": doc.get("tweet_id"),
                    "_source": doc
                }
                actions.append(action)
                
                # Process in batches
                if len(actions) >= batch_size:
                    success_count, failed_items = bulk(
                        self.client,
                        actions,
                        chunk_size=batch_size,
                        request_timeout=60
                    )
                    successful += success_count
                    if failed_items:
                        failed += len(failed_items)
                        errors.extend(failed_items)
                    actions = []
                    
            # Process remaining documents
            if actions:
                success_count, failed_items = bulk(
                    self.client,
                    actions,
                    chunk_size=len(actions),
                    request_timeout=60
                )
                successful += success_count
                if failed_items:
                    failed += len(failed_items)
                    errors.extend(failed_items)
                    
            # Refresh index to make documents searchable
            self.client.indices.refresh(index=index_name)
            
            logger.info(f"Bulk indexing completed: {successful}/{total_docs} successful")
            
            return {
                "total": total_docs,
                "successful": successful,
                "failed": failed,
                "errors": errors[:10] if errors else []  # Limit error details
            }
            
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return {
                "total": total_docs,
                "successful": successful,
                "failed": total_docs - successful,
                "errors": [str(e)]
            }


def main():
    """Command-line interface for index management."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="OpenSearch Index Management")
    parser.add_argument("--host", default="localhost", help="OpenSearch host")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port")
    parser.add_argument("--index", default="tweets-hybrid", help="Index name")
    parser.add_argument("--action", choices=["create", "delete", "info"], 
                       required=True, help="Action to perform")
    parser.add_argument("--force", action="store_true", 
                       help="Force recreate index if it exists")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize index manager
    manager = OpenSearchIndexManager(host=args.host, port=args.port)
    
    if args.action == "create":
        success = manager.create_index(args.index, force_recreate=args.force)
        sys.exit(0 if success else 1)
        
    elif args.action == "delete":
        success = manager.delete_index(args.index)
        sys.exit(0 if success else 1)
        
    elif args.action == "info":
        info = manager.get_index_info(args.index)
        if info:
            print(f"Index: {info['name']}")
            print(f"Document count: {info['stats']['total']['docs']['count']}")
            print(f"Store size: {info['stats']['total']['store']['size_in_bytes']} bytes")
        else:
            print(f"Index {args.index} not found")
            sys.exit(1)


if __name__ == "__main__":
    main()