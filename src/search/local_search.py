#!/usr/bin/env python3
"""
Local Hybrid Search Engine - In-memory implementation for testing.
Simulates OpenSearch behavior without requiring external dependencies.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import math
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from .schema import SearchSpec, SearchResult, SearchResponse, Entity, EntityType, MarketImpact, Language
from .cluster import RealTimeResultClusterer, ClusteredSearchResponse


class IntentType(Enum):
    """Extensible intent classification for search queries."""
    DECISION_MAKING = "decision_making"  # should I buy/sell XXX
    CAUSATION_ANALYSIS = "causation_analysis"  # why is XXX pumping/dumping
    FORECASTING = "forecasting"  # price prediction for XXX
    OPINION_SEEKING = "opinion_seeking"  # what is XXX saying about YYY
    SENTIMENT_FILTERING = "sentiment_filtering"  # bullish/bearish news for XXX
    COMPARISON = "comparison"  # XXX vs YYY
    IMPACT_ANALYSIS = "impact_analysis"  # XXX impact on YYY
    NEWS_SEEKING = "news_seeking"  # XXX news, show news for XXX, latest XXX updates
    GENERAL = "general"  # fallback for other queries


class RelationshipType(Enum):
    """Entity relationship types for multi-entity queries."""
    HAS_OPINION_ABOUT = "has_opinion_about"  # KOL → Token
    COMPARES_TO = "compares_to"  # Token → Token
    AFFECTS = "affects"  # Event → Token
    COLLABORATES_WITH = "collaborates_with"  # Entity → Entity


@dataclass
class DetectedEntity:
    """Entity detected in query with type and confidence."""
    entity_type: str  # token, kol, event, macro
    entity_id: str
    name: str
    confidence: float = 1.0


@dataclass 
class EntityRelationship:
    """Relationship between two entities in a query."""
    source: DetectedEntity
    relationship_type: RelationshipType
    target: DetectedEntity


@dataclass
class QueryAnalysis:
    """Complete analysis of user query intent and entity requirements."""
    original_query: str
    primary_intent: IntentType
    detected_entities: List[DetectedEntity]
    entity_relationships: List[EntityRelationship]
    entity_coverage_requirement: str  # "ALL", "ANY", "SINGLE_PRIMARY"
    content_preferences: List[str]  # analytical, sentiment_specific, etc.


class IntentClassifier:
    """Extensible intent pattern classifier."""
    
    def __init__(self):
        self.intent_patterns = {
            IntentType.DECISION_MAKING: {
                'keywords': ['should i', 'should buy', 'should sell', 'worth buying', 'good investment', 'recommend', 'buy', 'sell', 'invest in'],
                'entity_requirement': 'SINGLE_PRIMARY',
                'content_preference': 'BALANCED_ANALYSIS'
            },
            IntentType.CAUSATION_ANALYSIS: {
                'keywords': ['why', 'because', 'reason', 'due to', 'pumping', 'dumping', 'crashing', 'rallying'],
                'entity_requirement': 'SINGLE_PRIMARY', 
                'content_preference': 'CAUSAL_EXPLANATION'
            },
            IntentType.FORECASTING: {
                'keywords': ['prediction', 'forecast', 'target', 'will', 'expect', 'price target', 'future'],
                'entity_requirement': 'SINGLE_PRIMARY',
                'content_preference': 'PREDICTIVE_ANALYSIS'
            },
            IntentType.OPINION_SEEKING: {
                'keywords': ['saying', 'thinks', 'opinion', 'believes', 'view', 'thoughts', 'mentioned'],
                'entity_requirement': 'MULTI_ENTITY_RELATIONSHIP',
                'content_preference': 'DIRECT_STATEMENTS'
            },
            IntentType.SENTIMENT_FILTERING: {
                'keywords': ['bullish', 'bearish', 'positive', 'negative', 'optimistic', 'pessimistic'],
                'entity_requirement': 'SINGLE_PRIMARY',
                'content_preference': 'SENTIMENT_SPECIFIC'
            },
            IntentType.COMPARISON: {
                'keywords': ['vs', 'versus', 'compared to', 'better than', 'worse than', 'compare'],
                'entity_requirement': 'MULTI_ENTITY_RELATIONSHIP',
                'content_preference': 'COMPARATIVE_ANALYSIS'
            },
            IntentType.IMPACT_ANALYSIS: {
                'keywords': ['impact', 'effect', 'affects', 'influence', 'consequences'],
                'entity_requirement': 'MULTI_ENTITY_RELATIONSHIP', 
                'content_preference': 'CAUSAL_EXPLANATION'
            },
            IntentType.NEWS_SEEKING: {
                'keywords': ['news', 'show news', 'latest news', 'breaking news', 'updates', 'latest updates', 'show updates'],
                'entity_requirement': 'SINGLE_PRIMARY_STRONG',
                'content_preference': 'NEWS_CONTENT'
            }
        }
    
    def classify_intent(self, query: str) -> Tuple[IntentType, str, str]:
        """Classify query intent and return requirements."""
        query_lower = query.lower()
        
        # Check for compound patterns first (more specific)
        compound_patterns = [
            # Pattern: "why is X bullish/bearish" - requires both entity and sentiment
            (r'why.*(bullish|bearish)', IntentType.CAUSATION_ANALYSIS, 'COMPOUND_REQUIREMENT', 'CAUSAL_EXPLANATION'),
            # Pattern: "why.*(pumping|dumping)" - similar compound pattern
            (r'why.*(pumping|dumping)', IntentType.CAUSATION_ANALYSIS, 'COMPOUND_REQUIREMENT', 'CAUSAL_EXPLANATION'),
        ]
        
        import re
        for pattern, intent, requirement, preference in compound_patterns:
            if re.search(pattern, query_lower):
                return intent, requirement, preference
        
        # Fall back to regular pattern matching
        for intent_type, config in self.intent_patterns.items():
            if any(keyword in query_lower for keyword in config['keywords']):
                return (
                    intent_type,
                    config['entity_requirement'],
                    config['content_preference']
                )
        
        return IntentType.GENERAL, 'ANY', 'GENERAL_RELEVANCE'


class EntityExtractor:
    """Extract entities from queries with dynamic patterns."""
    
    def __init__(self):
        # Extensible entity patterns - can be loaded from config
        self.entity_patterns = {
            'token': {
                'btc': ['btc', 'bitcoin', 'bitcoin token'],
                'eth': ['eth', 'ethereum', 'ethereum token'],
                'sol': ['sol', 'solana', 'solana token', 'sol token'],
                'ada': ['ada', 'cardano', 'cardano token'],
                'matic': ['matic', 'polygon', 'polygon token'],
                'avax': ['avax', 'avalanche', 'avalanche token'],
                'link': ['link', 'chainlink', 'chainlink token'],
                'uni': ['uni', 'uniswap', 'uniswap token'],
                'doge': ['doge', 'dogecoin', 'dogecoin token']
            },
            'kol': {
                'elon': ['elon', 'musk', 'elonmusk', 'elon musk'],
                'vitalik': ['vitalik', 'vitalikbuterin', 'vitalik buterin'],
                'michael_saylor': ['saylor', 'michael saylor', 'michaelsaylor'],
                'cathie': ['cathie wood', 'cathie', 'arkk']
            },
            'event': {
                'fed_rate': ['fed rate', 'federal reserve', 'interest rate'],
                'halving': ['halving', 'bitcoin halving'],
                'upgrade': ['upgrade', 'fork', 'update']
            },
            'macro': {
                'inflation': ['inflation', 'cpi'],
                'recession': ['recession', 'economic downturn']
            },
            'sentiment': {
                'bullish': ['bullish', 'bull', 'positive', 'optimistic', 'moon', 'pump', 'rally'],
                'bearish': ['bearish', 'bear', 'negative', 'pessimistic', 'crash', 'dump', 'selloff']
            }
        }
    
    def extract_entities(self, query: str) -> List[DetectedEntity]:
        """Extract all entities from query."""
        query_lower = query.lower()
        detected = []
        
        for entity_type, entities in self.entity_patterns.items():
            for entity_id, patterns in entities.items():
                for pattern in patterns:
                    if pattern in query_lower:
                        detected.append(DetectedEntity(
                            entity_type=entity_type,
                            entity_id=entity_id,
                            name=entities[entity_id][0].title(),  # First pattern as canonical name
                            confidence=1.0
                        ))
                        break  # Found this entity, move to next
        
        return detected


class EntityRelationshipInferencer:
    """Infer relationships between entities based on query intent."""
    
    def infer_relationships(self, entities: List[DetectedEntity], intent: IntentType, query: str) -> List[EntityRelationship]:
        """Dynamically infer entity relationships."""
        relationships = []
        query_lower = query.lower()
        
        if len(entities) < 2:
            return relationships
        
        # OPINION_SEEKING: "what is XXX saying about YYY" → XXX HAS_OPINION_ABOUT YYY
        if intent == IntentType.OPINION_SEEKING:
            # Look for KOL + Token combination
            kol_entities = [e for e in entities if e.entity_type == 'kol']
            token_entities = [e for e in entities if e.entity_type == 'token']
            
            if kol_entities and token_entities:
                relationships.append(EntityRelationship(
                    source=kol_entities[0],
                    relationship_type=RelationshipType.HAS_OPINION_ABOUT,
                    target=token_entities[0]
                ))
        
        # COMPARISON: "XXX vs YYY" → XXX COMPARES_TO YYY
        elif intent == IntentType.COMPARISON:
            if len(entities) >= 2 and entities[0].entity_type == entities[1].entity_type:
                relationships.append(EntityRelationship(
                    source=entities[0],
                    relationship_type=RelationshipType.COMPARES_TO,
                    target=entities[1]
                ))
        
        # IMPACT_ANALYSIS: "XXX impact on YYY" → XXX AFFECTS YYY
        elif intent == IntentType.IMPACT_ANALYSIS:
            relationships.append(EntityRelationship(
                source=entities[0],
                relationship_type=RelationshipType.AFFECTS,
                target=entities[1]
            ))
        
        return relationships


class QueryAnalyzer:
    """Complete query analysis combining intent classification and entity extraction."""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.relationship_inferencer = EntityRelationshipInferencer()
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Perform complete query analysis."""
        # Extract entities
        entities = self.entity_extractor.extract_entities(query)
        
        # Classify intent
        intent, entity_requirement, content_preference = self.intent_classifier.classify_intent(query)
        
        # Infer relationships
        relationships = self.relationship_inferencer.infer_relationships(entities, intent, query)
        
        return QueryAnalysis(
            original_query=query,
            primary_intent=intent,
            detected_entities=entities,
            entity_relationships=relationships,
            entity_coverage_requirement=entity_requirement,
            content_preferences=[content_preference]
        )


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
    """Local hybrid search engine combining BM25 and vector search with intent-driven scoring."""
    
    def __init__(self):
        self.bm25_engine = LocalBM25Engine()
        self.vector_engine = LocalVectorEngine()
        self.documents = []
        self.clusterer = RealTimeResultClusterer(
            min_cluster_size=2,  # Smaller clusters for sample data
            min_samples=1,       # More permissive for small datasets
            cluster_selection_epsilon=0.3  # Better clustering for similar content
        )
        # Initialize intent-driven query analysis system
        self.query_analyzer = QueryAnalyzer()
        
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
        
        # Analyze query using intent-driven approach
        query_analysis = self.query_analyzer.analyze(search_spec.query)
        
        # For compound requirements, ensure entity-matching documents are in candidate set
        if query_analysis.detected_entities:
            combined_scores = self._expand_candidates_for_compound_requirements(
                combined_scores, query_analysis
            )
        
        # Calculate final scores with intent-driven entity scoring
        final_results = []
        
        for doc_id, scores in combined_scores.items():
            # Base hybrid score calculation
            base_score = (
                0.45 * scores["bm25"] +
                0.35 * scores["vector"] +
                0.20 * 0.5  # Placeholder for other factors
            )
            
            # Calculate intent fulfillment multiplier
            intent_multiplier = self._calculate_intent_fulfillment_score(
                doc_id, query_analysis, search_spec, base_score
            )
            
            # Apply intent-driven scoring
            final_score = max(0.0, min(base_score * intent_multiplier, 1.0))
            
            final_results.append((doc_id, final_score, scores))
        
        # Sort by final score and return top results
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        return final_results[:search_spec.size]
    
    def _expand_candidates_for_compound_requirements(self, combined_scores: dict, query_analysis: QueryAnalysis) -> dict:
        """Expand candidate set to include entity-matching documents for compound requirements."""
        # Check if this is a compound requirement query
        has_sentiment_requirement = any(e.entity_type == 'sentiment' for e in query_analysis.detected_entities)
        has_other_entities = any(e.entity_type != 'sentiment' for e in query_analysis.detected_entities)
        
        if not (has_sentiment_requirement and has_other_entities):
            return combined_scores  # Not a compound requirement, no expansion needed
        
        # Find all documents that match any of the required entities
        required_entity_ids = {(e.entity_type, e.entity_id) for e in query_analysis.detected_entities if e.entity_type != 'sentiment'}
        
        for doc_id, doc in enumerate(self.documents):
            # Skip if already in candidate set
            if doc_id in combined_scores:
                continue
            
            doc_entities = doc.get("entities", [])
            doc_entity_ids = {(e.get("entity_type", ""), e.get("entity_id", "")) for e in doc_entities}
            
            # Check if document has any required entities
            if required_entity_ids.intersection(doc_entity_ids):
                # Add to candidate set with minimal scores (will be boosted by compound logic if sentiment also matches)
                combined_scores[doc_id] = {"bm25": 0.0, "vector": 0.0}
        
        return combined_scores
    
    def _calculate_intent_fulfillment_score(self, doc_id: int, query_analysis: QueryAnalysis, search_spec: SearchSpec, base_score: float) -> float:
        """Calculate intent fulfillment multiplier based on entity coverage and intent requirements."""
        if doc_id >= len(self.documents):
            return 0.1  # Very low score for invalid documents
        
        doc = self.documents[doc_id]
        doc_entities = doc.get("entities", [])
        
        # Check if we have sentiment filters from the API (compound requirement)
        has_sentiment_filter = (search_spec.filters and 
                               hasattr(search_spec.filters, 'sentiment') and 
                               search_spec.filters.sentiment)
        
        
        # Calculate entity coverage score (enhanced for compound requirements)
        entity_coverage_multiplier = self._calculate_entity_coverage_score(
            doc_entities, doc, query_analysis, has_sentiment_filter, search_spec
        )
        
        # Calculate content preference score based on intent
        content_preference_multiplier = self._calculate_content_preference_score(
            doc, query_analysis
        )
        
        # Calculate relationship fulfillment score for multi-entity queries
        relationship_multiplier = self._calculate_relationship_fulfillment_score(
            doc_entities, query_analysis
        )
        
        # Combine multipliers - multiplicative approach for strict requirements
        final_multiplier = entity_coverage_multiplier * content_preference_multiplier * relationship_multiplier
        
        return final_multiplier
    
    def _calculate_entity_coverage_score(self, doc_entities: List[Dict], doc: Dict, query_analysis: QueryAnalysis, has_sentiment_filter: bool = False, search_spec: SearchSpec = None) -> float:
        """Calculate entity coverage score based on query requirements."""
        if not query_analysis.detected_entities:
            return 1.0  # No entity requirements, full score
        
        # Create mapping of detected entities for easy lookup
        doc_entity_map = {}
        for entity in doc_entities:
            entity_type = entity.get("entity_type", "")
            entity_id = entity.get("entity_id", "")
            if entity_type not in doc_entity_map:
                doc_entity_map[entity_type] = set()
            doc_entity_map[entity_type].add(entity_id)
        
        # Get document market impact for sentiment checking
        doc_market_impact = doc.get("market_impact", "neutral")
        
        # Check coverage based on query requirements
        required_entities = query_analysis.detected_entities
        coverage_scores = []
        
        for required_entity in required_entities:
            entity_type = required_entity.entity_type
            entity_id = required_entity.entity_id
            
            if entity_type == 'sentiment':
                # Special handling for sentiment entities - check against market_impact
                if doc_market_impact == entity_id:
                    coverage_scores.append(1.0)  # Perfect sentiment match
                else:
                    coverage_scores.append(0.0)  # Wrong sentiment
            else:
                # Regular entity checking
                if entity_type in doc_entity_map and entity_id in doc_entity_map[entity_type]:
                    coverage_scores.append(1.0)  # Perfect match
                else:
                    coverage_scores.append(0.0)  # Missing entity
        
        # Apply scoring based on entity coverage requirement
        coverage_ratio = sum(coverage_scores) / len(coverage_scores)
        
        # Check if we have compound requirements (entity + sentiment)
        # Check both EntityExtractor detection AND API sentiment filters
        has_sentiment_requirement = (any(e.entity_type == 'sentiment' for e in query_analysis.detected_entities) or 
                                   has_sentiment_filter)
        has_other_entities = any(e.entity_type != 'sentiment' for e in query_analysis.detected_entities)
        is_compound_requirement = has_sentiment_requirement and has_other_entities
        
        # REMOVED: Duplicate API sentiment filter checking to avoid double sentiment scoring
        # The EntityExtractor sentiment check above (lines 664-669) already handles sentiment requirements
        # Adding API sentiment filters on top creates duplicate sentiment scoring which breaks compound requirements
        
        if is_compound_requirement or query_analysis.entity_coverage_requirement == "COMPOUND_REQUIREMENT":
            # For compound requirements (e.g., "why is sol bullish"), ALL must be satisfied
            
            if coverage_ratio == 1.0:
                return 2.0  # Strong bonus for perfect compound match
            else:
                return 0.1  # Heavy penalty for any missing requirement
        
        elif query_analysis.entity_coverage_requirement == "MULTI_ENTITY_RELATIONSHIP":
            # For multi-entity queries, ALL entities must be present
            if coverage_ratio == 1.0:
                return 1.5  # Bonus for perfect multi-entity match
            elif coverage_ratio >= 0.5:
                return 0.4  # Partial match penalty
            else:
                return 0.1  # Heavy penalty for poor coverage
        
        elif query_analysis.entity_coverage_requirement in ["SINGLE_PRIMARY", "SINGLE_PRIMARY_STRONG"]:
            # For single entity queries, prioritize primary entity
            primary_entity_present = coverage_scores[0] if coverage_scores else 0.0
            if primary_entity_present:
                if query_analysis.entity_coverage_requirement == "SINGLE_PRIMARY_STRONG":
                    return 3.0  # Strong boost for news + correct entity
                else:
                    return 1.2  # Standard boost for primary entity match
            else:
                # Check if document has off-topic entities
                off_topic_penalty = self._calculate_off_topic_entity_penalty(
                    doc_entity_map, required_entities
                )
                if query_analysis.entity_coverage_requirement == "SINGLE_PRIMARY_STRONG":
                    return max(0.1, 0.2 - off_topic_penalty)  # Stronger penalty for news + wrong entity
                else:
                    return max(0.1, 0.8 - off_topic_penalty)  # Standard penalty for missing primary entity
        
        else:  # "ANY" requirement
            if coverage_ratio > 0:
                return 1.0 + (coverage_ratio * 0.3)  # Modest boost for any entity match
            else:
                return 1.0  # No penalty for general queries
    
    def _calculate_off_topic_entity_penalty(self, doc_entity_map: Dict, required_entities: List[DetectedEntity]) -> float:
        """Calculate penalty for off-topic entities (especially tokens)."""
        penalty = 0.0
        required_token_ids = {e.entity_id for e in required_entities if e.entity_type == 'token'}
        
        if 'token' in doc_entity_map:
            doc_token_ids = doc_entity_map['token']
            # Penalize documents with token entities that don't match the query
            off_topic_tokens = doc_token_ids - required_token_ids
            penalty += len(off_topic_tokens) * 0.3  # 0.3 penalty per off-topic token
        
        return penalty
    
    def _calculate_content_preference_score(self, doc: Dict, query_analysis: QueryAnalysis) -> float:
        """Calculate content preference score based on intent requirements."""
        content = doc.get("content", "").lower()
        market_impact = doc.get("market_impact", "neutral")
        
        multiplier = 1.0
        
        for preference in query_analysis.content_preferences:
            if preference == "BALANCED_ANALYSIS":
                # For investment decisions, prefer analytical content
                analytical_keywords = ["analysis", "fundamental", "technical", "risk", "strategy", "research"]
                speculative_keywords = ["moon", "pump", "dump", "rocket", "lambo"]
                
                if any(keyword in content for keyword in analytical_keywords):
                    multiplier *= 1.3  # Boost analytical content
                elif any(keyword in content for keyword in speculative_keywords):
                    multiplier *= 0.7  # Penalize speculative content
                
                # For investment queries, boost both bullish and bearish perspectives
                if market_impact in ["bullish", "bearish"]:
                    multiplier *= 1.15
            
            elif preference == "CAUSAL_EXPLANATION":
                # For "why" queries, prefer explanatory content
                causal_keywords = ["because", "due to", "caused by", "reason", "impact", "driven by"]
                if any(keyword in content for keyword in causal_keywords):
                    multiplier *= 1.4
            
            elif preference == "DIRECT_STATEMENTS":
                # For opinion seeking, prefer direct statements/quotes
                statement_keywords = ["said", "says", "believes", "thinks", "stated", "mentioned", "tweeted"]
                if any(keyword in content for keyword in statement_keywords):
                    multiplier *= 1.5
            
            elif preference == "SENTIMENT_SPECIFIC":
                # For sentiment filtering, exact sentiment match is crucial
                if query_analysis.primary_intent == IntentType.SENTIMENT_FILTERING:
                    query_sentiment = "bullish" if "bullish" in query_analysis.original_query.lower() else "bearish"
                    if market_impact == query_sentiment:
                        multiplier *= 1.6  # Strong boost for matching sentiment
                    else:
                        multiplier *= 0.3  # Strong penalty for non-matching sentiment
            
            elif preference == "COMPARATIVE_ANALYSIS":
                # For comparisons, prefer content that discusses both entities
                comparative_keywords = ["better", "worse", "versus", "compared", "difference", "advantage"]
                if any(keyword in content for keyword in comparative_keywords):
                    multiplier *= 1.4
            
            elif preference == "PREDICTIVE_ANALYSIS":
                # For forecasting, prefer future-oriented content
                predictive_keywords = ["will", "expect", "forecast", "target", "prediction", "future"]
                if any(keyword in content for keyword in predictive_keywords):
                    multiplier *= 1.3
            
            elif preference == "NEWS_CONTENT":
                # For news seeking, prefer news-style and recent content
                news_keywords = ["breaking", "announced", "reported", "confirmed", "update", "latest"]
                recent_keywords = ["today", "yesterday", "just", "now", "recent", "new"]
                
                if any(keyword in content for keyword in news_keywords):
                    multiplier *= 1.4  # Strong boost for news indicators
                elif any(keyword in content for keyword in recent_keywords):
                    multiplier *= 1.2  # Moderate boost for recency indicators
                
                # Additional boost for authoritative sources in news context
                source_handle = doc.get("source_handle", "").lower()
                news_sources = ["news", "reporter", "analyst", "tracker", "official"]
                if any(source in source_handle for source in news_sources):
                    multiplier *= 1.15
        
        return multiplier
    
    def _calculate_relationship_fulfillment_score(self, doc_entities: List[Dict], query_analysis: QueryAnalysis) -> float:
        """Calculate relationship fulfillment score for multi-entity queries."""
        if not query_analysis.entity_relationships:
            return 1.0  # No relationship requirements
        
        # For each required relationship, check if document fulfills it
        relationship_scores = []
        
        for relationship in query_analysis.entity_relationships:
            source_entity = relationship.source
            target_entity = relationship.target
            rel_type = relationship.relationship_type
            
            # Check if both entities are present in document
            source_present = any(
                e.get("entity_type") == source_entity.entity_type and 
                e.get("entity_id") == source_entity.entity_id 
                for e in doc_entities
            )
            
            target_present = any(
                e.get("entity_type") == target_entity.entity_type and 
                e.get("entity_id") == target_entity.entity_id 
                for e in doc_entities
            )
            
            if source_present and target_present:
                # Both entities present - perfect for relationship queries
                relationship_scores.append(1.0)
            elif source_present or target_present:
                # Only one entity present - partial relationship fulfillment
                relationship_scores.append(0.2)
            else:
                # Neither entity present - no relationship possible
                relationship_scores.append(0.0)
        
        # Return average relationship score
        if relationship_scores:
            avg_relationship_score = sum(relationship_scores) / len(relationship_scores)
            # For relationship queries, perfect fulfillment gets bonus
            if avg_relationship_score == 1.0:
                return 2.0  # Strong bonus for perfect relationship match
            elif avg_relationship_score > 0:
                return 0.5  # Partial relationship penalty
            else:
                return 0.1  # Heavy penalty for no relationship fulfillment
        
        return 1.0


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