#!/usr/bin/env python3
"""
LLM-based SearchSpec generation from natural language queries.
Converts user questions into structured SearchSpec for hybrid search.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from pydantic import BaseModel

# Import search schema
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from search.schema import SearchSpec, Entity, EntityType, MarketImpact, SearchFilters, SearchBoosts, TimeRange

logger = logging.getLogger(__name__)


class QueryAnalysis(BaseModel):
    """Analysis of natural language query."""
    intent: str  # "price", "news", "analysis", "sentiment", etc.
    entities_mentioned: List[Dict[str, Any]]
    sentiment_filter: Optional[str] = None
    time_scope: Optional[str] = None  # "recent", "today", "week", etc.
    query_complexity: str  # "simple", "moderate", "complex"


class LLMSearchSpecGenerator:
    """Generates SearchSpec from natural language using LLM or rule-based fallback."""
    
    def __init__(self, use_openai: bool = False, api_key: Optional[str] = None):
        """
        Initialize the SearchSpec generator.
        
        Args:
            use_openai: Whether to use OpenAI API (requires API key)
            api_key: OpenAI API key if using OpenAI
        """
        self.use_openai = use_openai
        self.api_key = api_key
        
        # Entity patterns for rule-based fallback
        self.entity_patterns = {
            "tokens": {
                "btc": ["btc", "bitcoin", "bitcoin"],
                "eth": ["eth", "ethereum", "ether"],
                "sol": ["sol", "solana"],
                "ada": ["ada", "cardano"],
                "dot": ["dot", "polkadot"],
                "link": ["link", "chainlink"],
                "matic": ["matic", "polygon"],
                "avax": ["avax", "avalanche"]
            },
            "kols": {
                "elon": ["elon", "musk", "elonmusk", "tesla"],
                "vitalik": ["vitalik", "buterin", "vitalikbuterin"],
                "cz": ["cz", "binance", "changpeng"],
                "michael_saylor": ["saylor", "microstrategy"]
            },
            "events": {
                "halving": ["halving", "halvening", "bitcoin halving"],
                "merger": ["merge", "merger", "ethereum merge"],
                "upgrade": ["upgrade", "fork", "hard fork"]
            },
            "macros": {
                "fed_rate": ["fed", "federal reserve", "interest rate", "rate decision", "fomc"],
                "inflation": ["inflation", "cpi", "consumer price"],
                "gdp": ["gdp", "growth", "economic growth"]
            }
        }
        
        self.sentiment_keywords = {
            "bullish": ["bullish", "pump", "pumping", "moon", "rally", "surge", "gain", "up", "rise", "bull"],
            "bearish": ["bearish", "dump", "dumping", "crash", "drop", "fall", "bear", "down", "decline"],
            "neutral": ["analysis", "technical", "chart", "data", "research"]
        }
        
        self.time_keywords = {
            "today": ["today", "today's"],
            "recent": ["recent", "recently", "latest", "new"],
            "week": ["week", "weekly", "past week", "this week"],
            "month": ["month", "monthly", "past month", "this month"]
        }
    
    async def generate_search_spec(self, natural_query: str) -> SearchSpec:
        """
        Generate SearchSpec from natural language query.
        
        Args:
            natural_query: Natural language search query
            
        Returns:
            SearchSpec object ready for hybrid search
        """
        logger.info(f"Generating SearchSpec for query: {natural_query}")
        
        if self.use_openai and self.api_key:
            try:
                return await self._generate_with_openai(natural_query)
            except Exception as e:
                logger.warning(f"OpenAI generation failed, falling back to rules: {e}")
        
        # Rule-based fallback
        return self._generate_with_rules(natural_query)
    
    def _generate_with_rules(self, natural_query: str) -> SearchSpec:
        """Generate SearchSpec using rule-based approach."""
        query_lower = natural_query.lower()
        
        # Analyze query
        analysis = self._analyze_query(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query_lower)
        
        # Determine sentiment filter
        sentiment_filter = self._extract_sentiment(query_lower)
        
        # Determine time range
        time_range = self._extract_time_range(query_lower)
        
        # Build filters
        filters = SearchFilters()
        if sentiment_filter:
            filters.sentiment = [MarketImpact(sentiment_filter)]
        if time_range:
            filters.time_range = time_range
        
        # Determine boosts based on query intent
        boosts = self._determine_boosts(analysis)
        
        # Clean the original query for search
        search_query = self._clean_query_for_search(natural_query)
        
        return SearchSpec(
            query=search_query,
            entities=entities,
            filters=filters,
            boosts=boosts,
            size=20
        )
    
    def _analyze_query(self, query_lower: str) -> QueryAnalysis:
        """Analyze query intent and complexity."""
        
        # Determine intent
        intent = "general"
        if any(word in query_lower for word in ["price", "pump", "dump", "cost", "value"]):
            intent = "price"
        elif any(word in query_lower for word in ["news", "happening", "update", "event"]):
            intent = "news"
        elif any(word in query_lower for word in ["analysis", "technical", "chart", "predict"]):
            intent = "analysis"
        elif any(word in query_lower for word in ["sentiment", "feeling", "opinion", "think"]):
            intent = "sentiment"
        
        # Extract mentioned entities
        entities_mentioned = []
        for entity_type, patterns in self.entity_patterns.items():
            for entity_id, keywords in patterns.items():
                if any(keyword in query_lower for keyword in keywords):
                    entities_mentioned.append({
                        "type": entity_type[:-1],  # Remove 's' from plural
                        "id": entity_id,
                        "keywords": keywords
                    })
        
        # Determine complexity
        complexity = "simple"
        if len(entities_mentioned) > 2 or len(query_lower.split()) > 10:
            complexity = "complex"
        elif len(entities_mentioned) > 0 or any(word in query_lower for word in ["and", "or", "but", "because"]):
            complexity = "moderate"
        
        return QueryAnalysis(
            intent=intent,
            entities_mentioned=entities_mentioned,
            query_complexity=complexity
        )
    
    def _extract_entities(self, query_lower: str) -> Dict[str, List[Entity]]:
        """Extract entities from query and categorize them."""
        entities = {"must": [], "should": []}
        
        for entity_type, patterns in self.entity_patterns.items():
            for entity_id, keywords in patterns.items():
                if any(keyword in query_lower for keyword in keywords):
                    # Determine entity type enum
                    if entity_type == "tokens":
                        ent_type = EntityType.TOKEN
                    elif entity_type == "kols":
                        ent_type = EntityType.KOL
                    elif entity_type == "events":
                        ent_type = EntityType.EVENT
                    elif entity_type == "macros":
                        ent_type = EntityType.MACRO
                    else:
                        continue
                    
                    # Create entity
                    entity = Entity(
                        entity_type=ent_type,
                        entity_id=entity_id,
                        name=keywords[0].title(),  # Use first keyword as name
                        confidence=0.8
                    )
                    
                    # Primary entities (tokens, major KOLs) are "must"
                    if entity_type in ["tokens"] or entity_id in ["elon", "vitalik"]:
                        entities["must"].append(entity)
                    else:
                        entities["should"].append(entity)
        
        return entities
    
    def _extract_sentiment(self, query_lower: str) -> Optional[str]:
        """Extract sentiment filter from query."""
        for sentiment, keywords in self.sentiment_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return sentiment
        return None
    
    def _extract_time_range(self, query_lower: str) -> Optional[TimeRange]:
        """Extract time range from query."""
        for time_scope, keywords in self.time_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if time_scope == "today":
                    return TimeRange(days_back=1)
                elif time_scope == "recent":
                    return TimeRange(days_back=3)
                elif time_scope == "week":
                    return TimeRange(days_back=7)
                elif time_scope == "month":
                    return TimeRange(days_back=30)
        
        return None
    
    def _determine_boosts(self, analysis: QueryAnalysis) -> SearchBoosts:
        """Determine boost weights based on query analysis."""
        
        if analysis.intent == "news":
            # News queries prioritize recency
            return SearchBoosts(
                recency=0.12,
                authority=0.04,
                engagement=0.04
            )
        elif analysis.intent == "sentiment":
            # Sentiment queries prioritize engagement
            return SearchBoosts(
                recency=0.06,
                authority=0.04,
                engagement=0.10
            )
        elif analysis.intent == "analysis":
            # Analysis queries prioritize authority
            return SearchBoosts(
                recency=0.04,
                authority=0.10,
                engagement=0.06
            )
        else:
            # Default balanced weights
            return SearchBoosts()
    
    def _clean_query_for_search(self, original_query: str) -> str:
        """Clean the original query for search terms."""
        # Remove question words and common phrases
        stopwords = ["why", "what", "how", "when", "where", "is", "are", "the", "a", "an"]
        
        words = original_query.lower().split()
        cleaned_words = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Keep minimum of 2 words
        if len(cleaned_words) < 2:
            return original_query
        
        return " ".join(cleaned_words)
    
    async def _generate_with_openai(self, natural_query: str) -> SearchSpec:
        """Generate SearchSpec using OpenAI (placeholder for future implementation)."""
        # TODO: Implement OpenAI integration with function calling
        # This would use OpenAI's function calling feature to generate SearchSpec
        logger.info("OpenAI integration not yet implemented, using rule-based fallback")
        return self._generate_with_rules(natural_query)


def main():
    """Test SearchSpec generation."""
    import asyncio
    
    async def test_queries():
        generator = LLMSearchSpecGenerator()
        
        test_queries = [
            "Why is BTC pumping?",
            "What's the latest news on Ethereum?",
            "Elon Musk thoughts on Bitcoin",
            "Fed rate decision impact on crypto",
            "Recent bearish sentiment on Solana"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            search_spec = await generator.generate_search_spec(query)
            print(f"📝 SearchSpec:")
            print(f"   Query: '{search_spec.query}'")
            print(f"   Entities: {[(e.entity_type, e.name) for e in search_spec.entities.get('must', [])]}")
            print(f"   Filters: sentiment={search_spec.filters.sentiment}, time={search_spec.filters.time_range}")
            print(f"   Boosts: recency={search_spec.boosts.recency}, authority={search_spec.boosts.authority}")
    
    asyncio.run(test_queries())


if __name__ == "__main__":
    main()