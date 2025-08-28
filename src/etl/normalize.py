#!/usr/bin/env python3
"""
Tweet normalization and entity extraction pipeline.
Converts raw tweets into clean, structured documents ready for indexing.
"""

import re
import json
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
import jsonlines
from tqdm import tqdm


@dataclass
class Entity:
    """Extracted entity from tweet content."""
    entity_type: str  # token, project, kol, event, macro
    entity_id: str
    name: str
    symbol: Optional[str] = None
    confidence: float = 1.0


@dataclass
class NormalizedTweet:
    """Normalized tweet structure ready for indexing."""
    tweet_id: str
    clean_content: str
    original_content: str
    created_at_iso: str
    day_bucket: str
    language: str
    entities: List[Entity]
    market_impact: str  # bullish, bearish, neutral
    has_tokens: bool
    has_projects: bool
    has_kols: bool
    has_events: bool
    has_macros: bool
    source_handle: Optional[str] = None
    source_followers: Optional[int] = None
    engagement_score: Optional[int] = None
    is_vectorized: bool = False


class TweetNormalizer:
    """Handles tweet cleaning and entity extraction."""
    
    def __init__(self, entity_registry_path: str):
        """Initialize with entity registry for lookups."""
        self.entity_registry = self._load_entity_registry(entity_registry_path)
        self._compile_patterns()
    
    def _load_entity_registry(self, registry_path: str) -> Dict[str, Dict]:
        """Load entity registry from YAML config."""
        with open(registry_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _compile_patterns(self):
        """Compile regex patterns for entity detection."""
        # Cashtag pattern: $BTC, $ETH, etc.
        self.cashtag_pattern = re.compile(r'\$([A-Z]{2,10})\b', re.IGNORECASE)
        
        # URL pattern for removal
        self.url_pattern = re.compile(
            r'https?://[^\s]+|www\.[^\s]+|\b[^\s]+\.(com|org|net|io|co)\b',
            re.IGNORECASE
        )
        
        # Handle/mention pattern
        self.handle_pattern = re.compile(r'@(\w+)', re.IGNORECASE)
        
        # Emoji pattern (basic)
        self.emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F700-\U0001F77F]'
        )
        
        # Build lookup tables for faster entity matching
        self._build_entity_lookups()
    
    def _build_entity_lookups(self):
        """Build efficient lookup tables from entity registry."""
        self.symbol_to_entity = {}
        self.name_to_entity = {}
        self.handle_to_entity = {}
        
        for entity_type, entities in self.entity_registry.items():
            for entity_key, entity_data in entities.items():
                # Symbol lookup (for tokens/projects)
                if 'symbol' in entity_data:
                    symbol = entity_data['symbol'].upper()
                    self.symbol_to_entity[symbol] = {
                        'type': entity_type.rstrip('s'),  # tokens -> token
                        'data': entity_data
                    }
                
                # Name lookup (fuzzy matching)
                name_normalized = entity_data['name'].lower()
                self.name_to_entity[name_normalized] = {
                    'type': entity_type.rstrip('s'),
                    'data': entity_data
                }
                
                # Handle lookup (for KOLs)
                if 'handle' in entity_data:
                    handle = entity_data['handle'].replace('@', '').lower()
                    self.handle_to_entity[handle] = {
                        'type': entity_type.rstrip('s'),
                        'data': entity_data
                    }
    
    def clean_content(self, raw_content: str) -> str:
        """Clean tweet content for BM25 indexing."""
        content = raw_content.strip()
        
        # Remove URLs
        content = self.url_pattern.sub('', content)
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Convert to lowercase for search (keep original case for display)
        content = content.lower()
        
        # Remove emojis for cleaner text matching
        content = self.emoji_pattern.sub('', content)
        
        return content.strip()
    
    def extract_entities(self, content: str) -> List[Entity]:
        """Extract and resolve entities from tweet content."""
        entities = []
        seen_entities = set()  # Avoid duplicates
        
        # Extract cashtags (highest confidence)
        cashtags = self.cashtag_pattern.findall(content)
        for cashtag in cashtags:
            symbol = cashtag.upper()
            if symbol in self.symbol_to_entity:
                entity_info = self.symbol_to_entity[symbol]
                entity_key = f"{entity_info['type']}:{entity_info['data']['id']}"
                if entity_key not in seen_entities:
                    entities.append(Entity(
                        entity_type=entity_info['type'],
                        entity_id=entity_info['data']['id'],
                        name=entity_info['data']['name'],
                        symbol=entity_info['data'].get('symbol'),
                        confidence=1.0
                    ))
                    seen_entities.add(entity_key)
        
        # Extract handles
        handles = self.handle_pattern.findall(content)
        for handle in handles:
            handle_normalized = handle.lower()
            if handle_normalized in self.handle_to_entity:
                entity_info = self.handle_to_entity[handle_normalized]
                entity_key = f"{entity_info['type']}:{entity_info['data']['id']}"
                if entity_key not in seen_entities:
                    entities.append(Entity(
                        entity_type=entity_info['type'],
                        entity_id=entity_info['data']['id'],
                        name=entity_info['data']['name'],
                        symbol=entity_info['data'].get('symbol'),
                        confidence=0.9
                    ))
                    seen_entities.add(entity_key)
        
        # Fuzzy name matching (lower confidence)
        content_lower = content.lower()
        for name, entity_info in self.name_to_entity.items():
            if name in content_lower and len(name) > 3:  # Avoid short false matches
                entity_key = f"{entity_info['type']}:{entity_info['data']['id']}"
                if entity_key not in seen_entities:
                    entities.append(Entity(
                        entity_type=entity_info['type'],
                        entity_id=entity_info['data']['id'],
                        name=entity_info['data']['name'],
                        symbol=entity_info['data'].get('symbol'),
                        confidence=0.7
                    ))
                    seen_entities.add(entity_key)
        
        return entities
    
    def infer_market_impact(self, content: str, entities: List[Entity]) -> str:
        """Infer market sentiment from content and entities."""
        content_lower = content.lower()
        
        # Bullish indicators
        bullish_terms = [
            'pump', 'rally', 'moon', 'bullish', 'buy', 'long', 'hodl',
            'green', 'up', 'rise', 'surge', 'breakout', 'ath', 'new high'
        ]
        
        # Bearish indicators  
        bearish_terms = [
            'dump', 'crash', 'bear', 'bearish', 'sell', 'short', 'red',
            'down', 'fall', 'drop', 'liquidation', 'hack', 'exploit', 'rekt'
        ]
        
        bullish_score = sum(1 for term in bullish_terms if term in content_lower)
        bearish_score = sum(1 for term in bearish_terms if term in content_lower)
        
        # Entity-based signals
        for entity in entities:
            if entity.entity_type == 'event':
                event_data = next(
                    (e for e in self.entity_registry.get('events', {}).values() 
                     if e.get('id') == entity.entity_id), 
                    None
                )
                if event_data and event_data.get('impact') == 'bullish':
                    bullish_score += 2
                elif event_data and event_data.get('impact') == 'bearish':
                    bearish_score += 2
        
        if bullish_score > bearish_score:
            return 'bullish'
        elif bearish_score > bullish_score:
            return 'bearish'
        else:
            return 'neutral'
    
    def normalize_tweet(self, raw_tweet: Dict[str, Any]) -> NormalizedTweet:
        """Convert raw tweet to normalized structure."""
        # Extract basic fields
        tweet_id = str(raw_tweet.get('id', raw_tweet.get('tweet_id', '')))
        original_content = raw_tweet.get('text', raw_tweet.get('content', ''))
        clean_content = self.clean_content(original_content)
        
        # Parse timestamp
        created_at = raw_tweet.get('created_at', raw_tweet.get('timestamp', ''))
        if isinstance(created_at, str):
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except:
                dt = datetime.now()
        else:
            dt = datetime.fromtimestamp(created_at) if created_at else datetime.now()
        
        created_at_iso = dt.isoformat()
        day_bucket = dt.strftime('%Y-%m-%d')
        
        # Extract entities
        entities = self.extract_entities(original_content)
        
        # Determine entity flags
        entity_types = {entity.entity_type for entity in entities}
        has_tokens = 'token' in entity_types
        has_projects = 'project' in entity_types
        has_kols = 'kol' in entity_types
        has_events = 'event' in entity_types
        has_macros = 'macro' in entity_types
        
        # Infer market impact
        market_impact = self.infer_market_impact(clean_content, entities)
        
        # Extract engagement metrics
        engagement_score = (
            raw_tweet.get('likes', 0) + 
            raw_tweet.get('retweets', 0) + 
            raw_tweet.get('replies', 0)
        )
        
        return NormalizedTweet(
            tweet_id=tweet_id,
            clean_content=clean_content,
            original_content=original_content,
            created_at_iso=created_at_iso,
            day_bucket=day_bucket,
            language=raw_tweet.get('lang', 'en'),
            entities=entities,
            market_impact=market_impact,
            has_tokens=has_tokens,
            has_projects=has_projects,
            has_kols=has_kols,
            has_events=has_events,
            has_macros=has_macros,
            source_handle=raw_tweet.get('user', {}).get('username'),
            source_followers=raw_tweet.get('user', {}).get('followers_count'),
            engagement_score=engagement_score
        )


def main():
    """CLI entry point for tweet normalization."""
    parser = argparse.ArgumentParser(description='Normalize tweets for search indexing')
    parser.add_argument('--input', required=True, help='Input JSONL file path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', default='config/entities.yml', help='Entity registry path')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Initialize normalizer
    normalizer = TweetNormalizer(args.config)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process tweets
    output_file = output_dir / 'normalized_tweets.jsonl'
    
    with jsonlines.open(args.input, 'r') as reader, \
         jsonlines.open(output_file, 'w') as writer:
        
        batch = []
        total_processed = 0
        
        for raw_tweet in tqdm(reader, desc="Normalizing tweets"):
            try:
                normalized_tweet = normalizer.normalize_tweet(raw_tweet)
                batch.append(asdict(normalized_tweet))
                
                if len(batch) >= args.batch_size:
                    writer.write_all(batch)
                    total_processed += len(batch)
                    batch = []
                    
            except Exception as e:
                print(f"Error processing tweet {raw_tweet.get('id', 'unknown')}: {e}")
                continue
        
        # Write remaining batch
        if batch:
            writer.write_all(batch)
            total_processed += len(batch)
    
    print(f"Processed {total_processed} tweets -> {output_file}")


if __name__ == '__main__':
    main()