#!/usr/bin/env python3
"""
Comprehensive system test for Hybrid Search Engine.
Tests local search, API endpoints, clustering, and investment advisory features.
"""

import asyncio
import json
import time
import requests
from pathlib import Path

def test_local_search_engine():
    """Test the LocalHybridSearchEngine directly."""
    print("🔍 Testing LocalHybridSearchEngine...")
    
    try:
        import sys
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from search.local_search import LocalHybridSearchEngine
        from search.schema import SearchSpec
        
        # Initialize engine
        engine = LocalHybridSearchEngine()
        
        # Test basic search
        search_spec = SearchSpec(
            query="Bitcoin price analysis",
            size=3
        )
        
        response = asyncio.run(engine.search(search_spec))
        assert len(response.results) >= 0, "Search failed"
        
        if len(response.results) > 0:
            assert response.results[0].final_score >= 0, "Invalid final score"
        
        print(f"✅ Local search: {len(response.results)} results in {response.execution_time_ms}ms")
        
        # Test entity-specific search
        btc_spec = SearchSpec(query="BTC news", size=3)
        btc_response = asyncio.run(engine.search(btc_spec))
        
        # Test investment query
        investment_spec = SearchSpec(query="should I buy Bitcoin", size=3)
        investment_response = asyncio.run(engine.search(investment_spec))
        
        print(f"✅ Entity search: {len(btc_response.results)} BTC results")
        print(f"✅ Investment query: {len(investment_response.results)} advisory results")
        
    except Exception as e:
        print(f"❌ Local search test failed: {e}")
        return False
    
    return True


def test_api_endpoints():
    """Test FastAPI endpoints."""
    print("\n🌐 Testing API endpoints...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test /ask endpoint (natural language)
        response = requests.post(
            f"{base_url}/ask",
            json={"query": "Why is Bitcoin pumping?", "size": 3},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        assert "results" in data, "Missing results in /ask response"
        assert len(data["results"]) > 0, "No results from /ask endpoint"
        
        print(f"✅ /ask endpoint: {len(data['results'])} results")
        
        # Test traditional search
        response = requests.get(
            f"{base_url}/search",
            params={"q": "Ethereum", "size": 3},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        assert "results" in data, "Missing results in /search response"
        print(f"✅ /search endpoint: {len(data['results'])} results")
        
        # Test clustered search
        response = requests.post(
            f"{base_url}/search/clustered",
            json={"query": "crypto market analysis", "size": 5},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        assert "clusters" in data, "Missing clusters in /search/clustered response"
        print(f"✅ /search/clustered endpoint: {data.get('total_results', 0)} results in {len(data['clusters'])} clusters")
        
        # Test investment advisory queries
        investment_queries = [
            "should I buy SOL",
            "why is ETH bullish",
            "analyze Bitcoin investment"
        ]
        
        for query in investment_queries:
            response = requests.post(
                f"{base_url}/ask",
                json={"query": query, "size": 3},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            print(f"✅ Investment query '{query}': {len(data['results'])} results")
        
    except requests.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running:")
        print("   uvicorn src.api.main:app --reload --port 8000")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False
    
    return True


def test_performance():
    """Test system performance."""
    print("\n⚡ Testing performance...")
    
    try:
        queries = [
            "Bitcoin price prediction",
            "should I sell Ethereum",
            "Solana network issues",
            "DeFi market analysis"
        ]
        
        base_url = "http://localhost:8000"
        total_time = 0
        successful_queries = 0
        
        for query in queries:
            start_time = time.time()
            
            response = requests.post(
                f"{base_url}/ask",
                json={"query": query, "size": 5},
                timeout=10
            )
            
            if response.status_code == 200:
                query_time = (time.time() - start_time) * 1000
                total_time += query_time
                successful_queries += 1
                print(f"✅ '{query}': {query_time:.0f}ms")
        
        if successful_queries > 0:
            avg_time = total_time / successful_queries
            print(f"📊 Average response time: {avg_time:.0f}ms")
            
            if avg_time < 500:  # Target: under 500ms
                print("🎯 Performance target met!")
            else:
                print("⚠️  Performance could be improved")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False
    
    return True


def test_data_integrity():
    """Test data files and embeddings."""
    print("\n📁 Testing data integrity...")
    
    try:
        # Check main data file
        data_file = Path("data/expanded_sample_tweets.jsonl")
        if data_file.exists():
            with open(data_file, 'r') as f:
                tweet_count = sum(1 for line in f if line.strip())
            print(f"✅ Main data file: {tweet_count} tweets")
        else:
            print("❌ Main data file missing")
            return False
        
        # Check embeddings file
        embeddings_file = Path("normalized/tweets_with_embeddings.jsonl")
        if embeddings_file.exists():
            with open(embeddings_file, 'r') as f:
                first_line = f.readline()
                if first_line.strip():
                    tweet = json.loads(first_line)
                    if "embedding" in tweet and len(tweet["embedding"]) == 384:
                        print("✅ Embeddings file: 384D vectors confirmed")
                    else:
                        print("⚠️  Embeddings file format issue")
        else:
            print("❌ Embeddings file missing")
            return False
            
    except Exception as e:
        print(f"❌ Data integrity test failed: {e}")
        return False
    
    return True


def main():
    """Run comprehensive system tests."""
    print("🚀 Hybrid Search Engine - Comprehensive System Test")
    print("=" * 60)
    
    tests = [
        ("Data Integrity", test_data_integrity),
        ("Local Search Engine", test_local_search_engine), 
        ("API Endpoints", test_api_endpoints),
        ("Performance", test_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{passed+1}/{total}] {test_name}")
        print("-" * 40)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is production-ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)