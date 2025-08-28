#!/usr/bin/env python3
"""
Basic API tests for the hybrid search engine.
"""

import pytest
import requests
import json


class TestAPI:
    """Test the search API endpoints."""
    
    BASE_URL = "http://localhost:8000"
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_search_endpoint(self):
        """Test basic search endpoint."""
        response = requests.get(f"{self.BASE_URL}/search?q=bitcoin&size=3")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= 3
    
    def test_ask_endpoint(self):
        """Test natural language endpoint."""
        payload = {"query": "Bitcoin pumping", "size": 3}
        response = requests.post(
            f"{self.BASE_URL}/ask",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= 3
        
    def test_clustered_search(self):
        """Test clustered search endpoint."""
        payload = {"query": "crypto market", "size": 5}
        response = requests.post(
            f"{self.BASE_URL}/search/clustered",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "clusters" in data or "results" in data


if __name__ == "__main__":
    # Run tests directly
    import subprocess
    import sys
    
    print("Running API tests...")
    print("Make sure the API server is running: uvicorn src.api.main:app --port 8000")
    
    # Simple test runner
    test_instance = TestAPI()
    
    try:
        test_instance.test_health_endpoint()
        print("✅ Health endpoint test passed")
        
        test_instance.test_search_endpoint() 
        print("✅ Search endpoint test passed")
        
        test_instance.test_ask_endpoint()
        print("✅ Ask endpoint test passed")
        
        test_instance.test_clustered_search()
        print("✅ Clustered search test passed")
        
        print("\n🎉 All API tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)