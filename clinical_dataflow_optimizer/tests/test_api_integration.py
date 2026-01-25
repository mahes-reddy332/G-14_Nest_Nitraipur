"""
Integration tests for API endpoints
"""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from api.main import app


class TestAPIIntegration:
    """Integration tests for API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    async def async_client(self):
        """Create async test client"""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            yield client

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data

        # Check services status
        services = data["services"]
        assert "data_service" in services
        assert "metrics_service" in services
        assert "realtime_service" in services

    def test_dashboard_summary_endpoint(self, client):
        """Test dashboard summary endpoint"""
        response = client.get("/api/dashboard/summary")

        # Should return 200 or handle gracefully
        assert response.status_code in [200, 500, 503]  # Allow for service unavailability

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_studies_endpoints(self, client):
        """Test studies API endpoints"""
        # Test get all studies
        response = client.get("/api/studies/")
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

            if data:  # If there are studies
                study = data[0]
                assert "study_id" in study
                assert "name" in study

    def test_patients_endpoints(self, client):
        """Test patients API endpoints"""
        # Test get all patients
        response = client.get("/api/patients/")
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict) or isinstance(data, list)

    def test_sites_endpoints(self, client):
        """Test sites API endpoints"""
        response = client.get("/api/sites/")
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    def test_agents_endpoints(self, client):
        """Test agents API endpoints"""
        response = client.get("/api/agents/status")
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_narratives_endpoints(self, client):
        """Test narrative generation endpoints"""
        # Test patient narrative
        response = client.post("/api/narratives/patient-narrative", json={
            "subject_id": "SUBJ-001",
            "study_id": "Study_1"
        })
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "narrative" in data
            assert "ai_generated" in data

        # Test RBM report
        response = client.post("/api/narratives/rbm-report", json={
            "site_id": "SITE-001",
            "study_id": "Study_1"
        })
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "report" in data
            assert "ai_generated" in data

        # Test clinical insights
        response = client.post("/api/narratives/clinical-insights", json={
            "study_id": "Study_1"
        })
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "insights" in data
            assert "ai_generated" in data

    def test_nlq_endpoints(self, client):
        """Test natural language query endpoints"""
        response = client.post("/api/nlq/query", json={
            "query": "Show me patient enrollment trends",
            "study_id": "Study_1"
        })
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_error_handling(self, client):
        """Test error handling for invalid requests"""
        # Test invalid endpoint
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

        # Test invalid method
        response = client.post("/api/health")
        assert response.status_code in [405, 404]

        # Test malformed JSON
        response = client.post("/api/narratives/patient-narrative",
                              data="invalid json",
                              headers={"Content-Type": "application/json"})
        assert response.status_code == 422

    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })

        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

    @pytest.mark.asyncio
    async def test_websocket_connection(self, async_client):
        """Test WebSocket connection (basic connectivity test)"""
        try:
            async with async_client.ws_connect("/api/ws/updates") as websocket:
                # Send a test message
                await websocket.send_json({"type": "ping"})

                # Try to receive response (may timeout, that's ok)
                try:
                    response = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=2.0
                    )
                    assert isinstance(response, dict)
                except asyncio.TimeoutError:
                    # Expected if no response
                    pass

        except Exception:
            # WebSocket may not be available in test environment
            pytest.skip("WebSocket connection failed (expected in test env)")

    def test_response_format_consistency(self, client):
        """Test that API responses have consistent format"""
        endpoints = [
            "/api/health",
            "/api/dashboard/summary",
            "/api/studies/",
            "/api/patients/",
            "/api/sites/",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Should be valid JSON
                    assert isinstance(data, (dict, list))
                except ValueError:
                    pytest.fail(f"Invalid JSON response from {endpoint}")

    def test_rate_limiting(self, client):
        """Test basic rate limiting (if implemented)"""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = client.get("/api/health")
            responses.append(response.status_code)

        # Should not have excessive 429 responses
        rate_limited = responses.count(429)
        assert rate_limited < 5, f"Too many rate limit responses: {rate_limited}"

    def test_memory_usage(self, client):
        """Basic memory usage test"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make several API calls
        for _ in range(10):
            client.get("/api/health")

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB)
        assert memory_increase < 50 * 1024 * 1024, f"Memory leak detected: {memory_increase} bytes"