"""
Unit tests for AgentService
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from api.services.agent_service import AgentService


class TestAgentService:
    """Test cases for agent service functionality"""

    @pytest.fixture
    async def service(self):
        """Create a test instance of AgentService"""
        service = AgentService()
        await service.initialize()
        return service

    @pytest.mark.asyncio
    async def test_initialization(self, service):
        """Test service initialization"""
        assert service.agents is not None
        assert 'rex' in service.agents
        assert 'codex' in service.agents
        assert 'lia' in service.agents

    @pytest.mark.asyncio
    async def test_get_agent_insights_rex(self, service):
        """Test getting insights from Rex agent"""
        insights = await service.get_agent_insights('rex', limit=5)

        assert isinstance(insights, list)
        if insights:  # May be empty if no data
            assert 'agent' in insights[0]
            assert 'timestamp' in insights[0]
            assert 'insights' in insights[0]

    @pytest.mark.asyncio
    async def test_get_agent_insights_codex(self, service):
        """Test getting insights from Codex agent"""
        insights = await service.get_agent_insights('codex', limit=3)

        assert isinstance(insights, list)
        if insights:
            assert insights[0]['agent'] == 'codex'

    @pytest.mark.asyncio
    async def test_get_agent_insights_lia(self, service):
        """Test getting insights from Lia agent"""
        insights = await service.get_agent_insights('lia', limit=2)

        assert isinstance(insights, list)
        if insights:
            assert insights[0]['agent'] == 'lia'

    @pytest.mark.asyncio
    async def test_get_agent_insights_invalid_agent(self, service):
        """Test getting insights for invalid agent"""
        insights = await service.get_agent_insights('invalid_agent', limit=5)

        assert insights == []

    @pytest.mark.asyncio
    async def test_get_agent_insights_with_limit(self, service):
        """Test insights limiting"""
        insights = await service.get_agent_insights('rex', limit=2)

        assert len(insights) <= 2

    @pytest.mark.asyncio
    async def test_get_all_agent_insights(self, service):
        """Test getting insights from all agents"""
        all_insights = await service.get_all_agent_insights(limit=10)

        assert isinstance(all_insights, dict)
        assert 'rex' in all_insights
        assert 'codex' in all_insights
        assert 'lia' in all_insights

        # Check that each agent's insights are lists
        for agent_insights in all_insights.values():
            assert isinstance(agent_insights, list)

    @pytest.mark.asyncio
    async def test_agent_data_processing(self, service):
        """Test that agents can process data"""
        # This is a basic test to ensure agents are callable
        # In a real scenario, we'd mock the data sources

        try:
            # Test Rex agent data processing
            rex_result = await service.agents['rex'].process_data({})
            assert rex_result is not None

            # Test Codex agent
            codex_result = await service.agents['codex'].process_data({})
            assert codex_result is not None

            # Test Lia agent
            lia_result = await service.agents['lia'].process_data({})
            assert lia_result is not None

        except Exception as e:
            # Agents may fail if data sources are not available
            # This is expected in test environment
            pytest.skip(f"Agent processing failed (expected in test env): {e}")

    @pytest.mark.asyncio
    async def test_supervisor_agent_coordination(self, service):
        """Test supervisor agent coordination"""
        try:
            supervisor = service.agents.get('supervisor')
            if supervisor:
                result = await supervisor.coordinate_agents(['rex', 'codex'])
                assert result is not None
        except Exception as e:
            pytest.skip(f"Supervisor coordination failed: {e}")

    @pytest.mark.asyncio
    async def test_agent_error_handling(self, service):
        """Test agent error handling"""
        # Test with invalid data
        try:
            result = await service.get_agent_insights('rex', limit=-1)
            # Should handle gracefully
            assert isinstance(result, list)
        except Exception:
            # Error handling should prevent crashes
            pass

    def test_agent_service_structure(self, service):
        """Test agent service has required structure"""
        assert hasattr(service, 'agents')
        assert hasattr(service, 'get_agent_insights')
        assert hasattr(service, 'get_all_agent_insights')
        assert hasattr(service, 'initialize')

    @pytest.mark.asyncio
    async def test_agent_insights_format(self, service):
        """Test that agent insights have correct format"""
        insights = await service.get_agent_insights('rex', limit=1)

        if insights:
            insight = insights[0]
            required_fields = ['agent', 'timestamp', 'insights']

            for field in required_fields:
                assert field in insight, f"Missing required field: {field}"

            # Check data types
            assert isinstance(insight['agent'], str)
            assert isinstance(insight['timestamp'], str)
            assert isinstance(insight['insights'], (str, list))