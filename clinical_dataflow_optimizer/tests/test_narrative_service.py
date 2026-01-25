"""
Unit tests for AINarrativeService
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from api.services.narrative_service import AINarrativeService


class TestAINarrativeService:
    """Test cases for AI-powered narrative service"""

    @pytest.fixture
    async def service(self):
        """Create a test instance of AINarrativeService"""
        service = AINarrativeService()
        # Mock the LLM to avoid requiring OpenAI API key
        service.llm = AsyncMock()
        service._initialized = True
        return service

    @pytest.mark.asyncio
    async def test_initialization_without_openai_key(self):
        """Test service initialization when OpenAI key is not available"""
        with patch.dict('os.environ', {}, clear=True):
            service = AINarrativeService()
            await service.initialize()
            assert service.llm is None
            assert not service._initialized

    @pytest.mark.asyncio
    async def test_generate_patient_narrative_ai_success(self, service):
        """Test successful AI-powered patient narrative generation"""
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "Patient shows stable condition with no adverse events."
        service.llm.ainvoke = AsyncMock(return_value=mock_response)

        # Mock data
        patient_data = {
            'subject_id': 'SUBJ-001',
            'age': 45,
            'gender': 'F',
            'site_id': 'SITE-001'
        }
        safety_data = [{'event': 'Headache', 'severity': 'mild'}]
        context = {'visit_date': '2024-01-15'}

        result = await service.generate_patient_narrative(
            patient_data, safety_data, [], [], context
        )

        assert result['generated_by'] == 'ai'
        assert 'insights' in result
        assert 'key_findings' in result
        assert 'recommendations' in result
        assert result['confidence'] > 0

    @pytest.mark.asyncio
    async def test_generate_patient_narrative_fallback(self, service):
        """Test fallback narrative generation when AI is unavailable"""
        service.llm = None  # Simulate no AI available

        patient_data = {'subject_id': 'SUBJ-001'}
        result = await service.generate_patient_narrative(patient_data, [], [], [], {})

        assert result['generated_by'] == 'template'
        assert 'insights' in result
        assert result['confidence'] < 0.6  # Lower confidence for fallback

    @pytest.mark.asyncio
    async def test_generate_rbm_report_ai_success(self, service):
        """Test successful AI-powered RBM report generation"""
        mock_response = Mock()
        mock_response.content = "Site performance is excellent with no critical issues."
        service.llm.ainvoke = AsyncMock(return_value=mock_response)

        site_data = {'site_id': 'SITE-001', 'investigator': 'Dr. Smith'}
        metrics = {'enrollment_rate': 0.85}
        issues = [{'description': 'Minor documentation issue', 'severity': 'low'}]
        achievements = [{'description': 'On-time enrollment target met'}]

        result = await service.generate_rbm_report(site_data, metrics, issues, achievements)

        assert result['generated_by'] == 'ai'
        assert 'report' in result
        assert 'risk_categories' in result
        assert 'prioritized_issues' in result

    @pytest.mark.asyncio
    async def test_generate_clinical_insights_ai_success(self, service):
        """Test successful AI-powered clinical insights generation"""
        mock_response = Mock()
        mock_response.content = "Study shows promising efficacy results with manageable safety profile."
        service.llm.ainvoke = AsyncMock(return_value=mock_response)

        study_data = {'study_id': 'STUDY-001', 'phase': 'Phase 2'}
        safety_data = [{'event': 'Nausea', 'frequency': 0.15}]
        efficacy_data = [{'endpoint': 'Primary', 'result': 'Positive'}]
        operational_data = [{'metric': 'Retention', 'value': 0.92}]

        result = await service.generate_clinical_insights(
            study_data, safety_data, efficacy_data, operational_data
        )

        assert result['generated_by'] == 'ai'
        assert 'insights' in result
        assert 'key_findings' in result
        assert 'recommendations' in result
        assert result['evidence_based'] is True

    @pytest.mark.asyncio
    async def test_ai_generation_error_handling(self, service):
        """Test error handling during AI generation"""
        # Mock LLM to raise an exception
        service.llm.ainvoke = AsyncMock(side_effect=Exception("AI service error"))

        patient_data = {'subject_id': 'SUBJ-001'}
        result = await service.generate_patient_narrative(patient_data, [], [], [], {})

        # Should fall back to template generation
        assert result['generated_by'] == 'template'
        assert 'insights' in result

    def test_extract_key_findings(self, service):
        """Test key findings extraction from AI response"""
        response_text = """
        Key findings from the analysis:
        1. Patient shows good response to treatment
        2. No serious adverse events reported
        3. Laboratory values within normal range

        Overall assessment: Patient is doing well.
        """

        findings = service._extract_key_findings(response_text)

        assert len(findings) > 0
        assert any('response' in finding.lower() for finding in findings)
        assert any('adverse' in finding.lower() for finding in findings)

    def test_extract_recommendations(self, service):
        """Test recommendations extraction from AI response"""
        response_text = """
        Recommendations:
        - Continue current treatment regimen
        - Schedule follow-up visit in 4 weeks
        - Monitor liver function tests

        Next steps: Regular monitoring required.
        """

        recommendations = service._extract_recommendations(response_text)

        assert len(recommendations) > 0
        assert any('treatment' in rec.lower() for rec in recommendations)
        assert any('follow-up' in rec.lower() for rec in recommendations)

    def test_format_patient_data(self, service):
        """Test patient data formatting for AI prompts"""
        data = {
            'subject_id': 'SUBJ-001',
            'age': 45,
            'gender': 'F',
            'site_id': 'SITE-001',
            'enrollment_date': '2024-01-01'
        }

        formatted = service._format_patient_data(data)

        assert 'Subject ID: SUBJ-001' in formatted
        assert 'Age: 45' in formatted
        assert 'Gender: F' in formatted
        assert 'Site: SITE-001' in formatted

    def test_format_adverse_events(self, service):
        """Test adverse events formatting for AI prompts"""
        events = [
            {'preferred_term': 'Headache', 'severity': 'mild', 'start_date': '2024-01-10'},
            {'preferred_term': 'Nausea', 'severity': 'moderate', 'start_date': '2024-01-12'}
        ]

        formatted = service._format_adverse_events(events)

        assert 'Headache' in formatted
        assert 'Nausea' in formatted
        assert 'mild' in formatted
        assert 'moderate' in formatted

    def test_format_empty_adverse_events(self, service):
        """Test formatting when no adverse events are present"""
        formatted = service._format_adverse_events([])
        assert 'No adverse events reported' in formatted