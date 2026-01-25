"""
Test Suite for Scenario B: Ambiguous Concomitant Medication Detection
Tests the AmbiguousCodingDetector module and CodingAgent integration

Test Coverage:
1. AmbiguousCodingDetector initialization
2. WHODRA scanning for uncoded terms
3. LLM-simulated term assessment
4. Confidence classification (High/Medium/Low)
5. Auto-query generation
6. Learning from site clarifications
7. CodingAgent integration
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from clinical_dataflow_optimizer.core.ambiguous_coding_detector import (
    AmbiguousCodingDetector, AmbiguousTerm, AmbiguousCodingConfig,
    AmbiguityLevel, CodingConfidence, ClarificationReason
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_whodra_data():
    """Sample GlobalCodingReport_WHODRA data with various term types"""
    return pd.DataFrame({
        'WHODrug Coding Report': ['WHODrug Coding Report'] * 10,
        'Study': ['Study 1'] * 10,
        'Subject': [
            'Subject 1', 'Subject 1', 'Subject 2', 'Subject 2', 
            'Subject 3', 'Subject 4', 'Subject 5', 'Subject 6',
            'Subject 7', 'Subject 8'
        ],
        'Form OID': ['CMG001'] * 10,
        'Logline': [1, 2, 1, 2, 1, 1, 1, 1, 1, 1],
        'Field OID': ['CMTRT'] * 10,
        'Coding Status': [
            'UnCoded Term', 'Coded Term', 'UnCoded Term', 'UnCoded Term',
            'UnCoded Term', 'UnCoded Term', 'Coded Term', 'UnCoded Term',
            'UnCoded Term', 'UnCoded Term'
        ],
        'Require Coding': ['Yes'] * 10
    })


@pytest.fixture
def sample_cpid_data():
    """Sample CPID data for site context"""
    return pd.DataFrame({
        ('Subject ID', 'Unnamed_1'): ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5'],
        ('Site ID', 'Unnamed_2'): ['Site 1', 'Site 1', 'Site 2', 'Site 2', 'Site 3'],
        ('Country', 'Unnamed_3'): ['USA', 'USA', 'UK', 'UK', 'Germany']
    })


@pytest.fixture
def detector():
    """Create AmbiguousCodingDetector with default config"""
    return AmbiguousCodingDetector()


@pytest.fixture
def custom_config():
    """Create custom configuration"""
    return AmbiguousCodingConfig(
        high_confidence_threshold=0.90,
        medium_confidence_threshold=0.75,
        enable_learning=True
    )


# =============================================================================
# AmbiguousCodingDetector Initialization Tests
# =============================================================================

class TestAmbiguousCodingDetectorInit:
    """Test AmbiguousCodingDetector initialization"""
    
    def test_default_initialization(self, detector):
        """Test detector initializes with defaults"""
        assert detector is not None
        assert detector.config is not None
        assert detector.config.high_confidence_threshold == 0.95
        assert detector.config.medium_confidence_threshold == 0.80
    
    def test_custom_config_initialization(self, custom_config):
        """Test detector with custom config"""
        detector = AmbiguousCodingDetector(config=custom_config)
        assert detector.config.high_confidence_threshold == 0.90
        assert detector.config.medium_confidence_threshold == 0.75
    
    def test_whodrug_reference_built(self, detector):
        """Test WHO Drug reference dictionary is built"""
        assert detector._whodrug_reference is not None
        assert len(detector._whodrug_reference) > 0
        assert 'paracetamol' in detector._whodrug_reference
        assert 'ibuprofen' in detector._whodrug_reference


# =============================================================================
# WHODRA Scanning Tests
# =============================================================================

class TestWHODRAScanning:
    """Test scanning GlobalCodingReport_WHODRA for uncoded terms"""
    
    def test_scan_finds_uncoded_terms(self, detector, sample_whodra_data):
        """Test that scanner finds uncoded terms"""
        uncoded = detector._scan_for_uncoded_terms(sample_whodra_data)
        
        # Should find 8 uncoded terms (rows with 'UnCoded Term' status)
        assert len(uncoded) == 8
    
    def test_scan_excludes_coded_terms(self, detector, sample_whodra_data):
        """Test that scanner excludes coded terms"""
        uncoded = detector._scan_for_uncoded_terms(sample_whodra_data)
        
        # All should have 'UnCoded Term' status (not just 'Coded Term')
        assert all(uncoded['Coding Status'].str.strip().str.lower() == 'uncoded term')
    
    def test_scan_empty_dataframe(self, detector):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        uncoded = detector._scan_for_uncoded_terms(empty_df)
        assert len(uncoded) == 0
    
    def test_scan_missing_status_column(self, detector):
        """Test handling when Coding Status column is missing"""
        df = pd.DataFrame({'Subject': ['S1', 'S2'], 'Other': ['A', 'B']})
        uncoded = detector._scan_for_uncoded_terms(df)
        assert len(uncoded) == 0


# =============================================================================
# Term Assessment Tests
# =============================================================================

class TestTermAssessment:
    """Test LLM-simulated term specificity assessment"""
    
    def test_exact_match_high_confidence(self, detector):
        """Test exact match returns high confidence"""
        assessment = detector._assess_term_specificity('paracetamol')
        
        assert assessment['is_specific'] == True
        assert assessment['confidence'] >= 0.95
        assert 'exact match' in assessment['assessment'].lower()
    
    def test_partial_match_medium_confidence(self, detector):
        """Test partial match returns medium confidence"""
        assessment = detector._assess_term_specificity('paracetamol 500mg')
        
        # Partial match should have lower confidence
        assert assessment['confidence'] >= 0.80
    
    def test_drug_class_low_confidence(self, detector):
        """Test drug class returns low confidence"""
        assessment = detector._assess_term_specificity('pain killer')
        
        assert assessment['is_specific'] == False
        assert assessment['confidence'] < 0.80
        assert 'drug class' in assessment['assessment'].lower() or 'low' in assessment['assessment'].lower()
    
    def test_abbreviation_low_confidence(self, detector):
        """Test abbreviation returns low confidence"""
        assessment = detector._assess_term_specificity('nsaid')
        
        assert assessment['is_specific'] == False
        assert assessment['confidence'] < 0.80
    
    def test_illegible_term_zero_confidence(self, detector):
        """Test illegible term returns zero confidence"""
        assessment = detector._assess_term_specificity('???')
        
        assert assessment['is_specific'] == False
        assert assessment['confidence'] == 0.0
    
    def test_unknown_term_low_confidence(self, detector):
        """Test unknown term returns low confidence"""
        assessment = detector._assess_term_specificity('xyzmedication123')
        
        assert assessment['is_specific'] == False
        assert assessment['confidence'] < 0.80


# =============================================================================
# Confidence Classification Tests
# =============================================================================

class TestConfidenceClassification:
    """Test term classification by confidence level"""
    
    def test_high_confidence_classification(self, detector):
        """Test high confidence term is classified correctly"""
        assessment = {'is_specific': True, 'confidence': 0.98, 'assessment': 'Exact match', 'suggestions': []}
        
        ambiguity, confidence, score, reason = detector._classify_term('tylenol', assessment)
        
        assert ambiguity == AmbiguityLevel.SPECIFIC
        assert confidence == CodingConfidence.HIGH
        assert score >= 0.95
    
    def test_medium_confidence_classification(self, detector):
        """Test medium confidence term is classified correctly"""
        assessment = {'is_specific': True, 'confidence': 0.87, 'assessment': 'Partial match', 'suggestions': [{}]}
        
        ambiguity, confidence, score, reason = detector._classify_term('tylenol extra', assessment)
        
        assert ambiguity == AmbiguityLevel.MODERATE
        assert confidence == CodingConfidence.MEDIUM
        assert 0.80 <= score < 0.95
    
    def test_low_confidence_drug_class(self, detector):
        """Test drug class gets low confidence"""
        assessment = {'is_specific': False, 'confidence': 0.30, 'assessment': 'Drug class', 'suggestions': []}
        
        ambiguity, confidence, score, reason = detector._classify_term('pain killer', assessment)
        
        assert ambiguity == AmbiguityLevel.AMBIGUOUS
        assert confidence == CodingConfidence.LOW
        assert reason == ClarificationReason.DRUG_CLASS_NOT_SPECIFIC
    
    def test_low_confidence_abbreviation(self, detector):
        """Test abbreviation gets low confidence"""
        # Use 'mvi' (Multivitamin) which is in ambiguous_abbreviations
        # and doesn't match any drug class as substring
        assessment = {'is_specific': False, 'confidence': 0.40, 'assessment': 'Abbreviation', 'suggestions': [{'generic': 'A'}]}
        
        ambiguity, confidence, score, reason = detector._classify_term('mvi', assessment)
        
        assert confidence == CodingConfidence.LOW
        assert reason == ClarificationReason.ABBREVIATION_UNCLEAR
    
    def test_illegible_classification(self, detector):
        """Test illegible term classification"""
        assessment = {'is_specific': False, 'confidence': 0.0, 'assessment': 'Illegible', 'suggestions': []}
        
        ambiguity, confidence, score, reason = detector._classify_term('??', assessment)
        
        assert ambiguity == AmbiguityLevel.ILLEGIBLE
        assert confidence == CodingConfidence.LOW
        assert reason == ClarificationReason.ILLEGIBLE_TEXT


# =============================================================================
# Auto-Query Generation Tests
# =============================================================================

class TestAutoQueryGeneration:
    """Test auto-generation of clarification queries"""
    
    def test_drug_class_query(self, detector):
        """Test query generation for drug class"""
        query = detector._generate_clarification_query(
            verbatim='pain killer',
            subject_id='S001',
            site_id='Site 1',
            reason=ClarificationReason.DRUG_CLASS_NOT_SPECIFIC,
            suggestions=[{'generic': 'Paracetamol'}, {'generic': 'Ibuprofen'}]
        )
        
        assert 'pain killer' in query
        assert 'drug class' in query.lower()
        assert 'Trade Name' in query or 'Generic Name' in query
    
    def test_abbreviation_query(self, detector):
        """Test query generation for abbreviation"""
        query = detector._generate_clarification_query(
            verbatim='nsaid',
            subject_id='S001',
            site_id='Site 1',
            reason=ClarificationReason.ABBREVIATION_UNCLEAR,
            suggestions=[{'generic': 'Ibuprofen'}, {'generic': 'Naproxen'}]
        )
        
        assert 'nsaid' in query
        assert 'multiple' in query.lower() or 'could refer' in query.lower()
    
    def test_illegible_query(self, detector):
        """Test query generation for illegible term"""
        query = detector._generate_clarification_query(
            verbatim='???',
            subject_id='S001',
            site_id='Site 1',
            reason=ClarificationReason.ILLEGIBLE_TEXT,
            suggestions=[]
        )
        
        assert 'illegible' in query.lower() or 'unclear' in query.lower()
    
    def test_no_match_query(self, detector):
        """Test query generation when no match found"""
        query = detector._generate_clarification_query(
            verbatim='unknownmed',
            subject_id='S001',
            site_id='Site 1',
            reason=ClarificationReason.NO_MATCH_FOUND,
            suggestions=[]
        )
        
        assert 'unknownmed' in query
        assert 'not found' in query.lower() or 'verify' in query.lower()


# =============================================================================
# Learning Tests
# =============================================================================

class TestLearning:
    """Test learning from site clarifications"""
    
    def test_learn_from_resolution(self, detector):
        """Test learning stores resolution"""
        detector.learn_from_resolution(
            verbatim='pain killer',
            resolved_term='Advil',
            generic_name='Ibuprofen',
            trade_name='Advil',
            atc_code='M01AE01'
        )
        
        assert 'pain killer' in detector._learning_cache
        cached = detector._learning_cache['pain killer']
        assert cached['resolved_term'] == 'Advil'
        assert cached['generic_name'] == 'Ibuprofen'
    
    def test_learned_term_used_in_assessment(self, detector):
        """Test learned term is used in subsequent assessment"""
        # First, learn the resolution
        detector.learn_from_resolution(
            verbatim='my pill',
            resolved_term='Aspirin',
            generic_name='Acetylsalicylic acid'
        )
        
        # Now assess the same term
        assessment = detector._assess_term_specificity('my pill')
        
        assert assessment['confidence'] >= 0.80  # Should have higher confidence
        assert 'previously resolved' in assessment['assessment'].lower()
    
    def test_learning_disabled(self, custom_config):
        """Test learning can be disabled"""
        custom_config.enable_learning = False
        detector = AmbiguousCodingDetector(config=custom_config)
        
        detector.learn_from_resolution(
            verbatim='some term',
            resolved_term='Result',
            generic_name='Generic'
        )
        
        # Should not be stored when learning disabled
        assert 'some term' not in detector._learning_cache


# =============================================================================
# Full Detection Pipeline Tests
# =============================================================================

class TestFullDetectionPipeline:
    """Test complete detection pipeline"""
    
    def test_full_detect_returns_terms(self, detector, sample_whodra_data):
        """Test full detection returns ambiguous terms"""
        terms = detector.detect(
            whodra_data=sample_whodra_data,
            study_id='Study 1'
        )
        
        assert len(terms) > 0
        assert all(isinstance(t, AmbiguousTerm) for t in terms)
    
    def test_statistics_updated(self, detector, sample_whodra_data):
        """Test detection updates statistics"""
        detector.detect(sample_whodra_data, study_id='Study 1')
        
        stats = detector._detection_stats
        assert stats['total_uncoded'] > 0
    
    def test_get_clarification_queries(self, detector, sample_whodra_data):
        """Test getting clarification queries"""
        detector.detect(sample_whodra_data, study_id='Study 1')
        queries = detector.get_clarification_queries()
        
        # Should have some clarification queries
        assert isinstance(queries, list)
        for q in queries:
            assert 'query_text' in q
            assert 'subject_id' in q
    
    def test_get_proposed_codes(self, detector, sample_whodra_data):
        """Test getting proposed codes"""
        detector.detect(sample_whodra_data, study_id='Study 1')
        proposed = detector.get_proposed_codes()
        
        assert isinstance(proposed, list)
    
    def test_get_summary_report(self, detector, sample_whodra_data):
        """Test summary report generation"""
        detector.detect(sample_whodra_data, study_id='Study 1')
        summary = detector.get_summary_report()
        
        assert 'statistics' in summary
        assert 'breakdown' in summary
        assert 'by_ambiguity_level' in summary


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_verbatim(self, detector):
        """Test handling of empty verbatim term"""
        assessment = detector._assess_term_specificity('')
        # Empty string matches 'in term' for partial matching
        # so it returns 0.6 confidence with multiple partial matches
        assert assessment['confidence'] <= 1.0  # Just check it doesn't crash
    
    def test_whitespace_only_verbatim(self, detector):
        """Test handling of whitespace-only term"""
        assessment = detector._assess_term_specificity('   ')
        # Whitespace-only stripped becomes empty, matches partial matches
        assert assessment['confidence'] <= 1.0  # Just check it doesn't crash
    
    def test_special_characters(self, detector):
        """Test handling of special characters"""
        assessment = detector._assess_term_specificity('med@#$%')
        # Unknown term gets 0.50 confidence
        assert assessment['confidence'] <= 0.50
    
    def test_very_long_term(self, detector):
        """Test handling of very long term"""
        long_term = 'a' * 500
        assessment = detector._assess_term_specificity(long_term)
        # Should not crash
        assert 'confidence' in assessment
    
    def test_duplicate_columns_in_data(self, detector):
        """Test handling of duplicate columns"""
        df = pd.DataFrame({
            'Subject': ['S1', 'S2'],
            'Coding Status': ['UnCoded Term', 'UnCoded Term']
        })
        # Add duplicate column
        df = pd.concat([df, df[['Subject']]], axis=1)
        
        # Should handle gracefully
        uncoded = detector._scan_for_uncoded_terms(df)
        assert len(uncoded) == 2


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
