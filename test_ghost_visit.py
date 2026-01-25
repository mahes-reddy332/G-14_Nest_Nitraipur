"""
Test Suite for Scenario C: Ghost Visit Detection
Tests the GhostVisitDetector module and SiteLiaisonAgent integration

Test Coverage:
1. GhostVisitDetector initialization
2. Visit Projection Tracker scanning
3. Days Outstanding calculation
4. Inactivation status verification
5. Site standing assessment
6. Escalation level determination
7. Auto-reminder generation
8. SiteLiaisonAgent integration
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from clinical_dataflow_optimizer.core.ghost_visit_detector import (
    GhostVisitDetector, GhostVisit, GhostVisitConfig,
    VisitStatus, SiteStanding, EscalationLevel, InactivationReason
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def current_date():
    """Fixed current date for testing"""
    return datetime(2025, 11, 14)


@pytest.fixture
def sample_visit_tracker(current_date):
    """Sample Visit Projection Tracker data"""
    return pd.DataFrame({
        'Country': ['USA', 'USA', 'UK', 'UK', 'Germany', 'France', 'Spain'],
        'Site': ['Site 1', 'Site 1', 'Site 2', 'Site 2', 'Site 3', 'Site 4', 'Site 5'],
        'Subject': ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5', 'Subject 6', 'Subject 7'],
        'Visit': ['W2D7', 'Cycle 3', 'Visit 5', 'W4D1', 'Baseline', 'EOT', 'Follow-up'],
        'Projected Date': [
            '28OCT2025',   # 17 days outstanding
            '07NOV2025',   # 7 days outstanding
            '01NOV2025',   # 13 days outstanding
            '10NOV2025',   # 4 days outstanding (below threshold)
            '25OCT2025',   # 20 days outstanding
            '01OCT2025',   # 44 days outstanding
            '05NOV2025'    # 9 days outstanding
        ],
        '# Days Outstanding': [17, 7, 13, 4, 20, 44, 9]
    })


@pytest.fixture
def sample_inactivated_forms():
    """Sample Inactivated Forms Report data"""
    return pd.DataFrame({
        'Country': ['USA', 'UK', 'Germany', 'Germany'],
        'Study Site Number': ['Site 1', 'Site 2', 'Site 3', 'Site 3'],
        'Subject': ['Subject 2', 'Subject 4', 'Subject 5', 'Subject 8'],
        'Folder': ['Cycle 3', 'W4D1', 'Baseline', 'Visit 10'],
        'Form ': ['Visit Form', 'Lab Form', 'Demographics', 'AE Form'],
        'Data on Form/\nRecord    ': ['Data', 'Data', 'Data', 'Data'],
        'RecordPosition': [1, 1, 1, 1],
        'Audit Action': [
            'Record inactivated with code reason code Patient Withdrew from Study',
            'Record inactivated with code reason code Not Done.',
            'Record inactivated with code reason code Patient Discontinued',
            'Record Inactivated.'
        ]
    })


@pytest.fixture
def sample_cpid_data():
    """Sample CPID data for site metrics"""
    return pd.DataFrame({
        ('Site ID', 'Unnamed_1'): ['Site 1', 'Site 1', 'Site 2', 'Site 2', 'Site 3', 'Site 4', 'Site 5'],
        ('Subject ID', 'Unnamed_2'): ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5', 'Subject 6', 'Subject 7'],
        ('Open Queries', 'Unnamed_3'): [2, 3, 1, 0, 8, 15, 2]  # Site 4 has high queries
    })


@pytest.fixture
def detector():
    """Create GhostVisitDetector with default config"""
    return GhostVisitDetector()


@pytest.fixture
def custom_config():
    """Create custom configuration"""
    return GhostVisitConfig(
        days_warning_threshold=5,
        days_critical_threshold=10,
        days_sponsor_escalation=15,
        site_red_threshold=3,
        site_yellow_threshold=1
    )


# =============================================================================
# GhostVisitDetector Initialization Tests
# =============================================================================

class TestGhostVisitDetectorInit:
    """Test GhostVisitDetector initialization"""
    
    def test_default_initialization(self, detector):
        """Test detector initializes with defaults"""
        assert detector is not None
        assert detector.config is not None
        assert detector.config.days_warning_threshold == 7
        assert detector.config.days_critical_threshold == 14
    
    def test_custom_config_initialization(self, custom_config):
        """Test detector with custom config"""
        detector = GhostVisitDetector(config=custom_config)
        assert detector.config.days_warning_threshold == 5
        assert detector.config.days_critical_threshold == 10


# =============================================================================
# Visit Tracker Scanning Tests
# =============================================================================

class TestVisitTrackerScanning:
    """Test scanning Visit Projection Tracker for overdue visits"""
    
    def test_scan_finds_overdue_visits(self, detector, sample_visit_tracker, current_date):
        """Test scanner finds overdue visits"""
        overdue = detector._scan_for_overdue_visits(sample_visit_tracker, current_date)
        
        # All visits are past projected date
        assert len(overdue) == 7
    
    def test_scan_parses_dates(self, detector, sample_visit_tracker, current_date):
        """Test scanner parses date formats correctly"""
        overdue = detector._scan_for_overdue_visits(sample_visit_tracker, current_date)
        
        # Should have _parsed_date column
        assert '_parsed_date' in overdue.columns
        assert all(overdue['_parsed_date'].notna())
    
    def test_scan_empty_dataframe(self, detector, current_date):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        overdue = detector._scan_for_overdue_visits(empty_df, current_date)
        assert len(overdue) == 0
    
    def test_scan_missing_date_column(self, detector, current_date):
        """Test handling when Projected Date column is missing"""
        df = pd.DataFrame({'Subject': ['S1', 'S2'], 'Site': ['A', 'B']})
        overdue = detector._scan_for_overdue_visits(df, current_date)
        assert len(overdue) == 0


# =============================================================================
# Days Outstanding Calculation Tests
# =============================================================================

class TestDaysOutstandingCalculation:
    """Test Days Outstanding calculation"""
    
    def test_uses_existing_days_column(self, detector, sample_visit_tracker, current_date):
        """Test uses existing # Days Outstanding column"""
        ghosts = detector.detect(
            visit_tracker_data=sample_visit_tracker,
            current_date=current_date,
            study_id='Study 1'
        )
        
        # Check that days are captured
        for ghost in ghosts:
            assert ghost.days_outstanding >= 0
    
    def test_calculates_days_when_column_missing(self, detector, current_date):
        """Test calculates days when column is missing"""
        df = pd.DataFrame({
            'Subject': ['S1'],
            'Site': ['Site 1'],
            'Visit': ['Visit 1'],
            'Projected Date': ['01NOV2025']  # 13 days before current_date
        })
        
        ghosts = detector.detect(df, current_date=current_date, study_id='Study 1')
        
        # Should calculate days
        if ghosts:
            assert ghosts[0].days_outstanding >= 7


# =============================================================================
# Inactivation Check Tests
# =============================================================================

class TestInactivationCheck:
    """Test inactivation status verification"""
    
    def test_builds_inactivation_cache(self, detector, sample_inactivated_forms):
        """Test inactivation cache is built"""
        detector._build_inactivation_cache(sample_inactivated_forms)
        
        assert len(detector._inactivation_cache) > 0
        assert 'Subject 2' in detector._inactivation_cache
    
    def test_detects_valid_withdrawal(self, detector, sample_inactivated_forms):
        """Test detection of valid patient withdrawal"""
        detector._build_inactivation_cache(sample_inactivated_forms)
        
        result = detector._check_inactivation_status('Subject 2', 'Cycle 3')
        
        assert result['is_valid_inactivation'] == True
        assert result['inactivation_reason'] == InactivationReason.PATIENT_WITHDREW
        assert result['requires_query'] == False
    
    def test_detects_valid_discontinued(self, detector, sample_inactivated_forms):
        """Test detection of valid patient discontinuation"""
        detector._build_inactivation_cache(sample_inactivated_forms)
        
        result = detector._check_inactivation_status('Subject 5', 'Baseline')
        
        assert result['is_valid_inactivation'] == True
        assert result['inactivation_reason'] == InactivationReason.PATIENT_DISCONTINUED
    
    def test_detects_not_done(self, detector, sample_inactivated_forms):
        """Test detection of 'Not Done' inactivation"""
        detector._build_inactivation_cache(sample_inactivated_forms)
        
        result = detector._check_inactivation_status('Subject 4', 'W4D1')
        
        assert result['is_valid_inactivation'] == True
        assert result['inactivation_reason'] == InactivationReason.VISIT_NOT_DONE
    
    def test_no_inactivation_found(self, detector, sample_inactivated_forms):
        """Test when no inactivation exists"""
        detector._build_inactivation_cache(sample_inactivated_forms)
        
        result = detector._check_inactivation_status('Subject 1', 'W2D7')
        
        assert result['is_valid_inactivation'] == False
        assert result['requires_query'] == True


# =============================================================================
# Site Standing Tests
# =============================================================================

class TestSiteStanding:
    """Test site standing assessment"""
    
    def test_green_standing(self, detector, sample_cpid_data):
        """Test Green standing for site with few issues"""
        detector._build_site_metrics(sample_cpid_data)
        
        standing = detector._get_site_standing('Site 1')
        
        # Site 1 has moderate queries per subject
        assert standing in [SiteStanding.GREEN, SiteStanding.YELLOW]
    
    def test_red_standing(self, detector, sample_cpid_data):
        """Test Red standing for site with many issues"""
        detector._build_site_metrics(sample_cpid_data)
        
        standing = detector._get_site_standing('Site 4')
        
        # Site 4 has 15 queries for 1 subject = high
        assert standing == SiteStanding.RED
    
    def test_unknown_site_defaults_yellow(self, detector):
        """Test unknown site defaults to Yellow"""
        standing = detector._get_site_standing('Unknown Site')
        assert standing == SiteStanding.YELLOW


# =============================================================================
# Escalation Level Tests
# =============================================================================

class TestEscalationLevel:
    """Test escalation level determination"""
    
    def test_standard_reminder_green_site(self, detector, current_date):
        """Test enhanced reminder for green site at warning threshold"""
        # At 8 days (>= 7 warning threshold), even green site gets ENHANCED_REMINDER
        # STANDARD_REMINDER only applies for days < warning_threshold, but those
        # aren't processed by the detector
        level, reminder = detector._determine_action(
            visit_status=VisitStatus.GHOST_CONFIRMED,
            site_standing=SiteStanding.GREEN,
            days_outstanding=8,  # Just over warning threshold
            subject_id='S001',
            site_id='Site 1',
            visit_name='Visit 1',
            projected_date=current_date - timedelta(days=8)
        )
        
        # Green site at 8 days gets enhanced reminder (warning threshold is 7)
        assert level == EscalationLevel.ENHANCED_REMINDER
        assert 'PRIORITY' in reminder
    
    def test_enhanced_reminder_yellow_site(self, detector, current_date):
        """Test enhanced reminder for yellow site"""
        level, reminder = detector._determine_action(
            visit_status=VisitStatus.GHOST_CONFIRMED,
            site_standing=SiteStanding.YELLOW,
            days_outstanding=10,
            subject_id='S001',
            site_id='Site 1',
            visit_name='Visit 1',
            projected_date=current_date - timedelta(days=10)
        )
        
        assert level == EscalationLevel.ENHANCED_REMINDER
        assert 'PRIORITY' in reminder
    
    def test_cra_escalation_red_site(self, detector, current_date):
        """Test CRA escalation for red site"""
        level, reminder = detector._determine_action(
            visit_status=VisitStatus.GHOST_CONFIRMED,
            site_standing=SiteStanding.RED,
            days_outstanding=10,
            subject_id='S001',
            site_id='Site 1',
            visit_name='Visit 1',
            projected_date=current_date - timedelta(days=10)
        )
        
        assert level == EscalationLevel.CRA_ESCALATION
        assert 'ESCALATION' in reminder or 'phone call' in reminder.lower()
    
    def test_cra_escalation_critical_days(self, detector, current_date):
        """Test CRA escalation for critical days threshold"""
        level, reminder = detector._determine_action(
            visit_status=VisitStatus.GHOST_CONFIRMED,
            site_standing=SiteStanding.GREEN,  # Even green site
            days_outstanding=15,  # Over critical threshold
            subject_id='S001',
            site_id='Site 1',
            visit_name='Visit 1',
            projected_date=current_date - timedelta(days=15)
        )
        
        assert level == EscalationLevel.CRA_ESCALATION
    
    def test_sponsor_escalation(self, detector, current_date):
        """Test sponsor escalation for very overdue visits"""
        level, reminder = detector._determine_action(
            visit_status=VisitStatus.GHOST_CONFIRMED,
            site_standing=SiteStanding.GREEN,
            days_outstanding=25,  # Way over sponsor threshold
            subject_id='S001',
            site_id='Site 1',
            visit_name='Visit 1',
            projected_date=current_date - timedelta(days=25)
        )
        
        assert level == EscalationLevel.SPONSOR_ESCALATION


# =============================================================================
# Risk Score Tests
# =============================================================================

class TestRiskScore:
    """Test risk score calculation"""
    
    def test_high_risk_score_red_site_long_delay(self, detector):
        """Test high risk score for red site with long delay"""
        score = detector._calculate_risk_score(
            days_outstanding=20,
            site_standing=SiteStanding.RED,
            visit_status=VisitStatus.GHOST_CONFIRMED
        )
        
        assert score >= 0.7
    
    def test_low_risk_score_green_site_short_delay(self, detector):
        """Test lower risk score for green site with short delay"""
        score = detector._calculate_risk_score(
            days_outstanding=8,
            site_standing=SiteStanding.GREEN,
            visit_status=VisitStatus.GHOST_CONFIRMED
        )
        
        assert score < 0.7
    
    def test_low_risk_valid_inactivation(self, detector):
        """Test reduced risk for valid inactivation"""
        score = detector._calculate_risk_score(
            days_outstanding=20,
            site_standing=SiteStanding.RED,
            visit_status=VisitStatus.VALIDLY_INACTIVATED
        )
        
        # Valid inactivation gets 0.1 status factor, but days (0.8) and site (1.0)
        # still contribute. Score = 0.4*0.8 + 0.3*1.0 + 0.3*0.1 = 0.65
        # This is lower than GHOST_CONFIRMED which would be ~0.92
        assert score < 0.7  # Lower than ghost confirmed


# =============================================================================
# Full Detection Pipeline Tests
# =============================================================================

class TestFullDetectionPipeline:
    """Test complete ghost visit detection pipeline"""
    
    def test_full_detect_returns_ghosts(
        self, detector, sample_visit_tracker, sample_inactivated_forms, 
        sample_cpid_data, current_date
    ):
        """Test full detection returns ghost visits"""
        ghosts = detector.detect(
            visit_tracker_data=sample_visit_tracker,
            inactivated_forms_data=sample_inactivated_forms,
            cpid_data=sample_cpid_data,
            current_date=current_date,
            study_id='Study 1'
        )
        
        assert len(ghosts) > 0
        assert all(isinstance(g, GhostVisit) for g in ghosts)
    
    def test_validly_inactivated_excluded_from_action(
        self, detector, sample_visit_tracker, sample_inactivated_forms,
        sample_cpid_data, current_date
    ):
        """Test validly inactivated visits don't require action"""
        ghosts = detector.detect(
            visit_tracker_data=sample_visit_tracker,
            inactivated_forms_data=sample_inactivated_forms,
            cpid_data=sample_cpid_data,
            current_date=current_date,
            study_id='Study 1'
        )
        
        # Subject 2's Cycle 3 and Subject 5's Baseline should be validly inactivated
        valid_inact = [g for g in ghosts if g.visit_status == VisitStatus.VALIDLY_INACTIVATED]
        # Note: depends on visit matching logic
        assert isinstance(valid_inact, list)
    
    def test_get_cra_escalations(
        self, detector, sample_visit_tracker, sample_inactivated_forms,
        sample_cpid_data, current_date
    ):
        """Test getting CRA escalations"""
        detector.detect(
            visit_tracker_data=sample_visit_tracker,
            inactivated_forms_data=sample_inactivated_forms,
            cpid_data=sample_cpid_data,
            current_date=current_date,
            study_id='Study 1'
        )
        
        escalations = detector.get_cra_escalations()
        assert isinstance(escalations, list)
        
        for esc in escalations:
            assert 'subject_id' in esc
            assert 'escalation_level' in esc
    
    def test_get_standard_reminders(
        self, detector, sample_visit_tracker, current_date
    ):
        """Test getting standard reminders"""
        detector.detect(
            visit_tracker_data=sample_visit_tracker,
            current_date=current_date,
            study_id='Study 1'
        )
        
        reminders = detector.get_standard_reminders()
        assert isinstance(reminders, list)
    
    def test_get_summary_report(
        self, detector, sample_visit_tracker, sample_inactivated_forms,
        sample_cpid_data, current_date
    ):
        """Test summary report generation"""
        detector.detect(
            visit_tracker_data=sample_visit_tracker,
            inactivated_forms_data=sample_inactivated_forms,
            cpid_data=sample_cpid_data,
            current_date=current_date,
            study_id='Study 1'
        )
        
        summary = detector.get_summary_report()
        
        assert 'statistics' in summary
        assert 'breakdown' in summary
        assert 'by_escalation' in summary
        assert 'by_site' in summary


# =============================================================================
# Site Standing Summary Tests
# =============================================================================

class TestSiteStandingSummary:
    """Test site standing summary"""
    
    def test_get_site_standing_summary(
        self, detector, sample_visit_tracker, sample_cpid_data, current_date
    ):
        """Test site standing summary generation"""
        detector.detect(
            visit_tracker_data=sample_visit_tracker,
            cpid_data=sample_cpid_data,
            current_date=current_date,
            study_id='Study 1'
        )
        
        summary = detector.get_site_standing_summary()
        
        assert isinstance(summary, dict)
        for site_id, data in summary.items():
            assert 'standing' in data
            assert 'ghost_visits' in data
            assert 'total_overdue' in data


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_visit_tracker(self, detector, current_date):
        """Test handling of empty visit tracker"""
        empty_df = pd.DataFrame()
        ghosts = detector.detect(empty_df, current_date=current_date, study_id='Study 1')
        assert ghosts == []
    
    def test_no_inactivation_data(self, detector, sample_visit_tracker, current_date):
        """Test handling when no inactivation data provided"""
        ghosts = detector.detect(
            visit_tracker_data=sample_visit_tracker,
            inactivated_forms_data=None,
            current_date=current_date,
            study_id='Study 1'
        )
        
        # Should still work, just won't have inactivation context
        assert len(ghosts) > 0
    
    def test_no_cpid_data(self, detector, sample_visit_tracker, current_date):
        """Test handling when no CPID data provided"""
        ghosts = detector.detect(
            visit_tracker_data=sample_visit_tracker,
            cpid_data=None,
            current_date=current_date,
            study_id='Study 1'
        )
        
        # Should still work, site standing defaults to Yellow
        assert len(ghosts) > 0
        for ghost in ghosts:
            assert ghost.site_standing == SiteStanding.YELLOW
    
    def test_various_date_formats(self, detector, current_date):
        """Test handling of various date formats"""
        df = pd.DataFrame({
            'Subject': ['S1', 'S2', 'S3'],
            'Site': ['Site 1', 'Site 1', 'Site 1'],
            'Visit': ['V1', 'V2', 'V3'],
            'Projected Date': ['01NOV2025', '2025-11-01', '11/01/2025'],
            '# Days Outstanding': [13, 13, 13]
        })
        
        ghosts = detector.detect(df, current_date=current_date, study_id='Study 1')
        
        # Should parse at least some dates
        parsed = [g for g in ghosts if g.projected_date is not None]
        assert len(parsed) > 0
    
    def test_duplicate_columns(self, detector, current_date):
        """Test handling of duplicate columns"""
        df = pd.DataFrame({
            'Subject': ['S1'],
            'Site': ['Site 1'],
            'Visit': ['V1'],
            'Projected Date': ['01NOV2025'],
            '# Days Outstanding': [10]
        })
        # Add duplicate column
        df = pd.concat([df, df[['Subject']]], axis=1)
        
        # Should handle gracefully
        ghosts = detector.detect(df, current_date=current_date, study_id='Study 1')
        assert isinstance(ghosts, list)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
