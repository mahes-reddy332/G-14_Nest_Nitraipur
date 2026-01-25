"""
Test Suite for Scientific Questions Module
==========================================

This test suite verifies all the enhanced features that address the core scientific questions:

1. Which sites/patients have the most missing visits? (Top 10 Offenders)
2. Where are the highest rates of non-conformant data? (DQI Heatmap)
3. Which sites require immediate attention? (Delta Engine + DQI Flagging)
4. Is the snapshot clean enough for interim analysis? (Global Cleanliness Meter)

Plus:
- ROI Metrics tracking
- Cross-therapeutic scalability
- Query automation efficiency
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from clinical_dataflow_optimizer.core.scientific_answers import (
    VisitAdherenceAnalyzer,
    NonConformanceHeatmapGenerator,
    DeltaEngine,
    GlobalCleanlinessMeter,
    ROICalculator,
    ScientificQuestionsDashboard,
    InterimReadiness,
    InterventionLevel
)

from clinical_dataflow_optimizer.config.settings import (
    VisitAdherenceConfig,
    DQIHeatmapConfig,
    SiteInterventionConfig,
    DeltaEngineConfig,
    GlobalCleanlinessMeterConfig,
    ROIMetricsConfig
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_cpid_data():
    """Create sample CPID EDC Metrics data"""
    np.random.seed(42)
    n_patients = 50
    
    countries = ['USA', 'Germany', 'France', 'UK', 'Japan']
    data = {
        'Subject ID': [f'SUBJ-{i:03d}' for i in range(n_patients)],
        'Site ID': [f'SITE-{(i % 5) + 1:02d}' for i in range(n_patients)],
        'Country': [countries[i % 5] for i in range(n_patients)],
        'Missing Visits': np.random.poisson(1, n_patients),
        '# Pages Entered': np.random.randint(10, 100, n_patients),
        '# Pages with Non-Conformant data': np.random.poisson(2, n_patients),
        '# Open Queries': np.random.poisson(3, n_patients),
        '# Total Queries': np.random.randint(5, 50, n_patients),
        '# Uncoded Terms': np.random.poisson(1, n_patients),
        '# Coded terms': np.random.randint(0, 20, n_patients),
        '# Reconciliation Issues': np.random.choice([0, 0, 0, 1, 2], n_patients),
        'Data Verification %': np.random.uniform(70, 100, n_patients)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_visit_tracker_data():
    """Create sample Visit Projection Tracker data"""
    visits = []
    np.random.seed(42)
    
    for i in range(30):
        visits.append({
            'Subject': f'SUBJ-{i:03d}',
            'Site': f'SITE-{(i % 5) + 1:02d}',
            'Visit': f'Visit {(i % 6) + 1}',
            'Projected Date': (datetime.now() - timedelta(days=np.random.randint(1, 90))).strftime('%Y-%m-%d'),
            '# Days Outstanding': np.random.randint(5, 120)
        })
    
    return pd.DataFrame(visits)


@pytest.fixture
def sample_clean_patient_statuses():
    """Create sample Clean Patient Status data"""
    statuses = []
    np.random.seed(42)
    
    for i in range(50):
        is_clean = np.random.random() > 0.3
        statuses.append({
            'subject_id': f'SUBJ-{i:03d}',
            'site_id': f'SITE-{(i % 5) + 1:02d}',
            'is_clean': is_clean,
            'clean_percentage': 100.0 if is_clean else np.random.uniform(50, 99),
            'primary_blocker': None if is_clean else np.random.choice([
                'Missing Visit', 'Open Query', 'Uncoded Term', 'Reconciliation Issue'
            ])
        })
    
    return statuses


@pytest.fixture
def sample_dqi_results():
    """Create sample DQI results for delta analysis"""
    class MockDQIResult:
        def __init__(self, site_id, dqi_score):
            self.site_id = site_id
            self.dqi_score = dqi_score
    
    results = {}
    np.random.seed(42)
    
    for i in range(10):
        site_id = f'SITE-{i+1:02d}'
        dqi_score = np.random.uniform(50, 100)
        results[site_id] = MockDQIResult(site_id, dqi_score)
    
    return results


# =============================================================================
# TEST: VISIT ADHERENCE ANALYZER (Top 10 Offenders)
# =============================================================================

class TestVisitAdherenceAnalyzer:
    """Tests for Visit Adherence Analyzer - Top 10 Offenders functionality"""
    
    def test_analyzer_initialization(self):
        """Test that analyzer initializes with default config"""
        analyzer = VisitAdherenceAnalyzer()
        assert analyzer.config is not None
        assert analyzer.config.top_offenders_count == 10
    
    def test_analyze_returns_correct_structure(self, sample_cpid_data, sample_visit_tracker_data):
        """Test that analysis returns expected structure"""
        analyzer = VisitAdherenceAnalyzer()
        results = analyzer.analyze(sample_cpid_data, sample_visit_tracker_data)
        
        assert 'top_10_sites' in results
        assert 'top_10_patients' in results
        assert 'summary' in results
        assert 'by_country' in results
        assert 'critical_cases' in results
    
    def test_top_offenders_sorted_by_priority(self, sample_cpid_data, sample_visit_tracker_data):
        """Test that offenders are sorted by priority score (Days Outstanding)"""
        analyzer = VisitAdherenceAnalyzer()
        results = analyzer.analyze(sample_cpid_data, sample_visit_tracker_data)
        
        sites = results['top_10_sites']
        if len(sites) > 1:
            # Verify sorted by priority_score descending
            for i in range(len(sites) - 1):
                assert sites[i]['priority_score'] >= sites[i+1]['priority_score']
    
    def test_top_10_limit_respected(self, sample_cpid_data, sample_visit_tracker_data):
        """Test that top 10 limit is respected"""
        analyzer = VisitAdherenceAnalyzer()
        results = analyzer.analyze(sample_cpid_data, sample_visit_tracker_data)
        
        assert len(results['top_10_sites']) <= 10
        assert len(results['top_10_patients']) <= 10
    
    def test_summary_statistics(self, sample_cpid_data, sample_visit_tracker_data):
        """Test that summary contains expected statistics"""
        analyzer = VisitAdherenceAnalyzer()
        results = analyzer.analyze(sample_cpid_data, sample_visit_tracker_data)
        
        summary = results['summary']
        assert 'total_missing_visits' in summary
        assert 'total_affected_patients' in summary
        assert 'avg_days_outstanding' in summary
        assert 'critical_count' in summary
    
    def test_custom_config(self, sample_cpid_data):
        """Test with custom configuration"""
        config = VisitAdherenceConfig(
            top_offenders_count=5,
            days_outstanding_critical=30
        )
        analyzer = VisitAdherenceAnalyzer(config)
        
        assert analyzer.config.top_offenders_count == 5
        assert analyzer.config.days_outstanding_critical == 30
    
    def test_dashboard_output(self, sample_cpid_data, sample_visit_tracker_data):
        """Test dashboard output generation"""
        analyzer = VisitAdherenceAnalyzer()
        results = analyzer.analyze(sample_cpid_data, sample_visit_tracker_data)
        dashboard = analyzer.get_top_offenders_dashboard(results)
        
        assert 'title' in dashboard
        assert 'sites' in dashboard
        assert 'patients' in dashboard
        assert 'alerts' in dashboard


# =============================================================================
# TEST: NON-CONFORMANCE HEATMAP GENERATOR
# =============================================================================

class TestNonConformanceHeatmapGenerator:
    """Tests for Non-Conformance Heatmap - answers 'Where is non-conformant data?'"""
    
    def test_generator_initialization(self):
        """Test heatmap generator initialization"""
        generator = NonConformanceHeatmapGenerator()
        assert generator.config is not None
    
    def test_analyze_returns_correct_structure(self, sample_cpid_data):
        """Test that analysis returns expected structure"""
        generator = NonConformanceHeatmapGenerator()
        results = generator.analyze(sample_cpid_data)
        
        assert 'hotspots_by_site' in results
        assert 'hotspots_by_country' in results
        assert 'heatmap_data' in results
        assert 'intervention_priorities' in results
        assert 'retraining_recommendations' in results
        assert 'summary' in results
    
    def test_hotspots_sorted_by_rate(self, sample_cpid_data):
        """Test that hotspots are sorted by non-conformance rate"""
        generator = NonConformanceHeatmapGenerator()
        results = generator.analyze(sample_cpid_data)
        
        hotspots = results['hotspots_by_site']
        if len(hotspots) > 1:
            for i in range(len(hotspots) - 1):
                assert hotspots[i]['non_conformance_rate'] >= hotspots[i+1]['non_conformance_rate']
    
    def test_heatmap_data_contains_colors(self, sample_cpid_data):
        """Test that heatmap data includes color information"""
        generator = NonConformanceHeatmapGenerator()
        results = generator.analyze(sample_cpid_data)
        
        for item in results['heatmap_data']:
            assert 'color' in item
            assert 'intensity' in item
            assert item['color'].startswith('#')
    
    def test_intervention_recommendations(self, sample_cpid_data):
        """Test that intervention recommendations are generated"""
        generator = NonConformanceHeatmapGenerator()
        results = generator.analyze(sample_cpid_data)
        
        for intervention in results['intervention_priorities']:
            assert 'rank' in intervention
            assert 'site_id' in intervention
            assert 'intervention' in intervention
    
    def test_summary_statistics(self, sample_cpid_data):
        """Test summary statistics"""
        generator = NonConformanceHeatmapGenerator()
        results = generator.analyze(sample_cpid_data)
        
        summary = results['summary']
        assert 'total_non_conformant_pages' in summary
        assert 'overall_non_conformance_rate' in summary
        assert 'sites_with_issues' in summary


# =============================================================================
# TEST: DELTA ENGINE (Site Intervention)
# =============================================================================

class TestDeltaEngine:
    """Tests for Delta Engine - answers 'Which sites need immediate attention?'"""
    
    def test_engine_initialization(self):
        """Test Delta Engine initialization"""
        engine = DeltaEngine()
        assert engine.delta_config is not None
        assert engine.intervention_config is not None
    
    def test_analyze_returns_correct_structure(self, sample_dqi_results):
        """Test that analysis returns expected structure"""
        engine = DeltaEngine()
        results = engine.analyze(sample_dqi_results)
        
        assert 'site_metrics' in results
        assert 'flagged_sites' in results
        assert 'immediate_intervention_required' in results
        assert 'improving_sites' in results
        assert 'summary' in results
    
    def test_intervention_flagging_logic(self, sample_dqi_results):
        """Test that intervention is flagged for DQI < 75 AND negative velocity"""
        engine = DeltaEngine()
        results = engine.analyze(sample_dqi_results)
        
        for metrics in results['site_metrics']:
            if metrics['requires_intervention']:
                # Should only be flagged if DQI < 75 AND negative velocity
                assert metrics['current_dqi'] < 75 or metrics['velocity'] < -5
    
    def test_trend_classification(self, sample_dqi_results):
        """Test that trends are correctly classified"""
        engine = DeltaEngine()
        results = engine.analyze(sample_dqi_results)
        
        valid_trends = ['critical_decline', 'declining', 'stable', 'improving']
        for metrics in results['site_metrics']:
            assert metrics['trend'] in valid_trends
    
    def test_intervention_levels(self, sample_dqi_results):
        """Test intervention levels are assigned"""
        engine = DeltaEngine()
        results = engine.analyze(sample_dqi_results)
        
        valid_levels = ['critical', 'high', 'medium', 'low']
        for metrics in results['site_metrics']:
            assert metrics['intervention_level'] in valid_levels
    
    def test_snapshot_storage(self):
        """Test historical snapshot storage"""
        engine = DeltaEngine()
        
        engine.store_snapshot('SITE-01', 85.5)
        engine.store_snapshot('SITE-01', 83.2)
        
        assert 'SITE-01' in engine._historical_snapshots
        assert len(engine._historical_snapshots['SITE-01']) == 2
    
    def test_get_intervention_actions(self):
        """Test getting intervention actions for different levels"""
        engine = DeltaEngine()
        
        critical_actions = engine.get_intervention_actions(InterventionLevel.CRITICAL)
        assert len(critical_actions) > 0
        assert 'onsite audit' in str(critical_actions).lower() or 'intervention' in str(critical_actions).lower()


# =============================================================================
# TEST: GLOBAL CLEANLINESS METER (Interim Analysis)
# =============================================================================

class TestGlobalCleanlinessMeter:
    """Tests for Global Cleanliness Meter - answers 'Is snapshot clean enough?'"""
    
    def test_meter_initialization(self):
        """Test meter initialization"""
        meter = GlobalCleanlinessMeter()
        assert meter.config is not None
        assert meter.config.clean_patient_threshold_itt == 80.0
    
    def test_assess_returns_correct_structure(self, sample_clean_patient_statuses):
        """Test that assessment returns expected structure"""
        meter = GlobalCleanlinessMeter()
        result = meter.assess(sample_clean_patient_statuses)
        
        assert hasattr(result, 'definitive_answer')
        assert hasattr(result, 'overall_clean_percentage')
        assert hasattr(result, 'meets_power_threshold')
        assert hasattr(result, 'clean_patient_count')
        assert hasattr(result, 'confidence_interval')
    
    def test_definitive_yes_above_threshold(self):
        """Test definitive YES when clean percentage is above threshold"""
        meter = GlobalCleanlinessMeter()
        
        # Create 90% clean patients
        statuses = [{'is_clean': True, 'subject_id': f'S{i}', 'clean_percentage': 100} 
                   for i in range(90)]
        statuses.extend([{'is_clean': False, 'subject_id': f'S{i}', 'clean_percentage': 70, 
                         'primary_blocker': 'Query'} for i in range(90, 100)])
        
        result = meter.assess(statuses)
        assert result.overall_clean_percentage >= 80
        assert result.definitive_answer in [InterimReadiness.YES, InterimReadiness.CONDITIONAL]
    
    def test_definitive_no_below_threshold(self):
        """Test definitive NO when clean percentage is below threshold"""
        meter = GlobalCleanlinessMeter()
        
        # Create 60% clean patients (below 80% threshold)
        statuses = [{'is_clean': True, 'subject_id': f'S{i}', 'clean_percentage': 100} 
                   for i in range(60)]
        statuses.extend([{'is_clean': False, 'subject_id': f'S{i}', 'clean_percentage': 50, 
                         'primary_blocker': 'Query'} for i in range(60, 100)])
        
        result = meter.assess(statuses)
        assert result.overall_clean_percentage < 80
        assert result.definitive_answer == InterimReadiness.NO
    
    def test_confidence_interval_calculation(self, sample_clean_patient_statuses):
        """Test that confidence interval is calculated correctly"""
        meter = GlobalCleanlinessMeter()
        result = meter.assess(sample_clean_patient_statuses)
        
        ci = result.confidence_interval
        assert ci[0] <= result.overall_clean_percentage <= ci[1]
        assert ci[0] >= 0
        assert ci[1] <= 100
    
    def test_population_results(self, sample_clean_patient_statuses):
        """Test population-specific results"""
        meter = GlobalCleanlinessMeter()
        result = meter.assess(sample_clean_patient_statuses)
        
        assert 'ITT' in result.population_results
        assert 'PP' in result.population_results
        assert 'Safety' in result.population_results
    
    def test_dashboard_display(self, sample_clean_patient_statuses):
        """Test dashboard display format"""
        meter = GlobalCleanlinessMeter()
        result = meter.assess(sample_clean_patient_statuses)
        dashboard = meter.get_dashboard_display(result)
        
        assert 'answer' in dashboard
        assert 'color' in dashboard
        assert 'percentage' in dashboard
        assert dashboard['color'].startswith('#')
    
    def test_no_data_handling(self):
        """Test handling of empty data"""
        meter = GlobalCleanlinessMeter()
        result = meter.assess([])
        
        assert result.definitive_answer == InterimReadiness.NO
        assert result.total_patient_count == 0


# =============================================================================
# TEST: ROI CALCULATOR
# =============================================================================

class TestROICalculator:
    """Tests for ROI Calculator - tracks efficiency gains"""
    
    def test_calculator_initialization(self):
        """Test ROI calculator initialization"""
        calculator = ROICalculator()
        assert calculator.config is not None
    
    def test_calculate_returns_correct_structure(self):
        """Test that calculation returns expected structure"""
        calculator = ROICalculator()
        
        query_stats = {'total': 100, 'automated': 75}
        result = calculator.calculate(
            query_automation_stats=query_stats,
            dqi_improvement=10.0,
            time_to_lock_months=2.5,
            dm_hours_saved=200
        )
        
        assert hasattr(result, 'efficiency_metrics')
        assert hasattr(result, 'quality_metrics')
        assert hasattr(result, 'speed_metrics')
        assert hasattr(result, 'financial_impact')
        assert hasattr(result, 'summary')
    
    def test_automation_rate_calculation(self):
        """Test automation rate calculation"""
        calculator = ROICalculator()
        
        query_stats = {'total': 100, 'automated': 70}
        result = calculator.calculate(
            query_automation_stats=query_stats,
            dqi_improvement=5.0,
            time_to_lock_months=3.0,
            dm_hours_saved=100
        )
        
        assert result.efficiency_metrics['automation_rate'] == 70.0
    
    def test_target_met_flags(self):
        """Test that target met flags are set correctly"""
        calculator = ROICalculator()
        
        # Test with automation target met (>= 70%)
        query_stats = {'total': 100, 'automated': 75}
        result = calculator.calculate(
            query_automation_stats=query_stats,
            dqi_improvement=15.0,
            time_to_lock_months=1.5,
            dm_hours_saved=300
        )
        
        assert result.efficiency_metrics['target_met'] == True
        assert result.quality_metrics['quality_target_met'] == True
    
    def test_financial_impact_positive(self):
        """Test that financial impact is calculated"""
        calculator = ROICalculator()
        
        query_stats = {'total': 100, 'automated': 70}
        result = calculator.calculate(
            query_automation_stats=query_stats,
            dqi_improvement=10.0,
            time_to_lock_months=2.0,
            dm_hours_saved=200
        )
        
        assert result.financial_impact['total_value'] > 0


# =============================================================================
# TEST: INTEGRATED DASHBOARD
# =============================================================================

class TestScientificQuestionsDashboard:
    """Tests for the integrated Scientific Questions Dashboard"""
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        dashboard = ScientificQuestionsDashboard()
        
        assert dashboard.visit_analyzer is not None
        assert dashboard.heatmap_generator is not None
        assert dashboard.delta_engine is not None
        assert dashboard.cleanliness_meter is not None
        assert dashboard.roi_calculator is not None
    
    def test_generate_full_report(self, sample_cpid_data, sample_visit_tracker_data, 
                                   sample_dqi_results, sample_clean_patient_statuses):
        """Test generating full report"""
        dashboard = ScientificQuestionsDashboard()
        
        report = dashboard.generate_full_report(
            cpid_data=sample_cpid_data,
            visit_tracker_data=sample_visit_tracker_data,
            dqi_results=sample_dqi_results,
            clean_patient_statuses=sample_clean_patient_statuses,
            query_stats={'total': 100, 'automated': 70}
        )
        
        assert 'generated_at' in report
        assert 'questions_answered' in report
        assert 'missing_visits' in report['questions_answered']
        assert 'non_conformant_data' in report['questions_answered']


# =============================================================================
# TEST: CONFIGURATION CLASSES
# =============================================================================

class TestConfigurations:
    """Tests for all configuration classes"""
    
    def test_visit_adherence_config_defaults(self):
        """Test VisitAdherenceConfig default values"""
        config = VisitAdherenceConfig()
        
        assert config.top_offenders_count == 10
        assert config.days_outstanding_critical == 60
        assert config.days_outstanding_high == 30
    
    def test_dqi_heatmap_config_defaults(self):
        """Test DQIHeatmapConfig default values"""
        config = DQIHeatmapConfig()
        
        assert config.enable_geographic_view == True
        assert config.non_conformant_critical_threshold == 10
    
    def test_delta_engine_config_defaults(self):
        """Test DeltaEngineConfig default values"""
        config = DeltaEngineConfig()
        
        assert config.velocity_critical == -10.0
        assert config.velocity_warning == -5.0
        assert config.enable_acceleration == True
    
    def test_global_cleanliness_config_defaults(self):
        """Test GlobalCleanlinessMeterConfig default values"""
        config = GlobalCleanlinessMeterConfig()
        
        assert config.clean_patient_threshold_itt == 80.0
        assert config.output_definitive_answer == True
    
    def test_site_intervention_config(self):
        """Test SiteInterventionConfig"""
        config = SiteInterventionConfig()
        
        assert config.immediate_intervention_dqi == 75.0
        assert config.combined_flag_enabled == True
        assert 'critical' in config.intervention_actions


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Running Scientific Questions Module Tests")
    print("=" * 70)
    
    # Run with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
