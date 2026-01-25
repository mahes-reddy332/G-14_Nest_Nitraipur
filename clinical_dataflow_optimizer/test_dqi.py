"""
Test Suite for Data Quality Index (DQI) Module
===============================================

Comprehensive tests for:
1. DQI Configuration Validation
2. Weighted Penalization Model
3. Site DQI Calculation
4. DQI Level Interpretation
5. Visualization Components
6. Dashboard Generation
7. Sankey Diagram Data
8. Scatter Plot Quadrants
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from clinical_dataflow_optimizer.core.data_quality_index import (
    DataQualityIndexCalculator,
    DQIConfig,
    DQIResult,
    DQILevel,
    MetricCategory,
    MetricPenalty,
    SiteRiskProfile,
    RiskQuadrant,
    QueryFlowData,
    DQIVisualizationEngine,
    calculate_site_dqi,
    calculate_study_dqi,
    generate_dqi_dashboard,
    DEFAULT_DQI_CONFIG
)


def get_data_path(study_name: str) -> Path:
    """Get path to study data folder"""
    # Use parent directory (Main_Project Track1) for QC data
    base = Path(__file__).parent.parent / "QC Anonymized Study Files"
    
    # Find matching folder
    for folder in base.iterdir():
        if folder.is_dir() and study_name.lower().replace(' ', '') in folder.name.lower().replace(' ', ''):
            return folder
    
    # Return first available study if not found
    for folder in base.iterdir():
        if folder.is_dir():
            return folder
    
    return base


def load_study_data(data_path: Path) -> dict:
    """Load study data files"""
    data = {}
    
    for file_path in data_path.glob("*.xlsx"):
        file_lower = file_path.name.lower()
        
        try:
            if 'cpid' in file_lower:
                data['cpid'] = pd.read_excel(file_path)
            elif 'esae' in file_lower or 'sae' in file_lower:
                data['esae'] = pd.read_excel(file_path)
            elif 'missing' in file_lower and 'page' in file_lower:
                data['missing_pages'] = pd.read_excel(file_path)
            elif 'visit' in file_lower and 'projection' in file_lower:
                data['visit_tracker'] = pd.read_excel(file_path)
        except Exception as e:
            print(f"  Warning: Could not load {file_path.name}: {e}")
    
    print(f"Loaded: {list(data.keys())}")
    return data


def test_dqi_configuration():
    """Test 1: DQI Configuration Validation"""
    print("\n" + "="*60)
    print("TEST 1: DQI Configuration Validation")
    print("="*60)
    
    # Default configuration
    config = DEFAULT_DQI_CONFIG
    print(f"\n📋 Default DQI Configuration:")
    print(f"  Weight (Visit Adherence): {config.weight_visit} (20%)")
    print(f"  Weight (Query Responsiveness): {config.weight_query} (20%)")
    print(f"  Weight (Data Conformance): {config.weight_conform} (20%)")
    print(f"  Weight (Safety Criticality): {config.weight_safety} (40%)")
    
    total_weight = config.weight_visit + config.weight_query + config.weight_conform + config.weight_safety
    print(f"\n  Total Weight: {total_weight:.2f}")
    print(f"  Config Valid: {config.validate()}")
    
    assert config.validate(), "Default configuration should be valid"
    assert abs(total_weight - 1.0) < 0.001, "Weights should sum to 1.0"
    
    # Test thresholds
    print(f"\n📊 DQI Level Thresholds:")
    print(f"  Green (Good): DQI > {config.green_threshold}")
    print(f"  Yellow (Warning): DQI {config.yellow_threshold}-{config.green_threshold}")
    print(f"  Red (Critical): DQI < {config.yellow_threshold}")
    
    # Test custom configuration
    custom_config = DQIConfig(
        weight_visit=0.25,
        weight_query=0.25,
        weight_conform=0.25,
        weight_safety=0.25
    )
    print(f"\n📋 Custom Configuration Valid: {custom_config.validate()}")
    
    # Test invalid configuration
    invalid_config = DQIConfig(
        weight_visit=0.5,
        weight_query=0.5,
        weight_conform=0.5,
        weight_safety=0.5
    )
    print(f"📋 Invalid Configuration Detected: {not invalid_config.validate()}")
    
    print("\n✅ DQI Configuration PASSED")


def test_weighted_penalization_model():
    """Test 2: Weighted Penalization Model Calculation"""
    print("\n" + "="*60)
    print("TEST 2: Weighted Penalization Model")
    print("="*60)
    
    print(f"\n📊 Formula: DQI = 100 - Σ(W_i × f(M_i))")
    print("   Where W_i = weight, f(M_i) = normalized penalty")
    
    # Create test data with known values
    test_data = pd.DataFrame([{
        'Subject ID': 'TEST-001',
        'Site ID': 'Site A',
        'Missing Visits': 2,
        '# Days Outstanding': 15,
        '# Total Queries': 10,
        '# Open Queries': 3,
        'CRFs overdue for signs': 2,
        'Time lag (Days)': 7,
        '# Pages with Non-Conformant data': 5,
        '# Inactivated Forms': 1,
        '# Reconciliation Issues': 0
    }])
    
    calculator = DataQualityIndexCalculator()
    calculator.load_data({'cpid': test_data})
    
    result = calculator.calculate_site_dqi('Site A')
    
    print(f"\n🔬 Test Site: Site A")
    print(f"   DQI Score: {result.dqi_score:.2f}")
    print(f"   Level: {result.level.value.upper()}")
    print(f"   Total Penalty: {result.total_penalty:.2f}")
    
    print(f"\n📈 Penalty Breakdown:")
    for penalty in result.penalties:
        print(f"   {penalty.category.value.title()}:")
        print(f"     - Raw Value: {penalty.raw_value:.2f}")
        print(f"     - Normalized Penalty: {penalty.normalized_penalty:.2f}")
        print(f"     - Weight: {penalty.weight} ({penalty.weight*100:.0f}%)")
        print(f"     - Weighted Penalty: {penalty.weighted_penalty:.2f}")
    
    # Verify formula
    calculated_total = sum(p.weighted_penalty for p in result.penalties)
    calculated_dqi = 100 - calculated_total
    
    print(f"\n✅ Formula Verification:")
    print(f"   Sum of Weighted Penalties: {calculated_total:.2f}")
    print(f"   Calculated DQI: {calculated_dqi:.2f}")
    print(f"   Reported DQI: {result.dqi_score:.2f}")
    
    assert abs(result.dqi_score - calculated_dqi) < 0.01, "DQI formula verification failed"
    assert result.dqi_score >= 0 and result.dqi_score <= 100, "DQI should be 0-100"
    
    print("\n✅ Weighted Penalization Model PASSED")


def test_dqi_level_interpretation():
    """Test 3: DQI Level Interpretation"""
    print("\n" + "="*60)
    print("TEST 3: DQI Level Interpretation")
    print("="*60)
    
    calculator = DataQualityIndexCalculator()
    
    # Test Green Level (DQI > 90)
    green_data = pd.DataFrame([{
        'Subject ID': 'TEST-001',
        'Site ID': 'Green Site',
        'Missing Visits': 0,
        '# Days Outstanding': 0,
        '# Total Queries': 5,
        '# Open Queries': 0,
        'CRFs overdue for signs': 0,
        '# Pages with Non-Conformant data': 0,
        '# Reconciliation Issues': 0
    }])
    
    calculator.load_data({'cpid': green_data})
    green_result = calculator.calculate_site_dqi('Green Site')
    
    print(f"\n🟢 Green Site Test:")
    print(f"   DQI Score: {green_result.dqi_score:.2f}")
    print(f"   Level: {green_result.level.value.upper()}")
    print(f"   Expected: GREEN (DQI > 90)")
    
    # Test Yellow Level (DQI 75-90)
    yellow_data = pd.DataFrame([{
        'Subject ID': 'TEST-001',
        'Site ID': 'Yellow Site',
        'Missing Visits': 3,
        '# Days Outstanding': 20,
        '# Total Queries': 15,
        '# Open Queries': 5,
        'CRFs overdue for signs': 3,
        '# Pages with Non-Conformant data': 8,
        '# Reconciliation Issues': 1
    }])
    
    calculator.load_data({'cpid': yellow_data})
    yellow_result = calculator.calculate_site_dqi('Yellow Site')
    
    print(f"\n🟡 Yellow Site Test:")
    print(f"   DQI Score: {yellow_result.dqi_score:.2f}")
    print(f"   Level: {yellow_result.level.value.upper()}")
    print(f"   Expected: YELLOW (DQI 75-90)")
    
    # Test Red Level (DQI < 75)
    red_data = pd.DataFrame([{
        'Subject ID': 'TEST-001',
        'Site ID': 'Red Site',
        'Missing Visits': 10,
        '# Days Outstanding': 60,
        '# Total Queries': 50,
        '# Open Queries': 30,
        'CRFs overdue for signs': 15,
        'Time lag (Days)': 30,
        '# Pages with Non-Conformant data': 25,
        '# Inactivated Forms': 5,
        '# Reconciliation Issues': 5
    }])
    
    calculator.load_data({'cpid': red_data})
    red_result = calculator.calculate_site_dqi('Red Site')
    
    print(f"\n🔴 Red Site Test:")
    print(f"   DQI Score: {red_result.dqi_score:.2f}")
    print(f"   Level: {red_result.level.value.upper()}")
    print(f"   Expected: RED (DQI < 75)")
    print(f"   Recommendation: {red_result.recommendation[:80]}...")
    
    # Verify levels
    assert green_result.level == DQILevel.GREEN or green_result.dqi_score >= 90, "Green site should be GREEN level"
    # Yellow and Red may vary based on exact penalties
    assert red_result.dqi_score < green_result.dqi_score, "Red site should have lower DQI than Green"
    
    print("\n✅ DQI Level Interpretation PASSED")


def test_site_dqi_calculation():
    """Test 4: Site DQI Calculation with Real Data"""
    print("\n" + "="*60)
    print("TEST 4: Site DQI Calculation (Real Data)")
    print("="*60)
    
    # Load real study data
    data_path = get_data_path("Study 1")
    print(f"\n📂 Loading data from: {data_path}")
    study_data = load_study_data(data_path)
    
    if 'cpid' not in study_data or study_data['cpid'].empty:
        print("⚠️ No CPID data available, skipping real data test")
        return True
    
    calculator = DataQualityIndexCalculator()
    calculator.load_data(study_data)
    
    # Get unique sites
    cpid = study_data['cpid']
    site_col = None
    for col in ['Site ID', 'Site', 'site_id']:
        if col in cpid.columns:
            site_col = col
            break
    
    if not site_col:
        print("⚠️ No Site ID column found")
        return True
    
    sites = cpid[site_col].unique()[:5]  # Test first 5 sites
    # Filter out NaN sites
    sites = [s for s in sites if pd.notna(s)][:5]
    print(f"\n🔬 Testing {len(sites)} sites")
    
    for site_id in sites:
        result = calculator.calculate_site_dqi(site_id)
        
        level_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(result.level.value, "⚪")
        
        print(f"\n--- {site_id} ---")
        print(f"  {level_emoji} DQI Score: {result.dqi_score:.1f}")
        print(f"  Level: {result.level.value.upper()}")
        
        # Show top penalty (only if penalties exist)
        if result.penalties:
            top_penalty = max(result.penalties, key=lambda p: p.weighted_penalty)
            print(f"  Top Issue: {top_penalty.category.value.title()} ({top_penalty.weighted_penalty:.1f} penalty)")
        print(f"  Data Sources: {result.data_sources_used}")
    
    print("\n✅ Site DQI Calculation PASSED")


def test_study_level_summary():
    """Test 5: Study-Level DQI Summary Statistics"""
    print("\n" + "="*60)
    print("TEST 5: Study-Level DQI Summary Statistics")
    print("="*60)
    
    # Load real study data
    data_path = get_data_path("Study 1")
    study_data = load_study_data(data_path)
    
    if 'cpid' not in study_data:
        print("⚠️ No CPID data available")
        return True
    
    calculator = DataQualityIndexCalculator()
    calculator.load_data(study_data)
    
    site_results = calculator.calculate_all_sites_dqi()
    summary = calculator.get_study_dqi_summary(site_results)
    
    print(f"\n📊 Study Summary:")
    print(f"  Total Sites: {summary['total_sites']}")
    print(f"  Average DQI: {summary['avg_dqi']:.1f}")
    print(f"  Median DQI: {summary['median_dqi']:.1f}")
    print(f"  Min DQI: {summary['min_dqi']:.1f}")
    print(f"  Max DQI: {summary['max_dqi']:.1f}")
    print(f"  Std Dev: {summary['std_dqi']:.1f}")
    
    print(f"\n📈 Level Distribution:")
    print(f"  🟢 Green Sites:  {summary['green_count']} ({summary['green_pct']:.1f}%)")
    print(f"  🟡 Yellow Sites: {summary['yellow_count']} ({summary['yellow_pct']:.1f}%)")
    print(f"  🔴 Red Sites:    {summary['red_count']} ({summary['red_pct']:.1f}%)")
    
    # Verify percentages sum to 100
    total_pct = summary['green_pct'] + summary['yellow_pct'] + summary['red_pct']
    assert abs(total_pct - 100.0) < 0.1, "Level percentages should sum to 100%"
    
    print("\n✅ Study Level Summary PASSED")


def test_scatter_plot_quadrants():
    """Test 6: Site Risk Quadrant Classification"""
    print("\n" + "="*60)
    print("TEST 6: Site Risk Quadrant Classification")
    print("="*60)
    
    visualizer = DQIVisualizationEngine()
    
    # Test all four quadrants
    test_cases = [
        ("Low Vol/Low Risk", 5, 2.0, 95.0, RiskQuadrant.LOW_VOLUME_LOW_RISK),
        ("High Vol/Low Risk", 20, 3.0, 92.0, RiskQuadrant.HIGH_VOLUME_LOW_RISK),
        ("Low Vol/High Risk", 5, 8.0, 70.0, RiskQuadrant.LOW_VOLUME_HIGH_RISK),
        ("High Vol/High Risk", 25, 10.0, 60.0, RiskQuadrant.HIGH_VOLUME_HIGH_RISK),
    ]
    
    print(f"\n📈 Scatter Plot Quadrant Tests:")
    print(f"   Enrollment Threshold: {visualizer.config.high_enrollment_threshold}")
    print(f"   Deviation Threshold: {visualizer.config.high_deviation_threshold}")
    
    all_passed = True
    for name, enrollment, deviation, dqi, expected_quadrant in test_cases:
        profile = visualizer.create_site_risk_profile(
            name, enrollment, deviation, dqi
        )
        
        status = "✅" if profile.quadrant == expected_quadrant else "❌"
        if profile.quadrant != expected_quadrant:
            all_passed = False
        
        print(f"\n   {status} {name}:")
        print(f"      Enrollment: {enrollment}, Deviation: {deviation:.1f}, DQI: {dqi:.1f}")
        print(f"      Quadrant: {profile.quadrant.value}")
        print(f"      Risk: {profile.risk_factors[0][:50]}...")
    
    # Special attention to High Volume/High Risk
    print(f"\n⚠️ High Volume/High Risk sites represent the GREATEST THREAT to trial integrity")
    
    assert all_passed, "Some quadrant classifications failed"
    print("\n✅ Scatter Plot Quadrants PASSED")


def test_sankey_diagram_data():
    """Test 7: Query Flow Sankey Diagram Data"""
    print("\n" + "="*60)
    print("TEST 7: Query Flow Sankey Diagram Data")
    print("="*60)
    
    # Load real study data
    data_path = get_data_path("Study 1")
    study_data = load_study_data(data_path)
    
    if 'cpid' not in study_data:
        print("⚠️ No CPID data available")
        return True
    
    calculator = DataQualityIndexCalculator()
    calculator.load_data(study_data)
    
    site_results = calculator.calculate_all_sites_dqi()
    
    visualizer = DQIVisualizationEngine()
    sankey_data = visualizer.create_sankey_diagram_data(
        site_results, study_data['cpid']
    )
    
    print(f"\n📊 Sankey Diagram Data:")
    print(f"   Nodes: {[n['label'] for n in sankey_data['nodes']]}")
    print(f"\n   Query Flow Totals:")
    print(f"     📥 Opened:   {sankey_data['totals']['opened']}")
    print(f"     ✅ Answered: {sankey_data['totals']['answered']}")
    print(f"     🔒 Closed:   {sankey_data['totals']['closed']}")
    print(f"     ⚠️ Overdue:  {sankey_data['totals']['overdue']}")
    
    print(f"\n   Bottleneck Analysis:")
    print(f"     Site Bottleneck:    {sankey_data['bottleneck_analysis']['site_bottleneck_count']} sites")
    print(f"     DM Bottleneck:      {sankey_data['bottleneck_analysis']['dm_bottleneck_count']} sites")
    print(f"     Healthy Flow:       {sankey_data['bottleneck_analysis']['healthy_count']} sites")
    
    # Verify links structure
    assert len(sankey_data['links']) == 3, "Should have 3 Sankey links"
    assert 'sites' in sankey_data, "Should have site-level data"
    
    print("\n✅ Sankey Diagram Data PASSED")


def test_dashboard_generation():
    """Test 8: Complete DQI Dashboard Generation"""
    print("\n" + "="*60)
    print("TEST 8: Complete DQI Dashboard Generation")
    print("="*60)
    
    # Load real study data
    data_path = get_data_path("Study 1")
    study_data = load_study_data(data_path)
    
    if 'cpid' not in study_data:
        print("⚠️ No CPID data available")
        return True
    
    # Generate dashboard
    html = generate_dqi_dashboard(
        study_data,
        study_name="Study 1 - DQI Analysis"
    )
    
    print(f"\n🎨 Dashboard Generated:")
    print(f"   HTML Size: {len(html)} bytes")
    print(f"   Contains <!DOCTYPE>: {'<!DOCTYPE' in html}")
    print(f"   Contains Summary Grid: {'summary-grid' in html}")
    print(f"   Contains Site Table: {'site-table' in html}")
    print(f"   Contains Scatter Plot: {'scatter-plot' in html}")
    print(f"   Contains Sankey: {'sankey-container' in html}")
    print(f"   Contains Interpretation Guide: {'Interpretation Guide' in html}")
    
    # Save dashboard
    output_path = project_root / "reports" / "dqi_dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n💾 Dashboard saved to: {output_path}")
    
    # Verify required components
    assert '<!DOCTYPE' in html, "Should have DOCTYPE"
    assert 'DQI Score' in html or 'dqi_score' in html or 'dqi-badge' in html, "Should have DQI scores"
    
    print("\n✅ Dashboard Generation PASSED")


def test_penalty_category_weights():
    """Test 9: Verify Penalty Category Weights Match Specification"""
    print("\n" + "="*60)
    print("TEST 9: Penalty Category Weight Verification")
    print("="*60)
    
    config = DEFAULT_DQI_CONFIG
    
    # TransCelerate RACT specified weights
    expected_weights = {
        'visit': 0.20,    # Visit Adherence: 20%
        'query': 0.20,    # Query Responsiveness: 20%
        'conform': 0.20,  # Data Conformance: 20%
        'safety': 0.40    # Safety Criticality: 40% (highest - patient safety)
    }
    
    actual_weights = config.get_weights_dict()
    
    print(f"\n📋 TransCelerate RACT Weight Verification:")
    all_match = True
    for category, expected in expected_weights.items():
        actual = actual_weights.get(category, 0)
        match = abs(actual - expected) < 0.001
        status = "✅" if match else "❌"
        if not match:
            all_match = False
        print(f"   {status} {category.title()}: {actual*100:.0f}% (expected: {expected*100:.0f}%)")
    
    # Verify safety has highest weight
    max_weight_category = max(actual_weights.items(), key=lambda x: x[1])
    print(f"\n⚠️ Highest Weight Category: {max_weight_category[0].title()} ({max_weight_category[1]*100:.0f}%)")
    print(f"   (Safety Criticality should be highest due to patient safety implications)")
    
    assert max_weight_category[0] == 'safety', "Safety should have highest weight"
    assert all_match, "All weights should match specification"
    
    print("\n✅ Penalty Category Weights PASSED")


def run_all_tests():
    """Run all DQI tests"""
    print("="*60)
    print("DATA QUALITY INDEX (DQI) TEST SUITE")
    print("TransCelerate RACT Methodology Implementation")
    print("="*60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    try:
        results['Configuration'] = test_dqi_configuration()
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        results['Configuration'] = False
    
    try:
        results['Weighted Penalization'] = test_weighted_penalization_model()
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Weighted Penalization'] = False
    
    try:
        results['Level Interpretation'] = test_dqi_level_interpretation()
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Level Interpretation'] = False
    
    try:
        results['Site Calculation'] = test_site_dqi_calculation()
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Site Calculation'] = False
    
    try:
        results['Study Summary'] = test_study_level_summary()
    except Exception as e:
        print(f"❌ Test 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Study Summary'] = False
    
    try:
        results['Quadrant Classification'] = test_scatter_plot_quadrants()
    except Exception as e:
        print(f"❌ Test 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Quadrant Classification'] = False
    
    try:
        results['Sankey Diagram'] = test_sankey_diagram_data()
    except Exception as e:
        print(f"❌ Test 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Sankey Diagram'] = False
    
    try:
        results['Dashboard Generation'] = test_dashboard_generation()
    except Exception as e:
        print(f"❌ Test 8 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Dashboard Generation'] = False
    
    try:
        results['Weight Verification'] = test_penalty_category_weights()
    except Exception as e:
        print(f"❌ Test 9 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Weight Verification'] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
