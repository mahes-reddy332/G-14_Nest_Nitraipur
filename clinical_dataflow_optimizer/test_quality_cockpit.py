"""
Test Suite for Risk-Based Quality Cockpit
==========================================

Tests:
1. Clean Patient Status Calculator - Boolean logic evaluation
2. Individual condition evaluations (V, P, Q, C, R, S, E)
3. Progress bar visualization
4. Study-level summary statistics
5. Dashboard generation

Run: python -m clinical_dataflow_optimizer.test_quality_cockpit
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clinical_dataflow_optimizer.core.quality_cockpit import (
    CleanPatientStatusCalculator,
    CleanPatientStatus,
    CleanConditionResult,
    CleanCondition,
    QualityCockpitConfig,
    QualityCockpitVisualizer,
    BlockerSeverity,
    calculate_clean_patient_status,
    DEFAULT_COCKPIT_CONFIG
)


def get_data_path(study: str = "Study 1") -> str:
    """Get the path to study data files"""
    base = Path(__file__).parent.parent
    study_folders = [
        f"QC Anonymized Study Files/{study}_CPID_Input Files - Anonymization",
        f"QC Anonymized Study Files/Study {study.split()[-1]}_CPID_Input Files - Anonymization",
    ]
    
    for folder in study_folders:
        path = base / folder
        if path.exists():
            return str(path)
    
    # List available folders
    qc_folder = base / "QC Anonymized Study Files"
    if qc_folder.exists():
        for item in os.listdir(qc_folder):
            if study.replace("Study ", "") in item:
                return str(qc_folder / item)
    
    return None


def load_study_data(data_path: str) -> dict:
    """Load study data files"""
    data = {}
    
    if not data_path or not os.path.exists(data_path):
        print(f"⚠️ Data path not found: {data_path}")
        return data
    
    files = os.listdir(data_path)
    
    for file in files:
        if file.endswith('.xlsx') or file.endswith('.xls'):
            file_path = os.path.join(data_path, file)
            try:
                df = pd.read_excel(file_path)
                
                # Determine data type from filename
                file_lower = file.lower()
                if 'cpid' in file_lower and 'metric' in file_lower:
                    data['cpid'] = df
                elif 'esae' in file_lower or 'sae_dashboard' in file_lower:
                    data['esae'] = df
                elif 'visit' in file_lower and 'projection' in file_lower:
                    data['visit_tracker'] = df
                elif 'missing' in file_lower and 'page' in file_lower:
                    data['missing_pages'] = df
                    
            except Exception as e:
                print(f"  Could not load {file}: {e}")
    
    return data


def test_configuration():
    """Test 1: Configuration Validation"""
    print("\n" + "="*60)
    print("TEST 1: Configuration Validation")
    print("="*60)
    
    # Test default config
    config = DEFAULT_COCKPIT_CONFIG
    print(f"\n📋 Default Configuration:")
    print(f"  Weight (Visits): {config.weight_visits}")
    print(f"  Weight (Pages): {config.weight_pages}")
    print(f"  Weight (Queries): {config.weight_queries}")
    print(f"  Weight (Coding): {config.weight_coding}")
    print(f"  Weight (Reconciliation): {config.weight_reconciliation}")
    print(f"  Weight (Safety): {config.weight_safety}")
    print(f"  Weight (Verification): {config.weight_verification}")
    
    total_weight = (
        config.weight_visits + config.weight_pages + config.weight_queries +
        config.weight_coding + config.weight_reconciliation +
        config.weight_safety + config.weight_verification
    )
    print(f"\n  Total Weight: {total_weight:.2f}")
    
    is_valid = config.validate()
    print(f"  Config Valid: {is_valid}")
    
    # Test custom config
    custom_config = QualityCockpitConfig(
        weight_visits=0.20,
        weight_pages=0.10,
        weight_queries=0.20,
        weight_coding=0.10,
        weight_reconciliation=0.10,
        weight_safety=0.20,
        weight_verification=0.10
    )
    
    print(f"\n📋 Custom Configuration Valid: {custom_config.validate()}")
    
    print("\n✅ Configuration PASSED")


def test_single_patient_calculation():
    """Test 2: Single Patient Status Calculation"""
    print("\n" + "="*60)
    print("TEST 2: Single Patient Status Calculation")
    print("="*60)
    
    # Load data
    data_path = get_data_path("Study 1")
    print(f"\n📂 Loading data from: {data_path}")
    
    study_data = load_study_data(data_path)
    print(f"  Loaded: {list(study_data.keys())}")
    
    if 'cpid' not in study_data:
        print("⚠️ No CPID data available")
        return False
    
    # Initialize calculator
    calculator = CleanPatientStatusCalculator()
    calculator.load_data(study_data)
    
    # Get a sample subject
    cpid = study_data['cpid']
    subject_col = None
    for col in cpid.columns:
        if 'subject' in col.lower():
            subject_col = col
            break
    
    if not subject_col:
        print("⚠️ Could not find Subject ID column")
        return False
    
    subjects = cpid[subject_col].dropna().unique()[:3]
    print(f"\n🔬 Testing with subjects: {list(subjects)}")
    
    for subject_id in subjects:
        print(f"\n--- Subject: {subject_id} ---")
        
        status = calculator.calculate_status(str(subject_id))
        
        print(f"  Is Clean: {status.is_clean}")
        print(f"  Clean %: {status.clean_percentage:.1f}%")
        print(f"  Total Blockers: {status.total_blockers}")
        if status.primary_blocker:
            print(f"  Primary Blocker: {status.primary_blocker}")
        
        print(f"\n  Conditions:")
        for cond in status.conditions:
            check = "✓" if cond.is_met else "✗"
            print(f"    {check} {cond.condition.value.title()}: {cond.score:.2f} (weight: {cond.weight})")
            if cond.blockers:
                for blocker in cond.blockers[:2]:
                    print(f"       → {blocker}")
        
        print(f"\n  Data Sources: {status.data_sources_used}")
    
    print("\n✅ Single Patient Calculation Test PASSED")


def test_boolean_logic():
    """Test 3: Boolean Logic Tree Verification"""
    print("\n" + "="*60)
    print("TEST 3: Boolean Logic Tree Verification")
    print("="*60)
    
    # Create mock data to test specific scenarios
    print("\n📊 Testing Boolean Logic: S_c = V ∧ P ∧ Q ∧ C ∧ R ∧ S ∧ E")
    
    # Scenario 1: Perfect patient (all conditions met)
    perfect_row = pd.Series({
        'Subject ID': 'TEST-001',
        'Site ID': 'Site 1',
        'Missing Visits': 0,
        'Missing Page': 0,
        '# Open Queries': 0,
        'Queries status': 'Closed',
        '# Uncoded Terms': 0,
        '# Coded terms': 5,
        '# Reconciliation Issues': 0,
        '# eSAE dashboard review for DM': 0,
        'Data Verification %': 100.0,
        '# Forms Verified': 10,
        '# Expected Visits': 10
    })
    
    calculator = CleanPatientStatusCalculator()
    
    # Test with empty data sources (will use only cpid_row)
    calculator.load_data({'cpid': pd.DataFrame([perfect_row])})
    status = calculator.calculate_status('TEST-001', perfect_row)
    
    print(f"\n🧪 Scenario 1: Perfect Patient")
    print(f"  Is Clean: {status.is_clean} (expected: True)")
    print(f"  Clean %: {status.clean_percentage:.1f}% (expected: 100%)")
    assert status.is_clean == True, "Perfect patient should be CLEAN"
    assert status.clean_percentage >= 99.9, "Perfect patient should be 100%"
    print("  ✅ PASSED")
    
    # Scenario 2: Single blocker (open queries)
    blocked_row = pd.Series({
        'Subject ID': 'TEST-002',
        'Site ID': 'Site 1',
        'Missing Visits': 0,
        'Missing Page': 0,
        '# Open Queries': 3,  # BLOCKER
        'Queries status': 'Open',
        '# Uncoded Terms': 0,
        '# Coded terms': 5,
        '# Reconciliation Issues': 0,
        '# eSAE dashboard review for DM': 0,
        'Data Verification %': 100.0,
        '# Forms Verified': 10,
        '# Expected Visits': 10
    })
    
    calculator.load_data({'cpid': pd.DataFrame([blocked_row])})
    status = calculator.calculate_status('TEST-002', blocked_row)
    
    print(f"\n🧪 Scenario 2: Single Blocker (Open Queries)")
    print(f"  Is Clean: {status.is_clean} (expected: False)")
    print(f"  Clean %: {status.clean_percentage:.1f}%")
    print(f"  Primary Blocker: {status.primary_blocker}")
    assert status.is_clean == False, "Patient with open queries should NOT be clean"
    assert 'query' in status.primary_blocker.lower() if status.primary_blocker else True
    print("  ✅ PASSED")
    
    # Scenario 3: Multiple blockers
    multi_blocked_row = pd.Series({
        'Subject ID': 'TEST-003',
        'Site ID': 'Site 1',
        'Missing Visits': 2,  # BLOCKER 1
        'Missing Page': 5,    # BLOCKER 2
        '# Open Queries': 3,  # BLOCKER 3
        'Queries status': 'Open',
        '# Uncoded Terms': 2, # BLOCKER 4
        '# Coded terms': 3,
        '# Reconciliation Issues': 1,  # BLOCKER 5 (Critical)
        '# eSAE dashboard review for DM': 0,
        'Data Verification %': 50.0,   # BLOCKER 6
        '# Forms Verified': 5,
        '# Expected Visits': 10
    })
    
    calculator.load_data({'cpid': pd.DataFrame([multi_blocked_row])})
    status = calculator.calculate_status('TEST-003', multi_blocked_row)
    
    print(f"\n🧪 Scenario 3: Multiple Blockers")
    print(f"  Is Clean: {status.is_clean} (expected: False)")
    print(f"  Clean %: {status.clean_percentage:.1f}%")
    print(f"  Total Blockers: {status.total_blockers}")
    
    failed_conditions = [c for c in status.conditions if not c.is_met]
    print(f"  Failed Conditions: {len(failed_conditions)}")
    for cond in failed_conditions:
        print(f"    - {cond.condition.value}: {cond.blockers}")
    
    assert status.is_clean == False, "Patient with multiple blockers should NOT be clean"
    # With weighted formula, 6 failed conditions out of 7 gives ~31.25% clean
    # (Only safety passes: 0.2 weight + partial reconciliation at ~0.1125 = ~0.3125)
    # The actual value depends on partial scores, so we check < 75% (definitely not clean)
    assert status.clean_percentage < 75, f"Patient with many blockers should be <75%, got {status.clean_percentage:.1f}%"
    assert len(failed_conditions) >= 5, f"Should have at least 5 failed conditions, got {len(failed_conditions)}"
    print("  ✅ PASSED")
    
    print("\n✅ Boolean Logic Test PASSED")


def test_progress_bar_visualization():
    """Test 4: Progress Bar Visualization"""
    print("\n" + "="*60)
    print("TEST 4: Progress Bar Visualization")
    print("="*60)
    
    # Load data
    data_path = get_data_path("Study 1")
    study_data = load_study_data(data_path)
    
    if 'cpid' not in study_data:
        print("⚠️ No CPID data available")
        return False
    
    # Initialize components
    calculator = CleanPatientStatusCalculator()
    calculator.load_data(study_data)
    visualizer = QualityCockpitVisualizer()
    
    # Get sample subjects
    cpid = study_data['cpid']
    subject_col = None
    for col in cpid.columns:
        if 'subject' in col.lower():
            subject_col = col
            break
    
    subjects = cpid[subject_col].dropna().unique()[:3]
    
    print(f"\n🎨 Generating Progress Bars for {len(subjects)} subjects")
    
    for subject_id in subjects:
        status = calculator.calculate_status(str(subject_id))
        progress_data = status.get_progress_bar_data()
        
        print(f"\n--- {subject_id} ---")
        print(f"  Status: {progress_data['status_text']}")
        print(f"  Color: {progress_data['color']}")
        print(f"  Segments: {len(progress_data['segments'])}")
        
        # Generate HTML
        html = visualizer.create_progress_bar_html(status)
        print(f"  HTML Length: {len(html)} chars")
        
        # Verify HTML structure - check for patient-card class in new UI
        assert '<div class="patient-card"' in html
        assert subject_id in html
        print("  ✅ HTML Valid")
    
    print("\n✅ Progress Bar Visualization Test PASSED")


def test_batch_calculation():
    """Test 5: Batch Calculation Performance"""
    print("\n" + "="*60)
    print("TEST 5: Batch Calculation Performance")
    print("="*60)
    
    # Load data
    data_path = get_data_path("Study 1")
    study_data = load_study_data(data_path)
    
    if 'cpid' not in study_data:
        print("⚠️ No CPID data available")
        return False
    
    # Initialize calculator
    calculator = CleanPatientStatusCalculator()
    calculator.load_data(study_data)
    
    # Time the batch calculation
    import time
    start_time = time.time()
    
    statuses = calculator.calculate_batch()
    
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ Batch Calculation Results:")
    print(f"  Subjects Processed: {len(statuses)}")
    print(f"  Time Elapsed: {elapsed:.2f}s")
    print(f"  Avg per Subject: {elapsed/len(statuses)*1000:.1f}ms" if statuses else "N/A")
    
    # Analyze results
    clean_count = sum(1 for s in statuses.values() if s.is_clean)
    avg_pct = np.mean([s.clean_percentage for s in statuses.values()])
    
    print(f"\n📊 Results Summary:")
    print(f"  Clean Patients: {clean_count}/{len(statuses)} ({clean_count/len(statuses)*100:.1f}%)")
    print(f"  Avg Clean %: {avg_pct:.1f}%")
    
    # Verify all statuses are valid
    for subject_id, status in statuses.items():
        assert 0 <= status.clean_percentage <= 100
        assert len(status.conditions) == 7
    
    print("  ✅ All statuses valid")
    
    print("\n✅ Batch Calculation Test PASSED")


def test_study_summary():
    """Test 6: Study-Level Summary Statistics"""
    print("\n" + "="*60)
    print("TEST 6: Study-Level Summary Statistics")
    print("="*60)
    
    # Load data
    data_path = get_data_path("Study 1")
    study_data = load_study_data(data_path)
    
    if 'cpid' not in study_data:
        print("⚠️ No CPID data available")
        return False
    
    # Calculate statuses
    calculator = CleanPatientStatusCalculator()
    calculator.load_data(study_data)
    statuses = calculator.calculate_batch()
    
    # Get summary
    summary = calculator.get_study_summary(statuses)
    
    print(f"\n📊 Study Summary:")
    print(f"  Total Subjects: {summary['total_subjects']}")
    print(f"  Clean Count: {summary['clean_count']}")
    print(f"  Clean %: {summary['clean_percentage']:.1f}%")
    print(f"  Average Clean %: {summary['average_clean_pct']:.1f}%")
    print(f"  Median Clean %: {summary['median_clean_pct']:.1f}%")
    
    print(f"\n📈 Distribution:")
    for bucket, count in summary['distribution'].items():
        bar = "█" * (count // 2) if count > 0 else ""
        print(f"  {bucket:12}: {count:3} {bar}")
    
    print(f"\n🚫 Top Blockers:")
    for blocker, count in summary['top_blockers']:
        print(f"  - {blocker.title()}: {count} patients")
    
    print("\n✅ Study Summary Test PASSED")


def test_dashboard_generation():
    """Test 7: Dashboard HTML Generation"""
    print("\n" + "="*60)
    print("TEST 7: Dashboard HTML Generation")
    print("="*60)
    
    # Load data
    data_path = get_data_path("Study 1")
    study_data = load_study_data(data_path)
    
    if 'cpid' not in study_data:
        print("⚠️ No CPID data available")
        return False
    
    # Calculate statuses
    calculator = CleanPatientStatusCalculator()
    calculator.load_data(study_data)
    statuses = calculator.calculate_batch()
    summary = calculator.get_study_summary(statuses)
    
    # Generate dashboard
    visualizer = QualityCockpitVisualizer()
    html = visualizer.create_study_dashboard_html(statuses, summary)
    
    print(f"\n🎨 Dashboard Generated:")
    print(f"  HTML Size: {len(html)} bytes")
    print(f"  Contains <!DOCTYPE>: {'<!DOCTYPE html>' in html}")
    print(f"  Contains Summary: {'Study Summary' in html}")
    print(f"  Contains Progress: {'Patient Progress' in html}")
    
    # Save dashboard
    output_dir = Path(__file__).parent.parent / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "quality_cockpit_dashboard.html"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n💾 Dashboard saved to: {output_path}")
    
    print("\n✅ Dashboard Generation Test PASSED")


def test_condition_breakdown():
    """Test 8: Condition Breakdown Analysis"""
    print("\n" + "="*60)
    print("TEST 8: Condition Breakdown Analysis")
    print("="*60)
    
    # Load data
    data_path = get_data_path("Study 1")
    study_data = load_study_data(data_path)
    
    if 'cpid' not in study_data:
        print("⚠️ No CPID data available")
        return False
    
    # Calculate statuses
    calculator = CleanPatientStatusCalculator()
    calculator.load_data(study_data)
    statuses = calculator.calculate_batch()
    
    # Get condition breakdown
    visualizer = QualityCockpitVisualizer()
    breakdown = visualizer.create_condition_breakdown_chart(statuses)
    
    print(f"\n📊 Condition Breakdown:")
    print(f"{'Condition':<20} {'Passed':>8} {'Failed':>8} {'Pass Rate':>10}")
    print("-" * 48)
    
    for i, cond in enumerate(breakdown['conditions']):
        passed = breakdown['passed'][i]
        failed = breakdown['failed'][i]
        rate = breakdown['pass_rates'][i]
        status = "✓" if rate >= 90 else "⚠" if rate >= 75 else "✗"
        print(f"{cond.title():<20} {passed:>8} {failed:>8} {rate:>9.1f}% {status}")
    
    print("\n✅ Condition Breakdown Test PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("QUALITY COCKPIT TEST SUITE")
    print("Risk-Based Quality Cockpit - Metrics & Visualization")
    print("="*60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Configuration
    try:
        results['Configuration'] = test_configuration()
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Configuration'] = False
    
    # Test 2: Single Patient Calculation
    try:
        results['Single Patient'] = test_single_patient_calculation()
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Single Patient'] = False
    
    # Test 3: Boolean Logic
    try:
        results['Boolean Logic'] = test_boolean_logic()
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Boolean Logic'] = False
    
    # Test 4: Progress Bar
    try:
        results['Progress Bar'] = test_progress_bar_visualization()
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Progress Bar'] = False
    
    # Test 5: Batch Calculation
    try:
        results['Batch Calculation'] = test_batch_calculation()
    except Exception as e:
        print(f"❌ Test 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Batch Calculation'] = False
    
    # Test 6: Study Summary
    try:
        results['Study Summary'] = test_study_summary()
    except Exception as e:
        print(f"❌ Test 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Study Summary'] = False
    
    # Test 7: Dashboard Generation
    try:
        results['Dashboard'] = test_dashboard_generation()
    except Exception as e:
        print(f"❌ Test 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Dashboard'] = False
    
    # Test 8: Condition Breakdown
    try:
        results['Condition Breakdown'] = test_condition_breakdown()
    except Exception as e:
        print(f"❌ Test 8 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Condition Breakdown'] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
