"""
Test Suite for Narratives Module
=================================

Tests:
1. Patient Narrative Generator - Medical Monitor narratives from SAE data
2. RBM Report Generator - CRA Visit Letters and Monitoring Reports

Run: python -m clinical_dataflow_optimizer.test_narratives
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clinical_dataflow_optimizer.narratives import (
    PatientNarrativeGenerator,
    PatientSafetyNarrative,
    NarrativeSeverity,
    RBMReportGenerator,
    MonitoringReport,
    CRAVisitLetter,
    RiskCategory,
    IssueType
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
                elif 'meddra' in file_lower:
                    data['meddra'] = df
                elif 'whodd' in file_lower or 'coding' in file_lower:
                    data['whodd'] = df
                elif 'missing' in file_lower and 'lab' in file_lower:
                    data['missing_lab'] = df
                elif 'visit' in file_lower:
                    data['visit_tracker'] = df
                    
            except Exception as e:
                print(f"  Could not load {file}: {e}")
    
    return data


def test_patient_narrative_generator():
    """Test 1: Patient Narrative Generator"""
    print("\n" + "="*60)
    print("TEST 1: Patient Narrative Generator")
    print("="*60)
    
    # Load Study 1 data
    data_path = get_data_path("Study 1")
    print(f"\n📂 Loading data from: {data_path}")
    
    study_data = load_study_data(data_path)
    print(f"  Loaded: {list(study_data.keys())}")
    
    # Initialize generator
    generator = PatientNarrativeGenerator()
    generator.load_data(study_data)
    
    # Get sample subjects
    cpid = study_data.get('cpid')
    if cpid is None:
        print("⚠️ No CPID data available")
        return False
    
    # Find subject column
    subject_col = None
    for col in cpid.columns:
        if 'subject' in col.lower():
            subject_col = col
            break
    
    if not subject_col:
        print("⚠️ Could not find Subject ID column")
        return False
    
    subjects = cpid[subject_col].dropna().unique()[:5]
    print(f"\n🔬 Testing with {len(subjects)} subjects")
    
    # Generate individual narrative
    print("\n--- Single Narrative Test ---")
    if len(subjects) > 0:
        test_subject = str(subjects[0])
        print(f"Generating narrative for: {test_subject}")
        
        narrative = generator.generate_narrative(test_subject)
        
        if narrative:
            print(f"\n✅ Generated Narrative:")
            print(f"  Subject: {narrative.subject_id}")
            print(f"  Severity: {narrative.severity.value}")
            print(f"  Key Findings: {len(narrative.key_findings)}")
            print(f"  Recommendations: {len(narrative.recommended_actions)}")
            print(f"\n📝 Narrative Preview (first 500 chars):")
            print("-" * 40)
            print(narrative.narrative_text[:500] + "...")
            print("-" * 40)
        else:
            print("⚠️ No narrative generated")
    
    # Generate batch narratives
    print("\n--- Batch Narrative Test ---")
    subject_list = [str(s) for s in subjects[:3]]
    narratives = generator.generate_batch_narratives(subject_list)
    
    print(f"Generated {len(narratives)} narratives")
    for narrative in narratives:
        print(f"  - {narrative.subject_id}: {narrative.severity.value} ({len(narrative.key_findings)} findings)")
    
    # Summary report
    print("\n--- Summary Report ---")
    summary = generator.get_summary_report(narratives)
    print(summary)
    
    print("\n✅ Patient Narrative Generator PASSED")


def test_rbm_report_generator():
    """Test 2: RBM Report Generator"""
    print("\n" + "="*60)
    print("TEST 2: RBM Report Generator")
    print("="*60)
    
    # Load Study 1 data
    data_path = get_data_path("Study 1")
    study_data = load_study_data(data_path)
    
    # Initialize generator
    generator = RBMReportGenerator()
    generator.load_data(study_data)
    
    # Get CPID data
    cpid = study_data.get('cpid')
    if cpid is None:
        print("⚠️ No CPID data available")
        return False
    
    # Find site column
    site_col = None
    for col in cpid.columns:
        if 'site' in col.lower():
            site_col = col
            break
    
    if not site_col:
        print("⚠️ Could not find Site ID column")
        return False
    
    sites = cpid[site_col].dropna().unique()[:3]
    print(f"\n🏥 Testing with {len(sites)} sites")
    
    # Test site risk analysis
    print("\n--- Site Risk Analysis Test ---")
    if len(sites) > 0:
        test_site = str(sites[0])
        print(f"Analyzing risk for site: {test_site}")
        
        profile = generator.analyze_site_risk(test_site)
        
        print(f"\n✅ Site Risk Profile:")
        print(f"  Site ID: {profile.site_id}")
        print(f"  Total Subjects: {profile.total_subjects}")
        print(f"  Overall Risk: {profile.overall_risk.value}")
        print(f"  Protocol Deviations: {profile.protocol_deviations}")
        print(f"  CRFs Overdue (>90 days): {profile.crfs_overdue_signature}")
        print(f"  Broken Signatures: {profile.broken_signatures}")
        print(f"  Open Queries: {profile.open_queries}")
        print(f"  Issues Identified: {len(profile.issues)}")
        
        for issue in profile.issues[:3]:
            print(f"    - [{issue.risk_category.value.upper()}] {issue.issue_type.value}: {issue.description[:60]}...")
    
    # Test CRA Visit Letter
    print("\n--- CRA Visit Letter Test ---")
    if len(sites) > 0:
        test_site = str(sites[0])
        print(f"Generating visit letter for site: {test_site}")
        
        letter = generator.generate_visit_letter(test_site)
        
        print(f"\n✅ CRA Visit Letter:")
        print(f"  Site: {letter.site_id}")
        print(f"  Total Issues: {letter.total_issues}")
        print(f"  Critical Issues: {letter.critical_issues}")
        print(f"  Priority Items: {len(letter.priority_items)}")
        
        print(f"\n📝 Letter Preview:")
        print("-" * 40)
        letter_text = letter.to_letter_format()
        print(letter_text[:800] + "..." if len(letter_text) > 800 else letter_text)
        print("-" * 40)
    
    # Test Monitoring Report
    print("\n--- Monitoring Report Test ---")
    if len(sites) > 0:
        test_site = str(sites[0])
        print(f"Generating monitoring report for site: {test_site}")
        
        report = generator.generate_monitoring_report(test_site)
        
        print(f"\n✅ Monitoring Report:")
        print(f"  Report ID: {report.report_id}")
        print(f"  Site: {report.site_id}")
        print(f"  Risk Rating: {report.overall_risk_rating.value}")
        print(f"  Findings: {len(report.compliance_findings)}")
        print(f"  Recommendations: {len(report.recommendations)}")
        print(f"  Follow-ups: {len(report.follow_up_items)}")
        
        print(f"\n📊 Metrics Summary:")
        for metric, value in report.metrics_summary.items():
            if isinstance(value, dict):
                print(f"  {metric}: {value['value']} {value['status']}")
            else:
                print(f"  {metric}: {value}")
        
        print(f"\n📝 Executive Summary:")
        print(report.executive_summary)
    
    # Test batch reports
    print("\n--- Batch Report Generation Test ---")
    site_list = [str(s) for s in sites[:2]]
    reports = generator.generate_batch_reports(site_list)
    
    print(f"Generated {len(reports)} reports")
    for site_id, report in reports.items():
        print(f"  - {site_id}: {report.overall_risk_rating.value} ({len(report.compliance_findings)} findings)")
    
    # Portfolio summary
    print("\n--- Portfolio Summary ---")
    summary = generator.get_portfolio_summary(reports)
    print(summary)
    
    print("\n✅ RBM Report Generator PASSED")


def test_integration():
    """Test 3: Integration test - combining both generators"""
    print("\n" + "="*60)
    print("TEST 3: Integration Test")
    print("="*60)
    
    # Load data
    data_path = get_data_path("Study 1")
    study_data = load_study_data(data_path)
    
    # Initialize both generators
    narrative_gen = PatientNarrativeGenerator()
    narrative_gen.load_data(study_data)
    
    rbm_gen = RBMReportGenerator()
    rbm_gen.load_data(study_data)
    
    # Get CPID data
    cpid = study_data.get('cpid')
    if cpid is None:
        print("⚠️ No CPID data available")
        return False
    
    # Find columns
    site_col = None
    subject_col = None
    for col in cpid.columns:
        if 'site' in col.lower() and site_col is None:
            site_col = col
        if 'subject' in col.lower() and subject_col is None:
            subject_col = col
    
    if not site_col or not subject_col:
        print("⚠️ Could not find required columns")
        return False
    
    # Get a site and its subjects
    test_site = str(cpid[site_col].dropna().iloc[0])
    site_subjects = cpid[cpid[site_col] == cpid[site_col].iloc[0]][subject_col].dropna().unique()[:3]
    
    print(f"\n🔗 Testing integration for site: {test_site}")
    print(f"   Subjects: {list(site_subjects)}")
    
    # Generate RBM report for site
    print("\n📊 Generating site monitoring report...")
    monitoring_report = rbm_gen.generate_monitoring_report(test_site)
    print(f"   Site Risk: {monitoring_report.overall_risk_rating.value}")
    print(f"   Findings: {len(monitoring_report.compliance_findings)}")
    
    # Generate patient narratives for subjects at that site
    print("\n📝 Generating patient narratives for site subjects...")
    narratives = narrative_gen.generate_batch_narratives([str(s) for s in site_subjects])
    print(f"   Narratives generated: {len(narratives)}")
    
    # Combined output
    print("\n" + "="*40)
    print("COMBINED SITE REPORT")
    print("="*40)
    
    print(f"\n## Site {test_site} - Overall Risk: {monitoring_report.overall_risk_rating.value.upper()}")
    print(f"\n### Site-Level Findings ({len(monitoring_report.compliance_findings)} issues)")
    for finding in monitoring_report.compliance_findings[:3]:
        print(f"- [{finding.risk_category.value}] {finding.description[:80]}...")
    
    print(f"\n### Subject Narratives ({len(narratives)} subjects)")
    for narrative in narratives:
        print(f"\n**{narrative.subject_id}** ({narrative.severity.value}):")
        print(f"  {narrative.narrative_text[:200]}...")
    
    print("\n✅ Integration PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("NARRATIVES MODULE TEST SUITE")
    print("="*60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Patient Narrative Generator
    try:
        results['Patient Narrative Generator'] = test_patient_narrative_generator()
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Patient Narrative Generator'] = False
    
    # Test 2: RBM Report Generator
    try:
        results['RBM Report Generator'] = test_rbm_report_generator()
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['RBM Report Generator'] = False
    
    # Test 3: Integration
    try:
        results['Integration'] = test_integration()
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Integration'] = False
    
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
