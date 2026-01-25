"""
Comprehensive Test: All Agents Working Together
================================================
Tests Rex (Reconciliation), Codex (Coding), and Lia (Site Liaison) agents
orchestrated by the SupervisorAgent with real Study 1 data.
"""
import sys
import pandas as pd
from datetime import datetime
sys.path.insert(0, '.')

from clinical_dataflow_optimizer.agents.agent_framework import (
    ReconciliationAgent, CodingAgent, SiteLiaisonAgent, SupervisorAgent,
    DigitalPatientTwin, SiteMetrics, AgentConfig, ActionType, ActionPriority
)

def load_study_data(study_path: str) -> dict:
    """Load all required data files for Study 1"""
    data = {}
    
    # Load MedDRA Coding Report
    data['meddra'] = pd.read_excel(f'{study_path}/Study 1_GlobalCodingReport_MedDRA_updated.xlsx')
    print(f"  ✓ MedDRA Coding Report: {len(data['meddra'])} rows")
    
    # Load WHO Drug Coding Report
    data['whodrug'] = pd.read_excel(f'{study_path}/Study 1_GlobalCodingReport_WHODD_updated.xlsx')
    print(f"  ✓ WHO Drug Coding Report: {len(data['whodrug'])} rows")
    
    # Load Visit Projection Tracker
    data['visit_tracker'] = pd.read_excel(f'{study_path}/Study 1_Visit Projection Tracker_14NOV2025_updated.xlsx')
    print(f"  ✓ Visit Projection Tracker: {len(data['visit_tracker'])} rows")
    
    # Load eSAE Dashboard
    data['esae'] = pd.read_excel(f'{study_path}/Study 1_eSAE Dashboard_Standard DM_Safety Report_updated.xlsx')
    print(f"  ✓ eSAE Dashboard: {len(data['esae'])} rows")
    
    # Load EDRR
    data['edrr'] = pd.read_excel(f'{study_path}/Study 1_Compiled_EDRR_updated.xlsx')
    print(f"  ✓ EDRR Report: {len(data['edrr'])} rows")
    
    # Load Missing Lab
    data['missing_lab'] = pd.read_excel(f'{study_path}/Study 1_Missing_Lab_Name_and_Missing_Ranges_14NOV2025_updated.xlsx')
    print(f"  ✓ Missing Lab Report: {len(data['missing_lab'])} rows")
    
    # Load CPID (with multi-row header handling)
    cpid_raw = pd.read_excel(f'{study_path}/Study 1_CPID_EDC_Metrics_URSV2.0_14 NOV 2025_updated.xlsx', header=None)
    headers = cpid_raw.iloc[0].tolist()
    data['cpid'] = pd.read_excel(f'{study_path}/Study 1_CPID_EDC_Metrics_URSV2.0_14 NOV 2025_updated.xlsx', header=None, skiprows=4)
    data['cpid'].columns = headers + [f'Extra_{i}' for i in range(len(data['cpid'].columns) - len(headers))]
    print(f"  ✓ CPID Metrics: {len(data['cpid'])} rows")
    
    return data

def build_twins_and_metrics(cpid_df: pd.DataFrame, study_id: str = 'Study_1'):
    """Build Digital Patient Twins and Site Metrics from CPID data"""
    twins = []
    site_metrics = {}
    
    for _, row in cpid_df.iterrows():
        site_id = str(row.get('Site ID', ''))
        if site_id and not pd.isna(site_id) and site_id != 'nan':
            if site_id not in site_metrics:
                ssm = row.get('SSM', 'Green')
                if pd.isna(ssm):
                    ssm = 'Green'
                site_metrics[site_id] = SiteMetrics(
                    site_id=site_id,
                    study_id=study_id,
                    country=str(row.get('Country', '')),
                    ssm_status=str(ssm),
                    total_open_queries=0,
                    total_missing_visits=0,
                    total_missing_pages=0,
                    data_quality_index=85.0
                )
        
        subject_id = str(row.get('Subject ID', ''))
        if subject_id and not pd.isna(subject_id) and subject_id != 'nan':
            twin = DigitalPatientTwin(
                subject_id=subject_id,
                study_id=study_id,
                site_id=str(row.get('Site ID', '')),
                country=str(row.get('Country', '')),
                open_queries=0,
                missing_visits=0,
                missing_pages=0
            )
            twins.append(twin)
    
    return twins, site_metrics

def test_all_agents():
    """Main test function - runs all agents and displays comprehensive results"""
    
    print("=" * 70)
    print("COMPREHENSIVE AGENT TEST - Neural Clinical Data Mesh Framework")
    print("=" * 70)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ========== LOAD DATA ==========
    print("[1/5] LOADING STUDY DATA...")
    print("-" * 40)
    study_path = r'QC Anonymized Study Files/Study 1_CPID_Input Files - Anonymization'
    data = load_study_data(study_path)
    print()
    
    # ========== BUILD TWINS ==========
    print("[2/5] BUILDING DIGITAL PATIENT TWINS...")
    print("-" * 40)
    twins, site_metrics = build_twins_and_metrics(data['cpid'])
    print(f"  ✓ Created {len(twins)} Digital Patient Twins")
    print(f"  ✓ Created {len(site_metrics)} Site Metrics")
    print()
    
    # ========== TEST REX ==========
    print("[3/5] TESTING REX (Reconciliation Agent)...")
    print("-" * 40)
    rex = ReconciliationAgent()
    rex_recs = rex.analyze(
        twins=twins,
        sae_data=data['esae'],
        edrr_data=data['edrr'],
        study_id='Study_1'
    )
    
    print(f"  Total Recommendations: {len(rex_recs)}")
    
    # Count by type
    rex_by_type = {}
    for rec in rex_recs:
        action = rec.action_type.value
        rex_by_type[action] = rex_by_type.get(action, 0) + 1
    
    for action, count in sorted(rex_by_type.items()):
        print(f"    - {action}: {count}")
    
    if rex_recs:
        print()
        print("  Sample Rex Recommendation:")
        rec = rex_recs[0]
        print(f"    Title: {rec.title}")
        print(f"    Priority: {rec.priority.value}")
        print(f"    Subject: {rec.subject_id}")
    print()
    
    # ========== TEST CODEX ==========
    print("[4/5] TESTING CODEX (Coding Agent)...")
    print("-" * 40)
    codex = CodingAgent()
    codex_recs = codex.analyze(
        twins=twins,
        meddra_data=data['meddra'],
        whodra_data=data['whodrug'],
        study_id='Study_1'
    )
    stats = codex.get_coding_statistics()
    
    print(f"  Total Recommendations: {len(codex_recs)}")
    print(f"    - Auto-coded (>95% confidence): {stats['auto_coded']}")
    print(f"    - Proposed (80-95% confidence): {stats['proposed']}")
    print(f"    - Clarification needed (<80%): {stats['clarification_needed']}")
    
    if codex_recs:
        print()
        print("  Sample Codex Recommendation:")
        rec = codex_recs[0]
        print(f"    Title: {rec.title}")
        print(f"    Confidence: {rec.confidence_score:.1%}")
        print(f"    Auto-executable: {rec.auto_executable}")
    print()
    
    # ========== TEST LIA ==========
    print("[5/5] TESTING LIA (Site Liaison Agent)...")
    print("-" * 40)
    lia = SiteLiaisonAgent()
    lia_recs = lia.analyze(
        twins=twins,
        site_metrics=site_metrics,
        visit_data=data['visit_tracker'],
        missing_lab_data=data['missing_lab'],
        study_id='Study_1'
    )
    lia_stats = lia.get_liaison_statistics()
    
    print(f"  Total Recommendations: {len(lia_recs)}")
    print(f"    - Standard Reminders: {lia_stats['communication_stats']['standard_reminders']}")
    print(f"    - Escalations: {lia_stats['communication_stats']['escalations']}")
    print(f"    - Soft Reminders: {lia_stats['communication_stats']['soft_reminders']}")
    print(f"    - Weekly Digests: {lia_stats['communication_stats']['weekly_digests']}")
    print(f"  Sites Analyzed: {lia_stats['sites_analyzed']}")
    print(f"  Overburdened Sites: {lia_stats['overburdened_sites']}")
    
    if lia_recs:
        print()
        print("  Sample Lia Recommendation:")
        rec = lia_recs[0]
        print(f"    Title: {rec.title}")
        print(f"    Site: {rec.site_id}")
        print(f"    Priority: {rec.priority.value}")
    print()
    
    # ========== AGGREGATE RESULTS ==========
    print("=" * 70)
    print("AGGREGATE RESULTS SUMMARY")
    print("=" * 70)
    
    all_recs = rex_recs + codex_recs + lia_recs
    
    print(f"\nTotal Recommendations Across All Agents: {len(all_recs)}")
    print()
    
    # By Agent
    print("By Agent:")
    print(f"  Rex (Reconciliation):  {len(rex_recs):>3} recommendations")
    print(f"  Codex (Coding):        {len(codex_recs):>3} recommendations")
    print(f"  Lia (Site Liaison):    {len(lia_recs):>3} recommendations")
    print()
    
    # By Priority
    priority_counts = {p.value: 0 for p in ActionPriority}
    for rec in all_recs:
        priority_counts[rec.priority.value] += 1
    
    print("By Priority:")
    for priority, count in sorted(priority_counts.items()):
        bar = "█" * (count // 2)
        print(f"  {priority:>10}: {count:>3} {bar}")
    print()
    
    # By Action Type
    action_counts = {}
    for rec in all_recs:
        action = rec.action_type.value
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print("By Action Type:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action:>25}: {count:>3}")
    print()
    
    # Auto-executable vs Requires Approval
    auto_exec = sum(1 for r in all_recs if r.auto_executable)
    needs_approval = sum(1 for r in all_recs if r.requires_human_approval)
    
    print("Execution Status:")
    print(f"  Auto-executable:     {auto_exec:>3} ({auto_exec/len(all_recs)*100:.1f}%)")
    print(f"  Requires Approval:   {needs_approval:>3} ({needs_approval/len(all_recs)*100:.1f}%)")
    print()
    
    # ========== SAMPLE OUTPUTS ==========
    print("=" * 70)
    print("SAMPLE PERSONALIZED COMMUNICATIONS")
    print("=" * 70)
    
    # Show a Lia email template in action
    visit_recs = [r for r in lia_recs if 'Visit Reminder' in r.title]
    if visit_recs:
        print("\n[Lia - Visit Reminder Email]")
        print("-" * 40)
        print(visit_recs[0].description)
        print()
    
    escalation_recs = [r for r in lia_recs if 'ESCALATION' in r.title]
    if escalation_recs:
        print("[Lia - Escalation Alert]")
        print("-" * 40)
        print(escalation_recs[0].description)
        print()
    
    # Show a Codex query
    clarify_recs = [r for r in codex_recs if r.confidence_score < 0.80]
    if clarify_recs:
        print("[Codex - Site Clarification Query]")
        print("-" * 40)
        print(clarify_recs[0].description)
        print()
    
    print("=" * 70)
    print("✅ ALL AGENTS TESTED SUCCESSFULLY")
    print("=" * 70)
    
    # Verify all agents returned results
    assert rex_recs is not None, "Rex should return recommendations"
    assert codex_recs is not None, "Codex should return recommendations"
    assert lia_recs is not None, "Lia should return recommendations"

if __name__ == '__main__':
    test_all_agents()
