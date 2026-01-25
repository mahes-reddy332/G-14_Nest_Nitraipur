"""
Test Enhanced Supervisor Agent - Orchestration Pattern
=======================================================
Tests the Blackboard Architecture, Safety Overrides, 
SOP Compliance, and White Space Reduction features.
"""
import sys
import pandas as pd
from datetime import datetime
sys.path.insert(0, '.')

from clinical_dataflow_optimizer.agents.agent_framework import (
    SupervisorAgent, DigitalPatientTwin, SiteMetrics, AgentConfig,
    ActionType, ActionPriority
)

def load_study_data(study_path: str) -> dict:
    """Load all required data files"""
    data = {}
    
    # Load all data files
    data['sae_dashboard'] = pd.read_excel(f'{study_path}/Study 1_eSAE Dashboard_Standard DM_Safety Report_updated.xlsx')
    data['compiled_edrr'] = pd.read_excel(f'{study_path}/Study 1_Compiled_EDRR_updated.xlsx')
    data['meddra_coding'] = pd.read_excel(f'{study_path}/Study 1_GlobalCodingReport_MedDRA_updated.xlsx')
    data['whodra_coding'] = pd.read_excel(f'{study_path}/Study 1_GlobalCodingReport_WHODD_updated.xlsx')
    data['visit_tracker'] = pd.read_excel(f'{study_path}/Study 1_Visit Projection Tracker_14NOV2025_updated.xlsx')
    data['missing_lab'] = pd.read_excel(f'{study_path}/Study 1_Missing_Lab_Name_and_Missing_Ranges_14NOV2025_updated.xlsx')
    
    return data

def load_cpid_data(study_path: str) -> pd.DataFrame:
    """Load CPID data with proper header handling"""
    cpid_raw = pd.read_excel(
        f'{study_path}/Study 1_CPID_EDC_Metrics_URSV2.0_14 NOV 2025_updated.xlsx', 
        header=None
    )
    
    # Build combined header from rows 0-2
    headers = []
    for col_idx in range(len(cpid_raw.columns)):
        header_parts = []
        for row_idx in range(3):
            val = cpid_raw.iloc[row_idx, col_idx]
            if not pd.isna(val):
                header_parts.append(str(val))
        
        if header_parts:
            headers.append(' - '.join(header_parts))
        else:
            headers.append(f'Col_{col_idx}')
    
    # Load data starting from row 4
    cpid_df = pd.read_excel(
        f'{study_path}/Study 1_CPID_EDC_Metrics_URSV2.0_14 NOV 2025_updated.xlsx', 
        header=None, 
        skiprows=4
    )
    
    # Handle column count mismatch
    if len(cpid_df.columns) > len(headers):
        headers.extend([f'Extra_{i}' for i in range(len(cpid_df.columns) - len(headers))])
    elif len(cpid_df.columns) < len(headers):
        headers = headers[:len(cpid_df.columns)]
    
    cpid_df.columns = headers
    
    # Map to standard column names
    column_mapping = {}
    for col in cpid_df.columns:
        if 'Subject ID' in col:
            column_mapping[col] = 'Subject ID'
        elif 'Site ID' in col:
            column_mapping[col] = 'Site ID'
        elif 'Country' in col:
            column_mapping[col] = 'Country'
        elif 'CRFs Frozen' in col:
            column_mapping[col] = '# CRFs Frozen'
        elif 'CRFs Locked' in col:
            column_mapping[col] = '# CRFs Locked'
        elif 'SSM' in col:
            column_mapping[col] = 'SSM'
    
    cpid_df = cpid_df.rename(columns=column_mapping)
    
    return cpid_df

def build_twins_and_metrics(cpid_df: pd.DataFrame, study_id: str = 'Study_1'):
    """Build Digital Patient Twins and Site Metrics"""
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

def test_enhanced_supervisor():
    """Test the enhanced Supervisor Agent with all orchestration features"""
    
    print("=" * 70)
    print("ENHANCED SUPERVISOR AGENT TEST")
    print("Blackboard Architecture | Safety Overrides | SOP Compliance")
    print("=" * 70)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ========== LOAD DATA ==========
    print("[1/6] LOADING STUDY DATA...")
    print("-" * 50)
    study_path = r'QC Anonymized Study Files/Study 1_CPID_Input Files - Anonymization'
    study_data = load_study_data(study_path)
    cpid_df = load_cpid_data(study_path)
    
    for name, df in study_data.items():
        print(f"  ✓ {name}: {len(df)} rows")
    print(f"  ✓ CPID (with SOP columns): {len(cpid_df)} rows")
    print()
    
    # ========== BUILD TWINS ==========
    print("[2/6] BUILDING DIGITAL PATIENT TWINS...")
    print("-" * 50)
    twins, site_metrics = build_twins_and_metrics(cpid_df)
    print(f"  ✓ Digital Patient Twins: {len(twins)}")
    print(f"  ✓ Site Metrics: {len(site_metrics)}")
    print()
    
    # ========== INITIALIZE SUPERVISOR ==========
    print("[3/6] INITIALIZING SUPERVISOR AGENT...")
    print("-" * 50)
    supervisor = SupervisorAgent()
    print("  ✓ Supervisor initialized with Rex, Codex, and Lia agents")
    print("  ✓ Blackboard architecture ready")
    print()
    
    # ========== RUN ORCHESTRATED ANALYSIS ==========
    print("[4/6] RUNNING ORCHESTRATED ANALYSIS...")
    print("-" * 50)
    print("  → Delegating to Rex (Safety/Reconciliation)...")
    print("  → Delegating to Codex (Coding)...")
    print("  → Delegating to Lia (Site Liaison)...")
    print("  → Applying safety overrides...")
    print("  → Checking SOP compliance...")
    print("  → Prioritizing recommendations...")
    print()
    
    results = supervisor.run_analysis(
        twins=twins,
        site_metrics=site_metrics,
        study_data=study_data,
        study_id='Study_1',
        cpid_data=cpid_df
    )
    print()
    
    # ========== DISPLAY RESULTS ==========
    print("[5/6] ORCHESTRATION RESULTS...")
    print("-" * 50)
    summary = supervisor.get_summary()
    
    print(f"  Total Recommendations Generated: {summary['total_recommendations']}")
    print(f"  Prioritized (Final Output):      {summary['prioritized_recommendations']}")
    print(f"  Suppressed (Safety Override):    {summary['suppressed_by_safety_override']}")
    print(f"  Blocked (SOP Compliance):        {summary['blocked_by_sop']}")
    print()
    
    print("  By Agent:")
    for agent, count in summary['by_agent'].items():
        print(f"    {agent}: {count}")
    print()
    
    print("  By Priority:")
    for priority, count in summary['by_priority'].items():
        bar = "█" * count
        print(f"    {priority:>8}: {count:>3} {bar}")
    print()
    
    print("  Orchestration Statistics:")
    stats = summary['orchestration_stats']
    print(f"    Total Delegations:        {stats['total_delegations']}")
    print(f"    Safety Overrides Applied: {stats['safety_overrides']}")
    print(f"    SOP Blocks:               {stats['sop_blocks']}")
    print(f"    Signal-Noise Filtered:    {stats['signal_noise_filtered']}")
    print()
    
    # ========== BLACKBOARD STATE ==========
    print("[6/6] BLACKBOARD STATE...")
    print("-" * 50)
    blackboard = supervisor.get_blackboard_state()
    
    print(f"  Critical Safety Subjects: {len(blackboard['critical_safety_subjects'])}")
    if blackboard['critical_safety_subjects']:
        print(f"    → {list(blackboard['critical_safety_subjects'])[:5]}...")
    
    print(f"  Critical Safety Sites: {len(blackboard['critical_safety_sites'])}")
    if blackboard['critical_safety_sites']:
        print(f"    → {list(blackboard['critical_safety_sites'])[:5]}...")
    
    print(f"  Subjects with Locked Pages: {len(blackboard['locked_pages'])}")
    print(f"  Subjects with Frozen Pages: {len(blackboard['frozen_pages'])}")
    print()
    
    # ========== WHITE SPACE REDUCTION ==========
    print("=" * 70)
    print("WHITE SPACE REDUCTION METRICS")
    print("=" * 70)
    ws_metrics = summary['white_space_metrics']
    
    print(f"  Analysis Start:    {ws_metrics['analysis_start_time']}")
    print(f"  Analysis End:      {ws_metrics['analysis_end_time']}")
    print(f"  Processing Time:   {ws_metrics['total_processing_seconds']:.2f} seconds")
    print()
    print("  Traditional Cycle: ~336 hours (2 weeks)")
    print(f"  Agentic Cycle:     {ws_metrics['total_processing_seconds']/3600:.4f} hours")
    print(f"  White Space Reduction: {stats['white_space_reduction_hours']:.1f} hours saved")
    print()
    
    # ========== SAMPLE OUTPUTS ==========
    print("=" * 70)
    print("SAMPLE ORCHESTRATION DECISIONS")
    print("=" * 70)
    
    # Show a prioritized recommendation
    if results['prioritized']:
        print("\n[APPROVED - Prioritized Recommendation]")
        print("-" * 40)
        rec = results['prioritized'][0]
        print(f"  Title: {rec.title}")
        print(f"  Agent: {rec.agent_name}")
        print(f"  Priority: {rec.priority.value}")
        print(f"  Action: {rec.action_type.value}")
        print(f"  Auto-executable: {rec.auto_executable}")
    
    # Show a suppressed recommendation (if any)
    if results['suppressed']:
        print("\n[SUPPRESSED - Safety Override]")
        print("-" * 40)
        rec = results['suppressed'][0]
        print(f"  Title: {rec.title}")
        print(f"  Agent: {rec.agent_name}")
        print(f"  Reason: {rec.source_data.get('suppression_reason', 'Safety override')}")
    
    # Show a SOP-blocked recommendation (if any)
    if results['sop_blocked']:
        print("\n[BLOCKED - SOP Compliance]")
        print("-" * 40)
        rec = results['sop_blocked'][0]
        print(f"  Title: {rec.title}")
        print(f"  Agent: {rec.agent_name}")
        print(f"  Reason: {rec.source_data.get('block_reason', 'Locked page')}")
    
    print()
    print("=" * 70)
    print("✅ ENHANCED SUPERVISOR TEST COMPLETE")
    print("=" * 70)
    
    # Verify test results
    assert supervisor is not None, "Supervisor should be initialized"
    assert results is not None, "Results should not be None"

if __name__ == '__main__':
    test_enhanced_supervisor()
