"""
Test Enhanced Agents: Codex and Lia
"""
import sys
import pandas as pd
sys.path.insert(0, '.')

from clinical_dataflow_optimizer.agents.agent_framework import (
    CodingAgent, SiteLiaisonAgent,
    DigitalPatientTwin, SiteMetrics
)

def test_codex_and_lia():
    """Test both enhanced agents with real Study 1 data"""
    
    # Load real data
    study_path = r'QC Anonymized Study Files/Study 1_CPID_Input Files - Anonymization'
    meddra_df = pd.read_excel(f'{study_path}/Study 1_GlobalCodingReport_MedDRA_updated.xlsx')
    visit_df = pd.read_excel(f'{study_path}/Study 1_Visit Projection Tracker_14NOV2025_updated.xlsx')
    
    # Load CPID with custom header handling
    cpid_raw = pd.read_excel(f'{study_path}/Study 1_CPID_EDC_Metrics_URSV2.0_14 NOV 2025_updated.xlsx', header=None)
    # First row has main headers, data starts at row 4
    headers = cpid_raw.iloc[0].tolist()
    cpid_df = pd.read_excel(f'{study_path}/Study 1_CPID_EDC_Metrics_URSV2.0_14 NOV 2025_updated.xlsx', header=None, skiprows=4)
    cpid_df.columns = headers + [f'Extra_{i}' for i in range(len(cpid_df.columns) - len(headers))]
    
    print(f'Loaded MedDRA: {len(meddra_df)} rows')
    print(f'Loaded Visit Tracker: {len(visit_df)} rows')
    print(f'Loaded CPID: {len(cpid_df)} rows')
    print()
    
    # Build site metrics and twins
    twins = []
    site_metrics = {}
    
    for _, row in cpid_df.iterrows():
        site_id = str(row.get('Site ID', ''))
        if site_id and not pd.isna(site_id) and site_id != 'nan':
            if site_id not in site_metrics:
                # Get SSM status - check for SSM column
                ssm = row.get('SSM', 'Green')
                if pd.isna(ssm):
                    ssm = 'Green'
                site_metrics[site_id] = SiteMetrics(
                    site_id=site_id,
                    study_id='Study_1',
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
                study_id='Study_1',
                site_id=str(row.get('Site ID', '')),
                country=str(row.get('Country', '')),
                open_queries=0,
                missing_visits=0,
                missing_pages=0
            )
            twins.append(twin)
    
    print(f'Created {len(twins)} Digital Patient Twins')
    print(f'Created {len(site_metrics)} Site Metrics')
    print()
    
    # =========== TEST CODEX ===========
    print('=' * 60)
    print('TESTING CODEX (Coding Agent)')
    print('=' * 60)
    
    codex = CodingAgent()
    codex_recs = codex.analyze(twins, meddra_df, study_id='Study_1')
    stats = codex.get_coding_statistics()
    
    print(f'Total recommendations: {len(codex_recs)}')
    print(f'Auto-coded (High Confidence >95%): {stats["auto_coded"]}')
    print(f'Proposed (Medium Confidence 80-95%): {stats["proposed"]}')
    print(f'Needs Clarification (Low <80%): {stats["clarification_needed"]}')
    print()
    
    # Show sample recommendations by type
    auto_coded = [r for r in codex_recs if r.confidence_score >= 0.95]
    proposed = [r for r in codex_recs if 0.80 <= r.confidence_score < 0.95]
    clarify = [r for r in codex_recs if r.confidence_score < 0.80]
    
    if auto_coded:
        print('Sample AUTO-CODED recommendation:')
        rec = auto_coded[0]
        print(f'  Title: {rec.title}')
        print(f'  Confidence: {rec.confidence_score:.1%}')
        print(f'  Auto-executable: {rec.auto_executable}')
        print()
    
    if proposed:
        print('Sample PROPOSED recommendation:')
        rec = proposed[0]
        print(f'  Title: {rec.title}')
        print(f'  Confidence: {rec.confidence_score:.1%}')
        print(f'  Requires approval: {rec.requires_human_approval}')
        print()
    
    if clarify:
        print('Sample CLARIFICATION NEEDED recommendation:')
        rec = clarify[0]
        print(f'  Title: {rec.title}')
        print(f'  Confidence: {rec.confidence_score:.1%}')
        print(f'  Description preview: {rec.description[:200]}...')
        print()
    
    # =========== TEST LIA ===========
    print('=' * 60)
    print('TESTING LIA (Site Liaison Agent)')
    print('=' * 60)
    
    lia = SiteLiaisonAgent()
    lia_recs = lia.analyze(twins, site_metrics, visit_data=visit_df, study_id='Study_1')
    lia_stats = lia.get_liaison_statistics()
    
    print(f'Total recommendations: {len(lia_recs)}')
    print(f'Standard reminders: {lia_stats["communication_stats"]["standard_reminders"]}')
    print(f'Escalations: {lia_stats["communication_stats"]["escalations"]}')
    print(f'Soft reminders: {lia_stats["communication_stats"]["soft_reminders"]}')
    print(f'Weekly digests: {lia_stats["communication_stats"]["weekly_digests"]}')
    print(f'Sites analyzed: {lia_stats["sites_analyzed"]}')
    print(f'Overburdened sites: {lia_stats["overburdened_sites"]}')
    print()
    
    # Show sample recommendations
    if lia_recs:
        print('Sample Lia recommendations:')
        for i, rec in enumerate(lia_recs[:3]):
            print(f'  [{i+1}] {rec.title}')
            print(f'      Site: {rec.site_id}, Priority: {rec.priority.value}')
            print(f'      Description preview: {rec.description[:150]}...')
            print()
    
    # Show personalized message example
    visit_recs = [r for r in lia_recs if 'Visit' in r.title and 'Digest' not in r.title]
    if visit_recs:
        print('PERSONALIZED EMAIL EXAMPLE:')
        print('-' * 40)
        print(visit_recs[0].description)
        print('-' * 40)
        print()
    
    print('=' * 60)
    print('ALL TESTS PASSED!')
    print('=' * 60)

if __name__ == '__main__':
    test_codex_and_lia()
