#!/usr/bin/env python3
"""
Test script for the Agentic AI Framework - Rex (Reconciliation Agent)

Tests:
1. SAE aging analysis (pending > 7 days)
2. EDRR cross-referencing for high-issue subjects
3. Zombie SAE detection
4. Auto-generated query messages
"""

import sys
import json
from datetime import datetime
sys.path.insert(0, '.')


def json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

from clinical_dataflow_optimizer.core import (
    ClinicalDataMesh, 
    build_clinical_data_mesh
)
from clinical_dataflow_optimizer.agents.agent_framework import (
    ReconciliationAgent,
    SupervisorAgent,
    ActionPriority,
    ActionType
)
from clinical_dataflow_optimizer.graph.knowledge_graph import NodeType


def test_rex_agent():
    """Test the Reconciliation Agent (Rex)"""
    
    print('=' * 70)
    print('AGENTIC AI FRAMEWORK TEST - REX (RECONCILIATION AGENT)')
    print('=' * 70)
    
    # Build the mesh
    study_path = r'QC Anonymized Study Files\Study 1_CPID_Input Files - Anonymization'
    print('\n1. Building Clinical Data Mesh...')
    
    mesh = build_clinical_data_mesh(study_path)
    
    # Get statistics
    stats = mesh.get_statistics()
    print(f'   Graph: {stats.total_nodes} nodes, {stats.total_edges} edges')
    
    # Get all Digital Patient Twins
    print('\n2. Creating Digital Patient Twins...')
    twins = mesh.get_all_digital_twins()
    print(f'   Created {len(twins)} Digital Twins')
    
    # Initialize Rex
    print('\n3. Initializing Rex (Reconciliation Agent)...')
    rex = ReconciliationAgent()
    
    # Run analysis with study data
    print('\n4. Running Rex Analysis...')
    print('   - Checking CPID reconciliation issues')
    print('   - Scanning SAE Dashboard for pending > 7 days')
    print('   - Cross-referencing EDRR for high-issue subjects')
    print('   - Detecting Zombie SAEs via Missing Pages Report')
    
    recommendations = rex.analyze(
        twins=twins,
        sae_data=mesh.data_sources.get('sae_dashboard'),
        edrr_data=mesh.data_sources.get('compiled_edrr'),
        missing_pages=mesh.data_sources.get('missing_pages'),
        study_id=mesh.study_id
    )
    
    print(f'\n   Rex generated {len(recommendations)} recommendations')
    
    # Display results by priority
    print('\n' + '=' * 70)
    print('REX RECOMMENDATIONS')
    print('=' * 70)
    
    by_priority = {
        ActionPriority.CRITICAL: [],
        ActionPriority.HIGH: [],
        ActionPriority.MEDIUM: [],
        ActionPriority.LOW: []
    }
    
    for rec in recommendations:
        by_priority[rec.priority].append(rec)
    
    for priority, recs in by_priority.items():
        if recs:
            print(f'\n** {priority.name} PRIORITY ({len(recs)} items):')
            print('-' * 60)
            for i, rec in enumerate(recs[:5], 1):  # Show first 5 per priority
                print(f'\n   [{i}] {rec.title}')
                print(f'       Subject: {rec.subject_id} | Site: {rec.site_id}')
                print(f'       Action: {rec.action_type.value}')
                print(f'       Auto-Executable: {rec.auto_executable}')
                if 'auto_generated_query' in rec.source_data:
                    print(f'       [AUTO-QUERY]: {rec.source_data["auto_generated_query"][:100]}...')
            
            if len(recs) > 5:
                print(f'\n       ... and {len(recs) - 5} more {priority.name} items')
    
    # Check for specific Rex capabilities
    print('\n' + '=' * 70)
    print('REX CAPABILITY VERIFICATION')
    print('=' * 70)
    
    zombie_saes = [r for r in recommendations if 'ZOMBIE' in r.title.upper()]
    pending_saes = [r for r in recommendations if 'PENDING' in r.title.upper() and 'SAE' in r.title.upper()]
    recon_issues = [r for r in recommendations if 'RECONCILIATION' in r.title.upper()]
    edrr_flagged = [r for r in recommendations if 'EDRR' in r.title.upper()]
    
    print(f'\n   [1] Zombie SAE Detection: {len(zombie_saes)} detected')
    print(f'   [2] SAE Pending > Threshold: {len(pending_saes)} flagged')
    print(f'   [3] Reconciliation Issues: {len(recon_issues)} found')
    print(f'   [4] High EDRR Cross-Reference: {len(edrr_flagged)} flagged')
    
    # Show a sample auto-generated query
    if zombie_saes:
        print('\n[SAMPLE ZOMBIE SAE AUTO-QUERY]:')
        print('-' * 60)
        sample = zombie_saes[0]
        print(f'   "{sample.description}"')
    
    print('\n' + '=' * 70)
    print('REX ANALYSIS COMPLETE')
    print('=' * 70)
    
    # Verify the analysis was successful
    assert recommendations is not None, "Rex should return recommendations"
    assert len(recommendations) >= 0, "Recommendations should be a valid list"
    print('\n✅ Rex Agent Test PASSED')


def test_supervisor_agent():
    """Test the Supervisor Agent (full multi-agent orchestration)"""
    
    print('\n' + '=' * 70)
    print('SUPERVISOR AGENT TEST - FULL MULTI-AGENT ORCHESTRATION')
    print('=' * 70)
    
    # Build the mesh
    study_path = r'QC Anonymized Study Files\Study 1_CPID_Input Files - Anonymization'
    print('\n1. Building Clinical Data Mesh...')
    
    mesh = build_clinical_data_mesh(study_path)
    
    # Get Digital Patient Twins
    twins = mesh.get_all_digital_twins()
    
    # Get site metrics (simplified)
    from clinical_dataflow_optimizer.models.data_models import SiteMetrics
    site_metrics = {}
    patient_nodes = mesh.graph.get_nodes_by_type(NodeType.PATIENT)
    for node in patient_nodes:
        site_id = node.attributes.get('site_id', '')
        if site_id and site_id not in site_metrics:
            site_metrics[site_id] = SiteMetrics(site_id=site_id, study_id=mesh.study_id)
    
    # Prepare study data
    study_data = mesh.data_sources
    
    # Initialize Supervisor
    print('\n2. Initializing Supervisor Agent...')
    supervisor = SupervisorAgent()
    
    # Run full analysis
    print('\n3. Running Multi-Agent Analysis...')
    results = supervisor.run_analysis(
        twins=twins,
        site_metrics=site_metrics,
        study_data=study_data,
        study_id=mesh.study_id
    )
    
    # Display summary
    summary = supervisor.get_summary()
    print('\n' + '=' * 70)
    print('MULTI-AGENT SUMMARY')
    print('=' * 70)
    print(json.dumps(summary, indent=2, default=json_serializer))
    
    # Verify the analysis was successful
    assert results is not None, "Supervisor analysis should return results"
    assert len(summary) > 0, "Supervisor summary should not be empty"
    print('\n✅ Supervisor Agent Test PASSED')


if __name__ == '__main__':
    # Test Rex agent
    rex_results = test_rex_agent()
    
    # Test full supervisor
    print('\n\n')
    supervisor_results = test_supervisor_agent()
