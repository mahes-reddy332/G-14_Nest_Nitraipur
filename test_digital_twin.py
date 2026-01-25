#!/usr/bin/env python3
"""Test script for Digital Patient Twin functionality."""

import sys
import json
sys.path.insert(0, '.')

from clinical_dataflow_optimizer.core import (
    ClinicalDataMesh, 
    build_clinical_data_mesh,
    DigitalTwinFactory
)
from graph.knowledge_graph import NodeType

def test_digital_twin():
    """Test Digital Patient Twin generation."""
    
    # Build the mesh
    study_path = r'QC Anonymized Study Files\Study 1_CPID_Input Files - Anonymization'
    print('=' * 60)
    print('DIGITAL PATIENT TWIN TEST')
    print('=' * 60)
    print('\n1. Building Clinical Data Mesh...')
    
    mesh = build_clinical_data_mesh(study_path)
    
    # Get statistics
    stats = mesh.get_statistics()
    print(f'   Graph: {stats.total_nodes} nodes, {stats.total_edges} edges')
    
    # Get all patient nodes from graph
    patient_nodes = mesh.graph.get_nodes_by_type(NodeType.PATIENT)
    print(f'   Patients found: {len(patient_nodes)}')
    
    if not patient_nodes:
        print('ERROR: No patients found!')
        return
    
    # Test with first patient
    test_patient_node = patient_nodes[0]
    test_patient = test_patient_node.attributes.get('subject_id', '')
    print(f'\n2. Creating Digital Twin for: {test_patient}')
    print('-' * 60)
    
    # Get AI-readable twin
    twin_dict = mesh.get_ai_readable_twin(test_patient)
    
    if twin_dict:
        print('\n   AI-READABLE TWIN OUTPUT:')
        print('   ' + '-' * 40)
        print(json.dumps(twin_dict, indent=4, default=str))
        
        # Verify required fields
        print('\n3. Verifying required fields:')
        required_fields = ['subject_id', 'status', 'clean_status', 'blocking_items', 'risk_metrics']
        for field in required_fields:
            if field in twin_dict:
                print(f'   [OK] {field}: {type(twin_dict[field]).__name__}')
            else:
                print(f'   [MISSING] {field}')
        
        # Check risk_metrics subfields
        if 'risk_metrics' in twin_dict:
            print('\n   Risk Metrics:')
            for key, value in twin_dict['risk_metrics'].items():
                print(f'      - {key}: {value}')
    else:
        print('   ERROR: Failed to create twin')
    
    # Test batch generation
    print('\n4. Testing batch Digital Twin generation...')
    all_twins = mesh.get_all_ai_readable_twins()
    print(f'   Generated {len(all_twins)} Digital Patient Twins')
    
    if all_twins:
        # Show summary
        clean_count = sum(1 for t in all_twins if t.get('clean_status', False))
        print(f'   Clean patients: {clean_count}/{len(all_twins)}')
        
        # Show first 3 patients summary
        print('\n   First 3 patients summary:')
        for twin in all_twins[:3]:
            status = twin.get('status', 'Unknown')
            clean = 'CLEAN' if twin.get('clean_status') else 'NOT CLEAN'
            blocking = len(twin.get('blocking_items', []))
            risk = twin.get('risk_metrics', {}).get('manipulation_risk_score', 'N/A')
            print(f'      - {twin["subject_id"]}: {status}, {clean}, {blocking} blockers, Risk: {risk}')
    
    print('\n' + '=' * 60)
    print('DIGITAL PATIENT TWIN TEST COMPLETE')
    print('=' * 60)

if __name__ == '__main__':
    test_digital_twin()
