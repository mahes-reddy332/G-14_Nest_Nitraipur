#!/usr/bin/env python3
"""
Integration test for enhanced LongCat client with agent framework
"""

import sys
import os
from pathlib import Path

# Import from the clinical_dataflow_optimizer package
from clinical_dataflow_optimizer.core.longcat_integration import longcat_client

def test_agent_reasoning():
    """Test agent reasoning with enhanced client"""
    context = 'Patient has incomplete visit schedule'
    task = 'Analyze patient data quality issues'
    data = {
        'patient_id': 'TEST001',
        'issues': ['Missing visit data', 'Query outstanding']
    }

    print('Testing agent reasoning with enhanced LongCat client...')
    try:
        result = longcat_client.generate_agent_reasoning(context, task, data)
        print('✓ Agent reasoning successful')
        print(f'Result type: {type(result)}')
        print(f'Result length: {len(result) if isinstance(result, str) else "N/A"}')
        return True
    except Exception as e:
        print(f'✗ Agent reasoning failed: {e}')
        return False

def test_narrative_generation():
    """Test narrative generation with enhanced client"""
    test_patient = {
        'subject_id': 'TEST001',
        'status': 'Active',
        'clean_status': True
    }

    print('\nTesting narrative generation with enhanced LongCat client...')
    try:
        result = longcat_client.generate_narrative(test_patient, ['Minor issue'], ['Review data'])
        print('✓ Narrative generation successful')
        print(f'Result type: {type(result)}')
        print(f'Result length: {len(result) if isinstance(result, str) else "N/A"}')
        return True
    except Exception as e:
        print(f'✗ Narrative generation failed: {e}')
        return False

def main():
    """Run integration tests"""
    print("Running LongCat Integration Tests")
    print("=" * 40)

    success = True
    success &= test_agent_reasoning()
    success &= test_narrative_generation()

    print("\n" + "=" * 40)
    if success:
        print("✅ All integration tests passed!")
    else:
        print("❌ Some integration tests failed!")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)