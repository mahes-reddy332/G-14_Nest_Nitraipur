"""
Test script to verify the reports endpoint works correctly
"""
import requests
import time
import json

def test_reports_endpoint():
    print("Waiting for backend to be ready...")
    
    # Wait for backend to be ready
    max_wait = 300  # 5 minutes
    start_time = time.time()
    
    while (time.time() - start_time) < max_wait:
        try:
            # Check if server is responding
            r = requests.get('http://127.0.0.1:8000/api/ready', timeout=5)
            if r.status_code == 200:
                print("Backend is ready!")
                break
        except Exception as e:
            pass
        
        time.sleep(5)
        print(f"Still waiting... ({int(time.time() - start_time)}s elapsed)")
    
    else:
        print("Timeout waiting for backend!")
        return
    
    # Test the reports/studies endpoint
    print("\nTesting /api/reports/studies endpoint...")
    try:
        r = requests.get('http://127.0.0.1:8000/api/reports/studies', timeout=15)
        print(f"Status Code: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            print(f"✓ SUCCESS: Got {len(data)} study reports")
            
            if data:
                print(f"\nFirst study report:")
                print(json.dumps(data[0], indent=2))
        else:
            print(f"✗ FAILED: {r.text[:500]}")
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
    
    # Test a specific study report
    print("\nTesting /api/reports/studies/Study_4 endpoint...")
    try:
        r = requests.get('http://127.0.0.1:8000/api/reports/studies/Study_4', timeout=15)
        print(f"Status Code: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            print(f"✓ SUCCESS: Got study report for {data.get('study_id')}")
            print(f"  - Study Name: {data.get('study_name')}")
            print(f"  - KPIs: {len(data.get('kpis', []))}")
            print(f"  - Insights: {len(data.get('insights', []))}")
        else:
            print(f"✗ FAILED: {r.text[:500]}")
            
    except Exception as e:
        print(f"✗ ERROR: {e}")

if __name__ == "__main__":
    test_reports_endpoint()
