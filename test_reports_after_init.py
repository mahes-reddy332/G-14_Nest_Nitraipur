"""Test script to verify /api/reports/studies endpoint after backend initialization."""

import time
import requests
from datetime import datetime

def test_reports_endpoint():
    """Test the reports/studies endpoint with retries until backend is ready."""
    
    base_url = "http://127.0.0.1:8000"
    endpoint = f"{base_url}/api/reports/studies"
    max_wait = 600  # 10 minutes max
    check_interval = 15  # Check every 15 seconds
    
    print(f"Testing endpoint: {endpoint}")
    print(f"Maximum wait time: {max_wait}s, checking every {check_interval}s\n")
    
    start_time = time.time()
    attempt = 0
    
    while time.time() - start_time < max_wait:
        attempt += 1
        elapsed = int(time.time() - start_time)
        
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Attempt {attempt} ({elapsed}s elapsed)...", end=" ")
            
            response = requests.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ SUCCESS!")
                print(f"\n{'='*60}")
                print(f"Status Code: {response.status_code}")
                print(f"Response Type: {type(data)}")
                print(f"Number of studies: {len(data) if isinstance(data, list) else 'N/A'}")
                
                if isinstance(data, list) and len(data) > 0:
                    print(f"\nFirst study preview:")
                    first_study = data[0]
                    for key, value in list(first_study.items())[:5]:
                        print(f"  {key}: {value}")
                
                print(f"{'='*60}\n")
                return True
            else:
                print(f"✗ Status {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            print("✗ Request timeout")
        except requests.exceptions.ConnectionError:
            print("✗ Connection error (backend may not be running)")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        if time.time() - start_time < max_wait:
            time.sleep(check_interval)
    
    print(f"\n✗ Timeout: Backend did not respond successfully within {max_wait}s")
    return False

if __name__ == "__main__":
    success = test_reports_endpoint()
    exit(0 if success else 1)
