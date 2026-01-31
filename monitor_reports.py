"""Monitor and test reports endpoint"""
import requests
import time
import sys

print("Monitoring reports endpoint...")
print("Waiting for backend to complete initialization...")

max_wait = 300  # 5 minutes
start = time.time()

while (time.time() - start) < max_wait:
    try:
        r = requests.get('http://127.0.0.1:8000/api/reports/studies', timeout=5)
        
        if r.status_code == 200:
            data = r.json()
            print(f"\n✓ SUCCESS! Reports endpoint is working")
            print(f"  Status: {r.status_code}")
            print(f"  Study reports: {len(data)}")
            
            if data:
                print(f"\n  Sample study report:")
                study = data[0]
                print(f"    - Study ID: {study.get('study_id')}")
                print(f"    - Study Name: {study.get('study_name')}")
                print(f"    - Total Patients: {study.get('total_patients')}")
                print(f"    - DQI Score: {study.get('dqi_score')}")
                print(f"    - Cleanliness Rate: {study.get('cleanliness_rate')}%")
                print(f"    - Open Queries: {study.get('open_queries')}")
            
            print("\n✓ Reports section is now ready! Refresh your browser.")
            sys.exit(0)
            
        elif r.status_code == 404:
            # Still 404, wait longer
            pass
        else:
            print(f"  Unexpected status: {r.status_code}")
            
    except requests.exceptions.RequestException:
        pass
    
    time.sleep(10)
    elapsed = int(time.time() - start)
    print(f"  Still waiting... ({elapsed}s elapsed)")

print("\n✗ Timeout: Backend did not respond successfully within 5 minutes")
sys.exit(1)
