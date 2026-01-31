"""Quick check of registered routes in OpenAPI spec"""
import requests
import json
import sys

try:
    r = requests.get('http://127.0.0.1:8000/openapi.json', timeout=5)
    if r.status_code != 200:
        print(f"Error: {r.status_code}", file=sys.stderr)
        sys.exit(1)
    
    paths = r.json().get('paths', {})
    report_routes = sorted([p for p in paths.keys() if '/reports' in p])
    
    print("=" * 50)
    print("REGISTERED /reports ROUTES:")
    print("=" * 50)
    for route in report_routes:
        methods = list(paths[route].keys())
        print(f"  {route} [{', '.join(methods).upper()}]")
    print("=" * 50)
    print(f"Total: {len(report_routes)} routes")
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
