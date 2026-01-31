"""
Test if reports router can be loaded and inspect its routes
"""

import sys
import traceback

try:
    print("=" * 60)
    print("Testing Reports Router Loading")
    print("=" * 60)
    
    # Change to clinical_dataflow_optimizer directory
    sys.path.insert(0, 'clinical_dataflow_optimizer')
    
    print("\n1. Importing reports module...")
    from api.routers import reports
    print("✓ Successfully imported reports module")
    
    print("\n2. Checking router object...")
    print(f"   Router type: {type(reports.router)}")
    print(f"   Router class: {reports.router.__class__.__name__}")
    
    print("\n3. Listing all routes in router...")
    routes = reports.router.routes
    print(f"   Total routes: {len(routes)}")
    
    for idx, route in enumerate(routes, 1):
        route_info = {
            'path': route.path,
            'methods': list(route.methods) if hasattr(route, 'methods') else [],
            'name': route.name if hasattr(route, 'name') else 'N/A'
        }
        print(f"   Route {idx}: {route_info}")
    
    print("\n4. Testing FastAPI app creation with router...")
    from fastapi import FastAPI
    test_app = FastAPI()
    test_app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
    print("✓ Successfully included router in FastAPI app")
    
    print("\n5. Checking routes in FastAPI app...")
    app_routes = [r for r in test_app.routes if '/reports' in r.path]
    print(f"   Total /reports routes in app: {len(app_routes)}")
    
    for route in app_routes:
        print(f"   - {route.path} ({route.methods if hasattr(route, 'methods') else 'N/A'})")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
