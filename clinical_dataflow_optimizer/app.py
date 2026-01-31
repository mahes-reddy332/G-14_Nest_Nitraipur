"""
Neural Clinical Data Mesh - Web Application
Flask-based dashboard for clinical trial data monitoring and analysis
"""

import sys
import os
import secrets
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request
from flask_socketio import SocketIO, emit
import json
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import core modules
try:
    from core.data_integration import ClinicalDataMesh, build_clinical_data_mesh
    from core.data_quality_index import DataQualityIndexCalculator as DataQualityIndex
    HAS_CORE = True
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    HAS_CORE = False

# Initialize Flask app
app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
secret_key = os.getenv("FLASK_SECRET_KEY")
if not secret_key:
    secret_key = secrets.token_hex(32)
    print("Warning: FLASK_SECRET_KEY not set. Using an ephemeral key.")
app.config['SECRET_KEY'] = secret_key

cors_origins_env = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:5174,http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:5174"
)
allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
# Force threading mode on Windows/Python 3.13 to avoid eventlet socket errors.
socketio = SocketIO(app, cors_allowed_origins=allowed_origins, async_mode="threading")

# Global state
clinical_mesh = None
graph_data = None

def _require_mesh():
    """Return an error response when real data is not initialized."""
    return jsonify({
        "error": "Clinical data mesh not initialized",
        "message": "Real data ingestion is required. Initialize the data mesh before using this endpoint.",
        "timestamp": datetime.now().isoformat()
    }), 503

def initialize_data_mesh():
    """Initialize the clinical data mesh on startup"""
    global clinical_mesh, graph_data
    
    if not HAS_CORE:
        return None
    
    try:
        base_path = Path(__file__).parent.parent / "QC Anonymized Study Files"
        study_folder = "Study 1_CPID_Input Files - Anonymization"
        study_path = base_path / study_folder
        
        if study_path.exists():
            print(f"\n🔄 Initializing Clinical Data Mesh from {study_path}...")
            clinical_mesh = build_clinical_data_mesh(str(study_path))
            
            if clinical_mesh and clinical_mesh.graph:
                # Access the underlying NetworkX graph via .graph.graph
                nx_graph = clinical_mesh.graph.graph
                graph_data = {
                    'nodes': nx_graph.number_of_nodes(),
                    'edges': nx_graph.number_of_edges(),
                    'studies_loaded': 1
                }
                print(f"✅ Data Mesh initialized: {graph_data['nodes']} nodes, {graph_data['edges']} edges")
                return clinical_mesh
    except Exception as e:
        print(f"⚠️ Could not initialize data mesh: {e}")
    
    return None

# ============================================================================
# API Routes
# ============================================================================

@app.route('/')
def serve_frontend():
    """Serve the React frontend"""
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({
        'message': 'Frontend not built. Use the Vite dev server at http://localhost:5173',
        'api_health': '/api/health',
        'api_docs': '/api/dashboard/summary'
    })

@app.route('/favicon.ico')
def serve_favicon():
    """Serve favicon"""
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, 'favicon.ico')):
        return send_from_directory(app.static_folder, 'favicon.ico')
    # Return empty response if no favicon
    return '', 204

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'name': 'Neural Clinical Data Mesh'
    })

@app.route('/api/ready')
def readiness_check():
    """Readiness probe for the frontend."""
    return jsonify({
        'status': 'ready',
        'data_mesh_initialized': clinical_mesh is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/dashboard/summary')
def get_dashboard_summary():
    """Get dashboard summary statistics"""
    global clinical_mesh, graph_data

    if clinical_mesh is None:
        return _require_mesh()

    summary = {
        'timestamp': datetime.now().isoformat(),
        'system_status': 'operational',
        'data_mesh': {
            'initialized': clinical_mesh is not None,
            'nodes': graph_data['nodes'] if graph_data else 0,
            'edges': graph_data['edges'] if graph_data else 0
        }
    }
    return jsonify(summary)

@app.route('/api/patients')
def get_patients():
    """Get patient list with quality indicators"""
    patients = []
    
    if clinical_mesh and clinical_mesh.graph:
        for node, data in clinical_mesh.graph.nodes(data=True):
            if data.get('type') == 'patient':
                patients.append({
                    'id': node,
                    'site': data.get('site_id', 'Unknown'),
                    'status': data.get('status', 'Active'),
                    'quality_score': data.get('quality_score', 85),
                    'open_queries': data.get('open_queries', 0),
                    'last_visit': data.get('last_visit', 'N/A')
                })
    else:
        return _require_mesh()
    
    return jsonify({'patients': patients, 'total': len(patients)})

@app.route('/api/sites')
def get_sites():
    """Get site performance data"""
    if clinical_mesh is None:
        return _require_mesh()
    return jsonify({'sites': [], 'total': 0})

@app.route('/api/saes')
def get_saes():
    """Get SAE dashboard data"""
    if clinical_mesh is None:
        return _require_mesh()
    return jsonify({
        'saes': [],
        'total': 0,
        'by_severity': {},
        'pending_coding': 0
    })

@app.route('/api/queries')
def get_queries():
    """Get open queries across studies"""
    if clinical_mesh is None:
        return _require_mesh()
    return jsonify({
        'queries': [],
        'total_open': 0,
        'avg_resolution_days': 0,
        'by_type': {}
    })

@app.route('/api/graph/query', methods=['POST'])
def execute_graph_query():
    """Execute a graph query on the clinical data mesh"""
    data = request.json or {}
    query_type = data.get('type', 'patients_with_issues')

    if clinical_mesh is None:
        return _require_mesh()

    result = {
        'query_type': query_type,
        'execution_time_ms': 0.0,
        'results': [],
        'count': 0
    }

    return jsonify(result)

@app.route('/api/agents/status')
def get_agent_status():
    """Get status of AI agents"""
    if clinical_mesh is None:
        return _require_mesh()

    return jsonify({
        'agents': [],
        'supervisor': {
            'status': 'unknown',
            'active_workflows': 0
        }
    })

@app.route('/api/agents/insights')
def get_agent_insights():
    """Return recent agent insights for the UI."""
    limit = int(request.args.get('limit', 5))
    return jsonify({
        'insights': [],
        'limit': limit
    })

@app.route('/api/alerts/recent')
def get_recent_alerts():
    """Return recent alerts for the UI."""
    return jsonify([])

@app.route('/api/alerts/critical')
def get_critical_alerts():
    """Return critical alerts for the UI."""
    return jsonify([])

@app.route('/api/alerts/summary')
def get_alert_summary():
    """Return alert summary counts for the UI."""
    return jsonify({
        'active_alerts': 0,
        'by_severity': {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
    })

@app.route('/api/alerts/')
def list_alerts():
    """Return alerts list for the UI."""
    return jsonify([])

@app.route('/api/alerts/history')
def get_alert_history():
    """Return alert history list for the UI."""
    return jsonify([])

@app.route('/api/metrics/heatmap')
def get_metrics_heatmap():
    """Return heatmap metric data for the UI."""
    metric = request.args.get('metric', 'dqi')
    return jsonify({
        'metric': metric,
        'rows': [],
        'columns': [],
        'values': []
    })

@app.route('/api/studies/')
def list_studies():
    """Return available studies for the UI."""
    return jsonify([])

@app.route('/api/dashboard/initial-load')
def get_dashboard_initial_load():
    """Return initial payload for the dashboard."""
    return jsonify({
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'study_filter': None,
        'summary': {
            'total_studies': 0,
            'total_patients': 0,
            'total_sites': 0,
            'clean_patients': 0,
            'dirty_patients': 0,
            'overall_dqi': 0,
            'open_queries': 0,
            'pending_saes': 0,
            'uncoded_terms': 0,
        },
        'query_metrics': {
            'total_queries': 0,
            'open_queries': 0,
            'closed_queries': 0,
            'resolution_rate': 0,
            'avg_resolution_time': 0,
            'aging_distribution': {
                '0-7': 0,
                '8-14': 0,
                '15-30': 0,
                '30+': 0,
            },
            'velocity_trend': [],
        },
        'cleanliness': {
            'cleanliness_rate': 0,
            'total_patients': 0,
            'clean_patients': 0,
            'dirty_patients': 0,
            'at_risk_count': 0,
            'trend': [],
        },
        'alerts': {
            'active_alerts': 0,
            'critical_count': 0,
            'high_count': 0,
        },
        '_cache_hit': False,
        '_response_time_ms': 0,
    })

# Catch-all route for SPA routing - must be after all other routes
@app.route('/<path:path>')
def serve_spa(path):
    """Serve the React SPA for any unmatched routes (except /api/)"""
    # Don't intercept API routes
    if path.startswith('api/'):
        return jsonify({'error': 'Not found', 'path': path}), 404
    
    # Try to serve static file first
    if app.static_folder:
        static_file = os.path.join(app.static_folder, path)
        if os.path.exists(static_file) and os.path.isfile(static_file):
            return send_from_directory(app.static_folder, path)
        
        # For SPA routes, serve index.html
        index_path = os.path.join(app.static_folder, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(app.static_folder, 'index.html')
    
    # Frontend not built - redirect to dev server info
    return jsonify({
        'message': f'Route /{path} not found. Frontend not built.',
        'suggestion': 'Use the Vite dev server at http://localhost:5173 for development',
        'api_health': '/api/health'
    }), 404

# ============================================================================
# WebSocket Events
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected', 'timestamp': datetime.now().isoformat()})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('subscribe_updates')
def handle_subscribe(data):
    """Subscribe to real-time updates"""
    channel = data.get('channel', 'all')
    print(f"Client {request.sid} subscribed to: {channel}")
    emit('subscribed', {'channel': channel, 'status': 'subscribed'})

# ============================================================================
# Main Entry Point
# ============================================================================

def print_banner():
    """Print startup banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗                            ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║                            ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║                            ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║                            ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗                       ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝                       ║
║                                                                              ║
║   ██████╗██╗     ██╗███╗   ██╗██╗ ██████╗ █████╗ ██╗                        ║
║  ██╔════╝██║     ██║████╗  ██║██║██╔════╝██╔══██╗██║                        ║
║  ██║     ██║     ██║██╔██╗ ██║██║██║     ███████║██║                        ║
║  ██║     ██║     ██║██║╚██╗██║██║██║     ██╔══██║██║                        ║
║  ╚██████╗███████╗██║██║ ╚████║██║╚██████╗██║  ██║███████╗                   ║
║   ╚═════╝╚══════╝╚═╝╚═╝  ╚═══╝╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝                   ║
║                                                                              ║
║   ██████╗  █████╗ ████████╗ █████╗     ███╗   ███╗███████╗███████╗██╗  ██╗  ║
║   ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ████╗ ████║██╔════╝██╔════╝██║  ██║  ║
║   ██║  ██║███████║   ██║   ███████║    ██╔████╔██║█████╗  ███████╗███████║  ║
║   ██║  ██║██╔══██║   ██║   ██╔══██║    ██║╚██╔╝██║██╔══╝  ╚════██║██╔══██║  ║
║   ██████╔╝██║  ██║   ██║   ██║  ██║    ██║ ╚═╝ ██║███████╗███████║██║  ██║  ║
║   ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝  ║
║                                                                              ║
║                    Clinical Trial Intelligence Platform                      ║
║                           Version 2.0 - Production                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

if __name__ == '__main__':
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description='Neural Clinical Data Mesh Web Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-init', action='store_true', help='Skip data mesh initialization')
    parser.add_argument('--backend', choices=['fastapi', 'flask'], default=os.getenv('APP_BACKEND', 'fastapi'), help='Backend server to run')
    
    args = parser.parse_args()
    
    print_banner()
    
    print(f"\n🚀 Starting Neural Clinical Data Mesh Server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Debug: {args.debug}")
    print(f"   Backend: {args.backend}")

    if args.backend == 'fastapi':
        from api.main import app as fastapi_app
        print(f"\n✅ Server ready!")
        print(f"   Dashboard: http://{args.host}:{args.port}")
        print(f"   API Health: http://{args.host}:{args.port}/api/health")
        print(f"   API Docs: http://{args.host}:{args.port}/api/docs")
        print("\n" + "=" * 80)
        uvicorn.run(fastapi_app, host=args.host, port=args.port, log_level="info")
    else:
        if not args.no_init:
            initialize_data_mesh()

        print(f"\n✅ Server ready!")
        print(f"   Dashboard: http://{args.host}:{args.port}")
        print(f"   API Health: http://{args.host}:{args.port}/api/health")
        print(f"   API Docs: http://{args.host}:{args.port}/api/dashboard/summary")
        print("\n" + "=" * 80)

        socketio.run(app, host=args.host, port=args.port, debug=args.debug)
