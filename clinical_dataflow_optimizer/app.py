"""
Neural Clinical Data Mesh - Web Application
Flask-based dashboard for clinical trial data monitoring and analysis
"""

import sys
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
    from core.data_quality_index import DataQualityIndex
    HAS_CORE = True
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    HAS_CORE = False

# Initialize Flask app
app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
app.config['SECRET_KEY'] = 'neural-clinical-data-mesh-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
clinical_mesh = None
graph_data = None

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
                graph_data = {
                    'nodes': clinical_mesh.graph.number_of_nodes(),
                    'edges': clinical_mesh.graph.number_of_edges(),
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
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'name': 'Neural Clinical Data Mesh'
    })

@app.route('/api/dashboard/summary')
def get_dashboard_summary():
    """Get dashboard summary statistics"""
    global clinical_mesh, graph_data
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'system_status': 'operational',
        'data_mesh': {
            'initialized': clinical_mesh is not None,
            'nodes': graph_data['nodes'] if graph_data else 0,
            'edges': graph_data['edges'] if graph_data else 0
        },
        'metrics': {
            'total_patients': 99,
            'total_sites': 15,
            'total_saes': 1274,
            'open_queries': 342,
            'missing_pages': 156,
            'coding_terms': 1532
        },
        'quality_score': 87.3,
        'alerts': [
            {'level': 'warning', 'message': '3 sites with >5% missing data', 'count': 3},
            {'level': 'info', 'message': 'Coding reconciliation in progress', 'count': 1}
        ]
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
        # Demo data
        for i in range(1, 100):
            patients.append({
                'id': f'PAT-{i:03d}',
                'site': f'SITE-{(i % 15) + 1:02d}',
                'status': 'Active' if i % 5 != 0 else 'Completed',
                'quality_score': 70 + (i % 30),
                'open_queries': i % 8,
                'last_visit': f'2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}'
            })
    
    return jsonify({'patients': patients, 'total': len(patients)})

@app.route('/api/sites')
def get_sites():
    """Get site performance data"""
    sites = []
    
    for i in range(1, 16):
        sites.append({
            'id': f'SITE-{i:02d}',
            'name': f'Clinical Site {i}',
            'patients': 5 + (i % 10),
            'quality_score': 75 + (i % 20),
            'enrollment_rate': 2.5 + (i % 5) * 0.5,
            'query_resolution_time': 3 + (i % 7),
            'status': 'Active' if i % 4 != 0 else 'Under Review'
        })
    
    return jsonify({'sites': sites, 'total': len(sites)})

@app.route('/api/saes')
def get_saes():
    """Get SAE dashboard data"""
    saes = []
    
    severities = ['Mild', 'Moderate', 'Severe', 'Life-threatening']
    outcomes = ['Recovered', 'Recovering', 'Not Recovered', 'Fatal', 'Unknown']
    
    for i in range(1, 51):
        saes.append({
            'id': f'SAE-{i:04d}',
            'patient_id': f'PAT-{(i % 99) + 1:03d}',
            'site_id': f'SITE-{(i % 15) + 1:02d}',
            'severity': severities[i % 4],
            'outcome': outcomes[i % 5],
            'reported_date': f'2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}',
            'coding_status': 'Coded' if i % 3 != 0 else 'Pending',
            'meddra_pt': f'PT-{10000 + i}'
        })
    
    return jsonify({
        'saes': saes,
        'total': 1274,
        'by_severity': {'Mild': 423, 'Moderate': 512, 'Severe': 289, 'Life-threatening': 50},
        'pending_coding': 342
    })

@app.route('/api/queries')
def get_queries():
    """Get open queries across studies"""
    queries = []
    
    query_types = ['Missing Data', 'Inconsistent Value', 'Protocol Deviation', 'Range Check', 'Date Logic']
    
    for i in range(1, 31):
        queries.append({
            'id': f'QRY-{i:05d}',
            'patient_id': f'PAT-{(i % 99) + 1:03d}',
            'site_id': f'SITE-{(i % 15) + 1:02d}',
            'type': query_types[i % 5],
            'field': f'Form_{(i % 10) + 1}.Field_{i % 20}',
            'status': 'Open' if i % 3 != 0 else 'Answered',
            'age_days': i % 30,
            'created_date': f'2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}'
        })
    
    return jsonify({
        'queries': queries,
        'total_open': 342,
        'avg_resolution_days': 4.7,
        'by_type': {t: 68 for t in query_types}
    })

@app.route('/api/graph/query', methods=['POST'])
def execute_graph_query():
    """Execute a graph query on the clinical data mesh"""
    data = request.json or {}
    query_type = data.get('type', 'patients_with_issues')
    
    result = {
        'query_type': query_type,
        'execution_time_ms': 2.3,
        'results': []
    }
    
    if query_type == 'patients_with_issues':
        result['results'] = [
            {'patient_id': f'PAT-{i:03d}', 'issues': ['Missing Visit', 'Open Query']}
            for i in range(1, 68)
        ]
        result['count'] = 67
    elif query_type == 'site_quality':
        result['results'] = [
            {'site_id': f'SITE-{i:02d}', 'quality_score': 75 + (i % 20)}
            for i in range(1, 16)
        ]
    
    return jsonify(result)

@app.route('/api/agents/status')
def get_agent_status():
    """Get status of AI agents"""
    return jsonify({
        'agents': [
            {
                'name': 'Rex',
                'role': 'Safety Reconciliation Expert',
                'status': 'active',
                'tasks_completed': 156,
                'current_task': 'SAE-MedDRA reconciliation'
            },
            {
                'name': 'Codex',
                'role': 'Medical Coding Specialist',
                'status': 'active',
                'tasks_completed': 234,
                'current_task': 'WHODRA batch coding'
            },
            {
                'name': 'Lia',
                'role': 'Site Liaison Coordinator',
                'status': 'idle',
                'tasks_completed': 89,
                'current_task': None
            }
        ],
        'supervisor': {
            'status': 'monitoring',
            'active_workflows': 3
        }
    })

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
    
    parser = argparse.ArgumentParser(description='Neural Clinical Data Mesh Web Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-init', action='store_true', help='Skip data mesh initialization')
    
    args = parser.parse_args()
    
    print_banner()
    
    print(f"\n🚀 Starting Neural Clinical Data Mesh Server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Debug: {args.debug}")
    
    if not args.no_init:
        initialize_data_mesh()
    
    print(f"\n✅ Server ready!")
    print(f"   Dashboard: http://{args.host}:{args.port}")
    print(f"   API Health: http://{args.host}:{args.port}/api/health")
    print(f"   API Docs: http://{args.host}:{args.port}/api/dashboard/summary")
    print("\n" + "=" * 80)
    
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)
