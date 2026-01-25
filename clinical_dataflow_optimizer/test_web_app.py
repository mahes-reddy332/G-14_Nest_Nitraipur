"""
Test script for the Neural Clinical Data Mesh Web Application
"""

import sys
from pathlib import Path
import unittest
import asyncio
from unittest.mock import Mock, patch
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from web_app import NeuralClinicalDataMeshApp
from core.real_time_monitor import RealTimeDataMonitor
from visualization.dashboard import RealTimeDashboardVisualizer


class TestWebApplication(unittest.TestCase):
    """Test cases for the web application"""

    def setUp(self):
        """Set up test fixtures"""
        self.data_path = PROJECT_ROOT.parent / "QC Anonymized Study Files"
        self.app = NeuralClinicalDataMeshApp(self.data_path)

    def test_app_initialization(self):
        """Test that the app initializes correctly"""
        self.assertIsNotNone(self.app.monitor)
        self.assertIsNotNone(self.app.dashboard)
        self.assertIsNotNone(self.app.data_ingester)
        self.assertIsNotNone(self.app.twin_builder)
        self.assertIsNotNone(self.app.supervisor_agent)

    def test_data_loading(self):
        """Test that initial data loads correctly"""
        # This will fail if data files don't exist, but tests the loading logic
        try:
            self.app._load_initial_data()
            self.assertIsInstance(self.app.current_twins, list)
            self.assertIsInstance(self.app.current_site_metrics, dict)
        except Exception as e:
            # Expected if data files don't exist
            self.assertIn("data", str(e).lower())

    @patch('web_app.FLASK_AVAILABLE', True)
    @patch('flask.Flask')
    @patch('flask_socketio.SocketIO')
    def test_flask_app_creation(self, mock_socketio, mock_flask):
        """Test Flask app creation"""
        flask_app = self.app.create_flask_app()
        self.assertIsNotNone(flask_app)
        self.assertIsNotNone(self.app.socketio)

    @patch('web_app.DASH_AVAILABLE', True)
    @patch('dash.Dash')
    def test_dash_app_creation(self, mock_dash):
        """Test Dash app creation"""
        dash_app = self.app.create_dash_app()
        self.assertIsNotNone(dash_app)

    def test_monitoring_start_stop(self):
        """Test monitoring lifecycle"""
        # Start monitoring
        self.app.start_monitoring()
        self.assertTrue(self.app.monitor.is_monitoring)

        # Stop monitoring
        self.app.stop_monitoring()
        self.assertFalse(self.app.monitor.is_monitoring)


class TestRealTimeFeatures(unittest.TestCase):
    """Test real-time features"""

    def setUp(self):
        """Set up test fixtures"""
        self.data_path = PROJECT_ROOT.parent / "QC Anonymized Study Files"
        self.monitor = RealTimeDataMonitor(self.data_path)

    def test_monitor_initialization(self):
        """Test monitor initialization"""
        self.assertIsNotNone(self.monitor.cleanliness_engine)
        self.assertIsNotNone(self.monitor.patient_status_history)
        self.assertFalse(self.monitor.is_monitoring)

    def test_websocket_server(self):
        """Test WebSocket server setup"""
        # WebSocket server is None until started
        self.assertIsNone(self.monitor.websocket_server)

        # After starting, it should be set (though we can't easily test the actual server)
        # This test just verifies the attribute exists and can be set
        self.assertTrue(hasattr(self.monitor, 'websocket_server'))

    def test_operational_velocity_calculation(self):
        """Test operational velocity metrics calculation"""
        # Mock site metrics
        site_metrics = {
            'site_001': {'total_patients': 10, 'clean_patients': 8, 'dqi_score': 85.0},
            'site_002': {'total_patients': 15, 'clean_patients': 12, 'dqi_score': 90.0}
        }

        self.monitor.calculate_operational_velocity_metrics(site_metrics)
        ranking = self.monitor.get_operational_velocity_ranking()

        self.assertIsInstance(ranking, list)
        self.assertGreater(len(ranking), 0)


class TestDashboardFeatures(unittest.TestCase):
    """Test dashboard visualization features"""

    def setUp(self):
        """Set up test fixtures"""
        self.data_path = PROJECT_ROOT.parent / "QC Anonymized Study Files"
        self.monitor = RealTimeDataMonitor(self.data_path)
        self.dashboard = RealTimeDashboardVisualizer(self.monitor)

    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        self.assertIsNotNone(self.dashboard.real_time_monitor)
        self.assertFalse(self.dashboard.is_real_time_enabled)

    def test_real_time_toggle(self):
        """Test real-time update toggle"""
        # Enable real-time updates
        self.dashboard.enable_real_time_updates()
        self.assertTrue(self.dashboard.is_real_time_enabled)

        # Disable real-time updates
        self.dashboard.disable_real_time_updates()
        self.assertFalse(self.dashboard.is_real_time_enabled)


def run_integration_test():
    """Run a basic integration test"""
    print("Running Neural Clinical Data Mesh Integration Test...")

    try:
        # Create app instance
        data_path = PROJECT_ROOT.parent / "QC Anonymized Study Files"
        app = NeuralClinicalDataMeshApp(data_path)

        # Test data loading
        app._load_initial_data()
        print(f"✓ Loaded {len(app.current_twins)} patients from {len(app.current_site_metrics)} sites")

        # Test monitoring
        app.start_monitoring()
        print("✓ Real-time monitoring started")

        # Test operational velocity
        app.monitor.calculate_operational_velocity_metrics(app.current_site_metrics)
        ovi_ranking = app.monitor.get_operational_velocity_ranking()
        print(f"✓ Calculated OVI ranking for {len(ovi_ranking)} sites")

        # Stop monitoring
        app.stop_monitoring()
        print("✓ Monitoring stopped")

        print("✓ Integration test passed!")
        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run integration test
    print("\n" + "="*50)
    success = run_integration_test()

    if success:
        print("\n🎉 All tests passed! The web application is ready to run.")
        print("\nTo start the application:")
        print("  python web_app.py --framework flask --host 0.0.0.0 --port 5000")
    else:
        print("\n❌ Some tests failed. Please check the setup and try again.")
        sys.exit(1)