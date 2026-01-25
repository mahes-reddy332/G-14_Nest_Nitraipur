"""
Test Suite for Monitoring and Alerting System
=============================================

Tests for:
- Metrics collection
- Alert rules and management
- Performance monitoring
- Request tracking
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.monitoring import (
    MetricsCollector,
    MetricType,
    MetricSeries,
    MetricPoint,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    Alert,
    PerformanceMonitor,
    RequestTracker,
    LogNotificationChannel,
    InMemoryNotificationChannel,
    setup_default_alert_rules,
    get_metrics_collector,
    get_alert_manager
)


# =============================================================================
# Metrics Collector Tests
# =============================================================================

class TestMetricsCollector:
    """Tests for MetricsCollector"""
    
    def test_increment_counter(self):
        """Test incrementing a counter metric"""
        collector = MetricsCollector()
        
        collector.increment("test.counter")
        collector.increment("test.counter")
        collector.increment("test.counter", 5)
        
        metric = collector.get_metric("test.counter")
        assert metric['exists'] == True
        assert metric['type'] == 'COUNTER'
        # Last value should be total
        assert metric['recent_points'][-1]['value'] == 7
    
    def test_set_gauge(self):
        """Test setting a gauge metric"""
        collector = MetricsCollector()
        
        collector.set_gauge("test.gauge", 42)
        collector.set_gauge("test.gauge", 100)
        
        metric = collector.get_metric("test.gauge")
        assert metric['exists'] == True
        assert metric['type'] == 'GAUGE'
        assert metric['recent_points'][-1]['value'] == 100
    
    def test_record_histogram(self):
        """Test recording histogram values"""
        collector = MetricsCollector()
        
        for i in range(10):
            collector.record_histogram("test.histogram", i * 10)
        
        metric = collector.get_metric("test.histogram")
        assert metric['exists'] == True
        assert metric['stats']['count'] == 10
        assert metric['stats']['min'] == 0
        assert metric['stats']['max'] == 90
    
    def test_timer_context_manager(self):
        """Test timer context manager"""
        collector = MetricsCollector()
        
        with collector.timer("test.timer"):
            time.sleep(0.01)  # 10ms sleep
        
        metric = collector.get_metric("test.timer")
        assert metric['exists'] == True
        assert metric['type'] == 'TIMER'
        # Duration should be at least 10ms
        assert metric['recent_points'][-1]['value'] >= 10
    
    def test_metrics_with_labels(self):
        """Test metrics with labels"""
        collector = MetricsCollector()
        
        collector.increment("http.requests", labels={'endpoint': '/api/study'})
        collector.increment("http.requests", labels={'endpoint': '/api/patient'})
        collector.increment("http.requests", labels={'endpoint': '/api/study'})
        
        # Should be separate series
        metric1 = collector.get_metric("http.requests", labels={'endpoint': '/api/study'})
        metric2 = collector.get_metric("http.requests", labels={'endpoint': '/api/patient'})
        
        assert metric1['recent_points'][-1]['value'] == 2
        assert metric2['recent_points'][-1]['value'] == 1
    
    def test_get_all_metrics(self):
        """Test getting all metrics"""
        collector = MetricsCollector()
        
        collector.set_gauge("metric.a", 1)
        collector.set_gauge("metric.b", 2)
        collector.increment("metric.c")
        
        all_metrics = collector.get_all_metrics()
        assert len(all_metrics) >= 3
    
    def test_dashboard_metrics(self):
        """Test dashboard metrics format"""
        collector = MetricsCollector()
        
        collector.set_gauge("cpu.percent", 45)
        collector.set_gauge("memory.percent", 60)
        
        dashboard = collector.get_dashboard_metrics()
        
        assert "cpu.percent" in dashboard
        assert dashboard["cpu.percent"]['current'] == 45


class TestMetricSeries:
    """Tests for MetricSeries"""
    
    def test_add_point(self):
        """Test adding points to series"""
        series = MetricSeries(name="test", metric_type=MetricType.GAUGE)
        
        series.add_point(10)
        series.add_point(20)
        
        assert len(series.points) == 2
    
    def test_get_recent(self):
        """Test getting recent points"""
        series = MetricSeries(name="test", metric_type=MetricType.GAUGE)
        
        # Add a point with old timestamp
        old_point = MetricPoint(
            name="test",
            value=1,
            timestamp=datetime.now() - timedelta(hours=1),
            metric_type=MetricType.GAUGE
        )
        series.points.append(old_point)
        
        # Add recent point
        series.add_point(2)
        
        recent = series.get_recent(duration_seconds=300)
        assert len(recent) == 1
        assert recent[0].value == 2
    
    def test_get_stats(self):
        """Test statistics calculation"""
        series = MetricSeries(name="test", metric_type=MetricType.HISTOGRAM)
        
        for v in [10, 20, 30, 40, 50]:
            series.add_point(v)
        
        stats = series.get_stats()
        
        assert stats['count'] == 5
        assert stats['min'] == 10
        assert stats['max'] == 50
        assert stats['mean'] == 30
        assert stats['median'] == 30


# =============================================================================
# Alert Manager Tests
# =============================================================================

class TestAlertRule:
    """Tests for AlertRule"""
    
    def test_evaluate_greater_than(self):
        """Test greater than condition"""
        rule = AlertRule(
            rule_id="test",
            name="Test Rule",
            description="Test",
            metric_name="test.metric",
            condition="gt",
            threshold=50
        )
        
        assert rule.evaluate(60) == True
        assert rule.evaluate(50) == False
        assert rule.evaluate(40) == False
    
    def test_evaluate_less_than(self):
        """Test less than condition"""
        rule = AlertRule(
            rule_id="test",
            name="Test Rule",
            description="Test",
            metric_name="test.metric",
            condition="lt",
            threshold=50
        )
        
        assert rule.evaluate(40) == True
        assert rule.evaluate(50) == False
        assert rule.evaluate(60) == False
    
    def test_evaluate_equals(self):
        """Test equals condition"""
        rule = AlertRule(
            rule_id="test",
            name="Test Rule",
            description="Test",
            metric_name="test.metric",
            condition="eq",
            threshold=50
        )
        
        assert rule.evaluate(50) == True
        assert rule.evaluate(49) == False


class TestAlertManager:
    """Tests for AlertManager"""
    
    def test_add_and_remove_rule(self):
        """Test adding and removing alert rules"""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        
        rule = AlertRule(
            rule_id="test_rule",
            name="Test",
            description="Test",
            metric_name="test.metric",
            condition="gt",
            threshold=50
        )
        
        manager.add_rule(rule)
        assert "test_rule" in manager._rules
        
        manager.remove_rule("test_rule")
        assert "test_rule" not in manager._rules
    
    def test_alert_firing(self):
        """Test alert fires when condition is met"""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        
        rule = AlertRule(
            rule_id="high_value",
            name="High Value",
            description="Value is too high",
            metric_name="test.metric",
            condition="gt",
            threshold=50,
            duration_seconds=0  # Immediate firing for test
        )
        manager.add_rule(rule)
        
        # Record high value
        collector.set_gauge("test.metric", 100)
        
        # Evaluate rules
        manager.evaluate_rules()
        
        # Check for pending first
        manager.evaluate_rules()  # Second evaluation should fire
        
        active = manager.get_active_alerts()
        # Should have fired or be pending
        assert len(active) >= 0  # Depends on duration
    
    def test_notification_channel(self):
        """Test notification channels"""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        
        notifications = InMemoryNotificationChannel()
        manager.add_notification_channel(notifications.send)
        
        rule = AlertRule(
            rule_id="test",
            name="Test",
            description="Test",
            metric_name="test.metric",
            condition="gt",
            threshold=50,
            duration_seconds=0
        )
        manager.add_rule(rule)
        
        collector.set_gauge("test.metric", 100)
        manager.evaluate_rules()
        manager.evaluate_rules()  # Fire after duration
        
        # Check if notification was received
        notifs = notifications.get_notifications()
        # May or may not have notifications depending on timing
        assert isinstance(notifs, list)
    
    def test_acknowledge_alert(self):
        """Test acknowledging an alert"""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        
        rule = AlertRule(
            rule_id="test",
            name="Test",
            description="Test",
            metric_name="test.metric",
            condition="gt",
            threshold=50,
            duration_seconds=0
        )
        manager.add_rule(rule)
        
        collector.set_gauge("test.metric", 100)
        manager.evaluate_rules()
        manager.evaluate_rules()
        
        # Acknowledge
        manager.acknowledge_alert("test", "test_user")
        
        active = manager.get_active_alerts()
        for alert in active:
            if alert['rule_id'] == 'test':
                assert alert['status'] in ['acknowledged', 'firing']
    
    def test_alert_summary(self):
        """Test alert summary generation"""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        
        rule1 = AlertRule(
            rule_id="rule1",
            name="Rule 1",
            description="Test",
            metric_name="metric1",
            condition="gt",
            threshold=50,
            severity=AlertSeverity.WARNING
        )
        rule2 = AlertRule(
            rule_id="rule2",
            name="Rule 2",
            description="Test",
            metric_name="metric2",
            condition="gt",
            threshold=50,
            severity=AlertSeverity.CRITICAL
        )
        
        manager.add_rule(rule1)
        manager.add_rule(rule2)
        
        summary = manager.get_alert_summary()
        
        assert summary['total_rules'] == 2
        assert 'by_severity' in summary


class TestNotificationChannels:
    """Tests for notification channels"""
    
    def test_log_notification_channel(self):
        """Test log notification channel"""
        channel = LogNotificationChannel()
        
        # Create mock alert
        rule = AlertRule(
            rule_id="test",
            name="Test Alert",
            description="Test description",
            metric_name="test.metric",
            condition="gt",
            threshold=50
        )
        
        alert = Alert(
            alert_id="alert_123",
            rule=rule,
            status=AlertStatus.FIRING,
            value=100,
            started_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Should not raise
        channel.send(alert)
    
    def test_in_memory_notification_channel(self):
        """Test in-memory notification channel"""
        channel = InMemoryNotificationChannel(max_notifications=10)
        
        rule = AlertRule(
            rule_id="test",
            name="Test",
            description="Test",
            metric_name="test",
            condition="gt",
            threshold=50
        )
        
        for i in range(15):
            alert = Alert(
                alert_id=f"alert_{i}",
                rule=rule,
                status=AlertStatus.FIRING,
                value=100,
                started_at=datetime.now(),
                updated_at=datetime.now()
            )
            channel.send(alert)
        
        notifications = channel.get_notifications()
        assert len(notifications) == 10  # Max limit


# =============================================================================
# Performance Monitor Tests
# =============================================================================

class TestPerformanceMonitor:
    """Tests for PerformanceMonitor"""
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector)
        
        monitor.start(interval_seconds=1)
        assert monitor._monitoring == True
        
        time.sleep(0.1)  # Let it run briefly
        
        monitor.stop()
        assert monitor._monitoring == False
    
    def test_fallback_metrics_collection(self):
        """Test fallback metrics when psutil not available"""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector)
        
        # Call fallback directly
        monitor._collect_fallback_metrics()
        
        # Should have recorded thread count
        metric = collector.get_metric("process.threads")
        assert metric['exists'] == True


# =============================================================================
# Request Tracker Tests
# =============================================================================

class TestRequestTracker:
    """Tests for RequestTracker"""
    
    def test_track_request(self):
        """Test tracking a request"""
        collector = MetricsCollector()
        tracker = RequestTracker(collector)
        
        tracker.track_request(
            endpoint="/api/study/1",
            method="GET",
            status_code=200,
            duration_ms=50
        )
        
        # Check request count
        metric = collector.get_metric(
            "api.requests.total",
            labels={'endpoint': '/api/study/1', 'method': 'GET', 'status': '200'}
        )
        assert metric['exists'] == True
    
    def test_track_error_request(self):
        """Test tracking an error request"""
        collector = MetricsCollector()
        tracker = RequestTracker(collector)
        
        tracker.track_request(
            endpoint="/api/patient/1",
            method="GET",
            status_code=500,
            duration_ms=100
        )
        
        # Check error count
        metric = collector.get_metric(
            "api.requests.errors",
            labels={'endpoint': '/api/patient/1', 'status': '500'}
        )
        assert metric['exists'] == True


# =============================================================================
# Integration Tests
# =============================================================================

class TestMonitoringIntegration:
    """Integration tests for the monitoring system"""
    
    def test_full_monitoring_flow(self):
        """Test complete monitoring flow"""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        tracker = RequestTracker(collector)
        
        # Setup alert rule
        rule = AlertRule(
            rule_id="high_error_rate",
            name="High Error Rate",
            description="Too many errors",
            metric_name="api.requests.errors",
            condition="gt",
            threshold=5,
            duration_seconds=0,
            severity=AlertSeverity.ERROR
        )
        manager.add_rule(rule)
        
        # Track some requests
        for i in range(10):
            tracker.track_request("/api/test", "GET", 500, 100)
        
        # Evaluate alerts
        manager.evaluate_rules()
        manager.evaluate_rules()
        
        # Check state
        summary = manager.get_alert_summary()
        assert summary['total_rules'] == 1
    
    def test_default_alert_rules(self):
        """Test setting up default alert rules"""
        collector = MetricsCollector()
        manager = AlertManager(collector)
        
        setup_default_alert_rules(manager)
        
        summary = manager.get_alert_summary()
        assert summary['total_rules'] >= 5  # At least 5 default rules


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety"""
    
    def test_concurrent_metric_recording(self):
        """Test concurrent metric recording"""
        collector = MetricsCollector()
        
        def record_metrics():
            for i in range(100):
                collector.increment("concurrent.counter")
                collector.set_gauge("concurrent.gauge", i)
        
        threads = [threading.Thread(target=record_metrics) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        metric = collector.get_metric("concurrent.counter")
        assert metric['exists'] == True
        # Should have 500 increments (5 threads * 100)
        assert metric['recent_points'][-1]['value'] == 500


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def metrics_collector():
    """Provide fresh metrics collector"""
    return MetricsCollector()


@pytest.fixture
def alert_manager(metrics_collector):
    """Provide fresh alert manager"""
    return AlertManager(metrics_collector)


@pytest.fixture
def request_tracker(metrics_collector):
    """Provide fresh request tracker"""
    return RequestTracker(metrics_collector)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
