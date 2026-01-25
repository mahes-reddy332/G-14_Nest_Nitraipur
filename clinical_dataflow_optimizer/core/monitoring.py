"""
Monitoring and Alerting System
==============================

Comprehensive monitoring, metrics collection, and alerting system
for the Clinical Dataflow Optimizer.

Features:
- Metrics collection and aggregation
- Performance monitoring
- Custom alert rules
- Alert notification channels
- Dashboard metrics API
"""

import logging
import time
import threading
import asyncio
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict, deque
import statistics
import json
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Metric Types
# =============================================================================

class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = auto()      # Monotonically increasing value
    GAUGE = auto()        # Point-in-time value
    HISTOGRAM = auto()    # Distribution of values
    TIMER = auto()        # Duration measurements


@dataclass
class MetricPoint:
    """A single metric data point"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class MetricSeries:
    """Time series of metric points"""
    name: str
    metric_type: MetricType
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, timestamp: datetime = None):
        """Add a data point to the series"""
        self.points.append(MetricPoint(
            name=self.name,
            value=value,
            timestamp=timestamp or datetime.now(),
            labels=self.labels,
            metric_type=self.metric_type
        ))
    
    def get_recent(self, duration_seconds: int = 300) -> List[MetricPoint]:
        """Get points from the last N seconds"""
        cutoff = datetime.now() - timedelta(seconds=duration_seconds)
        return [p for p in self.points if p.timestamp >= cutoff]
    
    def get_stats(self, duration_seconds: int = 300) -> Dict:
        """Get statistics for recent points"""
        recent = self.get_recent(duration_seconds)
        if not recent:
            return {'count': 0}
        
        values = [p.value for p in recent]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'sum': sum(values)
        }


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """
    Central metrics collection and aggregation system.
    """
    
    def __init__(self):
        self._series: Dict[str, MetricSeries] = {}
        self._counters: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def _get_series_key(self, name: str, labels: Dict = None) -> str:
        """Generate unique key for metric series"""
        label_str = ""
        if labels:
            label_str = "_" + "_".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{label_str}"
    
    def _get_or_create_series(
        self,
        name: str,
        metric_type: MetricType,
        labels: Dict = None
    ) -> MetricSeries:
        """Get or create a metric series"""
        key = self._get_series_key(name, labels)
        if key not in self._series:
            self._series[key] = MetricSeries(
                name=name,
                metric_type=metric_type,
                labels=labels or {}
            )
        return self._series[key]
    
    # Counter operations
    def increment(self, name: str, value: float = 1.0, labels: Dict = None):
        """Increment a counter metric"""
        with self._lock:
            key = self._get_series_key(name, labels)
            self._counters[key] += value
            series = self._get_or_create_series(name, MetricType.COUNTER, labels)
            series.add_point(self._counters[key])
    
    # Gauge operations
    def set_gauge(self, name: str, value: float, labels: Dict = None):
        """Set a gauge metric value"""
        with self._lock:
            series = self._get_or_create_series(name, MetricType.GAUGE, labels)
            series.add_point(value)
    
    # Histogram operations
    def record_histogram(self, name: str, value: float, labels: Dict = None):
        """Record a value in a histogram"""
        with self._lock:
            series = self._get_or_create_series(name, MetricType.HISTOGRAM, labels)
            series.add_point(value)
    
    # Timer operations
    def record_timer(self, name: str, duration_ms: float, labels: Dict = None):
        """Record a duration measurement"""
        with self._lock:
            series = self._get_or_create_series(name, MetricType.TIMER, labels)
            series.add_point(duration_ms)
    
    def timer(self, name: str, labels: Dict = None):
        """Context manager for timing operations"""
        return MetricTimer(self, name, labels)
    
    # Query operations
    def get_metric(
        self,
        name: str,
        labels: Dict = None,
        duration_seconds: int = 300
    ) -> Dict:
        """Get metric data and statistics"""
        key = self._get_series_key(name, labels)
        if key not in self._series:
            return {'exists': False}
        
        series = self._series[key]
        recent = series.get_recent(duration_seconds)
        
        return {
            'exists': True,
            'name': name,
            'type': series.metric_type.name,
            'labels': series.labels,
            'stats': series.get_stats(duration_seconds),
            'recent_points': [
                {'value': p.value, 'timestamp': p.timestamp.isoformat()}
                for p in recent[-50:]  # Last 50 points
            ]
        }
    
    def get_all_metrics(self, duration_seconds: int = 300) -> List[Dict]:
        """Get all metrics data"""
        with self._lock:
            return [
                self.get_metric(series.name, series.labels, duration_seconds)
                for series in self._series.values()
            ]
    
    def get_dashboard_metrics(self) -> Dict:
        """Get metrics formatted for dashboard display"""
        metrics = {}
        
        with self._lock:
            for key, series in self._series.items():
                stats = series.get_stats(300)  # Last 5 minutes
                metrics[key] = {
                    'name': series.name,
                    'type': series.metric_type.name,
                    'labels': series.labels,
                    'current': series.points[-1].value if series.points else None,
                    'avg': stats.get('mean'),
                    'min': stats.get('min'),
                    'max': stats.get('max'),
                    'count': stats.get('count', 0)
                }
        
        return metrics


class MetricTimer:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Dict = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        self.collector.record_timer(self.name, duration_ms, self.labels)
        return False


# =============================================================================
# Alert Definitions
# =============================================================================

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class AlertRule:
    """Definition of an alert rule"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "gt", "lt", "eq", "gte", "lte"
    threshold: float
    duration_seconds: int = 60  # Must be above threshold for this duration
    severity: AlertSeverity = AlertSeverity.WARNING
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    def evaluate(self, value: float) -> bool:
        """Evaluate if the condition is met"""
        ops = {
            'gt': lambda v: v > self.threshold,
            'lt': lambda v: v < self.threshold,
            'eq': lambda v: v == self.threshold,
            'gte': lambda v: v >= self.threshold,
            'lte': lambda v: v <= self.threshold,
            'ne': lambda v: v != self.threshold,
        }
        return ops.get(self.condition, lambda v: False)(value)


@dataclass
class Alert:
    """An active alert instance"""
    alert_id: str
    rule: AlertRule
    status: AlertStatus
    value: float
    started_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'rule_id': self.rule.rule_id,
            'name': self.rule.name,
            'description': self.rule.description,
            'severity': self.rule.severity.value,
            'status': self.status.value,
            'value': self.value,
            'threshold': self.rule.threshold,
            'started_at': self.started_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'labels': {**self.rule.labels, **self.labels}
        }


# =============================================================================
# Alert Manager
# =============================================================================

class AlertManager:
    """
    Manages alert rules, evaluates conditions, and triggers notifications.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._pending_alerts: Dict[str, datetime] = {}  # rule_id -> first_triggered_at
        self._alert_history: deque = deque(maxlen=1000)
        self._notification_channels: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        with self._lock:
            self._rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")
    
    def remove_rule(self, rule_id: str):
        """Remove an alert rule"""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")
    
    def add_notification_channel(self, channel: Callable[[Alert], None]):
        """Add a notification channel (function that receives Alert)"""
        self._notification_channels.append(channel)
    
    def evaluate_rules(self):
        """Evaluate all alert rules against current metrics"""
        now = datetime.now()
        
        with self._lock:
            for rule_id, rule in self._rules.items():
                if not rule.enabled:
                    continue
                
                # Get metric value
                metric = self.metrics.get_metric(rule.metric_name, rule.labels, 60)
                if not metric.get('exists') or metric['stats'].get('count', 0) == 0:
                    continue
                
                current_value = metric['stats'].get('mean', 0)
                condition_met = rule.evaluate(current_value)
                
                if condition_met:
                    self._handle_condition_met(rule, current_value, now)
                else:
                    self._handle_condition_cleared(rule, now)
    
    def _handle_condition_met(
        self,
        rule: AlertRule,
        value: float,
        now: datetime
    ):
        """Handle when an alert condition is met"""
        rule_id = rule.rule_id
        
        # Check if already firing
        if rule_id in self._active_alerts:
            # Update existing alert
            alert = self._active_alerts[rule_id]
            alert.value = value
            alert.updated_at = now
            return
        
        # Check if pending (waiting for duration)
        if rule_id in self._pending_alerts:
            first_triggered = self._pending_alerts[rule_id]
            elapsed = (now - first_triggered).total_seconds()
            
            if elapsed >= rule.duration_seconds:
                # Fire the alert
                self._fire_alert(rule, value, now)
                del self._pending_alerts[rule_id]
        else:
            # Start pending period
            self._pending_alerts[rule_id] = now
    
    def _handle_condition_cleared(self, rule: AlertRule, now: datetime):
        """Handle when an alert condition is cleared"""
        rule_id = rule.rule_id
        
        # Clear pending
        if rule_id in self._pending_alerts:
            del self._pending_alerts[rule_id]
        
        # Resolve active alert
        if rule_id in self._active_alerts:
            alert = self._active_alerts[rule_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = now
            alert.updated_at = now
            
            self._alert_history.append(alert)
            del self._active_alerts[rule_id]
            
            logger.info(f"Alert resolved: {rule.name}")
            self._send_notifications(alert)
    
    def _fire_alert(self, rule: AlertRule, value: float, now: datetime):
        """Fire a new alert"""
        alert_id = f"alert_{uuid.uuid4().hex[:12]}"
        
        alert = Alert(
            alert_id=alert_id,
            rule=rule,
            status=AlertStatus.FIRING,
            value=value,
            started_at=now,
            updated_at=now
        )
        
        self._active_alerts[rule.rule_id] = alert
        logger.warning(f"Alert firing: {rule.name} - {rule.description}")
        
        self._send_notifications(alert)
    
    def _send_notifications(self, alert: Alert):
        """Send alert to all notification channels"""
        for channel in self._notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")
    
    def acknowledge_alert(self, rule_id: str, acknowledged_by: str):
        """Acknowledge an active alert"""
        with self._lock:
            if rule_id in self._active_alerts:
                alert = self._active_alerts[rule_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                logger.info(f"Alert acknowledged: {alert.rule.name} by {acknowledged_by}")
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        with self._lock:
            return [alert.to_dict() for alert in self._active_alerts.values()]
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get alert history"""
        with self._lock:
            history = list(self._alert_history)[-limit:]
            return [alert.to_dict() for alert in reversed(history)]
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts"""
        with self._lock:
            active = list(self._active_alerts.values())
            return {
                'total_rules': len(self._rules),
                'active_alerts': len(active),
                'by_severity': {
                    sev.value: sum(1 for a in active if a.rule.severity == sev)
                    for sev in AlertSeverity
                },
                'by_status': {
                    status.value: sum(1 for a in active if a.status == status)
                    for status in AlertStatus
                }
            }


# =============================================================================
# Notification Channels
# =============================================================================

class NotificationChannel:
    """Base class for notification channels"""
    
    def send(self, alert: Alert):
        raise NotImplementedError


class LogNotificationChannel(NotificationChannel):
    """Log alerts to logging system"""
    
    def __init__(self, logger_name: str = "alerts"):
        self.logger = logging.getLogger(logger_name)
    
    def send(self, alert: Alert):
        level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.rule.severity, logging.WARNING)
        
        self.logger.log(
            level,
            f"[{alert.status.value.upper()}] {alert.rule.name}: "
            f"{alert.rule.description} (value={alert.value}, threshold={alert.rule.threshold})"
        )


class WebhookNotificationChannel(NotificationChannel):
    """Send alerts to a webhook URL"""
    
    def __init__(self, webhook_url: str, headers: Dict = None):
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    def send(self, alert: Alert):
        import urllib.request
        import urllib.error
        
        try:
            data = json.dumps(alert.to_dict()).encode('utf-8')
            request = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers=self.headers,
                method='POST'
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                logger.info(f"Webhook notification sent: {response.status}")
        except urllib.error.URLError as e:
            logger.error(f"Failed to send webhook notification: {e}")


class InMemoryNotificationChannel(NotificationChannel):
    """Store alerts in memory (for testing or UI display)"""
    
    def __init__(self, max_notifications: int = 100):
        self.notifications: deque = deque(maxlen=max_notifications)
    
    def send(self, alert: Alert):
        self.notifications.append({
            'alert': alert.to_dict(),
            'received_at': datetime.now().isoformat()
        })
    
    def get_notifications(self, limit: int = 20) -> List[Dict]:
        return list(self.notifications)[-limit:]


# =============================================================================
# Performance Monitor
# =============================================================================

class PerformanceMonitor:
    """
    Monitors system performance and resource usage.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._interval = 10  # seconds
    
    def start(self, interval_seconds: int = 10):
        """Start performance monitoring"""
        if self._monitoring:
            return
        
        self._interval = interval_seconds
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                self._collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting performance metrics: {e}")
            time.sleep(self._interval)
    
    def _collect_metrics(self):
        """Collect system performance metrics"""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.set_gauge("system.cpu.percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.set_gauge("system.memory.percent", memory.percent)
            self.metrics.set_gauge("system.memory.used_bytes", memory.used)
            self.metrics.set_gauge("system.memory.available_bytes", memory.available)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics.set_gauge("system.disk.percent", disk.percent)
            self.metrics.set_gauge("system.disk.used_bytes", disk.used)
            
            # Process metrics
            process = psutil.Process()
            self.metrics.set_gauge("process.cpu.percent", process.cpu_percent())
            self.metrics.set_gauge("process.memory.rss_bytes", process.memory_info().rss)
            self.metrics.set_gauge("process.threads", process.num_threads())
            
        except ImportError:
            # psutil not available, use fallback metrics
            self._collect_fallback_metrics()
    
    def _collect_fallback_metrics(self):
        """Collect basic metrics when psutil is not available"""
        import gc
        
        # Garbage collection stats
        gc_stats = gc.get_stats()
        for i, stat in enumerate(gc_stats):
            self.metrics.set_gauge(f"gc.generation_{i}.collections", stat['collections'])
            self.metrics.set_gauge(f"gc.generation_{i}.collected", stat['collected'])
        
        # Thread count
        self.metrics.set_gauge("process.threads", threading.active_count())


# =============================================================================
# Request Tracking
# =============================================================================

class RequestTracker:
    """
    Tracks API request metrics and performance.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    def track_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        labels: Dict = None
    ):
        """Track an API request"""
        base_labels = {
            'endpoint': endpoint,
            'method': method,
            'status': str(status_code),
            **(labels or {})
        }
        
        # Request count
        self.metrics.increment("api.requests.total", labels=base_labels)
        
        # Request duration
        self.metrics.record_timer(
            "api.requests.duration_ms",
            duration_ms,
            labels={'endpoint': endpoint, 'method': method}
        )
        
        # Error rate
        if status_code >= 400:
            self.metrics.increment(
                "api.requests.errors",
                labels={'endpoint': endpoint, 'status': str(status_code)}
            )
    
    def middleware(self):
        """Create FastAPI middleware for request tracking"""
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import Response
        
        tracker = self
        
        class RequestTrackingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next) -> Response:
                start_time = time.time()
                
                response = await call_next(request)
                
                duration_ms = (time.time() - start_time) * 1000
                tracker.track_request(
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=response.status_code,
                    duration_ms=duration_ms
                )
                
                return response
        
        return RequestTrackingMiddleware


# =============================================================================
# Global Instances
# =============================================================================

_metrics_collector: Optional[MetricsCollector] = None
_alert_manager: Optional[AlertManager] = None
_performance_monitor: Optional[PerformanceMonitor] = None
_request_tracker: Optional[RequestTracker] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_alert_manager() -> AlertManager:
    """Get or create global alert manager"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(get_metrics_collector())
    return _alert_manager


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(get_metrics_collector())
    return _performance_monitor


def get_request_tracker() -> RequestTracker:
    """Get or create global request tracker"""
    global _request_tracker
    if _request_tracker is None:
        _request_tracker = RequestTracker(get_metrics_collector())
    return _request_tracker


# =============================================================================
# Default Alert Rules
# =============================================================================

def setup_default_alert_rules(alert_manager: AlertManager):
    """Set up default alert rules for the system"""
    
    rules = [
        AlertRule(
            rule_id="high_cpu",
            name="High CPU Usage",
            description="CPU usage is above 80%",
            metric_name="system.cpu.percent",
            condition="gt",
            threshold=80,
            duration_seconds=60,
            severity=AlertSeverity.WARNING
        ),
        AlertRule(
            rule_id="critical_cpu",
            name="Critical CPU Usage",
            description="CPU usage is above 95%",
            metric_name="system.cpu.percent",
            condition="gt",
            threshold=95,
            duration_seconds=30,
            severity=AlertSeverity.CRITICAL
        ),
        AlertRule(
            rule_id="high_memory",
            name="High Memory Usage",
            description="Memory usage is above 85%",
            metric_name="system.memory.percent",
            condition="gt",
            threshold=85,
            duration_seconds=60,
            severity=AlertSeverity.WARNING
        ),
        AlertRule(
            rule_id="high_error_rate",
            name="High API Error Rate",
            description="API error rate is elevated",
            metric_name="api.requests.errors",
            condition="gt",
            threshold=10,
            duration_seconds=60,
            severity=AlertSeverity.ERROR
        ),
        AlertRule(
            rule_id="slow_api",
            name="Slow API Response",
            description="Average API response time is above 2000ms",
            metric_name="api.requests.duration_ms",
            condition="gt",
            threshold=2000,
            duration_seconds=120,
            severity=AlertSeverity.WARNING
        ),
    ]
    
    for rule in rules:
        alert_manager.add_rule(rule)
    
    logger.info(f"Set up {len(rules)} default alert rules")
