"""
System Monitoring and Health API Router
=======================================

Provides endpoints for:
- System health checks
- Metrics dashboard data
- Alert management
- Error tracking
- Performance monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["System Monitoring"])


# =============================================================================
# Request/Response Models
# =============================================================================

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    services: Dict[str, Any]
    version: str


class MetricResponse(BaseModel):
    """Single metric response"""
    name: str
    type: str
    current: Optional[float]
    avg: Optional[float]
    min: Optional[float]
    max: Optional[float]
    count: int


class AlertResponse(BaseModel):
    """Alert response"""
    alert_id: str
    rule_id: str
    name: str
    description: str
    severity: str
    status: str
    value: float
    threshold: float
    started_at: str
    updated_at: str


class AlertRuleRequest(BaseModel):
    """Request to create an alert rule"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str = Field(..., pattern="^(gt|lt|eq|gte|lte|ne)$")
    threshold: float
    duration_seconds: int = 60
    severity: str = "warning"


class ErrorSummaryResponse(BaseModel):
    """Error summary response"""
    total_errors: int
    unresolved: int
    error_counts: Dict[str, int]


class AcknowledgeAlertRequest(BaseModel):
    """Request to acknowledge an alert"""
    acknowledged_by: str


# =============================================================================
# Dependency Injection
# =============================================================================

def get_metrics_collector():
    """Get metrics collector instance"""
    from core.monitoring import get_metrics_collector
    return get_metrics_collector()


def get_alert_manager():
    """Get alert manager instance"""
    from core.monitoring import get_alert_manager
    return get_alert_manager()


def get_error_tracker():
    """Get error tracker instance"""
    from core.error_handling import get_error_tracker
    return get_error_tracker()


def get_health_checker():
    """Get health checker instance"""
    from core.error_handling import get_health_checker
    return get_health_checker()


# =============================================================================
# Health Endpoints
# =============================================================================

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Get system health status.
    
    Checks all registered services and returns overall health.
    """
    from core.error_handling import get_health_checker, ServiceHealth
    from api import __version__
    
    health_checker = get_health_checker()
    
    # Register default health checks if not already registered
    try:
        # Add basic checks
        health_checker.register_check("api", lambda: True)
        health_checker.register_check("database", lambda: True)  # Placeholder
    except Exception:
        pass  # Already registered
    
    # Run all checks
    results = health_checker.run_all_checks()
    overall = health_checker.get_overall_status()
    
    services = {}
    for name, result in results.items():
        services[name] = {
            'status': result.status.value,
            'latency_ms': round(result.latency_ms, 2),
            'details': result.details
        }
    
    return HealthCheckResponse(
        status=overall.value,
        timestamp=datetime.now().isoformat(),
        services=services,
        version=__version__
    )


@router.get("/health/liveness")
async def liveness_probe():
    """
    Kubernetes liveness probe.
    
    Returns 200 if the service is running.
    """
    return {"status": "ok"}


@router.get("/health/readiness")
async def readiness_probe():
    """
    Kubernetes readiness probe.
    
    Returns 200 if the service is ready to receive traffic.
    """
    from core.error_handling import get_health_checker, ServiceHealth
    
    health_checker = get_health_checker()
    overall = health_checker.get_overall_status()
    
    if overall == ServiceHealth.UNHEALTHY:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready"}


# =============================================================================
# Metrics Endpoints
# =============================================================================

@router.get("/metrics")
async def get_metrics(
    duration_seconds: int = Query(300, ge=60, le=3600, description="Time window in seconds"),
    metrics_collector=Depends(get_metrics_collector)
):
    """
    Get all system metrics.
    
    Returns metrics collected over the specified duration.
    """
    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration_seconds,
        "metrics": metrics_collector.get_all_metrics(duration_seconds)
    }


@router.get("/metrics/dashboard")
async def get_dashboard_metrics(
    metrics_collector=Depends(get_metrics_collector)
):
    """
    Get metrics formatted for dashboard display.
    
    Returns current values and statistics for all metrics.
    """
    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "data": metrics_collector.get_dashboard_metrics()
    }


@router.get("/metrics/{metric_name}")
async def get_metric(
    metric_name: str,
    duration_seconds: int = Query(300, ge=60, le=3600),
    metrics_collector=Depends(get_metrics_collector)
):
    """
    Get a specific metric by name.
    """
    metric = metrics_collector.get_metric(metric_name, duration_seconds=duration_seconds)
    
    if not metric.get('exists'):
        raise HTTPException(status_code=404, detail=f"Metric {metric_name} not found")
    
    return {
        "success": True,
        "metric": metric
    }


# =============================================================================
# Alert Endpoints
# =============================================================================

@router.get("/alerts")
async def get_active_alerts(
    alert_manager=Depends(get_alert_manager)
) -> Dict[str, Any]:
    """
    Get all active alerts.
    """
    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "alerts": alert_manager.get_active_alerts()
    }


@router.get("/alerts/summary")
async def get_alert_summary(
    alert_manager=Depends(get_alert_manager)
):
    """
    Get alert summary with counts by severity and status.
    """
    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "summary": alert_manager.get_alert_summary()
    }


@router.get("/alerts/history")
async def get_alert_history(
    limit: int = Query(50, ge=1, le=500),
    alert_manager=Depends(get_alert_manager)
):
    """
    Get alert history.
    """
    return {
        "success": True,
        "history": alert_manager.get_alert_history(limit)
    }


@router.post("/alerts/rules")
async def create_alert_rule(
    rule: AlertRuleRequest,
    alert_manager=Depends(get_alert_manager)
):
    """
    Create a new alert rule.
    """
    from core.monitoring import AlertRule, AlertSeverity
    
    severity_map = {
        'info': AlertSeverity.INFO,
        'warning': AlertSeverity.WARNING,
        'error': AlertSeverity.ERROR,
        'critical': AlertSeverity.CRITICAL,
    }
    
    alert_rule = AlertRule(
        rule_id=rule.rule_id,
        name=rule.name,
        description=rule.description,
        metric_name=rule.metric_name,
        condition=rule.condition,
        threshold=rule.threshold,
        duration_seconds=rule.duration_seconds,
        severity=severity_map.get(rule.severity, AlertSeverity.WARNING)
    )
    
    alert_manager.add_rule(alert_rule)
    
    return {
        "success": True,
        "message": f"Alert rule {rule.rule_id} created",
        "rule_id": rule.rule_id
    }


@router.delete("/alerts/rules/{rule_id}")
async def delete_alert_rule(
    rule_id: str,
    alert_manager=Depends(get_alert_manager)
):
    """
    Delete an alert rule.
    """
    alert_manager.remove_rule(rule_id)
    
    return {
        "success": True,
        "message": f"Alert rule {rule_id} deleted"
    }


@router.post("/alerts/{rule_id}/acknowledge")
async def acknowledge_alert(
    rule_id: str,
    request: AcknowledgeAlertRequest,
    alert_manager=Depends(get_alert_manager)
):
    """
    Acknowledge an active alert.
    """
    alert_manager.acknowledge_alert(rule_id, request.acknowledged_by)
    
    return {
        "success": True,
        "message": f"Alert {rule_id} acknowledged by {request.acknowledged_by}"
    }


# =============================================================================
# Error Tracking Endpoints
# =============================================================================

@router.get("/errors")
async def get_recent_errors(
    limit: int = Query(20, ge=1, le=100),
    error_tracker=Depends(get_error_tracker)
):
    """
    Get recent errors.
    """
    return {
        "success": True,
        "errors": error_tracker.get_recent_errors(limit)
    }


@router.get("/errors/summary", response_model=ErrorSummaryResponse)
async def get_error_summary(
    error_tracker=Depends(get_error_tracker)
):
    """
    Get error summary with counts.
    """
    return error_tracker.get_error_summary()


@router.post("/errors/{error_id}/resolve")
async def resolve_error(
    error_id: str,
    error_tracker=Depends(get_error_tracker)
):
    """
    Mark an error as resolved.
    """
    error_tracker.resolve_error(error_id)
    
    return {
        "success": True,
        "message": f"Error {error_id} marked as resolved"
    }


# =============================================================================
# Performance Endpoints
# =============================================================================

@router.get("/performance")
async def get_performance_metrics(
    metrics_collector=Depends(get_metrics_collector)
):
    """
    Get system performance metrics.
    
    Returns CPU, memory, disk, and process metrics.
    """
    performance_metrics = {}
    
    # System metrics
    for metric_name in ['system.cpu.percent', 'system.memory.percent', 'system.disk.percent']:
        metric = metrics_collector.get_metric(metric_name, duration_seconds=300)
        if metric.get('exists'):
            performance_metrics[metric_name] = {
                'current': metric['recent_points'][-1]['value'] if metric['recent_points'] else None,
                'avg': metric['stats'].get('mean'),
                'max': metric['stats'].get('max')
            }
    
    # Process metrics
    for metric_name in ['process.cpu.percent', 'process.memory.rss_bytes', 'process.threads']:
        metric = metrics_collector.get_metric(metric_name, duration_seconds=300)
        if metric.get('exists'):
            performance_metrics[metric_name] = {
                'current': metric['recent_points'][-1]['value'] if metric['recent_points'] else None,
                'avg': metric['stats'].get('mean')
            }
    
    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "metrics": performance_metrics
    }


# =============================================================================
# Circuit Breaker Status
# =============================================================================

@router.get("/circuit-breakers")
async def get_circuit_breaker_status():
    """
    Get status of all circuit breakers.
    """
    from core.error_handling import _circuit_breakers
    
    statuses = {}
    for name, cb in _circuit_breakers.items():
        statuses[name] = cb.get_state()
    
    return {
        "success": True,
        "circuit_breakers": statuses
    }


# =============================================================================
# Background Tasks
# =============================================================================

async def evaluate_alerts_background(alert_manager):
    """Background task to periodically evaluate alerts"""
    alert_manager.evaluate_rules()


@router.post("/alerts/evaluate")
async def trigger_alert_evaluation(
    background_tasks: BackgroundTasks,
    alert_manager=Depends(get_alert_manager)
):
    """
    Manually trigger alert evaluation.
    """
    background_tasks.add_task(evaluate_alerts_background, alert_manager)
    
    return {
        "success": True,
        "message": "Alert evaluation triggered"
    }


# =============================================================================
# Audit Log Endpoints
# =============================================================================

@router.get("/audit")
async def get_audit_log(
    user_id: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500)
):
    """
    Query audit log entries.
    """
    from core.security import get_audit_logger, AuditEventType
    
    audit_logger = get_audit_logger()
    
    # Convert event_type string to enum if provided
    event_type_enum = None
    if event_type:
        try:
            event_type_enum = AuditEventType(event_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
    
    events = audit_logger.query(
        user_id=user_id,
        event_type=event_type_enum,
        limit=limit
    )
    
    return {
        "success": True,
        "count": len(events),
        "events": events
    }


@router.get("/audit/integrity")
async def verify_audit_integrity():
    """
    Verify audit log integrity.
    """
    from core.security import get_audit_logger
    
    audit_logger = get_audit_logger()
    is_valid = audit_logger.verify_integrity()
    
    return {
        "success": True,
        "integrity_valid": is_valid,
        "timestamp": datetime.now().isoformat()
    }
