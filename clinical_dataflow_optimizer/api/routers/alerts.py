"""
Alerts API Router
Endpoints for real-time alerts and notifications
"""

from fastapi import APIRouter, HTTPException, Query, Body, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import logging

from api.config import get_service

logger = logging.getLogger(__name__)

router = APIRouter()


def get_alert_service():
    """Dependency to get the singleton AlertService"""
    return get_service("alert_service")


# ============== Enums ==============

class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertStatus(str, Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class AlertCategory(str, Enum):
    DATA_QUALITY = "data_quality"
    SAFETY = "safety"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    SYSTEM = "system"


# ============== Pydantic Models ==============

class Alert(BaseModel):
    alert_id: str
    category: AlertCategory
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    source: str  # Which system/agent generated
    affected_entity: Dict[str, str]  # {type: study|site|patient, id: xxx}
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    actions_taken: List[str] = Field(default_factory=list)


class AlertSummary(BaseModel):
    total: int
    by_severity: Dict[str, int]
    by_category: Dict[str, int]
    by_status: Dict[str, int]
    unacknowledged: int
    critical_unresolved: int


class AlertAcknowledgement(BaseModel):
    acknowledged_by: str
    note: Optional[str] = None


class AlertResolution(BaseModel):
    resolved_by: str
    resolution_note: Optional[str] = None
    actions_taken: List[str] = []


# ============== Endpoints ==============

@router.get("/", response_model=List[Alert])
async def get_alerts(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    category: Optional[AlertCategory] = Query(None),
    severity: Optional[AlertSeverity] = Query(None),
    status: Optional[AlertStatus] = Query(None),
    unacknowledged_only: bool = Query(False),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    service = Depends(get_alert_service)
):
    """Get alerts with filters"""
    try:
        await service.initialize()
        
        filters = {
            "study_id": study_id,
            "site_id": site_id,
            "category": category.value if category else None,
            "severity": severity.value if severity else None,
            "status": status.value if status else None,
            "unacknowledged_only": unacknowledged_only
        }
        filters = {k: v for k, v in filters.items() if v is not None and v is not False}
        
        alerts = await service.get_alerts(filters, limit, offset)
        return [Alert(**a) for a in alerts]
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=AlertSummary)
async def get_alert_summary(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    service = Depends(get_alert_service)
):
    """Get alert summary counts"""
    try:
        await service.initialize()
        summary = await service.get_alert_summary(study_id, site_id)
        return AlertSummary(**summary)
    except Exception as e:
        logger.error(f"Error getting alert summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/critical", response_model=List[Alert])
async def get_critical_alerts(
    study_id: Optional[str] = Query(None),
    unresolved_only: bool = Query(True),
    service = Depends(get_alert_service)
):
    """Get critical alerts requiring immediate attention"""
    try:
        await service.initialize()
        alerts = await service.get_critical_alerts(study_id, unresolved_only)
        return [Alert(**a) for a in alerts]
    except Exception as e:
        logger.error(f"Error getting critical alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent", response_model=List[Alert])
async def get_recent_alerts(
    hours: int = Query(24, description="Look back hours"),
    study_id: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    service = Depends(get_alert_service)
):
    """Get recently created alerts"""
    try:
        await service.initialize()
        alerts = await service.get_recent_alerts(hours, study_id, limit)
        return [Alert(**a) for a in alerts]
    except Exception as e:
        logger.error(f"Error getting recent alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{alert_id}", response_model=Alert)
async def get_alert_detail(
    alert_id: str,
    service = Depends(get_alert_service)
):
    """Get detailed information for a specific alert"""
    try:
        await service.initialize()
        alert = await service.get_alert(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        return Alert(**alert)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledgement: AlertAcknowledgement = Body(...),
    service = Depends(get_alert_service)
):
    """Acknowledge an alert"""
    try:
        await service.initialize()
        result = await service.acknowledge_alert(
            alert_id,
            acknowledgement.acknowledged_by,
            acknowledgement.note
        )
        if not result:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        return {
            "success": True,
            "alert_id": alert_id,
            "status": "acknowledged",
            "acknowledged_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolution: AlertResolution = Body(...),
    service = Depends(get_alert_service)
):
    """Resolve an alert"""
    try:
        await service.initialize()
        result = await service.resolve_alert(
            alert_id,
            resolution.resolved_by,
            resolution.resolution_note,
            resolution.actions_taken
        )
        if not result:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        return {
            "success": True,
            "alert_id": alert_id,
            "status": "resolved",
            "resolved_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/dismiss")
async def dismiss_alert(
    alert_id: str,
    dismissed_by: str = Query(..., description="User dismissing the alert"),
    reason: str = Query(..., description="Reason for dismissal"),
    service = Depends(get_alert_service)
):
    """Dismiss an alert (mark as not actionable)"""
    try:
        await service.initialize()
        result = await service.dismiss_alert(alert_id, dismissed_by, reason)
        if not result:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        return {
            "success": True,
            "alert_id": alert_id,
            "status": "dismissed",
            "dismissed_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error dismissing alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{entity_type}/{entity_id}")
async def get_alert_history(
    entity_type: str,  # study|site|patient
    entity_id: str,
    days: int = Query(30, description="History period"),
    service = Depends(get_alert_service)
):
    """Get alert history for an entity"""
    try:
        await service.initialize()
        history = await service.get_alert_history(entity_type, entity_id, days)
        return {
            "success": True,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "history": history,
            "total": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting alert history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_alert_trends(
    study_id: Optional[str] = Query(None),
    days: int = Query(30, description="Trend period"),
    group_by: str = Query("day", description="Grouping: day|week|category|severity"),
    service = Depends(get_alert_service)
):
    """Get alert trends over time"""
    try:
        await service.initialize()
        trends = await service.get_alert_trends(study_id, days, group_by)
        return {
            "success": True,
            "days": days,
            "group_by": group_by,
            "trends": trends
        }
    except Exception as e:
        logger.error(f"Error getting alert trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))
