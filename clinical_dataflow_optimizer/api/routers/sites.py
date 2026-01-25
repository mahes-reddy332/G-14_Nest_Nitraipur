"""
Sites API Router
Endpoints for site-level data and performance metrics
"""

from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Enums ==============

class SiteRiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SiteStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    CLOSED = "closed"


# ============== Pydantic Models ==============

class SitePerformance(BaseModel):
    query_resolution_rate: float
    query_resolution_velocity: float  # days to resolve
    enrollment_rate: float
    data_entry_timeliness: float  # % on time
    sae_reporting_timeliness: float
    overall_score: float


class SiteSummary(BaseModel):
    site_id: str
    site_name: str
    study_id: str
    country: str
    region: str
    status: SiteStatus
    total_patients: int
    clean_patients: int
    dirty_patients: int
    cleanliness_rate: float
    dqi_score: float
    open_queries: int
    pending_saes: int
    risk_level: SiteRiskLevel
    performance: SitePerformance


class SiteDetail(BaseModel):
    site_id: str
    site_name: str
    study_id: str
    country: str
    region: str
    address: Optional[str]
    principal_investigator: Optional[str]
    status: SiteStatus
    activation_date: Optional[str]
    total_patients: int
    target_enrollment: int
    clean_patients: int
    dirty_patients: int
    metrics: Dict[str, Any]
    performance: SitePerformance
    trends: Dict[str, List[float]]
    risk_indicators: List[Dict[str, Any]]
    cra_assignments: List[Dict[str, Any]]
    recent_activity: List[Dict[str, Any]]


class SiteComparison(BaseModel):
    site_id: str
    site_name: str
    dqi_score: float
    cleanliness_rate: float
    query_resolution_rate: float
    enrollment_progress: float
    rank: int


class CRAActivity(BaseModel):
    cra_id: str
    cra_name: str
    site_id: str
    activity_type: str
    description: str
    timestamp: str
    impact: Optional[str]


# ============== Endpoints ==============

@router.get("/", response_model=List[SiteSummary])
async def get_sites(
    study_id: Optional[str] = Query(None, description="Filter by study"),
    country: Optional[str] = Query(None, description="Filter by country"),
    region: Optional[str] = Query(None, description="Filter by region"),
    status: Optional[SiteStatus] = Query(None, description="Filter by status"),
    risk_level: Optional[SiteRiskLevel] = Query(None, description="Filter by risk level"),
    min_patients: Optional[int] = Query(None, description="Minimum patient count"),
    sort_by: str = Query("dqi_score", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order")
):
    """Get list of sites with filters"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        
        filters = {
            "study_id": study_id,
            "country": country,
            "region": region,
            "status": status.value if status else None,
            "risk_level": risk_level.value if risk_level else None,
            "min_patients": min_patients
        }
        filters = {k: v for k, v in filters.items() if v is not None}
        
        sites = await service.get_sites(filters, sort_by, sort_order)
        return [SiteSummary(**s) for s in sites]
    except Exception as e:
        logger.error(f"Error getting sites: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/high-risk", response_model=List[SiteSummary])
async def get_high_risk_sites(
    study_id: Optional[str] = Query(None),
    limit: int = Query(20, le=100)
):
    """Get sites with high risk indicators"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        sites = await service.get_high_risk_sites(study_id, limit)
        return [SiteSummary(**s) for s in sites]
    except Exception as e:
        logger.error(f"Error getting high-risk sites: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/slow-resolution", response_model=List[SiteSummary])
async def get_slow_resolution_sites(
    study_id: Optional[str] = Query(None),
    threshold_days: float = Query(7.0, description="Days threshold for slow resolution"),
    limit: int = Query(20, le=100)
):
    """Get sites with slow query resolution velocity"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        sites = await service.get_slow_resolution_sites(study_id, threshold_days, limit)
        return [SiteSummary(**s) for s in sites]
    except Exception as e:
        logger.error(f"Error getting slow resolution sites: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison", response_model=List[SiteComparison])
async def get_site_comparison(
    study_id: str = Query(..., description="Study ID for comparison"),
    metric: str = Query("dqi_score", description="Comparison metric")
):
    """Get site comparison/ranking within a study"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        comparison = await service.get_site_comparison(study_id, metric)
        return [SiteComparison(**c) for c in comparison]
    except Exception as e:
        logger.error(f"Error getting site comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{site_id}", response_model=SiteDetail)
async def get_site_detail(site_id: str = Path(..., description="Site ID")):
    """Get detailed information for a specific site"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        site = await service.get_site_detail(site_id)
        if not site:
            raise HTTPException(status_code=404, detail=f"Site {site_id} not found")
        return SiteDetail(**site)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting site detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{site_id}/performance", response_model=SitePerformance)
async def get_site_performance(site_id: str = Path(..., description="Site ID")):
    """Get performance metrics for a site"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        performance = await service.get_site_performance(site_id)
        return SitePerformance(**performance)
    except Exception as e:
        logger.error(f"Error getting site performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{site_id}/patients", response_model=List[Dict[str, Any]])
async def get_site_patients(
    site_id: str = Path(..., description="Site ID"),
    status: Optional[str] = Query(None, description="Filter by clean status")
):
    """Get patients for a specific site"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        patients = await service.get_site_patients(site_id, status)
        return patients
    except Exception as e:
        logger.error(f"Error getting site patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{site_id}/queries")
async def get_site_queries(
    site_id: str = Path(..., description="Site ID"),
    status: Optional[str] = Query(None, description="Filter: open|closed|all")
):
    """Get queries for a specific site"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        queries = await service.get_site_queries(site_id, status)
        return {
            "success": True,
            "site_id": site_id,
            "queries": queries,
            "total": len(queries)
        }
    except Exception as e:
        logger.error(f"Error getting site queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{site_id}/cra-activity", response_model=List[CRAActivity])
async def get_cra_activity(
    site_id: str = Path(..., description="Site ID"),
    days: int = Query(30, description="Activity lookback days")
):
    """Get CRA activity log for a site"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        activity = await service.get_cra_activity(site_id, days)
        return [CRAActivity(**a) for a in activity]
    except Exception as e:
        logger.error(f"Error getting CRA activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{site_id}/trends")
async def get_site_trends(
    site_id: str = Path(..., description="Site ID"),
    metrics: str = Query("dqi,cleanliness,queries", description="Comma-separated metrics"),
    days: int = Query(30, description="Trend period in days")
):
    """Get trend data for site metrics"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        
        metric_list = [m.strip() for m in metrics.split(",")]
        trends = await service.get_site_trends(site_id, metric_list, days)
        
        return {
            "success": True,
            "site_id": site_id,
            "metrics": metric_list,
            "days": days,
            "trends": trends
        }
    except Exception as e:
        logger.error(f"Error getting site trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{site_id}/issues")
async def get_site_issues(site_id: str = Path(..., description="Site ID")):
    """Get current issues and blockers for a site"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        issues = await service.get_site_issues(site_id)
        return {
            "success": True,
            "site_id": site_id,
            "issues": issues,
            "summary": {
                "critical": len([i for i in issues if i.get("severity") == "critical"]),
                "high": len([i for i in issues if i.get("severity") == "high"]),
                "medium": len([i for i in issues if i.get("severity") == "medium"]),
                "low": len([i for i in issues if i.get("severity") == "low"])
            }
        }
    except Exception as e:
        logger.error(f"Error getting site issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))
