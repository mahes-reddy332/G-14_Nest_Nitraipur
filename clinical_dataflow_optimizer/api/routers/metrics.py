"""
Metrics API Router
Endpoints for DQI, cleanliness, and operational metrics
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Pydantic Models ==============

class DQIMetrics(BaseModel):
    overall_dqi: float = Field(..., ge=0, le=100)
    completeness: float
    consistency: float
    timeliness: float
    accuracy: float
    conformity: float
    trend: List[float]
    comparison_to_benchmark: float


class CleanlinessMetrics(BaseModel):
    overall_rate: float
    total_patients: int
    clean_patients: int
    dirty_patients: int
    pending_patients: int
    trend: List[float]
    by_category: Dict[str, float]


class QueryMetrics(BaseModel):
    total_queries: int
    open_queries: int
    closed_queries: int
    resolution_rate: float
    avg_resolution_time_days: float
    velocity_trend: List[float]
    by_category: Dict[str, int]
    aging_distribution: Dict[str, int]


class SAEMetrics(BaseModel):
    total_saes: int
    reconciled: int
    pending: int
    overdue: int
    reconciliation_rate: float
    avg_reconciliation_days: float
    by_seriousness: Dict[str, int]


class CodingMetrics(BaseModel):
    total_terms: int
    coded: int
    uncoded: int
    completion_rate: float
    meddra_status: Dict[str, int]
    whodrug_status: Dict[str, int]
    uncoded_breakdown: List[Dict[str, Any]]


class OperationalVelocity(BaseModel):
    enrollment_velocity: float  # patients per week
    query_resolution_velocity: float  # queries per day
    form_completion_velocity: float  # forms per day
    sae_processing_velocity: float  # saes per day
    overall_velocity_index: float
    trend: List[float]


class KPITile(BaseModel):
    id: str
    title: str
    value: Any
    unit: Optional[str]
    trend: str  # up, down, stable
    trend_value: Optional[float]
    status: str  # good, warning, critical
    tooltip: str
    drill_down_available: bool


# ============== Endpoints ==============

@router.get("/kpi-tiles", response_model=List[KPITile])
async def get_kpi_tiles(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None)
):
    """Get KPI tiles for dashboard display"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        tiles = await service.get_kpi_tiles(study_id, site_id)
        return [KPITile(**t) for t in tiles]
    except Exception as e:
        logger.error(f"Error getting KPI tiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dqi", response_model=DQIMetrics)
async def get_dqi_metrics(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    days: int = Query(30, description="Trend period")
):
    """Get Data Quality Index metrics"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        metrics = await service.get_dqi_metrics(study_id, site_id, days)
        return DQIMetrics(**metrics)
    except Exception as e:
        logger.error(f"Error getting DQI metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cleanliness", response_model=CleanlinessMetrics)
async def get_cleanliness_metrics(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    days: int = Query(30, description="Trend period")
):
    """Get clean patient metrics"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        metrics = await service.get_cleanliness_metrics(study_id, site_id, days)
        return CleanlinessMetrics(**metrics)
    except Exception as e:
        logger.error(f"Error getting cleanliness metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queries", response_model=QueryMetrics)
async def get_query_metrics(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    days: int = Query(30, description="Trend period")
):
    """Get query management metrics"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        metrics = await service.get_query_metrics(study_id, site_id, days)
        return QueryMetrics(**metrics)
    except Exception as e:
        logger.error(f"Error getting query metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/saes", response_model=SAEMetrics)
async def get_sae_metrics(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None)
):
    """Get SAE reconciliation metrics"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        metrics = await service.get_sae_metrics(study_id, site_id)
        return SAEMetrics(**metrics)
    except Exception as e:
        logger.error(f"Error getting SAE metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/coding", response_model=CodingMetrics)
async def get_coding_metrics(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None)
):
    """Get medical coding completion metrics"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        metrics = await service.get_coding_metrics(study_id, site_id)
        return CodingMetrics(**metrics)
    except Exception as e:
        logger.error(f"Error getting coding metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/velocity", response_model=OperationalVelocity)
async def get_operational_velocity(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    days: int = Query(30, description="Calculation period")
):
    """Get operational velocity metrics"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        metrics = await service.get_velocity_metrics(study_id, site_id, days)
        return OperationalVelocity(**metrics)
    except Exception as e:
        logger.error(f"Error getting velocity metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/heatmap")
async def get_heatmap_data(
    metric: str = Query(..., description="Metric: dqi|cleanliness|queries|risk"),
    group_by: str = Query("site", description="Grouping: site|country|region"),
    study_id: Optional[str] = Query(None)
):
    """Get heatmap data for visualization"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        heatmap = await service.get_heatmap_data(metric, group_by, study_id)
        return heatmap
    except Exception as e:
        logger.error(f"Error getting heatmap data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_metric_trends(
    metrics: str = Query(..., description="Comma-separated metrics"),
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    days: int = Query(30, description="Trend period")
):
    """Get trend data for multiple metrics"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        
        metric_list = [m.strip() for m in metrics.split(",")]
        trends = await service.get_multiple_trends(metric_list, study_id, site_id, days)
        
        return {
            "success": True,
            "metrics": metric_list,
            "days": days,
            "trends": trends
        }
    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmarks")
async def get_benchmarks(
    study_id: Optional[str] = Query(None),
    metrics: str = Query("dqi,cleanliness,query_resolution", description="Metrics to benchmark")
):
    """Get benchmark comparisons"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        
        metric_list = [m.strip() for m in metrics.split(",")]
        benchmarks = await service.get_benchmarks(study_id, metric_list)
        
        return {
            "success": True,
            "benchmarks": benchmarks
        }
    except Exception as e:
        logger.error(f"Error getting benchmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies")
async def detect_anomalies(
    study_id: Optional[str] = Query(None),
    sensitivity: float = Query(0.8, description="Anomaly detection sensitivity")
):
    """Detect metric anomalies"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        anomalies = await service.detect_anomalies(study_id, sensitivity)
        return {
            "success": True,
            "anomalies": anomalies,
            "total": len(anomalies)
        }
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))
