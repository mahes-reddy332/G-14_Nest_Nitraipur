"""
Metrics API Router
Endpoints for DQI, cleanliness, and operational metrics
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

router = APIRouter()

# ============== In-memory Cache ==============
_metrics_cache: Dict[str, Any] = {}
_cache_timestamps: Dict[str, float] = {}
CACHE_TTL_SECONDS = 60  # 1 minute cache


def get_cached(key: str) -> Optional[Any]:
    """Get cached value if not expired"""
    if key in _metrics_cache:
        if time.time() - _cache_timestamps.get(key, 0) < CACHE_TTL_SECONDS:
            return _metrics_cache[key]
    return None


def set_cached(key: str, value: Any):
    """Set cached value with timestamp"""
    _metrics_cache[key] = value
    _cache_timestamps[key] = time.time()


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
    resolutions_per_day: float = Field(default=0.0) # Added field
    form_completion_velocity: float  # forms per day
    sae_processing_velocity: float  # saes per day
    overall_velocity_index: float
    trend: List[Dict[str, Any]] # Changed to list of dicts for frontend compatibility


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
        cache_key = f"kpi_tiles:{study_id}:{site_id}"
        cached = get_cached(cache_key)
        if cached:
            return cached
        
        from api.config import get_initialized_metrics_service
        service = await get_initialized_metrics_service()
        tiles = await service.get_kpi_tiles(study_id, site_id)
        result = [KPITile(**t) for t in tiles]
        set_cached(cache_key, result)
        return result
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
        cache_key = f"dqi:{study_id}:{site_id}:{days}"
        cached = get_cached(cache_key)
        if cached:
            return cached
        
        from api.config import get_initialized_metrics_service
        service = await get_initialized_metrics_service()
        metrics = await service.get_dqi_metrics(study_id, site_id, days)
        result = DQIMetrics(**metrics)
        set_cached(cache_key, result)
        return result
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
        cache_key = f"cleanliness:{study_id}:{site_id}:{days}"
        cached = get_cached(cache_key)
        if cached:
            return cached
        
        from api.config import get_initialized_metrics_service
        service = await get_initialized_metrics_service()
        metrics = await service.get_cleanliness_metrics(study_id, site_id, days)
        result = CleanlinessMetrics(**metrics)
        set_cached(cache_key, result)
        return result
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
        cache_key = f"queries:{study_id}:{site_id}:{days}"
        cached = get_cached(cache_key)
        if cached:
            return cached
        
        from api.config import get_initialized_metrics_service
        service = await get_initialized_metrics_service()
        metrics = await service.get_query_metrics(study_id, site_id, days)
        result = QueryMetrics(**metrics)
        set_cached(cache_key, result)
        return result
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
        cache_key = f"saes:{study_id}:{site_id}"
        cached = get_cached(cache_key)
        if cached:
            return cached
        
        from api.config import get_initialized_metrics_service
        service = await get_initialized_metrics_service()
        metrics = await service.get_sae_metrics(study_id, site_id)
        result = SAEMetrics(**metrics)
        set_cached(cache_key, result)
        return result
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
        cache_key = f"coding:{study_id}:{site_id}"
        cached = get_cached(cache_key)
        if cached:
            return cached
        
        from api.config import get_initialized_metrics_service
        service = await get_initialized_metrics_service()
        metrics = await service.get_coding_metrics(study_id, site_id)
        result = CodingMetrics(**metrics)
        set_cached(cache_key, result)
        return result
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
        cache_key = f"velocity:{study_id}:{site_id}:{days}"
        cached = get_cached(cache_key)
        if cached:
            return cached
        
        from api.config import get_initialized_metrics_service
        service = await get_initialized_metrics_service()
        metrics = await service.get_velocity_metrics(study_id, site_id, days)
        result = OperationalVelocity(**metrics)
        set_cached(cache_key, result)
        return result
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
        cache_key = f"heatmap:{metric}:{group_by}:{study_id}"
        cached = get_cached(cache_key)
        if cached:
            return cached
        
        from api.config import get_initialized_metrics_service
        service = await get_initialized_metrics_service()
        heatmap = await service.get_heatmap_data(metric, group_by, study_id)
        set_cached(cache_key, heatmap)
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
        cache_key = f"trends:{metrics}:{study_id}:{site_id}:{days}"
        cached = get_cached(cache_key)
        if cached:
            return cached
        
        from api.config import get_initialized_metrics_service
        service = await get_initialized_metrics_service()
        
        metric_list = [m.strip() for m in metrics.split(",")]
        trends = await service.get_multiple_trends(metric_list, study_id, site_id, days)
        
        result = {
            "success": True,
            "metrics": metric_list,
            "days": days,
            "trends": trends
        }
        set_cached(cache_key, result)
        return result
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
        cache_key = f"benchmarks:{study_id}:{metrics}"
        cached = get_cached(cache_key)
        if cached:
            return cached
        
        from api.config import get_initialized_metrics_service
        service = await get_initialized_metrics_service()
        
        metric_list = [m.strip() for m in metrics.split(",")]
        benchmarks = await service.get_benchmarks(study_id, metric_list)
        
        result = {
            "success": True,
            "benchmarks": benchmarks
        }
        set_cached(cache_key, result)
        return result
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
        cache_key = f"anomalies:{study_id}:{sensitivity}"
        cached = get_cached(cache_key)
        if cached:
            return cached
        
        from api.config import get_initialized_metrics_service
        service = await get_initialized_metrics_service()
        anomalies = await service.detect_anomalies(study_id, sensitivity)
        result = {
            "success": True,
            "anomalies": anomalies,
            "total": len(anomalies)
        }
        set_cached(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))
