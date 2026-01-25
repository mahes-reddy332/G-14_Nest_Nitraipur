"""
Studies API Router
Endpoints for study-level data and metrics
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Pydantic Models ==============

class StudySummary(BaseModel):
    study_id: str
    study_name: str
    total_patients: int
    total_sites: int
    clean_patients: int
    dirty_patients: int
    cleanliness_rate: float
    dqi_score: float
    open_queries: int
    pending_saes: int
    uncoded_terms: int
    enrollment_progress: float
    status: str
    last_updated: str


class StudyDetail(BaseModel):
    study_id: str
    study_name: str
    protocol: Optional[str]
    phase: Optional[str]
    therapeutic_area: Optional[str]
    start_date: Optional[str]
    target_enrollment: int
    current_enrollment: int
    regions: List[str]
    countries: List[str]
    total_sites: int
    active_sites: int
    metrics: Dict[str, Any]
    trends: Dict[str, List[float]]
    risk_indicators: List[Dict[str, Any]]


class StudyMetrics(BaseModel):
    study_id: str
    dqi_score: float
    dqi_trend: List[float]
    cleanliness_rate: float
    cleanliness_trend: List[float]
    query_count: int
    query_resolution_rate: float
    query_velocity: float  # queries resolved per day
    sae_count: int
    sae_reconciliation_rate: float
    coding_completion_rate: float
    visit_completion_rate: float
    form_completion_rate: float


class RegionBreakdown(BaseModel):
    region: str
    countries: List[str]
    total_sites: int
    total_patients: int
    clean_patients: int
    dqi_score: float
    risk_level: str


# ============== Endpoints ==============

@router.get("/", response_model=List[StudySummary])
async def get_all_studies():
    """Get summary of all studies"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        studies = await service.get_all_studies()
        return [StudySummary(**study) for study in studies]
    except Exception as e:
        logger.error(f"Error getting studies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{study_id}", response_model=StudyDetail)
async def get_study_detail(study_id: str):
    """Get detailed information for a specific study"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        study = await service.get_study_detail(study_id)
        if not study:
            raise HTTPException(status_code=404, detail=f"Study {study_id} not found")
        return StudyDetail(**study)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting study detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{study_id}/metrics", response_model=StudyMetrics)
async def get_study_metrics(study_id: str):
    """Get metrics for a specific study"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        metrics = await service.get_study_metrics(study_id)
        return StudyMetrics(**metrics)
    except Exception as e:
        logger.error(f"Error getting study metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{study_id}/regions", response_model=List[RegionBreakdown])
async def get_study_regions(study_id: str):
    """Get region breakdown for a study"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        regions = await service.get_study_regions(study_id)
        return [RegionBreakdown(**r) for r in regions]
    except Exception as e:
        logger.error(f"Error getting study regions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{study_id}/trends")
async def get_study_trends(
    study_id: str,
    metric: str = Query(..., description="Metric to get trends for"),
    days: int = Query(30, description="Number of days for trend data")
):
    """Get trend data for a specific metric"""
    try:
        from api.services.metrics_service import MetricsService
        from api.services.data_service import ClinicalDataService
        data_service = ClinicalDataService()
        await data_service.initialize()
        service = MetricsService(data_service)
        trends = await service.get_metric_trends(study_id, metric, days)
        return {
            "success": True,
            "study_id": study_id,
            "metric": metric,
            "days": days,
            "data": trends
        }
    except Exception as e:
        logger.error(f"Error getting study trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{study_id}/risk-assessment")
async def get_study_risk_assessment(study_id: str):
    """Get risk assessment for a study"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        assessment = await service.get_risk_assessment(study_id)
        return {
            "success": True,
            "study_id": study_id,
            "risk_assessment": assessment
        }
    except Exception as e:
        logger.error(f"Error getting risk assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{study_id}/heatmap")
async def get_study_heatmap(
    study_id: str,
    metric: str = Query("dqi", description="Metric for heatmap: dqi|cleanliness|queries|saes")
):
    """Get heatmap data for visualization"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        heatmap = await service.get_heatmap_data(study_id, metric)
        return {
            "success": True,
            "study_id": study_id,
            "metric": metric,
            "heatmap_data": heatmap
        }
    except Exception as e:
        logger.error(f"Error getting heatmap data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SourceFile(BaseModel):
    file_type: str
    display_name: str
    status: str  # 'loaded' or 'not_found'
    record_count: int
    loaded_at: Optional[str]


@router.get("/{study_id}/source-files", response_model=List[SourceFile])
async def get_study_source_files(study_id: str):
    """Get list of processed source files for a study"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        files = await service.get_study_source_files(study_id)
        return [SourceFile(**f) for f in files]
    except Exception as e:
        logger.error(f"Error getting source files for {study_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/source-files/all")
async def get_all_source_files():
    """Get processed source files for all studies"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        files = await service.get_all_source_files()
        return {
            "success": True,
            "data": files
        }
    except Exception as e:
        logger.error(f"Error getting all source files: {e}")
        raise HTTPException(status_code=500, detail=str(e))
