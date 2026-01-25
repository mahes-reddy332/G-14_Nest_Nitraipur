"""
Patients API Router
Endpoints for patient-level data and clean patient status
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import logging

# Import services
from api.services.data_service import ClinicalDataService

logger = logging.getLogger(__name__)

router = APIRouter()

# ============== Dependencies ==============

def get_data_service() -> 'ClinicalDataService':
    """Dependency injection for ClinicalDataService"""
    from api.main import data_service
    return data_service

# ============== Enums ==============

class CleanStatus(str, Enum):
    CLEAN = "clean"
    DIRTY = "dirty"
    PENDING = "pending"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============== Pydantic Models ==============

class CleanPatientStatus(BaseModel):
    """Clean Patient Status - dynamically calculated"""
    is_clean: bool
    cleanliness_score: float = Field(..., ge=0, le=100, description="Percentage score 0-100")
    status: CleanStatus
    blocking_factors: List[str] = Field(default_factory=list)
    missing_visits: int = 0
    open_queries: int = 0
    uncoded_terms: int = 0
    pending_saes: int = 0
    missing_forms: int = 0
    unsigned_forms: int = 0
    last_calculated: str


class PatientSummary(BaseModel):
    patient_id: str
    study_id: str
    site_id: str
    site_name: Optional[str]
    country: Optional[str]
    region: Optional[str]
    enrollment_date: Optional[str]
    clean_status: CleanPatientStatus
    risk_level: RiskLevel
    last_visit_date: Optional[str]
    next_scheduled_visit: Optional[str]
    total_visits: int
    completed_visits: int


class PatientDetail(BaseModel):
    patient_id: str
    study_id: str
    site_id: str
    site_name: Optional[str]
    country: Optional[str]
    region: Optional[str]
    enrollment_date: Optional[str]
    randomization_date: Optional[str]
    treatment_arm: Optional[str]
    current_status: str
    clean_status: CleanPatientStatus
    visits: List[Dict[str, Any]]
    queries: List[Dict[str, Any]]
    saes: List[Dict[str, Any]]
    coding_status: Dict[str, Any]
    forms_status: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    ai_insights: List[str]


class PatientListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    patients: List[PatientSummary]
    filters_applied: Dict[str, Any]


class PatientStatusChange(BaseModel):
    patient_id: str
    previous_status: CleanStatus
    new_status: CleanStatus
    previous_score: float
    new_score: float
    change_reason: str
    timestamp: str
    blocking_factors_added: List[str]
    blocking_factors_resolved: List[str]


# ============== Endpoints ==============

@router.get("/", response_model=PatientListResponse)
async def get_patients(
    study_id: Optional[str] = Query(None, description="Filter by study"),
    site_id: Optional[str] = Query(None, description="Filter by site"),
    status: Optional[CleanStatus] = Query(None, description="Filter by clean status"),
    risk_level: Optional[RiskLevel] = Query(None, description="Filter by risk level"),
    country: Optional[str] = Query(None, description="Filter by country"),
    has_open_queries: Optional[bool] = Query(None, description="Filter patients with open queries"),
    has_uncoded_terms: Optional[bool] = Query(None, description="Filter patients with uncoded terms"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=10, le=200, description="Items per page"),
    sort_by: str = Query("cleanliness_score", description="Sort field"),
    sort_order: str = Query("asc", description="Sort order: asc|desc")
):
    """Get paginated list of patients with filters"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        
        filters = {
            "study_id": study_id,
            "site_id": site_id,
            "status": status.value if status else None,
            "risk_level": risk_level.value if risk_level else None,
            "country": country,
            "has_open_queries": has_open_queries,
            "has_uncoded_terms": has_uncoded_terms
        }
        filters = {k: v for k, v in filters.items() if v is not None}
        
        result = await service.get_patients(
            filters=filters,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return PatientListResponse(**result)
    except Exception as e:
        logger.error(f"Error getting patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dirty", response_model=List[PatientSummary])
async def get_dirty_patients(
    study_id: Optional[str] = Query(None),
    limit: int = Query(100, le=500)
):
    """Get list of patients with dirty status (not clean)"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        patients = await service.get_dirty_patients(study_id, limit)
        return [PatientSummary(**p) for p in patients]
    except Exception as e:
        logger.error(f"Error getting dirty patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/at-risk", response_model=List[PatientSummary])
async def get_at_risk_patients(
    study_id: Optional[str] = Query(None),
    risk_threshold: float = Query(0.7, description="Risk score threshold")
):
    """Get patients at risk of becoming dirty or with high risk indicators"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        patients = await service.get_at_risk_patients(study_id, risk_threshold)
        return [PatientSummary(**p) for p in patients]
    except Exception as e:
        logger.error(f"Error getting at-risk patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status-changes", response_model=List[PatientStatusChange])
async def get_recent_status_changes(
    study_id: Optional[str] = Query(None),
    hours: int = Query(24, description="Look back hours"),
    limit: int = Query(50, le=200)
):
    """Get recent patient status changes"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        changes = await service.get_status_changes(study_id, hours, limit)
        return [PatientStatusChange(**c) for c in changes]
    except Exception as e:
        logger.error(f"Error getting status changes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/twins", response_model=List[Dict[str, Any]])
async def get_digital_twins(
    study_id: Optional[str] = Query(None, description="Filter by study ID"),
    service: ClinicalDataService = Depends(get_data_service)
):
    """
    Get all digital patient twins (real-time generated)
    
    Returns dynamically generated twins instead of static JSON files.
    Twins are created on-demand using NetworkX graph processing.
    """
    try:
        twins = await service.get_all_twins(study_id)
        return twins
    except Exception as e:
        logger.error(f"Error getting digital twins: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get digital twins: {str(e)}")


@router.post("/twins/{patient_id}/refresh")
async def refresh_patient_twin(
    patient_id: str = Path(..., description="Patient ID to refresh"),
    service: ClinicalDataService = Depends(get_data_service)
):
    """
    Refresh a specific patient's digital twin
    
    Forces regeneration of the twin and broadcasts update via WebSocket.
    """
    try:
        # Get updated twin data
        twin_data = await service.get_patient_detail(patient_id)
        if twin_data:
            await service.notify_twin_update(patient_id, twin_data)
            return {
                "success": True,
                "patient_id": patient_id,
                "refreshed_at": datetime.now().isoformat(),
                "twin_data": twin_data
            }
        else:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing twin for {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh twin: {str(e)}")


@router.get("/{patient_id}", response_model=PatientDetail)
async def get_patient_detail(patient_id: str = Path(..., description="Patient ID")):
    """Get detailed information for a specific patient"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        patient = await service.get_patient_detail(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        return PatientDetail(**patient)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patient detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{patient_id}/clean-status", response_model=CleanPatientStatus)
async def get_patient_clean_status(patient_id: str = Path(..., description="Patient ID")):
    """Get clean patient status with breakdown of blocking factors"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        status = await service.calculate_clean_status(patient_id)
        return CleanPatientStatus(**status)
    except Exception as e:
        logger.error(f"Error getting clean status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{patient_id}/timeline")
async def get_patient_timeline(patient_id: str = Path(..., description="Patient ID")):
    """Get patient event timeline"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        timeline = await service.get_patient_timeline(patient_id)
        return {
            "success": True,
            "patient_id": patient_id,
            "timeline": timeline
        }
    except Exception as e:
        logger.error(f"Error getting patient timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{patient_id}/blocking-factors")
async def get_blocking_factors(patient_id: str = Path(..., description="Patient ID")):
    """Get detailed breakdown of factors blocking patient from being clean"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        factors = await service.get_blocking_factors(patient_id)
        return {
            "success": True,
            "patient_id": patient_id,
            "blocking_factors": factors,
            "total_blockers": len(factors),
            "categories": {
                "visits": [f for f in factors if f.get("category") == "visits"],
                "queries": [f for f in factors if f.get("category") == "queries"],
                "coding": [f for f in factors if f.get("category") == "coding"],
                "saes": [f for f in factors if f.get("category") == "saes"],
                "forms": [f for f in factors if f.get("category") == "forms"]
            }
        }
    except Exception as e:
        logger.error(f"Error getting blocking factors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{patient_id}/lock-readiness")
async def get_lock_readiness(patient_id: str = Path(..., description="Patient ID")):
    """Get patient lock readiness assessment"""
    try:
        from api.services.data_service import ClinicalDataService
        service = ClinicalDataService()
        await service.initialize()
        readiness = await service.get_lock_readiness(patient_id)
        return {
            "success": True,
            "patient_id": patient_id,
            "lock_readiness": readiness
        }
    except Exception as e:
        logger.error(f"Error getting lock readiness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/twins", response_model=List[Dict[str, Any]])
async def get_digital_twins(
    study_id: Optional[str] = Query(None, description="Filter by study ID"),
    service: ClinicalDataService = Depends(get_data_service)
):
    """
    Get all digital patient twins (real-time generated)
    
    Returns dynamically generated twins instead of static JSON files.
    Twins are created on-demand using NetworkX graph processing.
    """
    try:
        twins = await service.get_all_twins(study_id)
        return twins
    except Exception as e:
        logger.error(f"Error getting digital twins: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get digital twins: {str(e)}")


@router.post("/twins/{patient_id}/refresh")
async def refresh_patient_twin(
    patient_id: str = Path(..., description="Patient ID to refresh"),
    service: ClinicalDataService = Depends(get_data_service)
):
    """
    Refresh a specific patient's digital twin
    
    Forces regeneration of the twin and broadcasts update via WebSocket.
    """
    try:
        # Get updated twin data
        twin_data = await service.get_patient_detail(patient_id)
        if twin_data:
            await service.notify_twin_update(patient_id, twin_data)
            return {
                "success": True,
                "patient_id": patient_id,
                "refreshed_at": datetime.now().isoformat(),
                "twin_data": twin_data
            }
        else:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing twin for {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh twin: {str(e)}")
