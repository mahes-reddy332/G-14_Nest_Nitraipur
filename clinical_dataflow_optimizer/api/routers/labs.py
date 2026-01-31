
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from api.config import get_service
from api.services.lab_service import LabService

from api.dependencies import require_permission, Permission

router = APIRouter(dependencies=[Depends(require_permission(Permission.VIEW_DASHBOARD))])

# Helper to get the service
async def get_lab_service():
    from api.config import get_initialized_data_service
    # Ensure data service is initialized as LabService depends on it
    await get_initialized_data_service()
    return get_service("lab_service")

@router.get("/missing", response_model=List[Dict[str, Any]])
async def get_missing_lab_data(
    study_id: Optional[str] = Query(None, description="Filter by study ID"),
    service: LabService = Depends(get_lab_service)
):
    """
    Get list of missing laboratory data.
    """
    return await service.get_missing_lab_data(study_id)

@router.get("/summary", response_model=Dict[str, Any])
async def get_lab_summary(
    study_id: Optional[str] = Query(None, description="Filter by study ID"),
    service: LabService = Depends(get_lab_service)
):
    """
    Get summary metrics for laboratory data reconciliation.
    """
    return await service.get_reconciliation_summary(study_id)
