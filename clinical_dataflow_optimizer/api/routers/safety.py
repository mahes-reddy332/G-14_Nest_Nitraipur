from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Dict, List, Optional, Any
from api.config import get_initialized_metrics_service

from api.dependencies import require_permission, Permission

router = APIRouter(tags=["safety"], dependencies=[Depends(require_permission(Permission.VIEW_DASHBOARD))])

@router.get("/saes")
async def get_sae_list(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None)
):
    """Get detailed list of SAEs"""
    try:
        service = await get_initialized_metrics_service()
        filters = {}
        if site_id: filters['site_id'] = site_id
        if status: filters['status'] = status
        if severity: filters['severity'] = severity
        
        return await service.get_sae_list(study_id, filters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_sae_summary(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None)
):
    """Get SAE summary metrics"""
    try:
        service = await get_initialized_metrics_service()
        return await service.get_sae_metrics(study_id, site_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
