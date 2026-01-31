from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Dict, List, Optional, Any
from api.config import get_initialized_metrics_service

router = APIRouter(tags=["coding"])

@router.get("/meddra")
async def get_meddra_coding(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """Get MedDRA coding items"""
    try:
        service = await get_initialized_metrics_service()
        filters = {}
        if status: filters['status'] = status
        
        # Site ID filtering would be applied in service if needed, passing simple filters for now
        
        return await service.get_coding_list(study_id, 'meddra', filters)
    except Exception as e:
        # Log error but return empty list to prevent frontend crash
        print(f"Error fetching MedDRA coding: {e}")
        return []

@router.get("/whodrug")
async def get_whodrug_coding(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """Get WHO Drug coding items"""
    try:
        service = await get_initialized_metrics_service()
        filters = {}
        if status: filters['status'] = status
        
        return await service.get_coding_list(study_id, 'whodrug', filters)
    except Exception as e:
        print(f"Error fetching WHODrug coding: {e}")
        return []

@router.get("/metrics")
async def get_coding_metrics(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None)
):
    """Get coding metrics"""
    try:
        service = await get_initialized_metrics_service()
        return await service.get_coding_metrics(study_id, site_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
