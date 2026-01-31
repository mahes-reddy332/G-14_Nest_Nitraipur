
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional
from core.audit.audit_service import AuditService, AuditLog
from api.dependencies import require_permission, Permission, get_current_user
from api.config import get_service

router = APIRouter(
    prefix="/api/audit",
    tags=["audit"],
    dependencies=[Depends(require_permission(Permission.EXPORT_AUDIT))] # Strict Export permission
)

# Helper to get service
async def get_audit_service():
    # Similar pattern to other services
    return get_service("audit_service")

@router.get("/logs", response_model=List[AuditLog])
async def get_audit_logs(
    user_id_filter: Optional[str] = Query(None, alias="userId"),
    event_type: Optional[str] = Query(None, alias="eventType"),
    limit: int = 100,
    service: AuditService = Depends(get_audit_service)
):
    """
    Retrieve system audit logs. 
    Only accessible users with EXPORT_AUDIT permission.
    """
    return await service.get_logs(user_id=user_id_filter, event_type=event_type, limit=limit)
