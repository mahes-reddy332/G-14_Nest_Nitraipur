from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
from ..services.nlq_service import NLQService

router = APIRouter()

class NLQQuery(BaseModel):
    query: str
    context: Dict[str, Any] = {}

class NLQResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    confidence: float = 0.0
    query_type: str = "general"

# Singleton instance
_nlq_service = None

async def get_nlq_service() -> NLQService:
    """Dependency injection for NLQ service"""
    global _nlq_service
    if _nlq_service is None:
        _nlq_service = NLQService()
        await _nlq_service.initialize()
    return _nlq_service

@router.post("/query", response_model=NLQResponse)
async def process_nlq_query(
    query: NLQQuery,
    nlq_service: NLQService = Depends(get_nlq_service)
):
    """
    Process a natural language query about clinical data.
    """
    try:
        result = await nlq_service.process_query(query.query, query.context)
        return NLQResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLQ processing failed: {str(e)}")

@router.get("/health")
async def nlq_health_check(
    nlq_service: NLQService = Depends(get_nlq_service)
):
    """
    Check NLQ service health and status.
    """
    try:
        status = await nlq_service.get_status()
        return {"status": "healthy", "details": status}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}