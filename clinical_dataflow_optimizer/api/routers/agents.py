"""
AI Agents API Router
Endpoints for AI agent insights and recommendations
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Enums ==============

class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    RECONCILIATION = "reconciliation"
    CODING = "coding"
    SITE_LIAISON = "site_liaison"
    QUALITY = "quality"


class InsightCategory(str, Enum):
    RISK = "risk"
    QUALITY = "quality"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    RECOMMENDATION = "recommendation"


class InsightSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ============== Pydantic Models ==============

class AgentInsight(BaseModel):
    insight_id: str
    agent: str
    title: str
    description: str
    category: str = "general"
    priority: str = "medium"
    confidence: float = Field(default=0.8, ge=0, le=1)
    affected_entities: Dict[str, Any] = Field(default_factory=dict)
    recommended_action: str = ""
    generated_at: str
    expires_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    nlq_context: Optional[str] = None


class AgentRecommendation(BaseModel):
    recommendation_id: str
    title: str
    description: str
    category: str = "general"
    impact: str = "medium"
    effort: str = "medium"
    priority_score: int = Field(default=3, ge=1, le=5)
    source_agent: str
    related_insights: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    estimated_completion_time: str = "N/A"
    generated_at: str
    status: str = "pending"


class AgentStatus(BaseModel):
    name: str
    status: str = "unknown"
    last_activity: str
    tasks_completed: int
    tasks_pending: int
    capabilities: List[str]


class ExplainabilityDetail(BaseModel):
    factor: str
    contribution: float
    description: str
    data_source: str
    evidence: List[str]


# ============== Endpoints ==============

@router.get("/status", response_model=Dict[str, AgentStatus])
async def get_agent_status():
    """Get status of all AI agents"""
    try:
        from api.services.agent_service import AgentService
        service = AgentService()
        await service.initialize()
        statuses = await service.get_agent_status()
        return {k: AgentStatus(**v) for k, v in statuses.items()}
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights", response_model=List[AgentInsight])
async def get_insights(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    patient_id: Optional[str] = Query(None),
    agent_type: Optional[AgentType] = Query(None),
    category: Optional[InsightCategory] = Query(None),
    severity: Optional[InsightSeverity] = Query(None),
    limit: int = Query(50, le=200)
):
    """Get AI-generated insights"""
    try:
        from api.services.agent_service import AgentService
        service = AgentService()
        
        await service.initialize()
        insights = await service.get_agent_insights(
            agent_type=agent_type.value if agent_type else None,
            priority=severity.value if severity else None,
            limit=limit
        )

        if category:
            insights = [i for i in insights if i.get('category') == category.value]

        return [AgentInsight(**i) for i in insights]
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations", response_model=List[AgentRecommendation])
async def get_recommendations(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    agent_type: Optional[AgentType] = Query(None),
    min_priority: int = Query(1, ge=1, le=5),
    limit: int = Query(20, le=100)
):
    """Get AI-generated recommendations"""
    try:
        from api.services.agent_service import AgentService
        service = AgentService()
        
        await service.initialize()
        recommendations = await service.get_recommendations(
            category=None,
            study_id=study_id,
            limit=limit
        )
        return [AgentRecommendation(**r) for r in recommendations]
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/{insight_id}/explain", response_model=List[ExplainabilityDetail])
async def explain_insight(insight_id: str):
    """Get detailed explainability for an insight"""
    try:
        from api.services.agent_service import AgentService
        service = AgentService()
        await service.initialize()
        explanation = await service.get_explainability(insight_id)
        return [ExplainabilityDetail(**e) for e in explanation] if explanation else []
    except Exception as e:
        logger.error(f"Error getting explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reconciliation/discrepancies")
async def get_sae_discrepancies(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None)
):
    """Get SAE reconciliation discrepancies identified by agent"""
    try:
        from api.services.agent_service import AgentService
        service = AgentService()
        discrepancies = await service.get_reconciliation_discrepancies(study_id, site_id)
        return {
            "success": True,
            "discrepancies": discrepancies,
            "total": len(discrepancies)
        }
    except Exception as e:
        logger.error(f"Error getting discrepancies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/coding/issues")
async def get_coding_issues(
    study_id: Optional[str] = Query(None),
    coding_type: str = Query("all", description="meddra|whodrug|all")
):
    """Get coding issues identified by agent"""
    try:
        from api.services.agent_service import AgentService
        service = AgentService()
        issues = await service.get_coding_issues(study_id, coding_type)
        return {
            "success": True,
            "issues": issues,
            "total": len(issues)
        }
    except Exception as e:
        logger.error(f"Error getting coding issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/site-liaison/flags")
async def get_site_flags(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None)
):
    """Get site flags and issues from liaison agent"""
    try:
        from api.services.agent_service import AgentService
        service = AgentService()
        flags = await service.get_site_flags(study_id, site_id)
        return {
            "success": True,
            "flags": flags,
            "total": len(flags)
        }
    except Exception as e:
        logger.error(f"Error getting site flags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality/assessment")
async def get_quality_assessment(
    study_id: Optional[str] = Query(None),
    level: str = Query("study", description="Assessment level: study|site|patient")
):
    """Get quality assessment from supervisor agent"""
    try:
        from api.services.agent_service import AgentService
        service = AgentService()
        assessment = await service.get_quality_assessment(study_id, level)
        return {
            "success": True,
            "level": level,
            "assessment": assessment
        }
    except Exception as e:
        logger.error(f"Error getting quality assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger/{agent_type}")
async def trigger_agent_analysis(
    agent_type: AgentType,
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None)
):
    """Manually trigger an agent analysis"""
    try:
        from api.services.agent_service import AgentService
        service = AgentService()
        result = await service.trigger_analysis(agent_type.value, study_id, site_id)
        return {
            "success": True,
            "agent_type": agent_type.value,
            "task_id": result.get("task_id"),
            "status": "triggered",
            "message": "Analysis started"
        }
    except Exception as e:
        logger.error(f"Error triggering agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cross-study-patterns")
async def get_cross_study_patterns():
    """Get patterns identified across studies by supervisor"""
    try:
        from api.services.agent_service import AgentService
        service = AgentService()
        patterns = await service.get_cross_study_patterns()
        return {
            "success": True,
            "patterns": patterns,
            "total": len(patterns)
        }
    except Exception as e:
        logger.error(f"Error getting cross-study patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/protocol-deviations")
async def get_protocol_deviations(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None)
):
    """Get potential protocol deviations inferred by agents"""
    try:
        from api.services.agent_service import AgentService
        service = AgentService()
        deviations = await service.get_protocol_deviations(study_id, site_id)
        return {
            "success": True,
            "deviations": deviations,
            "total": len(deviations)
        }
    except Exception as e:
        logger.error(f"Error getting protocol deviations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
