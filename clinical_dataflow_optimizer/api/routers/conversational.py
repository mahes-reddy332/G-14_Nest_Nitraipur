"""
Conversational API Router
Provides REST endpoints for the Generative AI Conversational Insight Engine
"""

import sys
from pathlib import Path
# Add parent directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json
import pandas as pd
import pickle

from nlq.conversational_engine import ConversationalEngine
from core.longcat_integration import LongCatClient
from config.settings import DEFAULT_LONGCAT_CONFIG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversational", tags=["conversational"])

# Global conversational engine instance
conversational_engines: Dict[str, ConversationalEngine] = {}
data_sources: Dict[str, pd.DataFrame] = {}
graph_cache: Dict[str, Dict[str, Any]] = {}

BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR_CANDIDATES = [
    BASE_DIR / "reports",
    BASE_DIR.parent / "reports",
]

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    study_id: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    understanding: str
    answer: str
    insights: List[Dict[str, Any]]
    visualizations: List[Dict[str, Any]]
    follow_up_questions: List[str]
    data_summary: Dict[str, Any]
    confidence: float
    processing_time_ms: float
    session_id: str
    timestamp: datetime

class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    turns: int
    active_filters: Dict[str, Any]


class SessionStartRequest(BaseModel):
    study_id: Optional[str] = None


def _normalize_study_id(study_id: Optional[str]) -> Optional[str]:
    if not study_id:
        return None
    normalized = study_id.replace(" ", "_")
    if not normalized.lower().startswith("study_"):
        normalized = f"Study_{normalized}"
    return normalized


def _find_reports_dir() -> Optional[Path]:
    for candidate in REPORTS_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _load_knowledge_graph(study_id: Optional[str]) -> Dict[str, Any]:
    """Load knowledge graph JSON from reports folder and build a NetworkX graph."""
    normalized_study_id = _normalize_study_id(study_id)
    cache_key = normalized_study_id or "__default__"
    if cache_key in graph_cache:
        return graph_cache[cache_key]

    reports_dir = _find_reports_dir()
    if not reports_dir:
        logger.warning("Reports directory not found; knowledge graph unavailable.")
        graph_cache[cache_key] = {"graph": None, "summary": {}}
        return graph_cache[cache_key]

    graph_file: Optional[Path] = None
    if normalized_study_id:
        candidate = reports_dir / f"{normalized_study_id}_knowledge_graph.json"
        if candidate.exists():
            graph_file = candidate
    if not graph_file:
        files = sorted(reports_dir.glob("*_knowledge_graph.json"))
        graph_file = files[0] if files else None

    if not graph_file or not graph_file.exists():
        logger.warning("No knowledge graph JSON found in reports directory.")
        graph_cache[cache_key] = {"graph": None, "summary": {}}
        return graph_cache[cache_key]

    try:
        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        import networkx as nx
        graph = nx.DiGraph()

        for node in data.get("nodes", []):
            node_id = node.get("node_id")
            if not node_id:
                continue
            attributes = node.get("attributes", {})
            node_type = node.get("node_type") or attributes.get("node_type")
            graph.add_node(
                node_id,
                type=node_type,
                node_type=node_type,
                **attributes,
            )

        for edge in data.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if not source or not target:
                continue
            edge_type = edge.get("edge_type")
            edge_attrs = {k: v for k, v in edge.items() if k not in {"source", "target"}}
            edge_attrs["type"] = edge_type
            graph.add_edge(source, target, **edge_attrs)

        summary = data.get("statistics") or data.get("stats") or {}
        summary = {
            "study_id": data.get("study_id"),
            "total_nodes": summary.get("total_nodes", graph.number_of_nodes()),
            "total_edges": summary.get("total_edges", graph.number_of_edges()),
            "node_breakdown": summary.get("node_breakdown", {}),
            "edge_breakdown": summary.get("edge_breakdown", {}),
        }

        graph_cache[cache_key] = {"graph": graph, "summary": summary}
        return graph_cache[cache_key]
    except Exception as e:
        logger.warning(f"Failed to load knowledge graph: {e}")
        graph_cache[cache_key] = {"graph": None, "summary": {}}
        return graph_cache[cache_key]

def get_conversational_engine(study_id: Optional[str] = None) -> ConversationalEngine:
    """Get or create a conversational engine scoped to a study graph."""
    global conversational_engines, data_sources

    engine_key = _normalize_study_id(study_id) or "default"
    if engine_key in conversational_engines:
        return conversational_engines[engine_key]

    # Load data sources from cache if not already loaded
    if not data_sources:
        try:
            cache_dir = BASE_DIR / "cache"
            cache_file = cache_dir / "data_cache.pkl"
            
            if cache_file.exists():
                logger.info("Loading data sources from cache for conversational engine...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Extract CPID metrics from all studies and combine them
                all_cpid_data = []
                for study_key, study_data in cached_data.get('studies', {}).items():
                    if isinstance(study_data, dict):
                        cpid_df = study_data.get('cpid_metrics')
                        if isinstance(cpid_df, pd.DataFrame) and not cpid_df.empty:
                            cpid_df = cpid_df.copy()
                            cpid_df['study_id'] = study_key
                            all_cpid_data.append(cpid_df)
                
                if all_cpid_data:
                    combined_cpid = pd.concat(all_cpid_data, ignore_index=True)
                    data_sources['CPID_EDC_Metrics'] = combined_cpid
                    logger.info(f"Loaded CPID_EDC_Metrics with {len(combined_cpid)} rows from cache")
                    
                # Also extract SAE dashboard data
                all_sae_data = []
                for study_key, study_data in cached_data.get('studies', {}).items():
                    if isinstance(study_data, dict):
                        sae_df = study_data.get('sae_dashboard')
                        if isinstance(sae_df, pd.DataFrame) and not sae_df.empty:
                            sae_df = sae_df.copy()
                            sae_df['study_id'] = study_key
                            all_sae_data.append(sae_df)
                
                if all_sae_data:
                    combined_sae = pd.concat(all_sae_data, ignore_index=True)
                    data_sources['eSAE_Dashboard'] = combined_sae
                    logger.info(f"Loaded eSAE_Dashboard with {len(combined_sae)} rows from cache")
                    
                logger.info(f"Loaded data sources from cache: {list(data_sources.keys())}")
            else:
                logger.warning("Cache file not found, conversational engine will use fallback responses")
        except Exception as e:
            logger.warning(f"Could not load data sources from cache: {e}")

    kg = _load_knowledge_graph(study_id)
    graph = kg.get("graph")

    engine = ConversationalEngine(data_sources=data_sources, graph=graph)
    engine.load_data(data_sources, graph=graph)
    conversational_engines[engine_key] = engine
    return engine


def _maybe_enhance_with_longcat(query: str, base_answer: str, graph_summary: Dict[str, Any]) -> str:
    if not DEFAULT_LONGCAT_CONFIG.api_key:
        return base_answer

    try:
        client = LongCatClient(DEFAULT_LONGCAT_CONFIG)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical data chatbot. Use the provided knowledge graph summary "
                    "and base analytical answer to produce a concise, grounded response."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Base answer: {base_answer}\n\n"
                    f"Knowledge graph summary: {json.dumps(graph_summary)[:4000]}"
                ),
            },
        ]
        result = client.chat_completion(messages)
        choices = result.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content") or choices[0].get("text")
            if content:
                return content.strip()
    except Exception as e:
        logger.warning(f"LongCat enhancement failed: {e}")

    return base_answer

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a natural language query and return conversational response
    """
    try:
        engine = get_conversational_engine(request.study_id)
        kg = _load_knowledge_graph(request.study_id)

        # Process the query
        response = engine.ask(request.query, request.session_id)
        enhanced_answer = _maybe_enhance_with_longcat(request.query, response.answer, kg.get("summary", {}))

        # Convert to response model
        return QueryResponse(
            query=response.query,
            understanding=response.understanding,
            answer=enhanced_answer,
            insights=[insight.to_dict() for insight in response.insights],
            visualizations=response.visualizations,
            follow_up_questions=response.follow_up_questions,
            data_summary=response.data_summary,
            confidence=response.confidence,
            processing_time_ms=response.processing_time_ms,
            session_id=engine.current_session.session_id if engine.current_session else "default",
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.post("/session/start")
async def start_session(request: SessionStartRequest | None = None):
    """Start a new conversation session"""
    try:
        study_id = request.study_id if request else None
        engine = get_conversational_engine(study_id)
        session_id = engine.start_session()
        return {"session_id": session_id, "status": "started"}
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get information about a conversation session"""
    try:
        engine = get_conversational_engine()
        if session_id not in engine.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = engine.sessions[session_id]
        return SessionInfo(
            session_id=session.session_id,
            created_at=session.created_at,
            turns=len(session.history),
            active_filters=session.active_filters
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=f"Session info retrieval failed: {str(e)}")

@router.get("/capabilities")
async def get_capabilities():
    """Get information about conversational engine capabilities"""
    return {
        "supported_queries": [
            "Show me sites with high missing visits",
            "What are the top 5 sites by open queries?",
            "Find correlations between metrics",
            "Show me patients with SAE issues",
            "Analyze trends in data quality",
            "Find sites with protocol deviations"
        ],
        "supported_entities": [
            "sites", "patients", "subjects", "countries", "visits", "studies"
        ],
        "supported_metrics": [
            "missing_visits", "open_queries", "data_quality_index",
            "sae_count", "uncoded_terms", "frozen_crfs", "locked_crfs"
        ],
        "features": [
            "Natural language understanding",
            "Multi-turn conversations",
            "RAG-powered insights",
            "Cross-dataset analysis",
            "Trend analysis",
            "Correlation detection"
        ]
    }

@router.get("/health")
async def health_check():
    """Health check for conversational service"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "data_sources_loaded": len(data_sources),
        "engine_initialized": len(conversational_engines) > 0
    }