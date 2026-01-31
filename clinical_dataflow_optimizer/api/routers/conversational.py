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
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import logging
from datetime import datetime
import json
import pandas as pd
import pickle

from config.settings import DEFAULT_LONGCAT_CONFIG

if TYPE_CHECKING:
    from nlq.conversational_engine import ConversationalEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversational", tags=["conversational"])

# Global conversational engine instance
conversational_engines: Dict[str, 'ConversationalEngine'] = {}
data_sources: Dict[str, pd.DataFrame] = {}
graph_cache: Dict[str, Dict[str, Any]] = {}

BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR_CANDIDATES = [
    BASE_DIR / "reports",
    BASE_DIR.parent / "reports",
]
# Also check graph_data folder for knowledge graphs
GRAPH_DATA_CANDIDATES = [
    BASE_DIR / "graph_data",
    BASE_DIR.parent / "graph_data",
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


def _find_graph_data_dir() -> Optional[Path]:
    """Find the graph_data directory."""
    for candidate in GRAPH_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _load_knowledge_graph(study_id: Optional[str]) -> Dict[str, Any]:
    """Load knowledge graph JSON from graph_data or reports folder and build a NetworkX graph."""
    normalized_study_id = _normalize_study_id(study_id)
    cache_key = normalized_study_id or "__default__"
    if cache_key in graph_cache:
        return graph_cache[cache_key]

    graph_file: Optional[Path] = None
    
    # First try graph_data folder with Study_*_graph.json pattern
    graph_data_dir = _find_graph_data_dir()
    if graph_data_dir:
        if normalized_study_id:
            candidate = graph_data_dir / f"{normalized_study_id}_graph.json"
            if candidate.exists():
                graph_file = candidate
        if not graph_file:
            # Look for any study graph file
            files = sorted(graph_data_dir.glob("Study_*_graph.json"))
            graph_file = files[0] if files else None
    
    # Fallback to reports folder with *_knowledge_graph.json pattern
    if not graph_file:
        reports_dir = _find_reports_dir()
        if reports_dir:
            if normalized_study_id:
                candidate = reports_dir / f"{normalized_study_id}_knowledge_graph.json"
                if candidate.exists():
                    graph_file = candidate
            if not graph_file:
                files = sorted(reports_dir.glob("*_knowledge_graph.json"))
                graph_file = files[0] if files else None

    if not graph_file or not graph_file.exists():
        logger.warning("No knowledge graph JSON found in graph_data or reports directory.")
        graph_cache[cache_key] = {"graph": None, "summary": {}}
        return graph_cache[cache_key]

    try:
        logger.info(f"Loading knowledge graph from: {graph_file}")
        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        import networkx as nx
        graph = nx.DiGraph()

        # Handle both node formats: new format with nodes array and old format with patient_index
        nodes_data = data.get("nodes", [])
        
        # If no nodes array, create from patient_index for backward compatibility
        if not nodes_data and "patient_index" in data:
            for patient_name, node_id in data["patient_index"].items():
                graph.add_node(
                    node_id,
                    type="Patient",
                    node_type="Patient",
                    patient_name=patient_name,
                )
        else:
            for node in nodes_data:
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

        # Build summary from statistics
        stats = data.get("statistics") or data.get("stats") or {}
        summary = {
            "study_id": data.get("study_id"),
            "total_nodes": stats.get("total_nodes", graph.number_of_nodes()),
            "total_edges": stats.get("total_edges", graph.number_of_edges()),
            "node_breakdown": stats.get("nodes_by_type", {}),
            "edge_breakdown": stats.get("edges_by_type", {}),
            "density": stats.get("density", 0),
            "avg_degree": stats.get("avg_degree", 0),
        }
        
        logger.info(f"Loaded knowledge graph with {summary['total_nodes']} nodes and {summary['total_edges']} edges")

        graph_cache[cache_key] = {"graph": graph, "summary": summary}
        return graph_cache[cache_key]
    except Exception as e:
        logger.warning(f"Failed to load knowledge graph: {e}")
        graph_cache[cache_key] = {"graph": None, "summary": {}}
        return graph_cache[cache_key]

def get_conversational_engine(study_id: Optional[str] = None) -> 'ConversationalEngine':
    """Get or create a conversational engine scoped to a study graph."""
    global conversational_engines, data_sources
    
    from nlq.conversational_engine import ConversationalEngine

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


def _get_site_metrics_summary(study_id: Optional[str] = None) -> Dict[str, Any]:
    """Extract site-level performance metrics from loaded data sources."""
    global data_sources
    
    if 'CPID_EDC_Metrics' not in data_sources:
        return {}
    
    df = data_sources['CPID_EDC_Metrics']
    
    # Filter by study if specified
    if study_id:
        normalized_study_id = _normalize_study_id(study_id)
        if 'study_id' in df.columns:
            df = df[df['study_id'] == normalized_study_id]
    
    if df.empty:
        return {}
    
    # Find site column (including lowercase variants)
    site_col = None
    for col in ['site_id', 'Site ID', 'Site_ID', 'SITE_ID', 'Site']:
        if col in df.columns:
            site_col = col
            break
    
    if not site_col:
        return {"error": f"No site column found in data. Available columns: {list(df.columns)[:10]}..."}
    
    # Find metrics columns - try various naming patterns including lowercase
    open_queries_col = None
    for col in ['open_queries', 'Open_Queries_Count', 'Open Queries', 'Open_Queries', 'Total Open Queries', 'total_queries']:
        if col in df.columns:
            open_queries_col = col
            break
    
    missing_visits_col = None
    for col in ['missing_visits', 'Missing Visits', 'Missing_Visits', 'Overdue Visits', 'Overdue_Visits']:
        if col in df.columns:
            missing_visits_col = col
            break
    
    dqi_col = None
    for col in ['Data_Quality_Index', 'DQI', 'Data Quality Index', 'SSM', 'SSM_Score']:
        if col in df.columns:
            dqi_col = col
            break
    
    # Calculate site-level aggregated metrics
    try:
        site_metrics = []
        for site_id in df[site_col].unique():
            if pd.isna(site_id):
                continue
                
            site_data = df[df[site_col] == site_id]
            metrics = {
                "site_id": str(site_id),
                "patient_count": len(site_data),
            }
            
            if open_queries_col and open_queries_col in site_data.columns:
                total_queries = pd.to_numeric(site_data[open_queries_col], errors='coerce').sum()
                metrics["total_open_queries"] = int(total_queries) if pd.notna(total_queries) else 0
            
            if missing_visits_col and missing_visits_col in site_data.columns:
                total_missing = pd.to_numeric(site_data[missing_visits_col], errors='coerce').sum()
                metrics["total_missing_visits"] = int(total_missing) if pd.notna(total_missing) else 0
            
            if dqi_col and dqi_col in site_data.columns:
                avg_dqi = pd.to_numeric(site_data[dqi_col], errors='coerce').mean()
                metrics["avg_data_quality_index"] = round(avg_dqi, 2) if pd.notna(avg_dqi) else None
            
            site_metrics.append(metrics)
        
        # Sort by worst performing (highest open queries or lowest DQI)
        if site_metrics:
            # Try to sort by total_open_queries descending (worst first)
            if any("total_open_queries" in m for m in site_metrics):
                site_metrics.sort(key=lambda x: x.get("total_open_queries", 0), reverse=True)
            elif any("avg_data_quality_index" in m for m in site_metrics):
                site_metrics.sort(key=lambda x: x.get("avg_data_quality_index", 100) or 100)
        
        return {
            "total_sites": len(site_metrics),
            "top_5_worst_sites": site_metrics[:5],
            "top_5_best_sites": site_metrics[-5:][::-1] if len(site_metrics) >= 5 else [],
            "available_metrics": [k for k in site_metrics[0].keys() if k != "site_id"] if site_metrics else []
        }
    except Exception as e:
        logger.warning(f"Error calculating site metrics: {e}")
        return {"error": str(e)}


def _maybe_enhance_with_longcat(query: str, base_answer: str, graph_summary: Dict[str, Any], study_id: Optional[str] = None) -> str:
    if not DEFAULT_LONGCAT_CONFIG.api_key:
        return base_answer

    try:
        from core.longcat_integration import LongCatClient
        
        # Get site-level metrics to provide more context
        site_metrics = _get_site_metrics_summary(study_id)
        
        client = LongCatClient(DEFAULT_LONGCAT_CONFIG)
        
        # Build a richer context including site metrics
        context_parts = []
        if graph_summary:
            context_parts.append(f"Knowledge graph: {json.dumps(graph_summary)[:2000]}")
        if site_metrics and not site_metrics.get("error"):
            context_parts.append(f"Site performance metrics: {json.dumps(site_metrics)[:2000]}")
        
        context_str = "\n\n".join(context_parts) if context_parts else "No additional context available."
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical data analyst chatbot. Use the provided data context "
                    "to produce accurate, data-driven responses. When asked about site performance, "
                    "reference specific sites from the provided metrics. Be concise but specific."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Base analytical answer: {base_answer}\n\n"
                    f"Data Context:\n{context_str}"
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
        enhanced_answer = _maybe_enhance_with_longcat(
            request.query, 
            response.answer, 
            kg.get("summary", {}),
            study_id=request.study_id
        )

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
        session = engine.sessions.get(session_id)

        if not session:
            for candidate_engine in conversational_engines.values():
                if session_id in candidate_engine.sessions:
                    session = candidate_engine.sessions[session_id]
                    break

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

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