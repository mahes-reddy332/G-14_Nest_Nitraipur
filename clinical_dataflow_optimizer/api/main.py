"""
FastAPI Main Application
Real-time Clinical Dashboard API Server
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# =============================================================================
# ASGI PROTOCOL COMPLIANCE NOTES
# =============================================================================
# 1. NEVER use JSONResponse(content=None) - it serializes to "null" (4 bytes)
# 2. HTTP 204 MUST have empty body - use Response(status_code=204) directly
# 3. NEVER manually set Content-Length headers - let FastAPI/Starlette calculate
# 4. NEVER mutate response.body after creation
# 5. For generators, ALWAYS use StreamingResponse
# =============================================================================

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Frontend build (optional)
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

# Import services that will be used via singletons
from api.services.realtime_service import RealtimeService, ConnectionManager
from api.services.alert_service import AlertService
from api.routers import studies, patients, sites, metrics, agents, alerts, conversational, narratives, reports, nlq, labs, edc, safety, coding
from api.config import get_settings, get_service, initialize_services, cleanup_services, get_initialized_data_service
from core.security import TokenManager, RateLimiter, RateLimitRule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _coerce_number(value: Any, field: str, scope: str) -> float:
    if value is None:
        raise HTTPException(status_code=503, detail=f"Missing numeric field {field} in {scope}")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Invalid numeric field {field} in {scope}") from exc


def _require_fields(payload: Dict[str, Any], fields: List[str], scope: str) -> Dict[str, float]:
    missing = [f for f in fields if f not in payload]
    if missing:
        raise HTTPException(status_code=503, detail=f"Missing required fields in {scope}: {', '.join(missing)}")
    return {field: _coerce_number(payload[field], field, scope) for field in fields}

# Get settings
settings = get_settings()

rate_limiter = RateLimiter(
    RateLimitRule(
        requests_per_minute=settings.rate_limit_requests_per_minute,
        requests_per_hour=settings.rate_limit_requests_per_hour,
        burst_size=settings.rate_limit_burst_size,
        cooldown_seconds=settings.rate_limit_cooldown_seconds
    )
)
token_manager = TokenManager(secret_key=settings.api_token_secret)

if settings.auth_enabled and not settings.api_token_secret:
    logger.warning("AUTH is enabled but API_TOKEN_SECRET is not set. Disabling auth for this runtime.")
    settings.auth_enabled = False

# Initialize services using singletons from config
# NOTE: These are lazy-loaded - actual initialization happens during startup
data_service = get_service("data_service")  # Singleton - prevents duplicate initialization
metrics_service = get_service("metrics_service")  # Singleton - uses same data_service
realtime_service = RealtimeService()
connection_manager = ConnectionManager()

# Global readiness state for startup tracking
class StartupState:
    """Tracks application startup state for readiness probes"""
    def __init__(self):
        self.is_ready = False
        self.is_starting = True
        self.startup_error: Optional[str] = None
        self.data_loaded = False
        self.services_initialized = False
        self.startup_time: Optional[datetime] = None
        self.ready_time: Optional[datetime] = None
    
    def mark_ready(self):
        self.is_ready = True
        self.is_starting = False
        self.ready_time = datetime.now()
    
    def mark_error(self, error: str):
        self.startup_error = error
        self.is_starting = False

startup_state = StartupState()

async def _background_initialization():
    """Non-blocking background initialization of heavy services"""
    global startup_state
    try:
        logger.info("Starting background data initialization...")
        await data_service.initialize()
        startup_state.data_loaded = True
        logger.info("Data service initialized in background")
        
        # Initialize other singleton services
        await initialize_services()
        startup_state.services_initialized = True
        
        startup_state.mark_ready()
        elapsed = (startup_state.ready_time - startup_state.startup_time).total_seconds()
        logger.info(f"Application fully ready in {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"Background initialization error: {e}")
        startup_state.mark_error(str(e))
        startup_state.is_ready = False
        startup_state.is_starting = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize services with non-blocking pattern
    logger.info("Starting Neural Clinical Data Mesh API...")
    startup_state.startup_time = datetime.now()
    
    try:
        # Start real-time service immediately (lightweight)
        await realtime_service.start_background_tasks()
        
        # Kick off heavy data loading in background
        asyncio.create_task(_background_initialization())
        
        logger.info("API server started - background initialization in progress")
        yield
    except asyncio.CancelledError:
        # Handle graceful shutdown on cancellation (Ctrl+C, SIGTERM, etc.)
        logger.info("Received shutdown signal...")
    finally:
        # Shutdown: Cleanup resources
        try:
            logger.info("Shutting down API server...")
            await realtime_service.stop_background_tasks()
            await connection_manager.disconnect_all()
            await cleanup_services()
            logger.info("API server shutdown complete")
        except asyncio.CancelledError:
            # Suppress CancelledError during cleanup
            logger.info("Cleanup interrupted, forcing shutdown...")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Neural Clinical Data Mesh API",
    description="Real-time clinical trial data visualization and monitoring API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# ============== Performance Middleware ==============
# GZip compression for response payloads (reduces bandwidth by 70-90%)
from starlette.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=500)  # Compress responses > 500 bytes

# CORS configuration for React frontend
# Support multiple frontend URLs and WebSocket connections
cors_origins_env = os.getenv(
    "CORS_ALLOWED_ORIGINS", 
    "http://localhost:3000,http://localhost:5173,http://localhost:5174,http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:5174"
)
allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

# Add expose headers for better debugging and custom headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
    max_age=600,  # Cache preflight for 10 minutes
)

# ============== Security Headers Middleware ==============
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    response.headers.setdefault("X-XSS-Protection", "0")
    if path.startswith(("/api/docs", "/api/redoc")):
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self' https:; script-src 'self' https: 'unsafe-inline'; style-src 'self' https: 'unsafe-inline'; img-src 'self' https: data:; frame-ancestors 'none'"
        )
    elif not path.startswith("/api"):
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:; connect-src 'self' ws: wss:; frame-ancestors 'none'"
        )
    else:
        response.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
    if request.url.scheme == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
    return response


def _is_auth_exempt(path: str) -> bool:
    return (
        path in {"/health", "/api/health", "/api/ready"}
        or path.startswith(("/api/docs", "/api/redoc", "/api/openapi.json"))
        or path == "/api/errors/report"
    )


async def _validate_ws_auth(websocket: WebSocket) -> bool:
    if not settings.auth_enabled:
        return True

    auth_header = websocket.headers.get("authorization", "")
    token = ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1]
    else:
        token = websocket.query_params.get("token", "")

    if not token or not token_manager.validate_token(token):
        await websocket.close(code=1008)
        return False

    return True


@app.middleware("http")
async def enforce_auth_and_rate_limit(request, call_next):
    path = request.url.path

    if settings.rate_limit_enabled and path.startswith("/api") and not _is_auth_exempt(path):
        client_id = request.client.host if request.client else "unknown"
        allowed, info = rate_limiter.is_allowed(client_id)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": "Rate limit exceeded",
                        "details": info,
                        "timestamp": datetime.now().isoformat(),
                        "path": path,
                    }
                }
            )

    if settings.auth_enabled and path.startswith("/api") and not _is_auth_exempt(path):
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "AUTH_REQUIRED",
                        "message": "Authorization header missing or invalid",
                        "timestamp": datetime.now().isoformat(),
                        "path": path,
                    }
                }
            )
        token = auth_header.split(" ", 1)[1]
        payload = token_manager.validate_token(token)
        if not payload:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "INVALID_TOKEN",
                        "message": "Invalid or expired token",
                        "timestamp": datetime.now().isoformat(),
                        "path": path,
                    }
                }
            )
        request.state.user = payload

    return await call_next(request)


# ============== Pydantic Models ==============

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]


class DashboardSummary(BaseModel):
    total_studies: int
    total_patients: int
    total_sites: int
    clean_patients: int
    dirty_patients: int
    overall_dqi: float
    open_queries: int
    pending_saes: int
    uncoded_terms: int
    last_updated: str


class FilterParams(BaseModel):
    study_id: Optional[str] = None
    site_id: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    status: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    risk_level: Optional[str] = None


class AuthTokenRequest(BaseModel):
    user_id: str
    role: str = "viewer"


class ClientErrorReport(BaseModel):
    message: str
    stack: Optional[str] = None
    component_stack: Optional[str] = None
    timestamp: Optional[str] = None
    user_agent: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============== Include Routers ==============

app.include_router(studies.router, prefix="/api/studies", tags=["Studies"])
app.include_router(patients.router, prefix="/api/patients", tags=["Patients"])
app.include_router(sites.router, prefix="/api/sites", tags=["Sites"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])
app.include_router(agents.router, prefix="/api/agents", tags=["AI Agents"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
app.include_router(conversational.router, tags=["Conversational AI"])
app.include_router(narratives.router, tags=["Narratives"])
app.include_router(nlq.router, prefix="/api/nlq", tags=["Natural Language Query"])
app.include_router(labs.router, prefix="/api/labs", tags=["Laboratory Data"])
app.include_router(edc.router, prefix="/api/edc-metrics", tags=["EDC Metrics"])
app.include_router(safety.router, prefix="/api/safety", tags=["Safety"])
app.include_router(coding.router, prefix="/api/coding", tags=["Coding"])


# ============== Core Endpoints ==============

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - serves the frontend if available, otherwise returns API info.
    """
    index_file = FRONTEND_DIST / "index.html"
    if FRONTEND_DIST.exists() and index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {
        "application": "Neural Clinical Data Mesh API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "documentation": {
            "swagger": "/api/docs",
            "redoc": "/api/redoc"
        },
        "endpoints": {
            "health": "/api/health",
            "dashboard_summary": "/api/dashboard/summary",
            "studies": "/api/studies/",
            "patients": "/api/patients/",
            "sites": "/api/sites/",
            "metrics": "/api/metrics/",
            "alerts": "/api/alerts/",
            "agents": "/api/agents/",
            "reports": "/api/reports/"
        }
    }


@app.get("/health", tags=["Root"])
async def root_health():
    """
    Root-level health check (alias for /api/health)
    Common convention for load balancers and health probes
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/auth/token")
async def issue_auth_token(payload: AuthTokenRequest, request: Request):
    """Issue a JWT for authenticated access (bootstrap only)."""
    if not settings.auth_bootstrap_key:
        raise HTTPException(status_code=403, detail="Auth bootstrap key not configured")

    bootstrap_key = request.headers.get("x-bootstrap-key")
    if not bootstrap_key or bootstrap_key != settings.auth_bootstrap_key:
        raise HTTPException(status_code=403, detail="Invalid bootstrap key")

    token = token_manager.generate_token(payload.user_id, claims={"role": payload.role})
    return {
        "access_token": token,
        "token_type": "bearer",
        "issued_at": datetime.now().isoformat(),
    }


@app.post("/api/errors/report")
async def report_client_error(payload: ClientErrorReport, request: Request):
    """Receive and persist frontend error reports."""
    from core.error_handling import ClinicalDataError, get_error_tracker

    context = {
        "source": "frontend",
        "url": payload.url,
        "user_agent": payload.user_agent,
        "client_ip": request.client.host if request.client else None,
        "timestamp": payload.timestamp or datetime.now().isoformat(),
        "component_stack": payload.component_stack,
        "metadata": payload.metadata,
    }

    error = ClinicalDataError(
        payload.message,
        error_code="CLIENT000",
        details={
            "stack": payload.stack,
            "component_stack": payload.component_stack,
        }
    )

    error_id = get_error_tracker().record_error(error, context=context)

    try:
        log_dir = PROJECT_ROOT / "audit_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "client_errors.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"error_id": error_id, **context}) + "\n")
    except Exception as e:
        logger.error(f"Failed to persist client error report: {e}")

    return {"success": True, "error_id": error_id}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Favicon handler.
    Returns 204 No Content (technically correct) or 200 OK with empty body.
    User preference is to "fix" the 204 log entry, so we return a dummy 200 OK.
    """
    return Response(status_code=200)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint (liveness probe) - always returns if server is running"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        services={
            "data_service": "active" if startup_state.data_loaded else "initializing",
            "metrics_service": "active" if startup_state.services_initialized else "initializing",
            "realtime_service": "active",
            "websocket": "active"
        }
    )


@app.get("/api/ready")
async def readiness_check():
    """Readiness probe - returns 200 only when fully ready to serve traffic"""
    if startup_state.is_ready:
        elapsed = None
        if startup_state.startup_time and startup_state.ready_time:
            elapsed = (startup_state.ready_time - startup_state.startup_time).total_seconds()
        return {
            "is_ready": True,  # Frontend expects is_ready
            "ready": True,     # Keep for backwards compatibility
            "status": "ready",
            "timestamp": datetime.now().isoformat(),
            "startup_duration_seconds": elapsed,
            "data_loaded": startup_state.data_loaded,
            "services_initialized": startup_state.services_initialized
        }
    elif startup_state.is_starting:
        return JSONResponse(
            status_code=503,
            content={
                "is_ready": False,  # Frontend expects is_ready
                "ready": False,     # Keep for backwards compatibility
                "status": "starting",
                "timestamp": datetime.now().isoformat(),
                "message": "Application is starting up",
                "data_loaded": startup_state.data_loaded,
                "services_initialized": startup_state.services_initialized
            }
        )
    else:
        return JSONResponse(
            status_code=503,
            content={
                "is_ready": False,  # Frontend expects is_ready
                "ready": False,     # Keep for backwards compatibility
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": startup_state.startup_error
            }
        )


@app.get("/api/startup-status")
async def startup_status():
    """Detailed startup status for frontend polling during initialization"""
    elapsed = None
    if startup_state.startup_time:
        elapsed = (datetime.now() - startup_state.startup_time).total_seconds()
    
    return {
        "is_ready": startup_state.is_ready,
        "is_starting": startup_state.is_starting,
        "data_loaded": startup_state.data_loaded,
        "services_initialized": startup_state.services_initialized,
        "startup_error": startup_state.startup_error,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """
    Real-time dashboard updates via WebSocket
    """
    # Verify auth token if enabled
    if not await _validate_ws_auth(websocket):
        return

    await connection_manager.connect(websocket)
    try:
        # Send initial connection success message
        await websocket.send_json({
            "type": "connection_established",
            "data": {"timestamp": datetime.now().isoformat()}
        })
        
        while True:
            # Wait for messages (heartbeats, subscriptions)
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                
                # Handle heartbeat
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                
                # Handle subscriptions
                elif msg_type == "subscribe":
                    await realtime_service.add_subscription(websocket, message.get("data", {}))
                    
            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        await realtime_service.remove_subscriptions(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)


# ============== Performance-Optimized Bundled Endpoint ==============
# In-memory cache for initial dashboard data
_dashboard_cache = {}
_dashboard_cache_time = None
DASHBOARD_CACHE_TTL = 60  # seconds


@app.get("/api/dashboard/initial-load")
async def get_initial_dashboard_data(
    study_id: Optional[str] = Query(None, description="Filter by study ID")
):
    """
    Bundled endpoint for initial dashboard load - returns ALL data needed for first render.
    
    This eliminates the waterfall of sequential API calls by bundling:
    - Dashboard summary metrics
    - Query metrics (velocity, resolution stats)
    - Cleanliness metrics
    - Recent alerts
    - Critical alerts count
    
    Performance target: <500ms response time (vs 3-5s with individual calls)
    """
    import time
    global _dashboard_cache, _dashboard_cache_time
    
    start_time = time.time()
    cache_key = f"initial_load:{study_id}"
    
    # Check cache
    if (cache_key in _dashboard_cache and 
        _dashboard_cache_time and 
        (time.time() - _dashboard_cache_time) < DASHBOARD_CACHE_TTL):
        cached_data = _dashboard_cache[cache_key]
        cached_data["_cache_hit"] = True
        cached_data["_response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return cached_data
    
    try:
        from api.config import get_initialized_metrics_service, get_initialized_data_service, get_service
        
        # Get initialized services (singleton - no reinitialization!)
        data_svc = await get_initialized_data_service()
        metrics_svc = await get_initialized_metrics_service()
        
        # Fetch all data in parallel using asyncio.gather for maximum speed
        summary_task = data_svc.get_dashboard_summary(study_id)
        query_metrics_task = metrics_svc.get_query_metrics(study_id, None, 30)
        cleanliness_task = metrics_svc.get_cleanliness_metrics(study_id, None, 30)
        
        # Execute all tasks concurrently
        summary, query_metrics, cleanliness = await asyncio.gather(
            summary_task,
            query_metrics_task,
            cleanliness_task,
            return_exceptions=True
        )
        
        # Handle any exceptions in the parallel tasks
        if isinstance(summary, Exception):
            logger.error(f"Summary fetch error: {summary}")
            raise HTTPException(status_code=503, detail="Dashboard summary unavailable")
        if isinstance(query_metrics, Exception):
            logger.error(f"Query metrics fetch error: {query_metrics}")
            raise HTTPException(status_code=503, detail="Query metrics unavailable")
        if isinstance(cleanliness, Exception):
            logger.error(f"Cleanliness fetch error: {cleanliness}")
            raise HTTPException(status_code=503, detail="Cleanliness metrics unavailable")
        
        # Get alerts separately (may not have initialized service)
        alert_summary = {"active_alerts": 0, "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0}}
        try:
            alert_service = get_service("alert_service")
            if hasattr(alert_service, 'get_summary'):
                alert_summary = await alert_service.get_summary()
        except Exception as e:
            logger.warning(f"Alert service not available: {e}")
        
        # Validate required fields and coerce numeric values
        summary_fields = _require_fields(
            summary,
            [
                "total_studies",
                "total_patients",
                "total_sites",
                "clean_patients",
                "dirty_patients",
                "overall_dqi",
                "open_queries",
                "pending_saes",
                "uncoded_terms",
            ],
            "dashboard_summary",
        )
        query_fields = _require_fields(
            query_metrics,
            [
                "total_queries",
                "open_queries",
                "closed_queries",
                "resolution_rate",
                "avg_resolution_time_days",
            ],
            "query_metrics",
        )
        cleanliness_fields = _require_fields(
            cleanliness,
            [
                "overall_rate",
                "total_patients",
                "clean_patients",
                "dirty_patients",
                "pending_patients",
            ],
            "cleanliness_metrics",
        )

        # Bundle response
        response_data = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "study_filter": study_id,
            
            # Dashboard Summary
            "summary": {
                "total_studies": int(summary_fields["total_studies"]),
                "total_patients": int(summary_fields["total_patients"]),
                "total_sites": int(summary_fields["total_sites"]),
                "clean_patients": int(summary_fields["clean_patients"]),
                "dirty_patients": int(summary_fields["dirty_patients"]),
                "overall_dqi": round(summary_fields["overall_dqi"], 1),
                "open_queries": int(summary_fields["open_queries"]),
                "pending_saes": int(summary_fields["pending_saes"]),
                "uncoded_terms": int(summary_fields["uncoded_terms"]),
            },
            
            # Query Metrics
            "query_metrics": {
                "total_queries": int(query_fields["total_queries"]),
                "open_queries": int(query_fields["open_queries"]),
                "closed_queries": int(query_fields["closed_queries"]),
                "resolution_rate": round(query_fields["resolution_rate"], 1),
                "avg_resolution_time": round(query_fields["avg_resolution_time_days"], 2),
                "aging_distribution": query_metrics.get("aging_distribution", {
                    "0-7": 0, "8-14": 0, "15-30": 0, "30+": 0
                }),
                "velocity_trend": query_metrics.get("velocity_trend", []),
            },
            
            # Cleanliness Metrics
            "cleanliness": {
                "cleanliness_rate": round(cleanliness_fields["overall_rate"], 1),
                "total_patients": int(cleanliness_fields["total_patients"]),
                "clean_patients": int(cleanliness_fields["clean_patients"]),
                "dirty_patients": int(cleanliness_fields["dirty_patients"]),
                "at_risk_count": int(cleanliness_fields["pending_patients"]),
                "trend": cleanliness.get("trend", []),
            },
            
            # Alert Summary
            "alerts": {
                "active_alerts": alert_summary.get("active_alerts", 0),
                "critical_count": alert_summary.get("by_severity", {}).get("critical", 0),
                "high_count": alert_summary.get("by_severity", {}).get("high", 0),
            },
            
            "_cache_hit": False,
            "_response_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        
        # Cache the response
        _dashboard_cache[cache_key] = response_data
        _dashboard_cache_time = time.time()
        
        logger.info(
            "Initial dashboard load completed in %sms (patients=%s, dqi=%s)",
            response_data['_response_time_ms'],
            response_data['summary']['total_patients'],
            response_data['summary']['overall_dqi'],
        )
        return response_data
        
    except Exception as e:
        logger.error(f"Error in initial dashboard load: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load dashboard data: {str(e)}"
        )


@app.get("/api/dashboard/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    study_id: Optional[str] = Query(None, description="Filter by study ID")
):
    """Get overall dashboard summary metrics"""
    try:
        summary = await data_service.get_dashboard_summary(study_id)
        logger.info(
            "Dashboard summary returned (study=%s): patients=%s sites=%s open_queries=%s overall_dqi=%s",
            study_id,
            summary.get("total_patients"),
            summary.get("total_sites"),
            summary.get("open_queries"),
            summary.get("overall_dqi"),
        )
        return DashboardSummary(**summary)
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/kpis")
async def get_kpi_metrics(
    study_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None)
):
    """Get KPI tiles data for dashboard"""
    try:
        kpis = await metrics_service.get_kpi_metrics(study_id, site_id)
        return {"success": True, "data": kpis}
    except Exception as e:
        logger.error(f"Error getting KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/drill-down")
async def get_drill_down_data(
    level: str = Query(..., description="Drill-down level: study|region|country|site|subject"),
    parent_id: Optional[str] = Query(None, description="Parent entity ID for filtering")
):
    """Get hierarchical drill-down data"""
    try:
        data = await data_service.get_drill_down_data(level, parent_id)
        return {"success": True, "level": level, "data": data}
    except Exception as e:
        logger.error(f"Error getting drill-down data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/filters")
async def get_filter_options():
    """Get available filter options for dropdowns"""
    try:
        filters = await data_service.get_filter_options()
        return {"success": True, "filters": filters}
    except Exception as e:
        logger.error(f"Error getting filter options: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Real-Time WebSocket ==============

async def _handle_dashboard_websocket(websocket: WebSocket):
    """Shared handler for dashboard WebSocket endpoints"""
    if not await _validate_ws_auth(websocket):
        return
    await connection_manager.connect(websocket)

    heartbeat_task = None

    async def send_heartbeat():
        """Send periodic heartbeat to keep connection alive"""
        while True:
            try:
                await asyncio.sleep(30)  # 30 second heartbeat
                await websocket.send_json({"type": "heartbeat", "timestamp": datetime.now().isoformat()})
            except Exception:
                break

    try:
        heartbeat_task = asyncio.create_task(send_heartbeat())

        initial_data = await data_service.get_dashboard_summary(None)
        await websocket.send_json({
            "type": "initial_state",
            "data": initial_data,
            "timestamp": datetime.now().isoformat()
        })

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "subscribe":
                subscription = message.get("subscription", {})
                await realtime_service.add_subscription(websocket, subscription)
                await websocket.send_json({
                    "type": "subscribed",
                    "subscription": subscription
                })

            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

            elif message.get("type") == "request_update":
                update_data = await data_service.get_dashboard_summary(
                    message.get("study_id")
                )
                await websocket.send_json({
                    "type": "update",
                    "data": update_data,
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        if heartbeat_task:
            heartbeat_task.cancel()
        connection_manager.disconnect(websocket)
        await realtime_service.remove_subscriptions(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        if heartbeat_task:
            heartbeat_task.cancel()
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)


@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await _handle_dashboard_websocket(websocket)


@app.websocket("/api/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """Compatibility WebSocket endpoint for dashboard updates"""
    await _handle_dashboard_websocket(websocket)


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts"""
    if not await _validate_ws_auth(websocket):
        return
    await connection_manager.connect(websocket)
    try:
        while True:
            # Get pending alerts
            alerts = await realtime_service.get_pending_alerts()
            if alerts:
                await websocket.send_json({
                    "type": "alerts",
                    "data": alerts,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Check for client messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=5.0
                )
                message = json.loads(data)
                if message.get("type") == "acknowledge":
                    alert_id = message.get("alert_id")
                    await realtime_service.acknowledge_alert(alert_id)
            except asyncio.TimeoutError:
                pass
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


@app.websocket("/ws/patient/{patient_id}")
async def websocket_patient_status(websocket: WebSocket, patient_id: str):
    """WebSocket endpoint for individual patient status updates"""
    if not await _validate_ws_auth(websocket):
        return
    await connection_manager.connect(websocket)
    try:
        # Send initial patient status
        patient_data = await data_service.get_patient_detail(patient_id)
        await websocket.send_json({
            "type": "patient_status",
            "data": patient_data,
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "refresh":
                patient_data = await data_service.get_patient_detail(patient_id)
                await websocket.send_json({
                    "type": "patient_status",
                    "data": patient_data,
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


# ============== Exception Handlers ==============

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback

class APIError(Exception):
    """Custom API error with structured error information"""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

@app.exception_handler(APIError)
async def api_error_handler(request, exc: APIError):
    """Handle custom API errors"""
    logger.error(f"API Error: {exc.code} - {exc.message}", extra={
        "error_code": exc.code,
        "status_code": exc.status_code,
        "details": exc.details,
        "path": str(request.url),
        "method": request.method,
    })

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url),
            }
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with structured response"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}", extra={
        "status_code": exc.status_code,
        "detail": exc.detail,
        "path": str(request.url),
        "method": request.method,
    })

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url),
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    logger.warning(f"Validation Error: {exc.errors()}", extra={
        "errors": exc.errors(),
        "path": str(request.url),
        "method": request.method,
    })

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {
                    "validation_errors": exc.errors(),
                    "body": exc.body if hasattr(exc, 'body') else None,
                },
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url),
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions"""
    error_id = f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(exc)) % 10000}"

    logger.error(f"Unexpected Error [{error_id}]: {str(exc)}", extra={
        "error_id": error_id,
        "exception_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
        "path": str(request.url),
        "method": request.method,
    })

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "error_id": error_id,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url),
            }
        }
    )

# ============== Health Check with Service Status ==============

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with service status"""
    services_status = {}

    try:
        # Check data service
        if hasattr(data_service, 'is_initialized'):
            services_status["data_service"] = "healthy" if data_service.is_initialized else "initializing"
        else:
            services_status["data_service"] = "unknown"

        # Check metrics service
        services_status["metrics_service"] = "healthy"

        # Check realtime service
        services_status["realtime_service"] = "healthy" if realtime_service else "unavailable"

        # Check connection manager
        services_status["connection_manager"] = "healthy"

        # Check circuit breakers
        from api.utils.circuit_breaker import get_all_circuit_breakers
        circuit_breakers = get_all_circuit_breakers()
        services_status["circuit_breakers"] = f"{len(circuit_breakers)} registered"

    except Exception as e:
        logger.error(f"Health check error: {e}")
        services_status["health_check"] = "error"

    overall_status = "healthy" if all(
        status in ["healthy", "initializing"] or status.startswith(("healthy", "registered"))
        for status in services_status.values()
    ) else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        services=services_status
    )

@app.get("/api/health/circuit-breakers")
async def circuit_breaker_health():
    """Get detailed circuit breaker status"""
    from api.utils.circuit_breaker import get_all_circuit_breakers
    return {
        "circuit_breakers": get_all_circuit_breakers(),
        "timestamp": datetime.now().isoformat()
    }

# ============== Frontend Static Serving (Optional) ==============
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


@app.get("/{full_path:path}", tags=["Frontend"])
async def spa_fallback(full_path: str):
    """Serve SPA routes from the frontend build when available."""
    if full_path.startswith(("api", "ws", "assets")):
        raise HTTPException(status_code=404, detail="Not found")
    index_file = FRONTEND_DIST / "index.html"
    if FRONTEND_DIST.exists() and index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    raise HTTPException(status_code=404, detail="Frontend build not found")

# ============== Startup/Shutdown Events ==============
# Handled by lifespan context manager above

# ============== Main Entry Point ==============

if __name__ == "__main__":
    # Ensure proper handling of double-reloads by ignoring cache files
    # Using 'api.main:app' import string to ensure reloading works correctly
    uvicorn.run(
        "api.main:app",  # Start from package root
        host="127.0.0.1", # Use localhost for dev security
        port=8000,
        reload=True,
        log_level="info",
        reload_excludes=["cache/*", "*.pkl", "__pycache__/*"],
        # Ensure we don't bind if already running
    )
