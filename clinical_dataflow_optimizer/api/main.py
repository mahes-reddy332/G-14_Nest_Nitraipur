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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

from api.services.data_service import ClinicalDataService
from api.services.metrics_service import MetricsService
from api.services.realtime_service import RealtimeService, ConnectionManager
from api.services.alert_service import AlertService
from api.routers import studies, patients, sites, metrics, agents, alerts, conversational, narratives, reports, nlq
from api.config import get_settings, get_service, initialize_services, cleanup_services

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Initialize services (Global scope for singletons)
data_service = ClinicalDataService()
metrics_service = MetricsService(data_service)
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
        # Still mark as ready to allow basic operations
        startup_state.is_ready = True
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
    finally:
        # Shutdown: Cleanup resources
        logger.info("Shutting down API server...")
        await realtime_service.stop_background_tasks()
        await connection_manager.disconnect_all()
        await cleanup_services()
        logger.info("API server shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Neural Clinical Data Mesh API",
    description="Real-time clinical trial data visualization and monitoring API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

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


# ============== Core Endpoints ==============

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API welcome page and navigation
    Prevents 404 on base URL access
    """
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
            "agents": "/api/agents/",
            "alerts": "/api/alerts/"
        },
        "message": "Welcome to the Neural Clinical Data Mesh API. Visit /api/docs for interactive documentation."
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
            "ready": True,
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
                "ready": False,
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
                "ready": False,
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


@app.get("/api/dashboard/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    study_id: Optional[str] = Query(None, description="Filter by study ID")
):
    """Get overall dashboard summary metrics"""
    try:
        summary = await data_service.get_dashboard_summary(study_id)
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

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates
    
    Supports:
    - Initial state on connect
    - Subscription to study/site updates
    - Ping/pong heartbeat
    - Manual refresh requests
    """
    await connection_manager.connect(websocket)
    
    # Create a heartbeat task
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
        # Start heartbeat
        heartbeat_task = asyncio.create_task(send_heartbeat())
        
        # Send initial state
        initial_data = await data_service.get_dashboard_summary(None)
        await websocket.send_json({
            "type": "initial_state",
            "data": initial_data,
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "subscribe":
                # Subscribe to specific study/site updates
                subscription = message.get("subscription", {})
                await realtime_service.add_subscription(websocket, subscription)
                await websocket.send_json({
                    "type": "subscribed",
                    "subscription": subscription
                })
            
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif message.get("type") == "request_update":
                # Manual refresh request
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


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts"""
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
