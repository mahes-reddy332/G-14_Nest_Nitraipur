"""
API Configuration
Centralized settings using Pydantic BaseSettings for environment-based configuration
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "Neural Clinical Data Mesh API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Server
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # CORS
    cors_allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173,http://localhost:5174,http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:5174",
        description="Comma-separated list of allowed CORS origins"
    )
    
    # Data paths
    data_path: Optional[str] = Field(default=None, description="Path to clinical data files")
    graph_data_path: Optional[str] = Field(default=None, description="Path to graph data files")
    cache_path: Optional[str] = Field(default=None, description="Path to cache directory")
    
    # Cache settings
    cache_timeout: int = Field(default=300, description="Cache timeout in seconds")
    force_reload_data: bool = Field(default=False, description="Force reload data from source")
    
    # OpenAI/LLM settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    llm_enabled: bool = Field(default=True, description="Enable LLM features")

    # Security settings
    auth_enabled: bool = Field(default=True, description="Require auth for API requests")
    api_token_secret: Optional[str] = Field(default=None, description="Secret for API bearer tokens")
    auth_bootstrap_key: Optional[str] = Field(default=None, description="Bootstrap key for token issuance")
    rate_limit_enabled: bool = Field(default=True, description="Enable API rate limiting")
    rate_limit_requests_per_minute: int = Field(default=60, description="Requests per minute limit")
    rate_limit_requests_per_hour: int = Field(default=1000, description="Requests per hour limit")
    rate_limit_burst_size: int = Field(default=10, description="Burst size for rate limiting")
    rate_limit_cooldown_seconds: int = Field(default=60, description="Cooldown seconds after rate limit exceeded")
    
    # WebSocket settings
    ws_heartbeat_interval: int = Field(default=30, description="WebSocket heartbeat interval in seconds")
    ws_max_connections: int = Field(default=100, description="Maximum WebSocket connections")
    
    # Alert settings
    alert_check_interval: int = Field(default=60, description="Alert check interval in seconds")
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list"""
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]
    
    @property
    def llm_available(self) -> bool:
        """Check if LLM is available (API key set and enabled)"""
        return self.llm_enabled and bool(self.openai_api_key)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Map environment variables to settings
        env_prefix = ""
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)"""
    return Settings()


# Service singletons
_services = {}
_data_service_instance = None
_data_service_initialized = False
_agent_service_initialized = False
_alert_service_initialized = False


def get_service(service_name: str):
    """Get or create a singleton service instance"""
    global _services
    
    if service_name not in _services:
        if service_name == "alert_service":
            from api.services.alert_service import AlertService
            _services[service_name] = AlertService()
        elif service_name == "agent_service":
            from api.services.agent_service import AgentService
            _services[service_name] = AgentService()
        elif service_name == "nlq_service":
            from api.services.nlq_service import NLQService
            _services[service_name] = NLQService()
        elif service_name == "narrative_service":
            from api.services.narrative_service import NarrativeService
            _services[service_name] = NarrativeService()
        elif service_name == "data_service":
            from api.services.data_service import ClinicalDataService
            _services[service_name] = ClinicalDataService()
        elif service_name == "metrics_service":
            # MetricsService needs data_service
            data_svc = get_service("data_service")
            from api.services.metrics_service import MetricsService
            _services[service_name] = MetricsService(data_svc)
        elif service_name == "lab_service":
            # LabService needs data_service
            data_svc = get_service("data_service")
            from api.services.lab_service import LabService
            _services[service_name] = LabService(data_svc)
        elif service_name == "audit_service":
            from core.audit.audit_service import AuditService
            _services[service_name] = AuditService()
        elif service_name == "risk_service":
            data_svc = get_service("data_service")
            audit_svc = get_service("audit_service")
            from api.services.risk_service import RiskScoringService
            _services[service_name] = RiskScoringService(data_svc, audit_svc)
        else:
            raise ValueError(f"Unknown service: {service_name}")
    
    return _services[service_name]


async def get_initialized_data_service():
    """Get a singleton data service that's guaranteed to be initialized"""
    global _data_service_initialized
    
    data_service = get_service("data_service")
    
    if not _data_service_initialized:
        await data_service.initialize()
        _data_service_initialized = True
    
    return data_service


async def get_initialized_metrics_service():
    """Get a singleton metrics service with initialized data service"""
    await get_initialized_data_service()  # Ensure data service is initialized
    return get_service("metrics_service")


async def get_initialized_agent_service():
    """Get a singleton agent service that's guaranteed to be initialized"""
    global _agent_service_initialized
    
    agent_service = get_service("agent_service")
    
    if not _agent_service_initialized:
        await agent_service.initialize()
        _agent_service_initialized = True
    
    return agent_service


async def get_initialized_alert_service():
    """Get a singleton alert service that's guaranteed to be initialized"""
    global _alert_service_initialized
    
    alert_service = get_service("alert_service")
    
    if not _alert_service_initialized:
        await alert_service.initialize()
        _alert_service_initialized = True
    
    return alert_service


async def initialize_services():
    """Initialize all services during startup"""
    global _agent_service_initialized, _alert_service_initialized
    
    # Initialize data service first (most critical for performance)
    await get_initialized_data_service()
    
    # Initialize alert service
    await get_initialized_alert_service()
    
    # Initialize agent service
    await get_initialized_agent_service()


async def cleanup_services():
    """Cleanup services during shutdown"""
    global _services, _data_service_initialized, _agent_service_initialized, _alert_service_initialized
    _services.clear()
    _data_service_initialized = False
    _agent_service_initialized = False
    _alert_service_initialized = False
