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
        else:
            raise ValueError(f"Unknown service: {service_name}")
    
    return _services[service_name]


async def initialize_services():
    """Initialize all services during startup"""
    from api.services.alert_service import AlertService
    
    # Initialize alert service
    alert_service = get_service("alert_service")
    await alert_service.initialize()


async def cleanup_services():
    """Cleanup services during shutdown"""
    global _services
    _services.clear()
