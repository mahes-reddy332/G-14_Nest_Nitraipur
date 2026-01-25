# Neural Clinical Data Mesh - Core Processing
"""Package initialization for core module"""

from .data_ingestion import ClinicalDataIngester
from .metrics_calculator import CleanPatientCalculator, DataQualityIndexCalculator
from .feature_engineering import (
    SiteFeatureEngineer,
    OperationalVelocityIndex,
    NormalizedDataDensity,
    ManipulationRiskScore,
    ManipulationRiskLevel,
    VelocityTrend,
    FeatureEngineeringConfig,
    engineer_study_features,
    DEFAULT_FEATURE_CONFIG
)
from .data_integration import (
    ClinicalDataMesh,
    build_clinical_data_mesh,
    MultiHopQueryResult,
    GraphStatistics
)
from .digital_twin import (
    DigitalTwinFactory,
    DigitalTwinConfig,
    BlockingItemType,
    BlockingSeverity,
    create_digital_twins
)
from .quality_cockpit import (
    CleanPatientStatusCalculator,
    CleanPatientStatus,
    CleanConditionResult,
    CleanCondition,
    QualityCockpitConfig,
    QualityCockpitVisualizer,
    BlockerSeverity,
    calculate_clean_patient_status,
    DEFAULT_COCKPIT_CONFIG
)
from .data_quality_index import (
    DataQualityIndexCalculator as DQICalculator,
    DQIConfig,
    DQIResult,
    DQILevel,
    MetricCategory,
    MetricPenalty,
    SiteRiskProfile,
    RiskQuadrant,
    QueryFlowData,
    QueryFlowStage,
    PatientTimelineEvent,
    DQIVisualizationEngine,
    calculate_site_dqi,
    calculate_study_dqi,
    generate_dqi_dashboard,
    DEFAULT_DQI_CONFIG
)

# Error Handling
from .error_handling import (
    ClinicalDataError,
    DataIngestionError,
    DataValidationError,
    GraphProcessingError,
    AgentExecutionError,
    LLMServiceError,
    APIError,
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryConfig,
    retry_with_backoff,
    with_fallback,
    FallbackResult,
    GracefulDegradationManager,
    ErrorTracker,
    get_error_tracker,
    HealthChecker,
    ServiceHealth,
    api_error_handler
)

# Monitoring
from .monitoring import (
    MetricsCollector,
    MetricType,
    AlertManager,
    AlertRule,
    AlertSeverity,
    PerformanceMonitor,
    RequestTracker,
    get_metrics_collector,
    get_alert_manager,
    get_performance_monitor,
    get_request_tracker,
    setup_default_alert_rules
)

# Real-time Twin Processing
from .realtime_twin_processor import (
    RealTimeTwinProcessor,
    TwinCache
)

__all__ = [
    'ClinicalDataIngester',
    'CleanPatientCalculator',
    'DataQualityIndexCalculator',
    'SiteFeatureEngineer',
    'OperationalVelocityIndex',
    'NormalizedDataDensity', 
    'ManipulationRiskScore',
    'ManipulationRiskLevel',
    'VelocityTrend',
    'FeatureEngineeringConfig',
    'engineer_study_features',
    'DEFAULT_FEATURE_CONFIG',
    'ClinicalDataMesh',
    'build_clinical_data_mesh',
    'MultiHopQueryResult',
    'GraphStatistics',
    'DigitalTwinFactory',
    'DigitalTwinConfig',
    'BlockingItemType',
    'BlockingSeverity',
    'create_digital_twins',
    # Quality Cockpit
    'CleanPatientStatusCalculator',
    'CleanPatientStatus',
    'CleanConditionResult',
    'CleanCondition',
    'QualityCockpitConfig',
    'QualityCockpitVisualizer',
    'BlockerSeverity',
    'calculate_clean_patient_status',
    'DEFAULT_COCKPIT_CONFIG',
    # Data Quality Index
    'DQICalculator',
    'DQIConfig',
    'DQIResult',
    'DQILevel',
    'MetricCategory',
    'MetricPenalty',
    'SiteRiskProfile',
    'RiskQuadrant',
    'QueryFlowData',
    'QueryFlowStage',
    'PatientTimelineEvent',
    'DQIVisualizationEngine',
    'calculate_site_dqi',
    'calculate_study_dqi',
    'generate_dqi_dashboard',
    'DEFAULT_DQI_CONFIG',
    # Error Handling
    'ClinicalDataError',
    'DataIngestionError',
    'DataValidationError',
    'GraphProcessingError',
    'AgentExecutionError',
    'LLMServiceError',
    'APIError',
    'CircuitBreaker',
    'RetryConfig',
    'retry_with_backoff',
    'with_fallback',
    'FallbackResult',
    'GracefulDegradationManager',
    'ErrorTracker',
    'get_error_tracker',
    'HealthChecker',
    'ServiceHealth',
    'api_error_handler',
    # Monitoring
    'MetricsCollector',
    'MetricType',
    'AlertManager',
    'AlertRule',
    'AlertSeverity',
    'PerformanceMonitor',
    'RequestTracker',
    'get_metrics_collector',
    'get_alert_manager',
    'get_performance_monitor',
    'get_request_tracker',
    'setup_default_alert_rules',
    # Real-time Twin
    'RealTimeTwinProcessor',
    'TwinCache'
]
