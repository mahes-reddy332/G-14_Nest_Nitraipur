# Clinical Dataflow Optimizer - Production Documentation

## Overview

The Clinical Dataflow Optimizer is a comprehensive system for managing and analyzing clinical trial data flows. This document covers the production-ready implementation including AI capabilities, error handling, monitoring, and security features.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Error        │  │ Dashboard    │  │ Alert        │          │
│  │ Boundaries   │  │ Components   │  │ Display      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ API Routers  │  │ Rate Limiter │  │ Auth/Authz   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Services                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Agent        │  │ Digital Twin │  │ RAG/NLQ      │          │
│  │ Framework    │  │ Processor    │  │ Engine       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Narrative    │  │ Monitoring   │  │ Security     │          │
│  │ Generator    │  │ System       │  │ Module       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Module Documentation

### 1. Agent Framework (`agents/`)

#### LLM Integration (`llm_integration.py`)

Multi-provider LLM support with intelligent fallbacks:

```python
from agents.llm_integration import LLMClientFactory, LLMConfig

# Configure LLM client
config = LLMConfig(
    provider="openai",
    model_name="gpt-4",
    api_key="your-api-key",
    temperature=0.7
)

# Create client
client = LLMClientFactory.create(config)

# Generate response
response = await client.generate("Analyze this clinical data...")
```

**Supported Providers:**
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Mock (for testing)

**Features:**
- Response caching with TTL
- Circuit breaker for fault tolerance
- Automatic retry with exponential backoff

#### Inter-Agent Communication (`inter_agent_comm.py`)

Event-driven messaging system for agent coordination:

```python
from agents.inter_agent_comm import MessageBus, EventType, AgentCoordinator

# Initialize message bus
bus = MessageBus()

# Subscribe to events
bus.subscribe(EventType.ANOMALY_DETECTED, handle_anomaly)

# Publish event
bus.publish(EventType.ANOMALY_DETECTED, {
    'patient_id': 'PAT001',
    'anomaly_type': 'data_gap'
})
```

### 2. Digital Twin Processing (`core/realtime_twin_processor.py`)

Real-time patient digital twin with WebSocket support:

```python
from core.realtime_twin_processor import RealTimeTwinProcessor

processor = RealTimeTwinProcessor()

# Process patient update
result = await processor.process_patient_update(
    patient_id="PAT001",
    study_id="STUDY001",
    update_data={'visit_completed': True}
)
```

**Features:**
- Graph-based state evolution using NetworkX
- Intelligent caching with TTL
- WebSocket broadcasting for real-time updates

### 3. Enhanced NLQ Processing (`nlq/enhanced_nlq_processor.py`)

LLM-powered natural language query processing:

```python
from nlq.enhanced_nlq_processor import EnhancedQueryParser, EnhancedRAGQueryExecutor

parser = EnhancedQueryParser()
executor = EnhancedRAGQueryExecutor()

# Parse query
intent = await parser.parse_query("Show me all patients with pending visits at Site 001")

# Execute with RAG
results = await executor.execute(intent.to_dict(), context={'study_id': 'STUDY001'})
```

### 4. Generative Narrative Engine (`narratives/generative_narrative_engine.py`)

AI-powered clinical narrative generation:

```python
from narratives.generative_narrative_engine import GenerativeNarrativeEngine, NarrativeType

engine = GenerativeNarrativeEngine()

# Generate study narrative
narrative = await engine.generate_narrative(
    narrative_type=NarrativeType.STUDY_SUMMARY,
    context={'study_id': 'STUDY001'},
    include_recommendations=True
)
```

### 5. Error Handling (`core/error_handling.py`)

Comprehensive error handling system:

```python
from core.error_handling import (
    ClinicalDataError,
    CircuitBreaker,
    retry_with_backoff,
    with_fallback,
    api_error_handler
)

# Custom exception
raise DataValidationError("Invalid date format", field="visit_date", value="2024-13-01")

# Circuit breaker
cb = CircuitBreaker("llm_service")
if cb.can_execute():
    try:
        result = call_llm()
        cb.record_success()
    except Exception:
        cb.record_failure()
        raise

# Retry decorator
@retry_with_backoff(RetryConfig(max_retries=3))
def unreliable_operation():
    ...

# Fallback decorator
@with_fallback(fallback_value="default")
def operation_with_fallback():
    ...
```

### 6. Monitoring System (`core/monitoring.py`)

Metrics collection and alerting:

```python
from core.monitoring import (
    MetricsCollector,
    AlertManager,
    AlertRule,
    AlertSeverity
)

# Record metrics
collector = MetricsCollector()
collector.increment("api.requests.total")
collector.set_gauge("system.memory.percent", 75.5)

# Timer context manager
with collector.timer("api.response_time"):
    process_request()

# Configure alerts
manager = AlertManager(collector)
manager.add_rule(AlertRule(
    rule_id="high_cpu",
    name="High CPU Usage",
    description="CPU above 80%",
    metric_name="system.cpu.percent",
    condition="gt",
    threshold=80,
    severity=AlertSeverity.WARNING
))
```

### 7. Security Module (`core/security.py`)

HIPAA/GxP compliant security features:

```python
from core.security import (
    InputValidator,
    RateLimiter,
    AuditLogger,
    TokenManager,
    PermissionChecker
)

# Input validation
validated = InputValidator.validate_clinical_id(user_input, "patient_id")

# Rate limiting
limiter = RateLimiter(RateLimitRule(requests_per_minute=60))
allowed, info = limiter.is_allowed(client_id)

# Audit logging
audit = AuditLogger()
audit.log(
    event_type=AuditEventType.ACCESS,
    user_id="user123",
    action="view",
    resource_type="patient_data",
    resource_id="PAT001"
)

# Token management
token_mgr = TokenManager()
token = token_mgr.generate_token("user123", claims={'role': 'analyst'})
payload = token_mgr.validate_token(token)
```

## API Endpoints

### Health & Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system/health` | GET | Full health check |
| `/api/system/health/liveness` | GET | Kubernetes liveness probe |
| `/api/system/health/readiness` | GET | Kubernetes readiness probe |
| `/api/system/metrics` | GET | All system metrics |
| `/api/system/metrics/dashboard` | GET | Dashboard-formatted metrics |
| `/api/system/performance` | GET | Performance metrics |

### Alerts

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system/alerts` | GET | Active alerts |
| `/api/system/alerts/summary` | GET | Alert summary |
| `/api/system/alerts/history` | GET | Alert history |
| `/api/system/alerts/rules` | POST | Create alert rule |
| `/api/system/alerts/{rule_id}/acknowledge` | POST | Acknowledge alert |

### Error Tracking

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system/errors` | GET | Recent errors |
| `/api/system/errors/summary` | GET | Error summary |
| `/api/system/errors/{error_id}/resolve` | POST | Resolve error |

### Audit

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system/audit` | GET | Query audit log |
| `/api/system/audit/integrity` | GET | Verify audit integrity |

## Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
DEFAULT_LLM_PROVIDER=openai

# Security
JWT_SECRET_KEY=your-secret-key
TOKEN_EXPIRY_HOURS=24

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_REQUESTS_PER_HOUR=1000

# Monitoring
METRICS_RETENTION_SECONDS=3600
ALERT_CHECK_INTERVAL_SECONDS=60

# Logging
LOG_LEVEL=INFO
AUDIT_LOG_PATH=./audit_logs
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_error_handling.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Test Coverage Targets

| Module | Target Coverage |
|--------|----------------|
| core/error_handling.py | 90% |
| core/monitoring.py | 85% |
| core/security.py | 90% |
| agents/* | 80% |
| nlq/* | 85% |

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Health Checks

```yaml
livenessProbe:
  httpGet:
    path: /api/system/health/liveness
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /api/system/health/readiness
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

## Compliance

### HIPAA Compliance

- All patient data access is logged via audit system
- Data encryption utilities for PII
- Role-based access control (RBAC)
- Audit log integrity verification with hash chains

### GxP Compliance

- Immutable audit trails
- 21 CFR Part 11 compliant logging
- Electronic signature support (via tokens)
- Complete traceability of data modifications

## Performance Considerations

1. **Caching**: LLM responses and digital twin states are cached
2. **Circuit Breakers**: Prevent cascade failures from external services
3. **Rate Limiting**: Protect against DoS and ensure fair usage
4. **Async Processing**: All I/O operations are async
5. **Connection Pooling**: Database and HTTP connections are pooled

## Troubleshooting

### Common Issues

1. **LLM Rate Limits**: Check circuit breaker status at `/api/system/circuit-breakers`
2. **High Memory Usage**: Review metrics at `/api/system/performance`
3. **Alert Storm**: Adjust alert thresholds via `/api/system/alerts/rules`

### Logging

All components use Python's standard logging:

```python
import logging
logger = logging.getLogger(__name__)

# Set log level
logging.basicConfig(level=logging.INFO)
```

## Support

For issues and feature requests, please refer to the project's issue tracker.
