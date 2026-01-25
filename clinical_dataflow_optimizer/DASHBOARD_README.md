# Clinical Dashboard - React + FastAPI

A real-time clinical data management dashboard built with React TypeScript frontend and FastAPI backend.

## Architecture

```
clinical_dataflow_optimizer/
├── api/                          # FastAPI Backend
│   ├── main.py                   # Application entry point
│   ├── routers/                  # API route handlers
│   │   ├── studies.py            # Study endpoints
│   │   ├── patients.py           # Patient endpoints  
│   │   ├── sites.py              # Site endpoints
│   │   ├── metrics.py            # Metrics endpoints
│   │   ├── agents.py             # AI Agent endpoints
│   │   └── alerts.py             # Alert endpoints
│   └── services/                 # Business logic services
│       ├── data_service.py       # Clinical data integration
│       ├── metrics_service.py    # Metrics calculations
│       ├── realtime_service.py   # WebSocket management
│       ├── agent_service.py      # AI agent interface
│       └── alert_service.py      # Alert management
│
└── frontend/                     # React TypeScript Frontend
    ├── src/
    │   ├── api/                  # API client
    │   ├── components/           # Reusable components
    │   │   ├── Layout/           # Sidebar, Header
    │   │   └── Dashboard/        # KPI tiles, charts, etc.
    │   ├── pages/                # Page components
    │   ├── hooks/                # Custom hooks (WebSocket)
    │   ├── store/                # Zustand state management
    │   └── types/                # TypeScript type definitions
    └── package.json
```

## Features

### Dashboard
- **KPI Tiles**: Real-time metrics with trend indicators
- **Cleanliness Gauge**: Patient clean/dirty status visualization
- **Data Heatmap**: Site DQI scores in a heatmap view
- **Query Velocity Chart**: Query creation vs resolution trends
- **Alerts Panel**: Recent alerts with severity indicators
- **Agent Insights**: AI-generated recommendations

### Patient Management
- **Clean Patient Status**: Track patient cleanliness with blocking factors
- **Blocking Factor Analysis**: Identify issues preventing database lock
- **Lock Readiness Assessment**: Monitor patient data readiness

### Site Performance
- **Site Metrics**: DQI scores, query resolution times
- **High-Risk Site Identification**: Flag underperforming sites
- **CRA Activity Tracking**: Monitor site liaison activities

### AI Agents
- **Agent Status Monitoring**: Real-time agent health
- **Insight Generation**: Automated data quality insights
- **Explainability**: Understand agent decision paths
- **Recommendations**: Actionable suggestions with priority scores

### Real-Time Updates
- **WebSocket Connection**: Live dashboard updates
- **Alert Notifications**: Instant critical alert delivery
- **Patient Status Changes**: Real-time status tracking

## Getting Started

### Backend Setup

```bash
cd clinical_dataflow_optimizer/api

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd clinical_dataflow_optimizer/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000` and will proxy API requests to `http://localhost:8000`.

## API Endpoints

### Dashboard
- `GET /api/health` - Health check
- `GET /api/dashboard/summary` - Dashboard summary

### Studies
- `GET /api/studies` - List all studies
- `GET /api/studies/{study_id}` - Get study details
- `GET /api/studies/{study_id}/metrics` - Get study metrics
- `GET /api/studies/{study_id}/heatmap` - Get heatmap data

### Patients
- `GET /api/patients` - List patients
- `GET /api/patients/{patient_id}` - Get patient details
- `GET /api/patients/{patient_id}/clean-status` - Get clean status
- `GET /api/patients/dirty` - Get dirty patients
- `GET /api/patients/blocking-factors` - Get blocking factor summary

### Sites
- `GET /api/sites` - List sites
- `GET /api/sites/{site_id}` - Get site details
- `GET /api/sites/{site_id}/performance` - Get site performance
- `GET /api/sites/high-risk` - Get high-risk sites

### Metrics
- `GET /api/metrics/kpi-tiles` - Get KPI tile data
- `GET /api/metrics/dqi` - Get DQI metrics
- `GET /api/metrics/cleanliness` - Get cleanliness metrics
- `GET /api/metrics/queries` - Get query metrics
- `GET /api/metrics/velocity` - Get operational velocity

### Alerts
- `GET /api/alerts` - List alerts
- `GET /api/alerts/summary` - Get alert summary
- `GET /api/alerts/critical` - Get critical alerts
- `POST /api/alerts/acknowledge` - Acknowledge alert
- `POST /api/alerts/resolve` - Resolve alert

### Agents
- `GET /api/agents/status` - Get agent status
- `GET /api/agents/insights` - Get AI insights
- `GET /api/agents/recommendations` - Get recommendations
- `GET /api/agents/explain/{insight_id}` - Get insight explanation

### WebSocket
- `WS /ws/dashboard` - Dashboard real-time updates
- `WS /ws/alerts` - Alert notifications
- `WS /ws/patient/{patient_id}` - Patient-specific updates

## Technology Stack

### Backend
- **FastAPI**: Modern async Python web framework
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server for async support
- **WebSockets**: Real-time communication

### Frontend
- **React 18**: UI library with concurrent features
- **TypeScript**: Type-safe JavaScript
- **Ant Design**: Enterprise UI component library
- **React Query**: Server state management
- **Zustand**: Client state management
- **Recharts**: Charting library
- **Vite**: Fast build tool

## Preserving Existing Features

This dashboard is designed to **preserve and integrate** with existing components:

- ✅ **ClinicalDataIngester**: Original data loading pipeline
- ✅ **PatientTwinBuilder**: Patient digital twin construction
- ✅ **DataQualityIndexCalculator**: DQI calculation logic
- ✅ **AI Agents**: Supervisor, Reconciliation, Coding, Site Liaison
- ✅ **Business Rules**: All existing validation and processing logic

The API services act as an **integration layer** that connects the new dashboard to existing backend logic without modifying core business rules.

## Contributing

1. Follow TypeScript strict mode conventions
2. Use async/await for all API calls
3. Maintain type definitions in `types/index.ts`
4. Test WebSocket functionality manually

## License

Internal use only.
